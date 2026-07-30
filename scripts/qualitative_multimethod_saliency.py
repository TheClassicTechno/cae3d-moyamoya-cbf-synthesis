#!/usr/bin/env python3
"""
Multi-method saliency comparison grid — single-subject figure.

Layout (2 rows x 4 cols):
  Row 1: Pre-ACZ | GT post-ACZ | CAE3D pred | Pred - GT (signed error)
  Row 2: Vanilla Gradient | SmoothGrad | Input x Gradient | Guided Backprop | GradCAM
         (we render the four saliency methods plus optionally GradCAM)

The figure mirrors the layout of multi-method saliency comparison figures in
the medical-imaging literature (e.g., binary brain-tumour classification
saliency overviews) but adapted for a 3D regression model: the target for
all gradient-based methods is the mean of in-brain predicted CBF.

Usage:
  PYTHONNOUSERSITE=1 \\
  PYTHONPATH=/data1/julih/scripts:/data1/julih/UNet_3D \\
  /data1/julih/miniconda3/envs/julih_monai/bin/python \\
      scripts/qualitative_multimethod_saliency.py \\
          [--subject_id 2020_042] [--model "CAE3D baseline"] [--axial_slice 45]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

ROOT = "/data1/julih"
SCRIPTS = os.path.join(ROOT, "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

import eval_comprehensive_all_models as ev


# --------------------------------------------------------------------------- #
# Saliency / attribution implementations                                       #
# --------------------------------------------------------------------------- #

def _score(pred, mask):
    """Regression target = mean predicted CBF inside the brain mask."""
    return (pred * mask).sum() / mask.sum().clamp(min=1.0)


def vanilla_gradient(model, x, mask):
    """|d score / d x| — basic saliency (Simonyan et al. 2013)."""
    import torch
    x = x.clone().detach().requires_grad_(True)
    model.zero_grad()
    pred = model(x)
    s = _score(pred, mask)
    s.backward()
    g = x.grad.detach().squeeze().cpu().numpy().astype(np.float32)
    return np.abs(g)


def smoothgrad(model, x, mask, n_samples=15, sigma=0.10):
    """Smilkov et al. 2017 — average |grad| over noisy versions of x."""
    import torch
    accum = None
    rng = np.random.default_rng(42)
    for _ in range(n_samples):
        noise = torch.from_numpy(
            rng.normal(0.0, sigma, size=tuple(x.shape)).astype(np.float32)
        ).to(x.device)
        g = vanilla_gradient(model, x + noise, mask)
        accum = g if accum is None else accum + g
    return accum / n_samples


def input_times_gradient(model, x, mask):
    """Shrikumar et al. — input * gradient, then |.| for visualisation."""
    import torch
    x_d = x.clone().detach().requires_grad_(True)
    model.zero_grad()
    pred = model(x_d)
    s = _score(pred, mask)
    s.backward()
    ig = (x_d * x_d.grad).detach().squeeze().cpu().numpy().astype(np.float32)
    return np.abs(ig)


def integrated_gradients(model, x, mask, baseline=None, n_steps=30):
    """
    Integrated Gradients (Sundararajan et al. 2017):
      IG_i(x) = (x_i - x_baseline_i) * integral_{a=0}^{1} d(F(x_baseline + a*(x - x_baseline))) / dx_i  da

    Numerical approximation: sample n_steps interpolations between baseline and x,
    compute gradient at each, average, then multiply by (x - baseline).

    Default baseline is zeros (standard choice for normalized inputs).

    Less prone to checkerboard artifacts than vanilla gradient because (a) the
    integration averages many gradient evaluations, and (b) the (x - baseline)
    multiplier suppresses voxels with zero input intensity (outside brain / padding).
    """
    import torch
    if baseline is None:
        baseline = torch.zeros_like(x)
    diff = x - baseline
    accum_grad = torch.zeros_like(x)
    for i in range(n_steps):
        alpha = float(i + 1) / n_steps   # avoid alpha=0 (baseline) for stability
        x_a = (baseline + alpha * diff).detach().requires_grad_(True)
        model.zero_grad()
        pred = model(x_a)
        s = _score(pred, mask)
        s.backward()
        accum_grad = accum_grad + x_a.grad.detach()
    avg_grad = accum_grad / n_steps
    ig = (diff * avg_grad).squeeze().cpu().numpy().astype(np.float32)
    return np.abs(ig)


def guided_backprop(model, x, mask):
    """
    Guided Backprop (Springenberg et al. 2014):
    in the backward pass through every activation module, zero out gradients
    that are negative (so only positive evidence flows back).

    We register backward hooks on every LeakyReLU/ReLU module. The model in
    this codebase uses inplace=True LeakyReLU, which conflicts with backward
    hooks (autograd raises a "view+inplace" error). We therefore temporarily
    flip inplace -> False on all activation modules and restore afterwards.
    """
    import torch
    import torch.nn as nn

    handles = []
    saved_inplace = []  # (module, prev_inplace_flag)

    def hook(module, grad_in, grad_out):
        # Clip negative gradients to zero (standard guided BP rule)
        return tuple(
            None if g is None else g.clamp(min=0.0)
            for g in grad_in
        )

    for m in model.modules():
        if isinstance(m, (nn.ReLU, nn.LeakyReLU)):
            saved_inplace.append((m, getattr(m, "inplace", False)))
            m.inplace = False
            handles.append(m.register_full_backward_hook(hook))

    try:
        x_d = x.clone().detach().requires_grad_(True)
        model.zero_grad()
        pred = model(x_d)
        s = _score(pred, mask)
        s.backward()
        g = x_d.grad.detach().squeeze().cpu().numpy().astype(np.float32)
    finally:
        for h in handles:
            h.remove()
        for m, prev in saved_inplace:
            m.inplace = prev

    return np.abs(g)


def gradcam(model, x, mask, layer_name="model.1.submodule.1.submodule.1.submodule"):
    """3D regression GradCAM at the bottleneck."""
    import torch
    import torch.nn.functional as F

    named = dict(model.named_modules())
    if layer_name not in named:
        raise RuntimeError(f"Layer '{layer_name}' not found in model")
    layer = named[layer_name]

    state = {"acts": None, "grads": None}
    h_f = layer.register_forward_hook(lambda m, i, o: state.update(acts=o))
    h_b = layer.register_full_backward_hook(lambda m, gi, go: state.update(grads=go[0]))

    try:
        x_d = x.clone().detach().requires_grad_(True)
        model.zero_grad()
        pred = model(x_d)
        s = _score(pred, mask)
        s.backward()

        weights = state["grads"].mean(dim=(2, 3, 4), keepdim=True)
        cam = (weights * state["acts"]).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam_up = F.interpolate(cam, size=x.shape[2:], mode="trilinear", align_corners=False)
        out = cam_up.squeeze().detach().cpu().numpy().astype(np.float32)
    finally:
        h_f.remove(); h_b.remove()

    return out


def normalize_minmax(a, percentile=99.0):
    """Normalize to [0,1] using a percentile cap to suppress outliers."""
    a = a.astype(np.float32)
    if a.size == 0:
        return a
    hi = np.percentile(np.abs(a), percentile)
    if hi <= 0:
        return np.zeros_like(a)
    a = np.clip(a / hi, -1.0, 1.0)
    a = np.abs(a)  # methods are abs-valued at this point
    return a


def smooth_saliency(a, sigma=1.0):
    """Gaussian-blur a saliency map to suppress strided-conv checkerboard
    artifacts. Cosmetic only — does not alter the regional attention pattern."""
    if sigma <= 0:
        return a
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(a.astype(np.float32), sigma=sigma)


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--subject_id", default="2020_042",
                   help="Subject ID from the 32-subject test fold.")
    p.add_argument("--model", default="CAE3D baseline")
    p.add_argument("--axial_slice", type=int, default=None)
    p.add_argument("--out", default=os.path.join(ROOT, "results", "qualitative", "multimethod_saliency.png"))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--blur_sigma", type=float, default=1.0,
                   help="Gaussian-blur sigma (voxels) applied to saliency maps "
                        "before display to suppress strided-conv checkerboard "
                        "artifacts. 0 disables. Default 1.0.")
    p.add_argument("--ig_steps", type=int, default=30,
                   help="Integration steps for Integrated Gradients.")
    args = p.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    pairs = ev.load_pairs_32()
    match = [(sid, pre, post) for sid, pre, post in pairs if sid == args.subject_id]
    if not match:
        print(f"ERROR: subject {args.subject_id!r} not in test fold.", file=sys.stderr)
        return 1
    sid, pre_p, post_p = match[0]

    meta = next((m for m in ev.MODELS if m["name"] == args.model), None)
    if meta is None:
        print(f"ERROR: model {args.model!r} not found.", file=sys.stderr)
        return 1
    print(f"Loading {meta['name']} ({meta['arch']}) ...")
    model = ev.load_model(meta["arch"], meta["ckpt"], args.device)
    model.eval()

    pre_padded  = ev.load_and_prep(pre_p)
    post_padded = ev.load_and_prep(post_p)
    mask_padded = ev.load_brain_mask()

    import torch
    x_t = torch.from_numpy(pre_padded).unsqueeze(0).unsqueeze(0).to(args.device)
    m_t = torch.from_numpy(mask_padded).unsqueeze(0).unsqueeze(0).to(args.device)

    # Forward pass for prediction (no grad)
    with torch.no_grad():
        pred_padded = model(x_t).squeeze().cpu().numpy().astype(np.float32)

    # Compute saliency maps
    print("\nComputing saliency methods:")
    methods = {}

    print("  [1/5] Vanilla Gradient ...")
    methods["Vanilla Gradient"] = vanilla_gradient(model, x_t, m_t)

    print("  [2/5] SmoothGrad      ...")
    methods["SmoothGrad"] = smoothgrad(model, x_t, m_t, n_samples=15, sigma=0.10)

    print("  [3/5] Integrated Gradients ...")
    methods["Integrated Gradients"] = integrated_gradients(model, x_t, m_t, n_steps=args.ig_steps)

    print("  [4/5] Guided Backprop ...")
    methods["Guided Backprop"] = guided_backprop(model, x_t, m_t)

    print("  [5/5] GradCAM         ...")
    methods["GradCAM"] = gradcam(model, x_t, m_t)

    # Crop everything from PAD_SHAPE -> TARGET_SHAPE
    h, w, d = ev.TARGET_SHAPE
    pre  = pre_padded[:h, :w, :d]
    gt   = post_padded[:h, :w, :d]
    pred = pred_padded[:h, :w, :d]
    mask = mask_padded[:h, :w, :d]
    sal_maps = {name: m[:h, :w, :d] * mask for name, m in methods.items()}

    z = args.axial_slice if args.axial_slice is not None else d // 2

    # SSIM/MAE for figure title
    metrics = ev.metrics_in_brain(pred, gt, mask)
    print(f"\nSubject {sid}: SSIM={metrics['ssim']:.4f}  MAE={metrics['mae']:.4f}  PSNR={metrics['psnr']:.2f}")

    # --------------------------------------------------------------------------- #
    # Figure                                                                       #
    # --------------------------------------------------------------------------- #

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    cmap_data = "viridis"
    cmap_err  = "magma"
    cmap_diff = "RdBu_r"

    # 3x3 layout:
    #   Row 1: Pre-ACZ | GT post-ACZ | CAE3D pred
    #   Row 2: Vanilla Grad | SmoothGrad | Integrated Gradients
    #   Row 3: Guided Backprop | GradCAM | Pred − GT (signed)
    fig, axes = plt.subplots(3, 3, figsize=(11, 11))

    def show_grayscale(ax, vol, title):
        slc = (vol * mask)[:, :, z].T[::-1]
        ax.imshow(slc, cmap=cmap_data, vmin=0, vmax=1, aspect="equal")
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    # Row 1: input + GT + prediction
    show_grayscale(axes[0, 0], pre,  f"Pre-ACZ\n({sid})")
    show_grayscale(axes[0, 1], gt,   "GT post-ACZ")
    show_grayscale(axes[0, 2], pred, f"{args.model}\nSSIM={metrics['ssim']:.3f}  MAE={metrics['mae']:.3f}")

    # Saliency normalisation (shared for all 5 methods)
    sal_norm = Normalize(vmin=0.0, vmax=1.0)
    sal_cmap = plt.get_cmap(cmap_err)

    def draw_saliency(ax, name):
        anat_slc = (pre * mask)[:, :, z].T[::-1]
        ax.imshow(anat_slc, cmap="gray", vmin=0, vmax=1, aspect="equal")
        # Apply Gaussian blur BEFORE percentile-normalisation
        sal = smooth_saliency(sal_maps[name], sigma=args.blur_sigma)
        sal = normalize_minmax(sal)
        sal_slc = sal[:, :, z].T[::-1]
        rgba = sal_cmap(sal_norm(sal_slc))
        rgba[..., -1] = np.clip(sal_slc, 0, 1) * 0.75
        ax.imshow(rgba, aspect="equal")
        ax.set_title(name, fontsize=10)
        ax.axis("off")

    # Row 2: gradient-based saliency family
    draw_saliency(axes[1, 0], "Vanilla Gradient")
    draw_saliency(axes[1, 1], "SmoothGrad")
    draw_saliency(axes[1, 2], "Integrated Gradients")

    # Row 3: Guided BP, GradCAM, signed prediction error
    draw_saliency(axes[2, 0], "Guided Backprop")
    draw_saliency(axes[2, 1], "GradCAM")

    # Signed difference panel
    diff = (pred - gt) * mask
    diff_max = float(np.percentile(np.abs(diff), 99))
    if diff_max <= 0: diff_max = 0.3
    diff_norm = Normalize(vmin=-diff_max, vmax=diff_max)
    ax = axes[2, 2]
    slc = diff[:, :, z].T[::-1]
    ax.imshow(slc, cmap=cmap_diff, norm=diff_norm, aspect="equal")
    ax.set_title("Pred − GT (signed)", fontsize=10)
    ax.axis("off")

    # Colorbars
    cbar_a = fig.add_axes([0.92, 0.10, 0.013, 0.22])
    sm_a = ScalarMappable(norm=diff_norm, cmap=cmap_diff); sm_a.set_array([])
    cb_a = fig.colorbar(sm_a, cax=cbar_a)
    cb_a.set_label("Pred − GT", fontsize=9)

    cbar_b = fig.add_axes([0.92, 0.40, 0.013, 0.45])
    sm_b = ScalarMappable(norm=sal_norm, cmap=cmap_err); sm_b.set_array([])
    cb_b = fig.colorbar(sm_b, cax=cbar_b)
    cb_b.set_label("Saliency (norm.)", fontsize=9)

    title = (
        f"Multi-method saliency comparison — {args.model}, subject {sid}, axial slice z={z}\n"
        f"(saliency maps Gaussian-blurred with σ={args.blur_sigma} voxel "
        f"to suppress strided-conv checkerboard artifacts; "
        f"IG = {args.ig_steps} integration steps)"
    )
    plt.suptitle(title, fontsize=10)
    plt.tight_layout(rect=[0, 0, 0.91, 0.93])
    plt.savefig(args.out, dpi=300, bbox_inches="tight")
    pdf = os.path.splitext(args.out)[0] + ".pdf"
    plt.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved: {args.out}")
    print(f"Saved: {pdf}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
