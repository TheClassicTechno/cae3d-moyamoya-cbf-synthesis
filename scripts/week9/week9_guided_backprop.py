#!/usr/bin/env python3
"""
Guided backpropagation for 3D UNet or ResNet3D: attribute predicted post-ACZ to pre-ACZ input.
Only positive gradients are propagated; gradient w.r.t. input = attribution map.
Usage (from repo root):
  python scripts/week9/week9_guided_backprop.py --model unet3d --output_dir week9_stats/guided_backprop
  python scripts/week9/week9_guided_backprop.py --model resnet3d --checkpoint scripts/week7_results/week7_resnet3d_best.pt --output_dir week9_stats/guided_backprop_resnet3d
  python scripts/week9/week9_guided_backprop.py --subject_id 2022_046 --n_subjects 5 --save_nifti
  python scripts/week9/week9_guided_backprop.py --n_subjects 1 --export_paper_panel week9_stats/guided_backprop_unet3d_panel.png
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from pipeline_model_registry import PRIMARY_RECON_CHECKPOINT

TARGET_3D_PAD = (96, 112, 96)


def _pad_3d(pre_t, post_t, target_shape):
    import torch.nn.functional as F
    _, _, h, w, d = pre_t.shape
    th, tw, td = target_shape
    if h < th or w < tw or d < td:
        pd = (0, max(0, td - d), 0, max(0, tw - w), 0, max(0, th - h))
        pre_t = F.pad(pre_t, pd, mode="constant", value=0)
        post_t = F.pad(post_t, pd, mode="constant", value=0)
    return pre_t[:, :, :th, :tw, :td], post_t[:, :, :th, :tw, :td]


def _disable_inplace_activations(model):
    """Avoid view+inplace conflict with backward hooks (PyTorch forbids modifying hook output inplace)."""
    for module in model.modules():
        if isinstance(module, nn.ReLU):
            module.inplace = False
        elif isinstance(module, nn.LeakyReLU):
            module.inplace = False


def register_guided_backprop_hooks(model):
    """Only positive gradients flow back (guided backprop). Return new tensor to avoid view/inplace error."""
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.ReLU, nn.LeakyReLU)):
            def _hook(m, gi, go):
                g = go[0]
                out = torch.where(g > 0, g, torch.zeros_like(g, device=g.device)).clone()
                return (out,)
            h = module.register_full_backward_hook(_hook)
            handles.append(h)
    return handles


def run_guided_backprop(model, pre_t, brain_mask_t, device):
    model.eval()
    pre_t = pre_t.to(device).requires_grad_(True)
    if brain_mask_t is not None:
        brain_mask_t = brain_mask_t.to(device)
    handles = register_guided_backprop_hooks(model)
    try:
        pred = model(pre_t)
        scalar = (pred * brain_mask_t).sum() / (brain_mask_t.sum() + 1e-8) if brain_mask_t is not None else pred.mean()
        scalar.backward()
        out = pre_t.grad.detach().cpu().float().numpy() if pre_t.grad is not None else np.zeros_like(pre_t.cpu().numpy())
    finally:
        for h in handles:
            h.remove()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unet3d", choices=("unet3d", "resnet3d"), help="Model architecture")
    ap.add_argument(
        "--checkpoint",
        default="",
        help="Path to .pt; default: PRIMARY_RECON_CHECKPOINT (unet3d) or week7_results/week7_<model>_best.pt",
    )
    ap.add_argument("--output_dir", default=str(ROOT / "week9_stats" / "guided_backprop"))
    ap.add_argument("--subject_id", default="")
    ap.add_argument("--n_subjects", type=int, default=5)
    ap.add_argument("--save_nifti", action="store_true")
    ap.add_argument(
        "--export_paper_panel",
        default="",
        help="Path to save 3-panel PNG for paper: pre-ACZ | attribution overlay | post-ACZ (first processed subject, or --subject_id). Example: week9_stats/guided_backprop_unet3d_panel.png",
    )
    args = ap.parse_args()
    if args.checkpoint.strip():
        ckpt_path = Path(args.checkpoint)
    elif args.model == "unet3d":
        ckpt_path = Path(PRIMARY_RECON_CHECKPOINT)
    else:
        ckpt_path = ROOT / "scripts" / "week7_results" / ("week7_%s_best.pt" % args.model)
    if not ckpt_path.is_file():
        print("Checkpoint not found:", ckpt_path)
        return 1
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.model == "unet3d":
        from monai.networks.nets import UNet
        model = UNet(
            spatial_dims=3, in_channels=1, out_channels=1,
            channels=(16, 32, 64, 128), strides=(2, 2, 2), num_res_units=2,
            act=("LeakyReLU", {"inplace": True}), norm="INSTANCE", dropout=0.0,
        )
    else:
        from week7_train_resnet3d import ResNet3DCVR
        model = ResNet3DCVR(pretrained=False)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)
    _disable_inplace_activations(model)
    model = model.to(device).eval()
    from week7_data import Week7VolumePairs3D
    from week7_preprocess import TARGET_SHAPE, get_brain_mask, get_pre_post_pairs_with_subject_id
    # Use canonical subject_id from split so each test subject gets a unique filename (e.g. 2022_046).
    test_triples = get_pre_post_pairs_with_subject_id("test")  # (subject_id, pre_path, post_path)
    test_pairs = [(p, q) for (_, p, q) in test_triples]
    test_sids = [sid for (sid, _, _) in test_triples]
    test_ds = Week7VolumePairs3D(test_pairs, augment=False)
    mask_vol = get_brain_mask()
    import torch.nn.functional as F
    mask_t = torch.from_numpy(mask_vol).float().unsqueeze(0).unsqueeze(0)
    _, _, h, w, d = mask_t.shape
    th, tw, td = TARGET_3D_PAD
    if h < th or w < tw or d < td:
        pd = (0, max(0, td - d), 0, max(0, tw - w), 0, max(0, th - h))
        mask_t = F.pad(mask_t, pd, mode="constant", value=0)
    mask_t = mask_t[:, :, :th, :tw, :td]
    count = 0
    for idx in range(len(test_pairs)):
        if count >= args.n_subjects and not args.subject_id:
            break
        sid = test_sids[idx]
        if args.subject_id and sid != args.subject_id:
            continue
        pre_t, post_t = test_ds[idx]
        pre_t, post_t = _pad_3d(pre_t.unsqueeze(0), post_t.unsqueeze(0), TARGET_3D_PAD)
        attribution = run_guided_backprop(model, pre_t, mask_t, device)
        att_np = attribution[0, 0]
        if att_np.shape != TARGET_SHAPE:
            att_np = att_np[:TARGET_SHAPE[0], :TARGET_SHAPE[1], :TARGET_SHAPE[2]]
        att_flat = att_np.ravel()
        att_flat = att_flat[np.isfinite(att_flat)]
        p99 = np.percentile(att_flat, 99) if att_flat.size > 0 else 1.0
        att_vis = np.clip(att_np / (p99 + 1e-8), 0, 1).astype(np.float32)
        mid_slice = att_vis.shape[2] // 2
        # Same MNI crop as attribution (model I/O may be padded to TARGET_3D_PAD)
        zh, zw, zd = TARGET_SHAPE
        pre_vol = pre_t[0, 0, :zh, :zw, :zd].detach().cpu().numpy()
        post_vol = post_t[0, 0, :zh, :zw, :zd].detach().cpu().numpy()
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            ax.imshow(att_vis[:, :, mid_slice].T, origin="lower", cmap="hot", aspect="equal")
            ax.set_title("Guided backprop " + sid)
            ax.axis("off")
            fig.savefig(out_dir / ("guided_backprop_%s.png" % sid), dpi=150, bbox_inches="tight")
            plt.close(fig)
            print("Wrote", out_dir / ("guided_backprop_%s.png" % sid))
            if args.export_paper_panel.strip() and (args.subject_id == sid or (not args.subject_id and count == 0)):
                pre_s = pre_vol[:, :, mid_slice].T
                post_s = post_vol[:, :, mid_slice].T
                att_s = att_vis[:, :, mid_slice].T
                pre_rgb = np.stack([pre_s, pre_s, pre_s], axis=-1)
                hot = plt.cm.hot(np.clip(att_s / (att_s.max() + 1e-8), 0, 1))[:, :, :3]
                overlay = np.clip(0.55 * pre_rgb + 0.45 * hot, 0, 1)
                panel_path = Path(args.export_paper_panel.strip())
                panel_path.parent.mkdir(parents=True, exist_ok=True)
                fig2, axes = plt.subplots(1, 3, figsize=(5.5, 1.75), constrained_layout=True)
                titles = ("Pre-ACZ (input)", "Attribution on pre-ACZ", "Post-ACZ (reference)")
                axes[0].imshow(pre_s, origin="lower", cmap="gray", aspect="equal")
                axes[1].imshow(overlay, origin="lower", aspect="equal")
                axes[2].imshow(post_s, origin="lower", cmap="gray", aspect="equal")
                for ax2, ttl in zip(axes, titles):
                    ax2.set_title(ttl, fontsize=16)
                    ax2.axis("off")
                fig2.suptitle("UNet3D guided backprop: " + sid, fontsize=14)
                fig2.savefig(panel_path, dpi=220, bbox_inches="tight")
                plt.close(fig2)
                print("Paper panel:", panel_path)
        except Exception as e:
            print("Plot failed:", e)
        if args.save_nifti:
            try:
                import nibabel as nib
                nib.save(nib.Nifti1Image(att_vis, np.eye(4)), str(out_dir / ("guided_backprop_%s.nii.gz" % sid)))
            except Exception:
                pass
        count += 1
        if args.subject_id:
            break
    print("Done.", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
