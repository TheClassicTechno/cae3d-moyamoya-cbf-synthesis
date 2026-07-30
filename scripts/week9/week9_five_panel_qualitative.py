#!/usr/bin/env python3
"""
Five-panel qualitative figure for Statistical validation slide: pre-ACZ, post-ACZ GT, prediction, brain mask, error map.
Runs UNet 3D on test set, picks best subject by PSNR, saves one figure (middle axial slice).

Usage (from repo root):
  python scripts/week9/week9_five_panel_qualitative.py --output_dir scripts/week9/slide_visuals
  python scripts/week9/week9_five_panel_qualitative.py --subject_id 2022_046  # optional: specific subject
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/data1/julih")
sys.path.insert(0, str(ROOT / "scripts"))
from pipeline_model_registry import PRIMARY_RECON_CHECKPOINT
from week7_preprocess import (
    TARGET_SHAPE,
    USE_AFFINE,
    get_brain_mask,
    get_week7_splits,
    load_pre_post_pair,
    _subject_id_from_path,
)

TARGET_3D_PAD = (96, 112, 96)


def _crop(vol):
    h, w, d = TARGET_SHAPE
    if vol.shape == TARGET_SHAPE:
        return vol
    return vol[:h, :w, :d].copy()


def main():
    ap = argparse.ArgumentParser(description="Five-panel: pre, post GT, pred, mask, error")
    ap.add_argument("--checkpoint", default="", help="Path to .pt; default: PRIMARY_RECON_CHECKPOINT (week7_unet3d)")
    ap.add_argument("--output_dir", default=str(ROOT / "scripts" / "week9" / "slide_visuals"))
    ap.add_argument("--pick_by", default="psnr", choices=("psnr", "ssim"))
    ap.add_argument("--subject_id", default="", help="If set, use this subject instead of best by metric")
    args = ap.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = args.checkpoint or str(PRIMARY_RECON_CHECKPOINT)
    if not os.path.isfile(ckpt):
        print("Checkpoint not found:", ckpt)
        return

    import torch
    from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from monai.networks.nets import UNet
    model = UNet(
        spatial_dims=3, in_channels=1, out_channels=1,
        channels=(16, 32, 64, 128), strides=(2, 2, 2), num_res_units=2,
        act=("LeakyReLU", {"inplace": True}), norm="INSTANCE", dropout=0.0,
    )
    model.load_state_dict(torch.load(ckpt, map_location=device)["model"])
    model = model.to(device).eval()

    _, _, test_pairs = get_week7_splits()

    def _pair_to_tensors(idx):
        pre_path, post_path = test_pairs[idx]
        pre, post = load_pre_post_pair(
            pre_path, post_path, use_affine=USE_AFFINE, target_shape=TARGET_SHAPE
        )
        pre_t = torch.from_numpy(pre).unsqueeze(0).float()
        post_t = torch.from_numpy(post).unsqueeze(0).float()
        return pre_t, post_t

    def _pad_3d(pre_t, post_t, target_shape):
        import torch.nn.functional as F
        _, _, h, w, d = pre_t.shape
        th, tw, td = target_shape
        if h < th or w < tw or d < td:
            pd = (0, max(0, td - d), 0, max(0, tw - w), 0, max(0, th - h))
            pre_t = F.pad(pre_t, pd, mode="constant", value=0)
            post_t = F.pad(post_t, pd, mode="constant", value=0)
        return pre_t[:, :, :th, :tw, :td], post_t[:, :, :th, :tw, :td]

    target_sid = args.subject_id.strip()
    best_sid = None
    best_val = -1.0
    best_pre = best_pred = best_gt = None
    best_mae = best_ssim = best_psnr = None

    with torch.no_grad():
        for idx in range(len(test_pairs)):
            pre_t, post_t = _pair_to_tensors(idx)
            sid = _subject_id_from_path(test_pairs[idx][0])
            if target_sid and sid != target_sid:
                continue
            pre_t = pre_t.unsqueeze(0).to(device)
            post_t = post_t.unsqueeze(0).to(device)
            pre_t, post_t = _pad_3d(pre_t, post_t, TARGET_3D_PAD)
            pred_t = model(pre_t)
            pred_vol = _crop(pred_t[0, 0].cpu().numpy())
            gt_vol = _crop(post_t[0, 0].cpu().numpy())
            pre_vol = _crop(pre_t[0, 0].cpu().numpy())
            mae = float(np.abs(pred_vol - gt_vol).mean())
            ssim_val = float(ssim(gt_vol, pred_vol, data_range=1.0))
            psnr_val = float(psnr(gt_vol, pred_vol, data_range=1.0))
            val = psnr_val if args.pick_by == "psnr" else ssim_val
            if not target_sid and val <= best_val:
                continue
            if target_sid:
                best_val = val
                best_sid = sid
                best_pre = pre_vol.copy()
                best_pred = pred_vol.copy()
                best_gt = gt_vol.copy()
                best_mae = mae
                best_ssim = ssim_val
                best_psnr = psnr_val
                break
            if val > best_val:
                best_val = val
                best_sid = sid
                best_pre = pre_vol.copy()
                best_pred = pred_vol.copy()
                best_gt = gt_vol.copy()
                best_mae = mae
                best_ssim = ssim_val
                best_psnr = psnr_val

    if best_sid is None:
        print("No test subject found.")
        return

    mask_vol = get_brain_mask()
    if mask_vol.shape != TARGET_SHAPE:
        from scipy.ndimage import zoom
        mask_vol = zoom(mask_vol, [TARGET_SHAPE[i] / mask_vol.shape[i] for i in range(3)], order=0)
    mask_vol = mask_vol[:TARGET_SHAPE[0], :TARGET_SHAPE[1], :TARGET_SHAPE[2]]
    error_vol = np.abs(best_pred.astype(np.float64) - best_gt.astype(np.float64))

    sl = TARGET_SHAPE[2] // 2  # middle axial
    pre_sl = best_pre[:, :, sl]
    gt_sl = best_gt[:, :, sl]
    pred_sl = best_pred[:, :, sl]
    mask_sl = mask_vol[:, :, sl]
    error_sl = error_vol[:, :, sl]

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available")
        return

    fig, axes = plt.subplots(1, 5, figsize=(6.5, 2.0))
    for ax, img, title in [
        (axes[0], pre_sl, "Pre-ACZ"),
        (axes[1], gt_sl, "Post-ACZ (GT)"),
        (axes[2], pred_sl, "Prediction"),
        (axes[3], mask_sl, "Brain mask"),
        (axes[4], error_sl, "Error |pred−GT|"),
    ]:
        if "Error" in title:
            im = ax.imshow(img.T, origin="lower", cmap="hot", vmin=0, vmax=0.3)
        else:
            im = ax.imshow(img.T, origin="lower", cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=16, fontweight="semibold")
        ax.axis("off")
    fig.suptitle(
        "Qualitative example (%s): %s  |  MAE=%.4f  SSIM=%.4f  PSNR=%.2f dB"
        % (args.pick_by, best_sid, best_mae, best_ssim, best_psnr),
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    out_path = out_dir / "five_panel_qualitative.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("Saved", out_path)


if __name__ == "__main__":
    main()
