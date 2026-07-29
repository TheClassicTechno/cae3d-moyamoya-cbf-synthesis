#!/usr/bin/env python3
"""
Week 9: Best-case figure (pre-ACZ, predicted post-ACZ, ground-truth post-ACZ) for paper.

Runs inference with a script-based model (unet3d or resnet3d) on the test set, selects
the subject with best PSNR (or SSIM), and saves one figure: three panels (pre, pred, GT)
middle axial slice + caption with MAE, SSIM, PSNR.

Usage (from repo root):
  python scripts/week9/week9_best_case_figure.py --model unet3d --output_dir week9_stats
  python scripts/week9/week9_best_case_figure.py --model resnet3d --output_dir week9_stats
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path("<repo-root>")
sys.path.insert(0, str(ROOT / "scripts"))
from pipeline_model_registry import PRIMARY_RECON_CHECKPOINT
from week7_data import get_week7_splits, _subject_id_from_path, Week7VolumePairs3D
from week7_preprocess import TARGET_SHAPE, load_volume

DATA_DIR = ROOT
TARGET_3D_PAD = (96, 112, 96)


def _crop(vol):
    h, w, d = TARGET_SHAPE
    if vol.shape == TARGET_SHAPE:
        return vol
    return vol[:h, :w, :d].copy()


def main():
    ap = argparse.ArgumentParser(description="Best-case figure: pre / pred / GT slice")
    ap.add_argument("--model", default="unet3d", choices=("unet3d", "resnet3d"))
    ap.add_argument("--checkpoint", default="", help="Path to .pt; default: week7_results/week7_<model>_best.pt")
    ap.add_argument("--output_dir", default=str(ROOT / "week9_stats"))
    ap.add_argument("--pick_by", default="psnr", choices=("psnr", "ssim"), help="Pick best subject by PSNR or SSIM")
    args = ap.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.checkpoint.strip():
        ckpt = args.checkpoint
    elif args.model == "unet3d":
        ckpt = str(PRIMARY_RECON_CHECKPOINT)
    else:
        ckpt = str(ROOT / "scripts" / "week7_results" / ("week7_%s_best.pt" % args.model))
    if not os.path.isfile(ckpt):
        print("Checkpoint not found:", ckpt)
        return

    import torch
    from torch.utils.data import DataLoader
    from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr

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
    model.load_state_dict(torch.load(ckpt, map_location=device)["model"])
    model = model.to(device).eval()

    _, _, test_pairs = get_week7_splits()
    test_ds = Week7VolumePairs3D(test_pairs, augment=False)

    def _pad_3d(pre_t, post_t, target_shape):
        import torch.nn.functional as F
        _, _, h, w, d = pre_t.shape
        th, tw, td = target_shape
        if h < th or w < tw or d < td:
            pd = (0, max(0, td - d), 0, max(0, tw - w), 0, max(0, th - h))
            pre_t = F.pad(pre_t, pd, mode="constant", value=0)
            post_t = F.pad(post_t, pd, mode="constant", value=0)
        return pre_t[:, :, :th, :tw, :td], post_t[:, :, :th, :tw, :td]

    best_sid = None
    best_val = -1.0
    best_pre = best_pred = best_gt = None
    best_mae = best_ssim = best_psnr = None
    with torch.no_grad():
        for idx in range(len(test_pairs)):
            pre_t, post_t = test_ds[idx]
            pre_t = pre_t.unsqueeze(0).to(device)
            post_t = post_t.unsqueeze(0).to(device)
            pre_t, post_t = _pad_3d(pre_t, post_t, TARGET_3D_PAD)
            pred_t = model(pre_t)
            pred_vol = _crop(pred_t[0, 0].cpu().numpy())
            gt_vol = _crop(post_t[0, 0].cpu().numpy())
            pre_vol = _crop(pre_t[0, 0].cpu().numpy())
            sid = _subject_id_from_path(test_pairs[idx][0])
            mae = float(np.abs(pred_vol - gt_vol).mean())
            ssim_val = float(ssim(gt_vol, pred_vol, data_range=1.0))
            psnr_val = float(psnr(gt_vol, pred_vol, data_range=1.0))
            val = psnr_val if args.pick_by == "psnr" else ssim_val
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
        print("No test subject processed.")
        return

    # Middle axial slice (D//2)
    sl = TARGET_SHAPE[2] // 2
    pre_sl = best_pre[:, :, sl]
    pred_sl = best_pred[:, :, sl]
    gt_sl = best_gt[:, :, sl]
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping figure.")
        return
    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    for ax, img, title in zip(axes, [pre_sl, pred_sl, gt_sl], ["Pre-ACZ", "Predicted post-ACZ", "Ground-truth post-ACZ"]):
        ax.imshow(img.T, origin="lower", cmap="gray", vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle("Best case (%s): %s  |  MAE=%.4f  SSIM=%.4f  PSNR=%.2f dB" % (args.pick_by, best_sid, best_mae, best_ssim, best_psnr), fontsize=10)
    out_path = out_dir / ("best_case_%s_%s.png" % (args.model, best_sid))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Wrote", out_path)


if __name__ == "__main__":
    main()
