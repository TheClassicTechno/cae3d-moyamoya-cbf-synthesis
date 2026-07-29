#!/usr/bin/env python3
"""
Step 2: Brain-only crop experiment - train 3D UNet on cropped volumes (same architecture, same eval definition).

- Data: Week7VolumePairs3DCropped (load 91x109x91, mask, crop to brain bbox, pad to CROP_PAD_SHAPE).
- Model: Same MONAI UNet as week7_train_unet3d.py but input size = CROP_PAD_SHAPE.
- Eval: At test time, run model on cropped pre; place prediction back into full 91x109x91; run metrics_in_brain
  so results are comparable to full-volume runs (same metric definition).

Usage:
  cd <repo-root>/scripts && python week7_train_unet3d_crop.py
  SEED=42 python week7_train_unet3d_crop.py

Results: week7_results/week7_unet3d_crop_results.json
"""
import os
import sys
import json
import random

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_REPO_ROOT, "pyproject.toml")):
    _parent = os.path.dirname(_REPO_ROOT)
    if _parent == _REPO_ROOT:
        raise RuntimeError("Could not locate repository root (pyproject.toml not found)")
    _REPO_ROOT = _parent

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from week7_data import get_week7_splits, Week7VolumePairs3DCropped
from week7_preprocess import (
    TARGET_SHAPE,
    metrics_in_brain,
    get_brain_bounding_box,
    get_brain_mask,
    get_brain_crop_shape,
    load_volume,
    load_volume_cropped,
)

from monai.networks.nets import UNet
from monai.losses import SSIMLoss

DATA_DIR = _REPO_ROOT
OUT_DIR = os.path.join(DATA_DIR, "scripts", "week7_results")
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = int(os.environ.get("SEED", 42))
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Brain crop: pad to multiple of 8 for UNet. Crop shape from MNI mask; pad at least that.
_CROP_RAW = get_brain_crop_shape()
CROP_PAD_SHAPE = tuple(max(_CROP_RAW[i], ((_CROP_RAW[i] + 7) // 8) * 8) for i in range(3))
CROP_PAD_SHAPE = tuple(min(96, CROP_PAD_SHAPE[i]) for i in range(3))

BATCH_SIZE = 2
EPOCHS = int(os.environ.get("WEEK7_EPOCHS", "50"))
LR = 1e-3


def make_unet_3d(spatial_shape):
    """UNet that accepts spatial_shape (D, H, W) - same architecture as full-volume, different input size."""
    return UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128),
        strides=(2, 2, 2),
        num_res_units=2,
        act=("LeakyReLU", {"inplace": True}),
        norm="INSTANCE",
        dropout=0.0,
    )


def _pad_3d(pre_t, post_t, target_shape):
    import torch.nn.functional as F
    _, _, h, w, d = pre_t.shape
    th, tw, td = target_shape
    if h < th or w < tw or d < td:
        pd = (0, max(0, td - d), 0, max(0, tw - w), 0, max(0, th - h))
        pre_t = F.pad(pre_t, pd, mode="constant", value=0)
        post_t = F.pad(post_t, pd, mode="constant", value=0)
    return pre_t[:, :, :th, :tw, :td], post_t[:, :, :th, :tw, :td]


def train_epoch(model, loader, criterion_l1, criterion_ssim, optimizer, mask_t=None):
    model.train()
    total = 0.0
    n = 0
    for batch in loader:
        pre, post = batch
        pre, post = _pad_3d(pre, post, CROP_PAD_SHAPE)
        pre, post = pre.to(DEVICE), post.to(DEVICE)
        optimizer.zero_grad()
        out = model(pre)
        if mask_t is not None:
            l1_masked = (torch.abs(out - post) * mask_t).sum() / (mask_t.sum() + 1e-8)
            loss = l1_masked + criterion_ssim(out, post)
        else:
            loss = criterion_l1(out, post) + criterion_ssim(out, post)
        loss.backward()
        optimizer.step()
        total += loss.item()
        n += 1
    return total / max(n, 1)


def evaluate(model, loader):
    """Run model on cropped input; place pred back into full volume; metrics_in_brain(full_pred, full_GT)."""
    model.eval()
    mae_list, ssim_list, psnr_list = [], [], []
    sl_d, sl_h, sl_w = get_brain_bounding_box(get_brain_mask())
    crop_shape = (sl_d.stop - sl_d.start, sl_h.stop - sl_h.start, sl_w.stop - sl_w.start)

    with torch.no_grad():
        for batch in loader:
            pre, post = batch
            pre, _ = _pad_3d(pre, post, CROP_PAD_SHAPE)
            pre = pre.to(DEVICE)
            pred = model(pre)
            pred = pred.cpu().numpy()

            for i in range(pred.shape[0]):
                pred_crop = pred[i, 0]  # (D, H, W) in CROP_PAD_SHAPE
                # Take only the crop region (same size as crop_shape; may be smaller than CROP_PAD)
                cd, ch, cw = crop_shape
                pred_crop = pred_crop[:cd, :ch, :cw]

                # Load full-volume GT for this sample (we need full 91x109x91 post)
                # Batch from Week7VolumePairs3DCropped doesn't have paths; we need full-volume test set
                # So we evaluate by loading test pairs again as full volume for GT.
                # Simpler: pass test_loader that yields (pre_crop, post_full) or we do two loaders.
                # Easiest: use same test_loader (cropped), compute metrics only inside crop (crop is all brain).
                # Then metrics are "brain-only in crop" which is comparable.
                t_full = post[i, 0].cpu().numpy()  # (D,H,W) padded to CROP_PAD_SHAPE
                t_crop = t_full[:cd, :ch, :cw]
                p_crop = pred_crop[:cd, :ch, :cw]
                m = metrics_in_brain(p_crop, t_crop, data_range=1.0)
                mae_list.append(m["mae_mean"])
                ssim_list.append(m["ssim_mean"])
                psnr_list.append(m["psnr_mean"])
    return {
        "mae_mean": float(np.mean(mae_list)),
        "ssim_mean": float(np.mean(ssim_list)),
        "psnr_mean": float(np.mean(psnr_list)),
    }


def evaluate_full_volume_metric(model, test_pairs):
    """Load each test sample as full volume; run model on cropped pre; fill full-volume pred; metrics_in_brain."""
    model.eval()
    mae_list, ssim_list, psnr_list = [], [], []
    sl_d, sl_h, sl_w = get_brain_bounding_box(get_brain_mask())
    cd, ch, cw = sl_d.stop - sl_d.start, sl_h.stop - sl_h.start, sl_w.stop - sl_w.start

    with torch.no_grad():
        for pre_path, post_path in test_pairs:
            pre_crop = load_volume_cropped(pre_path, pad_to_shape=CROP_PAD_SHAPE)
            post_full = load_volume(post_path)
            pre_t = torch.from_numpy(pre_crop).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
            pred = model(pre_t)
            pred_np = pred[0, 0].cpu().numpy()[:cd, :ch, :cw]

            full_pred = np.zeros(TARGET_SHAPE, dtype=np.float32)
            full_pred[sl_d, sl_h, sl_w] = pred_np
            m = metrics_in_brain(full_pred, post_full, data_range=1.0)
            mae_list.append(m["mae_mean"])
            ssim_list.append(m["ssim_mean"])
            psnr_list.append(m["psnr_mean"])

    return {
        "mae_mean": float(np.mean(mae_list)),
        "ssim_mean": float(np.mean(ssim_list)),
        "psnr_mean": float(np.mean(psnr_list)),
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-only", action="store_true", help="Load best.pt and run test eval only (no training)")
    args = ap.parse_args()

    print("Step 2: UNet 3D brain-only crop experiment (same architecture, crop train, full-volume metric)")
    print("CROP_PAD_SHAPE =", CROP_PAD_SHAPE, "| crop shape =", _CROP_RAW)
    train_pairs, val_pairs, test_pairs = get_week7_splits()

    ckpt_path = os.path.join(OUT_DIR, "week7_unet3d_crop_best.pt")
    if args.eval_only:
        if not os.path.isfile(ckpt_path):
            print("No checkpoint found at", ckpt_path)
            return
        model = make_unet_3d(CROP_PAD_SHAPE).to(DEVICE)
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model"])
        test_metrics_crop = evaluate(model, DataLoader(
            Week7VolumePairs3DCropped(test_pairs, augment=False, crop_pad_shape=CROP_PAD_SHAPE),
            batch_size=BATCH_SIZE, num_workers=0,
        ))
        test_metrics_full = evaluate_full_volume_metric(model, test_pairs)
        print("Test (metrics in crop):", test_metrics_crop)
        print("Test (full-volume metric):", test_metrics_full)
        out = {
            "model": "UNet3D_crop",
            "crop_pad_shape": [int(x) for x in CROP_PAD_SHAPE],
            "seed": int(SEED),
            "test_in_crop": {k: float(v) for k, v in test_metrics_crop.items()},
            "mae_mean": float(test_metrics_full["mae_mean"]),
            "ssim_mean": float(test_metrics_full["ssim_mean"]),
            "psnr_mean": float(test_metrics_full["psnr_mean"]),
        }
        out_path = os.path.join(OUT_DIR, "week7_unet3d_crop_results.json")
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print("Saved", out_path)
        return
    train_ds = Week7VolumePairs3DCropped(train_pairs, augment=True, crop_pad_shape=CROP_PAD_SHAPE)
    val_ds = Week7VolumePairs3DCropped(val_pairs, augment=False, crop_pad_shape=CROP_PAD_SHAPE)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=0)

    model = make_unet_3d(CROP_PAD_SHAPE).to(DEVICE)
    criterion_l1 = nn.L1Loss()
    criterion_ssim = SSIMLoss(spatial_dims=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    # Crop is already brain-only; no extra mask needed (or use ones)
    mask_t = None

    best_val_psnr = -1.0
    for ep in range(EPOCHS):
        loss = train_epoch(model, train_loader, criterion_l1, criterion_ssim, optimizer, mask_t=mask_t)
        metrics = evaluate(model, val_loader)
        if metrics["psnr_mean"] > best_val_psnr:
            best_val_psnr = metrics["psnr_mean"]
            torch.save(
                {"model": model.state_dict(), "epoch": ep, "crop_pad_shape": CROP_PAD_SHAPE},
                os.path.join(OUT_DIR, "week7_unet3d_crop_best.pt"),
            )
        if (ep + 1) % 10 == 0:
            print(
                "Epoch %d loss=%.4f val MAE=%.4f SSIM=%.4f PSNR=%.2f"
                % (ep + 1, loss, metrics["mae_mean"], metrics["ssim_mean"], metrics["psnr_mean"])
            )

    ckpt = torch.load(os.path.join(OUT_DIR, "week7_unet3d_crop_best.pt"), map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    test_metrics_crop = evaluate(model, DataLoader(
        Week7VolumePairs3DCropped(test_pairs, augment=False, crop_pad_shape=CROP_PAD_SHAPE),
        batch_size=BATCH_SIZE, num_workers=0,
    ))
    test_metrics_full = evaluate_full_volume_metric(model, test_pairs)
    print("Test (metrics in crop):", test_metrics_crop)
    print("Test (full-volume metric, comparable to baseline):", test_metrics_full)

    out = {
        "model": "UNet3D_crop",
        "crop_pad_shape": list(CROP_PAD_SHAPE),
        "seed": SEED,
        "test_in_crop": test_metrics_crop,
        "mae_mean": test_metrics_full["mae_mean"],
        "ssim_mean": test_metrics_full["ssim_mean"],
        "psnr_mean": test_metrics_full["psnr_mean"],
    }
    out_path = os.path.join(OUT_DIR, "week7_unet3d_crop_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print("Saved", out_path)


if __name__ == "__main__":
    main()
