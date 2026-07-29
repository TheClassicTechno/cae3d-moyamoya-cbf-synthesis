#!/usr/bin/env python3
"""
Train 2D UNet with Week 7 pipeline: combined 2020-2023, brain mask, 91x109, same augmentations.
Uses MONAI UNet (spatial_dims=2). Saves checkpoint and metrics to scripts/week7_results/.
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

# Add parent so we can import from week7_data (same dir) and optionally UNet_baseline
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from week7_data import get_week7_splits, Week7SlicePairs2D
from week7_preprocess import metrics_in_brain_2d, get_brain_mask_for_shape, get_region_weight_mask_for_shape

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

# Week7: middle slice is 91x109; pad to multiple of 8 for UNet (2^3 downsampling) -> 96x112
TARGET_2D = (96, 112)
BATCH_SIZE = 8
EPOCHS = int(os.environ.get("WEEK7_EPOCHS", "50"))
LR = 1e-3


def make_unet_2d():
    return UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128),
        strides=(2, 2, 2),
        num_res_units=2,
        act=("LeakyReLU", {"inplace": True}),
        norm="INSTANCE",
        dropout=0.0,
    )


def _pad_to_target(pre_t, post_t, target_hw):
    """Pad (B, 1, 91, 109) to (B, 1, th, tw) with zeros for UNet divisibility."""
    import torch.nn.functional as F
    th, tw = target_hw
    _, _, h, w = pre_t.shape
    if h < th or w < tw:
        pd = (0, max(0, tw - w), 0, max(0, th - h))  # last dim then second-to-last
        pre_t = F.pad(pre_t, pd, mode='constant', value=0)
        post_t = F.pad(post_t, pd, mode='constant', value=0)
    return pre_t[:, :, :th, :tw], post_t[:, :, :th, :tw]


def train_epoch(model, loader, criterion_l1, criterion_ssim, optimizer, mask_t=None):
    model.train()
    total = 0.0
    n = 0
    for pre, post in loader:
        pre, post = _pad_to_target(pre, post, TARGET_2D)
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
    model.eval()
    mae_list, ssim_list, psnr_list = [], [], []
    with torch.no_grad():
        for pre, post in loader:
            pre, post = _pad_to_target(pre, post, TARGET_2D)
            pre, post = pre.to(DEVICE), post.to(DEVICE)
            pred = model(pre)
            for i in range(pred.shape[0]):
                p = pred[i, 0].cpu().numpy()
                t = post[i, 0].cpu().numpy()
                m = metrics_in_brain_2d(p, t, data_range=1.0)
                mae_list.append(m["mae_mean"])
                ssim_list.append(m["ssim_mean"])
                psnr_list.append(m["psnr_mean"])
    return {
        "mae_mean": float(np.mean(mae_list)),
        "ssim_mean": float(np.mean(ssim_list)),
        "psnr_mean": float(np.mean(psnr_list)),
    }


def main():
    print("Week7 2D UNet: combined 2020-2023, brain mask, 91x109, same aug")
    train_pairs, val_pairs, test_pairs = get_week7_splits()
    train_ds = Week7SlicePairs2D(train_pairs, augment=True)
    val_ds = Week7SlicePairs2D(val_pairs, augment=False)
    test_ds = Week7SlicePairs2D(test_pairs, augment=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, num_workers=0)

    model = make_unet_2d().to(DEVICE)
    criterion_l1 = nn.L1Loss()
    criterion_ssim = SSIMLoss(spatial_dims=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    use_region_weight = os.environ.get("WEEK7_REGION_WEIGHT", "").lower() in ("1", "true", "yes")
    mask_np = get_region_weight_mask_for_shape(TARGET_2D, vascular_weight=1.5) if use_region_weight else get_brain_mask_for_shape(TARGET_2D)
    mask_t = torch.from_numpy(mask_np).float().to(DEVICE).unsqueeze(0).unsqueeze(0)

    best_val_psnr = -1.0
    for ep in range(EPOCHS):
        loss = train_epoch(model, train_loader, criterion_l1, criterion_ssim, optimizer, mask_t=mask_t)
        metrics = evaluate(model, val_loader)
        if metrics["psnr_mean"] > best_val_psnr:
            best_val_psnr = metrics["psnr_mean"]
            torch.save({"model": model.state_dict(), "epoch": ep}, os.path.join(OUT_DIR, "week7_unet2d_best.pt"))
        if (ep + 1) % 10 == 0:
            print(f"Epoch {ep+1} loss={loss:.4f} val MAE={metrics['mae_mean']:.4f} SSIM={metrics['ssim_mean']:.4f} PSNR={metrics['psnr_mean']:.2f}")

    # Load best and evaluate on test
    ckpt = torch.load(os.path.join(OUT_DIR, "week7_unet2d_best.pt"), map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    test_metrics = evaluate(model, test_loader)
    print("Test:", test_metrics)
    use_region = os.environ.get("WEEK7_REGION_WEIGHT", "").lower() in ("1", "true", "yes")
    out_name = "week7_unet2d_phase2_results.json" if use_region else "week7_unet2d_results.json"
    with open(os.path.join(OUT_DIR, out_name), "w") as f:
        json.dump(test_metrics, f, indent=2)
    print("Saved", os.path.join(OUT_DIR, out_name))


if __name__ == "__main__":
    main()
