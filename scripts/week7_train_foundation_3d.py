#!/usr/bin/env python3
"""
Minimal "foundation-style" 3D trainer for Week7 pre->post (from opensourcemodels.txt).
Uses a small 3D CNN backbone; can be swapped for Triad/Med3DVLM/STU-Net encoder once installed.
Same data pipeline as week7_train_unet3d: get_week7_splits(), Week7VolumePairs3D, 96x112x96 pad.
Saves scripts/week7_results/week7_foundation3d_best.pt and week7_foundation3d_results.json.
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
from week7_data import get_week7_splits, Week7VolumePairs3D, Week7VolumePairs3DWithMasks
from week7_preprocess import TARGET_SHAPE, metrics_in_brain, get_brain_mask_for_shape, get_region_weight_mask_for_shape

DATA_DIR = _REPO_ROOT
OUT_DIR = os.path.join(DATA_DIR, "scripts", "week7_results")
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

TARGET_3D_PAD = (96, 112, 96)
BATCH_SIZE = 2
EPOCHS = 40
LR = 1e-3


class Small3DCNN(nn.Module):
    """Small 3D CNN for pre->post (foundation-style backbone; replace with Triad/Med3DVLM encoder if desired)."""

    def __init__(self, in_ch=1, out_ch=1, base=32):
        super().__init__()
        self.encoder = nn.Sequential(
            _conv3(in_ch, base, 3, 1, 1),
            _conv3(base, base * 2, 3, 2, 1),
            _conv3(base * 2, base * 4, 3, 2, 1),
            _conv3(base * 4, base * 4, 3, 1, 1),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(base * 4, base * 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(base * 2, base, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(base, out_ch, 3, padding=1),
        )

    def forward(self, x):
        f = self.encoder(x)
        return self.decoder(f)


def _conv3(in_c, out_c, k, s, p):
    return nn.Sequential(
        nn.Conv3d(in_c, out_c, k, stride=s, padding=p),
        nn.BatchNorm3d(out_c),
        nn.ReLU(inplace=True),
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


def _pad_3d_mask(mask_t, target_shape):
    import torch.nn.functional as F
    _, _, h, w, d = mask_t.shape
    th, tw, td = target_shape
    if h != th or w != tw or d != td:
        if h < th or w < tw or d < td:
            pd = (0, max(0, td - d), 0, max(0, tw - w), 0, max(0, th - h))
            mask_t = F.pad(mask_t, pd, mode="constant", value=0)
        mask_t = mask_t[:, :, :th, :tw, :td]
    return mask_t


def train_epoch(model, loader, criterion_l1, criterion_ssim, optimizer, mask_t=None):
    model.train()
    total, n = 0.0, 0
    for batch in loader:
        if len(batch) == 3:
            pre, post, mask_batch = batch
            pre, post = _pad_3d(pre, post, TARGET_3D_PAD)
            mask_batch = _pad_3d_mask(mask_batch.to(DEVICE), TARGET_3D_PAD)
        else:
            pre, post = batch
            pre, post = _pad_3d(pre, post, TARGET_3D_PAD)
            mask_batch = mask_t
        pre, post = pre.to(DEVICE), post.to(DEVICE)
        optimizer.zero_grad()
        out = model(pre)
        if mask_batch is not None:
            l1_masked = (torch.abs(out - post) * mask_batch).sum() / (mask_batch.sum() + 1e-8)
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
        for batch in loader:
            pre, post = batch[0], batch[1]
            pre, post = _pad_3d(pre, post, TARGET_3D_PAD)
            pre, post = pre.to(DEVICE), post.to(DEVICE)
            pred = model(pre)
            for i in range(pred.shape[0]):
                p = pred[i, 0].cpu().numpy()
                t = post[i, 0].cpu().numpy()
                m = metrics_in_brain(p, t, data_range=1.0)
                mae_list.append(m["mae_mean"])
                ssim_list.append(m["ssim_mean"])
                psnr_list.append(m["psnr_mean"])
    return {
        "mae_mean": float(np.mean(mae_list)),
        "ssim_mean": float(np.mean(ssim_list)),
        "psnr_mean": float(np.mean(psnr_list)),
    }


def main():
    use_subject_masks = os.environ.get("WEEK7_SUBJECT_MASKS", "").lower() in ("1", "true", "yes")
    print("Week7 foundation-style 3D (small 3D CNN); same data as UNet 3D." + (" + subject masks" if use_subject_masks else ""))
    train_pairs, val_pairs, test_pairs = get_week7_splits()
    if use_subject_masks:
        train_ds = Week7VolumePairs3DWithMasks(train_pairs, augment=True, pad_shape=TARGET_3D_PAD)
        val_ds = Week7VolumePairs3DWithMasks(val_pairs, augment=False, pad_shape=TARGET_3D_PAD)
        test_ds = Week7VolumePairs3DWithMasks(test_pairs, augment=False, pad_shape=TARGET_3D_PAD)
    else:
        train_ds = Week7VolumePairs3D(train_pairs, augment=True)
        val_ds = Week7VolumePairs3D(val_pairs, augment=False)
        test_ds = Week7VolumePairs3D(test_pairs, augment=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, num_workers=0)

    model = Small3DCNN(in_ch=1, out_ch=1, base=32).to(DEVICE)
    criterion_l1 = nn.L1Loss()
    try:
        from monai.losses import SSIMLoss
        criterion_ssim = SSIMLoss(spatial_dims=3)
    except Exception:
        criterion_ssim = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    mask_t = None
    if not use_subject_masks:
        use_region_weight = os.environ.get("WEEK7_REGION_WEIGHT", "").lower() in ("1", "true", "yes")
        mask_np = get_region_weight_mask_for_shape(TARGET_3D_PAD, vascular_weight=1.5) if use_region_weight else get_brain_mask_for_shape(TARGET_3D_PAD)
        mask_t = torch.from_numpy(mask_np).float().to(DEVICE).unsqueeze(0).unsqueeze(0)

    best_val_psnr = -1.0
    for ep in range(EPOCHS):
        loss = train_epoch(model, train_loader, criterion_l1, criterion_ssim, optimizer, mask_t=mask_t)
        metrics = evaluate(model, val_loader)
        if metrics["psnr_mean"] > best_val_psnr:
            best_val_psnr = metrics["psnr_mean"]
            torch.save(
                {"model": model.state_dict(), "epoch": ep},
                os.path.join(OUT_DIR, "week7_foundation3d_best.pt"),
            )
        if (ep + 1) % 10 == 0:
            print(
                f"Epoch {ep+1} loss={loss:.4f} val MAE={metrics['mae_mean']:.4f} SSIM={metrics['ssim_mean']:.4f} PSNR={metrics['psnr_mean']:.2f}"
            )

    ckpt = torch.load(os.path.join(OUT_DIR, "week7_foundation3d_best.pt"), map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    test_metrics = evaluate(model, test_loader)
    print("Test:", test_metrics)
    use_region = os.environ.get("WEEK7_REGION_WEIGHT", "").lower() in ("1", "true", "yes")
    out_name = "week7_foundation3d_phase2_phase3_results.json" if (use_region or use_subject_masks) else "week7_foundation3d_results.json"
    with open(os.path.join(OUT_DIR, out_name), "w") as f:
        json.dump(test_metrics, f, indent=2)
    print("Saved", os.path.join(OUT_DIR, out_name))


if __name__ == "__main__":
    main()
