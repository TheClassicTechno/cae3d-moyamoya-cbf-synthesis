#!/usr/bin/env python3
"""
Train 3D UNet with Week 7 pipeline: combined 2020-2023, brain mask, 91x109x91, same augmentations.
Uses MONAI UNet (spatial_dims=3). Saves checkpoint and metrics to scripts/week7_results/.

Environment (optional):
  WEEK7_EVAL_ONLY=1 — skip training; load WEEK7_CKPT (default week7_unet3d_best.pt) and run test evaluate();
    set WEEK7_EVAL_OUT_JSON to write a structured JSON (e.g. under week9_stats/eval_runs/).
  WEEK7_CKPT_NAME — override checkpoint filename saved during training (avoids overwriting week7_unet3d_best.pt).
  WEEK7_RESULTS_JSON — override path for the final test metrics JSON (basename under OUT_DIR or absolute).
"""
import os
import sys
import json
import random
import time

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
from week7_preprocess import (
    TARGET_SHAPE,
    metrics_in_brain,
    get_brain_mask_for_shape,
    get_region_weight_mask_for_shape,
    load_territory_masks,
    LOW_BASELINE_THRESHOLD_DEFAULT,
    LOW_BASELINE_WEIGHT_DEFAULT,
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

# Week7: 91x109x91; pad to multiple of 8 for UNet -> 96x112x96
TARGET_3D_PAD = (96, 112, 96)
BATCH_SIZE = 2  # 3D is memory heavy
EPOCHS = int(os.environ.get("WEEK7_EPOCHS", "50"))
LR = 1e-3


def make_unet_3d():
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
        pre_t = F.pad(pre_t, pd, mode='constant', value=0)
        post_t = F.pad(post_t, pd, mode='constant', value=0)
    return pre_t[:, :, :th, :tw, :td], post_t[:, :, :th, :tw, :td]


def _pad_3d_mask(mask_t, target_shape):
    import torch.nn.functional as F
    _, _, h, w, d = mask_t.shape
    th, tw, td = target_shape
    if h != th or w != tw or d != td:
        if h < th or w < tw or d < td:
            pd = (0, max(0, td - d), 0, max(0, tw - w), 0, max(0, th - h))
            mask_t = F.pad(mask_t, pd, mode='constant', value=0)
        mask_t = mask_t[:, :, :th, :tw, :td]
    return mask_t


def train_epoch(
    model,
    loader,
    criterion_l1,
    criterion_ssim,
    optimizer,
    mask_t=None,
    use_low_baseline=False,
    low_baseline_threshold=LOW_BASELINE_THRESHOLD_DEFAULT,
    low_baseline_weight=LOW_BASELINE_WEIGHT_DEFAULT,
    brain_mask_t=None,
    region_masks_t=None,
    regional_loss_weight=0.0,
):
    model.train()
    total = 0.0
    n = 0
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
        if use_low_baseline and brain_mask_t is not None:
            # Per-batch weight: down-weight voxels where pre < threshold (low baseline)
            w = torch.where(pre < low_baseline_threshold, low_baseline_weight, 1.0).float() * brain_mask_t
            mask_batch = w
        optimizer.zero_grad()
        out = model(pre)
        loss_type = os.environ.get("WEEK7_LOSS", "l1_ssim").strip().lower()
        if mask_batch is not None:
            l1_masked = (torch.abs(out - post) * mask_batch).sum() / (mask_batch.sum() + 1e-8)
            if loss_type == "l1_only":
                loss = l1_masked
            elif loss_type == "ssim_only":
                loss = criterion_ssim(out, post)
            else:
                loss = l1_masked + criterion_ssim(out, post)
        else:
            if loss_type == "l1_only":
                loss = criterion_l1(out, post)
            elif loss_type == "ssim_only":
                loss = criterion_ssim(out, post)
            else:
                loss = criterion_l1(out, post) + criterion_ssim(out, post)
        if region_masks_t is not None and regional_loss_weight > 0:
            oh, ow, od = TARGET_SHAPE
            pred_crop = out[:, :, :oh, :ow, :od]
            post_crop = post[:, :, :oh, :ow, :od]
            diff = (pred_crop - post_crop).abs()
            # region_masks_t (1, K, 91, 109, 91), diff (B, 1, 91, 109, 91) -> (B, K, 91, 109, 91)
            denom = region_masks_t.sum(dim=(2, 3, 4)).clamp(min=1e-8)
            region_loss = ((diff * region_masks_t).sum(dim=(2, 3, 4)) / denom).mean()
            loss = loss + regional_loss_weight * region_loss.mean()
        loss.backward()
        grad_clip = os.environ.get("WEEK7_GRAD_CLIP", "").strip()
        if grad_clip:
            try:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip))
            except ValueError:
                pass
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


def _eval_only_unet3d():
    """WEEK7_EVAL_ONLY=1: load checkpoint, run test-set metrics (same as post-train evaluate).

    Env:
      WEEK7_CKPT — checkpoint path (default: OUT_DIR/week7_unet3d_best.pt)
      WEEK7_EVAL_OUT_JSON — if set, write JSON with metrics (recommended under week9_stats/eval_runs/)
    """
    _, _, test_pairs = get_week7_splits()
    test_ds = Week7VolumePairs3D(test_pairs, augment=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    model = make_unet_3d().to(DEVICE)
    ckpt_path = os.environ.get("WEEK7_CKPT", "").strip() or os.path.join(OUT_DIR, "week7_unet3d_best.pt")
    if not os.path.isfile(ckpt_path):
        print("Eval-only: checkpoint not found:", ckpt_path)
        sys.exit(1)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt)
    test_metrics = evaluate(model, test_loader)
    print("Eval-only test:", test_metrics, "ckpt=", ckpt_path)
    out_json = os.environ.get("WEEK7_EVAL_OUT_JSON", "").strip()
    if out_json:
        os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
        payload = {
            "model": "week7_unet3d",
            "split": "test",
            "protocol_note": (
                "Metrics use get_week7_splits() test set (2020–2023 combined protocol). "
                "This is not the Week 11 table (2020–2024, seed-aggregated cohort means)."
            ),
            "checkpoint": ckpt_path,
            "test_metrics": test_metrics,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        with open(out_json, "w") as f:
            json.dump(payload, f, indent=2)
        print("Wrote", out_json)


def main():
    if os.environ.get("WEEK7_EVAL_ONLY", "").lower() in ("1", "true", "yes"):
        _eval_only_unet3d()
        return

    use_subject_masks = os.environ.get("WEEK7_SUBJECT_MASKS", "").lower() in ("1", "true", "yes")
    print("Week7 3D UNet: combined 2020-2023, brain mask, 91x109x91, same aug" + (" + subject masks" if use_subject_masks else ""))
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

    model = make_unet_3d().to(DEVICE)
    criterion_l1 = nn.L1Loss()
    criterion_ssim = SSIMLoss(spatial_dims=3)
    use_region_weight = os.environ.get("WEEK7_REGION_WEIGHT", "").lower() in ("1", "true", "yes")
    phase2_lr = os.environ.get("WEEK7_PHASE2_LR", "").strip()
    lr = float(phase2_lr) if (use_region_weight and phase2_lr) else LR
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mask_t = None
    brain_mask_t = None
    use_low_baseline = False
    low_baseline_weight = LOW_BASELINE_WEIGHT_DEFAULT
    low_baseline_threshold = LOW_BASELINE_THRESHOLD_DEFAULT
    if not use_subject_masks and os.environ.get("WEEK7_LOW_BASELINE_WEIGHT", "").strip():
        try:
            low_baseline_weight = float(os.environ.get("WEEK7_LOW_BASELINE_WEIGHT", str(LOW_BASELINE_WEIGHT_DEFAULT)))
        except ValueError:
            low_baseline_weight = LOW_BASELINE_WEIGHT_DEFAULT
        use_low_baseline = True
        brain_mask_np = get_brain_mask_for_shape(TARGET_3D_PAD)
        brain_mask_t = torch.from_numpy(brain_mask_np).float().to(DEVICE).unsqueeze(0).unsqueeze(0)
        print("Low-baseline loss weight: threshold=%.2f weight=%.2f" % (low_baseline_threshold, low_baseline_weight))
    if not use_subject_masks:
        vascular_weight = 1.2 if os.environ.get("WEEK7_REGION_WEIGHT_LOW", "").lower() in ("1", "true", "yes") else 1.5
        mask_np = get_region_weight_mask_for_shape(TARGET_3D_PAD, vascular_weight=vascular_weight) if use_region_weight else get_brain_mask_for_shape(TARGET_3D_PAD)
        mask_t = torch.from_numpy(mask_np).float().to(DEVICE).unsqueeze(0).unsqueeze(0)
        if use_low_baseline and mask_t is not None:
            mask_t = None  # use per-batch low-baseline weight instead

    regional_loss_weight = float(os.environ.get("WEEK7_REGIONAL_LOSS_WEIGHT", "0"))
    region_masks_t = None
    if regional_loss_weight > 0:
        masks_dir = os.path.join(DATA_DIR, "Masks")
        if os.path.isdir(masks_dir):
            territory_list = load_territory_masks(masks_dir, TARGET_SHAPE)
            kept = [(n, m) for n, m in territory_list if m.sum() >= 10]
            if kept:
                stack = np.stack([m for _, m in kept], axis=0).astype(np.float32)
                region_masks_t = torch.from_numpy(stack).float().to(DEVICE).unsqueeze(0)
                print("Regional loss: mu=%.3f, %d territories" % (regional_loss_weight, len(kept)))
            else:
                regional_loss_weight = 0.0
        else:
            regional_loss_weight = 0.0

    # Checkpoint filename: keep week7_unet3d_best.pt for Phase 1 only; variants get distinct files
    loss_type = os.environ.get("WEEK7_LOSS", "l1_ssim").strip().lower()
    use_region_env = os.environ.get("WEEK7_REGION_WEIGHT", "").lower() in ("1", "true", "yes")
    use_phase2_or_3 = use_region_env or use_subject_masks
    if loss_type == "l1_only":
        ckpt_name = "week7_unet3d_best_l1_only.pt"
    elif loss_type == "ssim_only":
        ckpt_name = "week7_unet3d_best_ssim_only.pt"
    elif use_low_baseline:
        ckpt_name = "week7_unet3d_best_lowbaseline.pt"
    elif use_phase2_or_3:
        ckpt_name = "week7_unet3d_best_phase2_phase3.pt"
    else:
        ckpt_name = "week7_unet3d_best.pt"

    if os.environ.get("WEEK7_CKPT_NAME", "").strip():
        ckpt_name = os.environ.get("WEEK7_CKPT_NAME").strip()

    best_val_psnr = -1.0
    log_val = os.environ.get("WEEK7_LOG_VAL", "").lower() in ("1", "true", "yes")
    val_log = []  # list of {epoch, train_loss, val_mae, val_ssim, val_psnr} for Phase 2/3 stability analysis
    for ep in range(EPOCHS):
        loss = train_epoch(
            model,
            train_loader,
            criterion_l1,
            criterion_ssim,
            optimizer,
            mask_t=mask_t,
            use_low_baseline=use_low_baseline,
            low_baseline_threshold=low_baseline_threshold,
            low_baseline_weight=low_baseline_weight,
            brain_mask_t=brain_mask_t,
            region_masks_t=region_masks_t,
            regional_loss_weight=regional_loss_weight,
        )
        metrics = evaluate(model, val_loader)
        if log_val:
            val_log.append({
                "epoch": ep,
                "train_loss": loss,
                "val_mae": metrics["mae_mean"],
                "val_ssim": metrics["ssim_mean"],
                "val_psnr": metrics["psnr_mean"],
            })
        if metrics["psnr_mean"] > best_val_psnr:
            best_val_psnr = metrics["psnr_mean"]
            torch.save({"model": model.state_dict(), "epoch": ep}, os.path.join(OUT_DIR, ckpt_name))
            if os.environ.get("WEEK7_SAVE_SEED_CKPT", "").lower() in ("1", "true", "yes") and ckpt_name == "week7_unet3d_best.pt":
                seed_ckpt = os.path.join(OUT_DIR, "week7_unet3d_seed%d_best.pt" % SEED)
                torch.save({"model": model.state_dict(), "epoch": ep}, seed_ckpt)
                print("Also saved", seed_ckpt)
        if (ep + 1) % 10 == 0:
            print(f"Epoch {ep+1} loss={loss:.4f} val MAE={metrics['mae_mean']:.4f} SSIM={metrics['ssim_mean']:.4f} PSNR={metrics['psnr_mean']:.2f}")

    ckpt_full = os.path.join(OUT_DIR, ckpt_name)
    print("Best checkpoint:", ckpt_full, "best_val_psnr=%.4f" % best_val_psnr)
    ckpt = torch.load(ckpt_full, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    test_metrics = evaluate(model, test_loader)
    print("Test:", test_metrics)
    use_region = os.environ.get("WEEK7_REGION_WEIGHT", "").lower() in ("1", "true", "yes")
    use_subject = os.environ.get("WEEK7_SUBJECT_MASKS", "").lower() in ("1", "true", "yes")
    if loss_type == "l1_only":
        out_name = "week7_unet3d_l1_only_results.json"
    elif loss_type == "ssim_only":
        out_name = "week7_unet3d_ssim_only_results.json"
    else:
        out_name = "week7_unet3d_phase2_phase3_results.json" if (use_region or use_subject) else "week7_unet3d_results.json"
    results_override = os.environ.get("WEEK7_RESULTS_JSON", "").strip()
    if results_override:
        out_path = results_override if os.path.isabs(results_override) else os.path.join(OUT_DIR, results_override)
    else:
        out_path = os.path.join(OUT_DIR, out_name)
    with open(out_path, "w") as f:
        json.dump(test_metrics, f, indent=2)
    print("Saved", out_path)
    if log_val and val_log:
        log_name = "week7_unet3d_phase2_phase3_val_log.json" if (use_region or use_subject) else "week7_unet3d_val_log.json"
        with open(os.path.join(OUT_DIR, log_name), "w") as f:
            json.dump({"epochs": val_log}, f, indent=2)
        print("Saved val log", os.path.join(OUT_DIR, log_name))


if __name__ == "__main__":
    main()
