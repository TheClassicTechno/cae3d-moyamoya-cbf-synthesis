#!/usr/bin/env python3
"""
Train UNet 3D for one K-fold fold: train on K-1 folds, validate on the held-out fold.
Test set is never used during training. Saves checkpoint and (optionally) test results.

Usage (run from scripts/):
  python week9/week9_train_unet3d_kfold.py --fold 0 --kfold_splits week9/kfold_splits.json
  python week9/week9_train_unet3d_kfold.py --fold 0 --kfold_splits week9/kfold_splits.json --epochs 2  # quick test

Checkpoint: week7_results/unet3d_fold{N}_best.pt
Test results (if --eval_test): week9/unet3d_fold{N}_test_results.json
"""
import os
import sys
import json
import argparse

# Run from scripts/ so week7_* and week9.kfold_build_splits are importable
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from week7_train_unet3d import (
    make_unet_3d,
    train_epoch,
    evaluate,
    TARGET_3D_PAD,
    OUT_DIR,
    DEVICE,
    BATCH_SIZE,
    LR,
)
from week7_data import get_week7_splits, Week7VolumePairs3D
from week9.kfold_build_splits import get_train_val_pairs_for_fold

# Optional: fewer epochs for quick verification
EPOCHS_DEFAULT = int(os.environ.get("WEEK9_KFOLD_EPOCHS", "50"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True, help="Fold index 0..K-1")
    ap.add_argument("--kfold_splits", default="week9/kfold_splits.json", help="Path to kfold_splits.json (used if WEEK7_SPLIT_PATH not set)")
    ap.add_argument("--epochs", type=int, default=EPOCHS_DEFAULT, help="Max epochs")
    ap.add_argument("--eval_test", action="store_true", help="Run test set eval and save JSON after training")
    ap.add_argument("--out_dir", default="", help="Checkpoint dir (default week7_results)")
    args = ap.parse_args()

    # Full K-fold / per-fold JSON: train+val+test from WEEK7_SPLIT_PATH (see week9_write_kfold_split_files_full.py).
    split_path = os.environ.get("WEEK7_SPLIT_PATH", "").strip()
    if split_path and os.path.isfile(split_path):
        train_pairs, val_pairs, test_pairs = get_week7_splits()
    else:
        kfold_path = os.path.join(SCRIPT_DIR, args.kfold_splits) if not os.path.isabs(args.kfold_splits) else args.kfold_splits
        if not os.path.isfile(kfold_path):
            raise FileNotFoundError("K-fold splits not found: %s (set WEEK7_SPLIT_PATH or pass --kfold_splits)" % kfold_path)
        train_pairs, val_pairs = get_train_val_pairs_for_fold(kfold_path, args.fold)
        _, _, test_pairs = get_week7_splits()

    train_ds = Week7VolumePairs3D(train_pairs, augment=True)
    val_ds = Week7VolumePairs3D(val_pairs, augment=False)
    test_ds = Week7VolumePairs3D(test_pairs, augment=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, num_workers=0)

    out_dir = args.out_dir or OUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    ckpt_name = "unet3d_fold%d_best.pt" % args.fold
    ckpt_path = os.path.join(out_dir, ckpt_name)

    model = make_unet_3d().to(DEVICE)
    criterion_l1 = nn.L1Loss()
    from monai.losses import SSIMLoss
    criterion_ssim = SSIMLoss(spatial_dims=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_psnr = -1.0
    print("Fold %d: train %d val %d test %d" % (args.fold, len(train_pairs), len(val_pairs), len(test_pairs)))
    for ep in range(args.epochs):
        loss = train_epoch(model, train_loader, criterion_l1, criterion_ssim, optimizer, mask_t=None)
        metrics = evaluate(model, val_loader)
        if metrics["psnr_mean"] > best_val_psnr:
            best_val_psnr = metrics["psnr_mean"]
            torch.save({"model": model.state_dict(), "epoch": ep, "fold": args.fold}, ckpt_path)
        if (ep + 1) % 10 == 0:
            print("Fold %d Epoch %d loss=%.4f val MAE=%.4f SSIM=%.4f PSNR=%.2f" % (
                args.fold, ep + 1, loss, metrics["mae_mean"], metrics["ssim_mean"], metrics["psnr_mean"]))
    print("Saved best checkpoint (val PSNR=%.2f) to %s" % (best_val_psnr, ckpt_path))

    if args.eval_test:
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model"])
        test_metrics = evaluate(model, test_loader)
        result_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "unet3d_fold%d_test_results.json" % args.fold)
        with open(result_path, "w") as f:
            json.dump({"fold": args.fold, "mae_mean": test_metrics["mae_mean"], "ssim_mean": test_metrics["ssim_mean"], "psnr_mean": test_metrics["psnr_mean"]}, f, indent=2)
        print("Test (hold-out): MAE=%.4f SSIM=%.4f PSNR=%.2f -> %s" % (
            test_metrics["mae_mean"], test_metrics["ssim_mean"], test_metrics["psnr_mean"], result_path))


if __name__ == "__main__":
    main()
