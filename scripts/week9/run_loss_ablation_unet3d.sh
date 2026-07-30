#!/bin/bash
# Run UNet3D loss ablation: L1-only and SSIM-only (same protocol as main: 50 epochs, best by val PSNR).
# Run from repo root. Results go to scripts/week7_results/ (week7_unet3d_l1_only_results.json, week7_unet3d_ssim_only_results.json).
# Then add the test MAE/SSIM/PSNR to the supplementary table (see week9_stats/supplementary_loss_ablation.md).

set -e
ROOT="/data1/julih"
cd "$ROOT/scripts"
PYTHON="${PYTHON:-python3}"

echo "=== UNet3D L1-only (50 epochs) ==="
WEEK7_LOSS=l1_only WEEK7_EPOCHS=50 $PYTHON week7_train_unet3d.py

echo "=== UNet3D SSIM-only (50 epochs) ==="
WEEK7_LOSS=ssim_only WEEK7_EPOCHS=50 $PYTHON week7_train_unet3d.py

echo "=== Done. Check scripts/week7_results/week7_unet3d_l1_only_results.json and week7_unet3d_ssim_only_results.json ==="
echo "Update week9_stats/supplementary_loss_ablation.md with the test MAE, SSIM, PSNR from those files."
