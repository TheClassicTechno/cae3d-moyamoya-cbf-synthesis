#!/usr/bin/env python3
"""
Regional evaluation for 3D volumes: apply MNI territory masks to prediction and GT,
compute per-region MAE, SSIM, PSNR, and mean perfusion. For evaluation only (not training).

Usage:
  python regional_eval_3d.py --pre-dir <repo-root>/pre --post-dir <repo-root>/post \\
    --pred-dir <repo-root>/NeuralOperators/nifti_exports --masks-dir <repo-root>/Masks \\
    --out regional_results.json

Prediction files should be named post_{id}_pred.nii.gz (and post_{id}_gt.nii.gz optional);
or provide --pred-glob "post_*_pred.nii.gz". Masks are resized to match volume shape.
"""
import os
import sys
import json
import glob
import argparse
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_REPO_ROOT, "pyproject.toml")):
    _parent = os.path.dirname(_REPO_ROOT)
    if _parent == _REPO_ROOT:
        raise RuntimeError("Could not locate repository root (pyproject.toml not found)")
    _REPO_ROOT = _parent

DATA_DIR = _REPO_ROOT
MASKS_DIR_DEFAULT = os.path.join(_REPO_ROOT, "Masks")
# Optional secondary search location (env override)
MASKS_DIR_EXTRA = os.environ.get("MOYAMOYA_MASKS_DIR", "")


def load_nifti(path):
    img = nib.load(path)
    return np.asarray(img.dataobj).astype(np.float32).squeeze()


def load_mask_resized(mask_path, target_shape):
    """Load mask and resize to target_shape (H,W,D). Returns binary (0/1) float32."""
    data = load_nifti(mask_path)
    if data.ndim == 4:
        data = data[..., 0]
    mask = (data > 0).astype(np.float32)
    if mask.shape != target_shape:
        factors = [target_shape[i] / mask.shape[i] for i in range(3)]
        mask = zoom(mask, factors, order=1)
        mask = (mask > 0.5).astype(np.float32)
    return mask


def safe_ssim(gt, pred, mask, data_range=1.0):
    """SSIM only within mask region (crop to bounding box or use full volume with mask weighting)."""
    if mask.sum() < 10:
        return float("nan")
    gt_m = gt * mask
    pred_m = pred * mask
    return ssim(gt_m, pred_m, data_range=data_range)


def safe_psnr(gt, pred, mask, data_range=1.0):
    if mask.sum() < 10:
        return float("nan")
    gt_m = gt[mask > 0.5]
    pred_m = pred[mask > 0.5]
    mse = np.mean((gt_m - pred_m) ** 2)
    if mse <= 0:
        return 60.0
    return 10 * np.log10((data_range ** 2) / mse)


def main():
    ap = argparse.ArgumentParser(description="Regional (left/right ACA, MCA, PCA) evaluation for 3D predictions")
    ap.add_argument("--pre-dir", default=os.path.join(DATA_DIR, "pre"))
    ap.add_argument("--post-dir", default=os.path.join(DATA_DIR, "post"))
    ap.add_argument("--pred-dir", required=True, help="Directory containing post_*_pred.nii.gz (and optionally post_*_gt.nii.gz)")
    ap.add_argument("--masks-dir", default="", help="MNI masks dir (default: MASKS_DIR_DEFAULT, or $MOYAMOYA_MASKS_DIR if set)")
    ap.add_argument("--pred-glob", default="post_*_pred.nii.gz")
    ap.add_argument("--out", default="regional_results.json")
    args = ap.parse_args()

    pred_dir = args.pred_dir
    masks_dir = args.masks_dir or (MASKS_DIR_DEFAULT if os.path.isdir(MASKS_DIR_DEFAULT) else MASKS_DIR_EXTRA)
    if not os.path.isdir(masks_dir):
        print("Masks dir not found:", masks_dir)
        return
    pred_files = sorted(glob.glob(os.path.join(pred_dir, args.pred_glob)))
    if not pred_files:
        print("No prediction files found in", pred_dir, "with glob", args.pred_glob)
        return

    # Mask names (exclude test_set_patients_moss.txt and non-2mm)
    mask_files = sorted([f for f in glob.glob(os.path.join(masks_dir, "*.nii.gz")) if "2mm" in f])
    if not mask_files:
        mask_files = sorted(glob.glob(os.path.join(masks_dir, "MNI_*.nii.gz")))
    mask_names = [os.path.basename(m).replace(".nii.gz", "") for m in mask_files]

    results = {"per_volume": [], "per_region": {}, "mask_names": mask_names}

    for pred_path in pred_files:
        base = os.path.basename(pred_path)
        # post_2021_001_pred.nii.gz -> id 2021_001
        id_part = base.replace("post_", "").replace("_pred.nii.gz", "").replace("_gt.nii.gz", "")
        gt_path = os.path.join(pred_dir, f"post_{id_part}_gt.nii.gz")
        if not os.path.exists(gt_path):
            gt_path = os.path.join(args.post_dir, f"post_{id_part}.nii.gz")
        if not os.path.exists(gt_path):
            continue
        pred_vol = load_nifti(pred_path)
        gt_vol = load_nifti(gt_path)
        if pred_vol.ndim == 4:
            pred_vol = pred_vol[..., 0]
        if gt_vol.ndim == 4:
            gt_vol = gt_vol[..., 0]
        if pred_vol.shape != gt_vol.shape:
            print("Shape mismatch", pred_path, pred_vol.shape, gt_vol.shape)
            continue
        target_shape = pred_vol.shape

        rec = {"id": id_part, "full_mae": float(np.abs(pred_vol - gt_vol).mean()),
               "full_ssim": float(ssim(gt_vol, pred_vol, data_range=1.0)),
               "full_psnr": float(psnr(gt_vol, pred_vol, data_range=1.0))}
        for mname, mpath in zip(mask_names, mask_files):
            mask = load_mask_resized(mpath, target_shape)
            n = float(mask.sum())
            if n < 10:
                rec[mname] = {"mae": None, "ssim": None, "psnr": None, "mean_perfusion_pred": None, "mean_perfusion_gt": None}
                continue
            mae = float((np.abs(pred_vol - gt_vol) * mask).sum() / n)
            rec[mname] = {
                "mae": mae,
                "ssim": float(safe_ssim(gt_vol, pred_vol, mask)),
                "psnr": float(safe_psnr(gt_vol, pred_vol, mask)),
                "mean_perfusion_pred": float((pred_vol * mask).sum() / n),
                "mean_perfusion_gt": float((gt_vol * mask).sum() / n),
                "n_voxels": int(n),
            }
        results["per_volume"].append(rec)

    # Aggregate per-region over volumes
    for mname in mask_names:
        maes, ssims, psnrs = [], [], []
        for rec in results["per_volume"]:
            r = rec.get(mname, {})
            if r.get("mae") is not None:
                maes.append(r["mae"])
            if r.get("ssim") is not None and not np.isnan(r["ssim"]):
                ssims.append(r["ssim"])
            if r.get("psnr") is not None and not np.isnan(r["psnr"]):
                psnrs.append(r["psnr"])
        results["per_region"][mname] = {
            "mae_mean": float(np.mean(maes)) if maes else None,
            "mae_std": float(np.std(maes)) if maes else None,
            "ssim_mean": float(np.mean(ssims)) if ssims else None,
            "psnr_mean": float(np.mean(psnrs)) if psnrs else None,
            "n_volumes": len(maes),
        }

    out_path = args.out if os.path.isabs(args.out) else os.path.join(os.path.dirname(pred_dir), args.out)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print("Regional evaluation done. Results:", out_path)
    for mname in mask_names:
        r = results["per_region"][mname]
        if r["n_volumes"]:
            print("  %s: MAE %.4f  SSIM %.4f  PSNR %.2f dB (n=%d)" % (
                mname, r["mae_mean"] or 0, r["ssim_mean"] or 0, r["psnr_mean"] or 0, r["n_volumes"]))


if __name__ == "__main__":
    main()
