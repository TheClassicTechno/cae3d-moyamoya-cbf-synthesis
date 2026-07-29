#!/usr/bin/env python3
"""
NOT the pre→post CBF finetuning baseline — do not use these metrics next to CAE3D Table 1 rows.

This script: pretrained SAM-Med3D **inference** on pre volumes vs **brain mask** overlap (Dice/IoU).
It does **not** train to predict **post-ACZ** from pre-ACZ.

For **fair** SAM comparison to CAE3D, use:
  third_party_foundation_3d/run_sam_med3d_week7_cbf_regressor.py
See: third_party_foundation_3d/FOUNDATION_PRE_TO_POST_FINETUNING.md

---
Run SAM-Med3D on full Week7 test set; compute overlap + legacy pixel metrics (pred vs Week7 brain mask).
Saves third_party_foundation_3d/sam_med3d_week7_results.json.
MAE/SSIM/PSNR on binary maps are not comparable to continuous CBF metrics (see JSON note).
"""
import glob
import os
import re
import sys
import json
import shutil
import numpy as np
import nibabel as nib

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_REPO_ROOT, "pyproject.toml")):
    _parent = os.path.dirname(_REPO_ROOT)
    if _parent == _REPO_ROOT:
        raise RuntimeError("Could not locate repository root (pyproject.toml not found)")
    _REPO_ROOT = _parent

SAM_ROOT = os.path.dirname(os.path.abspath(__file__))
from foundation_mask_diagnostic_metrics import (  # noqa: E402
    PIXEL_METRICS_NOTE,
    align_pred_to_shape,
    mask_diagnostic_metrics,
)
WEEK7_SPLIT = os.path.join(_REPO_ROOT, "combined_subject_split.json")
BRAIN_MASK = os.path.join(_REPO_ROOT, "MNI152_T1_2mm_brain_mask_dil.nii.gz")
OUT_DIR = os.path.join(SAM_ROOT, "test_data", "week7_sam_test_set")
IMAGES_DIR = os.path.join(OUT_DIR, "imagesVa")
LABELS_DIR = os.path.join(OUT_DIR, "labelsVa")
PRED_DIR = os.path.join(OUT_DIR, "pred")
RESULTS_JSON = os.path.join(SAM_ROOT, "sam_med3d_week7_results.json")


def load_nii(path):
    return np.asarray(nib.load(path).get_fdata()).squeeze().astype(np.float32)


def main():
    with open(WEEK7_SPLIT) as f:
        data = json.load(f)
    pairs = data.get("test", [])
    if not pairs:
        print("No Week7 test pairs.")
        sys.exit(1)

    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(LABELS_DIR, exist_ok=True)
    os.makedirs(PRED_DIR, exist_ok=True)

    sam_dir = os.path.join(SAM_ROOT, "SAM-Med3D")
    if os.path.isdir(sam_dir):
        sys.path.insert(0, sam_dir)
    else:
        sys.path.insert(0, SAM_ROOT)
    try:
        import medim
        from utils.infer_utils import validate_paired_img_gt
    except ImportError as e:
        print("Import error:", e)
        sys.exit(1)

    ckpt_path = "https://huggingface.co/blueyo0/SAM-Med3D/blob/main/sam_med3d_turbo.pth"
    for loc in [os.path.join(SAM_ROOT, "SAM-Med3D", "ckpt", "sam_med3d_turbo.pth"),
                os.path.join(SAM_ROOT, "ckpt", "sam_med3d_turbo.pth")]:
        if os.path.isfile(loc):
            ckpt_path = loc
            break
    print("Loading SAM-Med3D...")
    model = medim.create_model("SAM-Med3D", pretrained=True, checkpoint_path=ckpt_path)

    n = len(pairs)
    print("Running inference on", n, "test volumes...")
    for i, item in enumerate(pairs):
        pre_path = item.get("pre_path")
        if not pre_path or not os.path.isfile(pre_path):
            continue
        base = os.path.splitext(os.path.splitext(os.path.basename(pre_path))[0])[0]
        sid = f"test_{i:03d}_{base}"[:60]
        img_dest = os.path.join(IMAGES_DIR, sid + ".nii.gz")
        gt_dest = os.path.join(LABELS_DIR, sid + ".nii.gz")
        out_path = os.path.join(PRED_DIR, sid + ".nii.gz")
        shutil.copy2(pre_path, img_dest)
        shutil.copy2(BRAIN_MASK, gt_dest)
        validate_paired_img_gt(model, img_dest, gt_dest, out_path, num_clicks=1)
        if (i + 1) % 8 == 0:
            print("  ", i + 1, "/", n)

    # Metrics: match preds by test index (test_000_*.nii.gz) so basename changes in split do not break pairing
    pred_by_i: dict[int, str] = {}
    for path in glob.glob(os.path.join(PRED_DIR, "test_*.nii.gz")):
        m = re.match(r"test_(\d+)_", os.path.basename(path))
        if m:
            pred_by_i[int(m.group(1))] = path

    mae_list, ssim_list, psnr_list, dice_list, iou_list = [], [], [], [], []
    for i, item in enumerate(pairs):
        pre_path = item.get("pre_path")
        if not pre_path or not os.path.isfile(pre_path):
            continue
        pred_path = pred_by_i.get(i)
        if not pred_path or not os.path.isfile(pred_path):
            continue
        pred = load_nii(pred_path)
        pre_vol = load_nii(pre_path)
        if pred.shape != pre_vol.shape:
            pred = align_pred_to_shape(pred, pre_vol.shape)
        pred = np.clip(pred, 0, 1).astype(np.float32)
        m = mask_diagnostic_metrics(pred)
        mae_list.append(m["mae"])
        ssim_list.append(m["ssim"])
        psnr_list.append(m["psnr"])
        dice_list.append(m["dice"])
        iou_list.append(m["iou"])

    if not mae_list:
        print("No predictions found.")
        sys.exit(1)

    results = {
        "model": "SAM-Med3D",
        "task": "mask_vs_brain_mask",
        "note": "Segmentation (mask). Primary: Dice/IoU vs Week7 brain mask on pre grid. Not CBF.",
        "primary_metrics": ["dice", "iou"],
        "pixel_image_metrics_note": PIXEL_METRICS_NOTE,
        "n_test": len(mae_list),
        "mae_mean": float(np.mean(mae_list)),
        "mae_std": float(np.std(mae_list)),
        "ssim_mean": float(np.mean(ssim_list)),
        "ssim_std": float(np.std(ssim_list)),
        "psnr_mean": float(np.mean(psnr_list)),
        "psnr_std": float(np.std(psnr_list)),
        "dice_mean": float(np.mean(dice_list)),
        "dice_std": float(np.std(dice_list)),
        "iou_mean": float(np.mean(iou_list)),
        "iou_std": float(np.std(iou_list)),
    }
    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print("Results:", results)
    print("Saved", RESULTS_JSON)


if __name__ == "__main__":
    main()
