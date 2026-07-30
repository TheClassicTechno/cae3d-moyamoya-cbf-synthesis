#!/usr/bin/env python3
"""Recompute sam_med3d_week7_results.json and stunet_week7_results.json from saved predictions (no re-inference)."""
from __future__ import annotations

import glob
import json
import os
import re
import sys

import numpy as np
import nibabel as nib

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from foundation_mask_diagnostic_metrics import (  # noqa: E402
    PIXEL_METRICS_NOTE,
    align_pred_to_shape,
    mask_diagnostic_metrics,
)

WEEK7_SPLIT = "/data1/julih/combined_subject_split.json"
SAM_PRED_DIR = os.path.join(ROOT, "test_data", "week7_sam_test_set", "pred")
SAM_JSON = os.path.join(ROOT, "sam_med3d_week7_results.json")
STU_INPUT = os.path.join(ROOT, "stunet_week7_test_input")
STU_OUTPUT = os.path.join(ROOT, "stunet_week7_test_output")
STU_JSON = os.path.join(ROOT, "stunet_week7_results.json")


def load_nii(path: str) -> np.ndarray:
    return np.asarray(nib.load(path).get_fdata()).squeeze().astype(np.float32)


def _sam_pred_by_test_index() -> dict[int, str]:
    """Map test split index -> pred path (basename may differ from current pre_path names)."""
    out: dict[int, str] = {}
    for path in glob.glob(os.path.join(SAM_PRED_DIR, "test_*.nii.gz")):
        m = re.match(r"test_(\d+)_", os.path.basename(path))
        if m:
            out[int(m.group(1))] = path
    return out


def recompute_sam() -> bool:
    with open(WEEK7_SPLIT) as f:
        pairs = json.load(f).get("test", [])
    pred_by_i = _sam_pred_by_test_index()
    mae_l, ssim_l, psnr_l, dice_l, iou_l = [], [], [], [], []
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
        mae_l.append(m["mae"])
        ssim_l.append(m["ssim"])
        psnr_l.append(m["psnr"])
        dice_l.append(m["dice"])
        iou_l.append(m["iou"])
    if not mae_l:
        return False
    out = {
        "model": "SAM-Med3D",
        "task": "mask_vs_brain_mask",
        "note": "Segmentation (mask). Primary: Dice/IoU vs Week7 brain mask on pre grid. Not CBF.",
        "primary_metrics": ["dice", "iou"],
        "pixel_image_metrics_note": PIXEL_METRICS_NOTE,
        "n_test": len(mae_l),
        "mae_mean": float(np.mean(mae_l)),
        "mae_std": float(np.std(mae_l)),
        "ssim_mean": float(np.mean(ssim_l)),
        "ssim_std": float(np.std(ssim_l)),
        "psnr_mean": float(np.mean(psnr_l)),
        "psnr_std": float(np.std(psnr_l)),
        "dice_mean": float(np.mean(dice_l)),
        "dice_std": float(np.std(dice_l)),
        "iou_mean": float(np.mean(iou_l)),
        "iou_std": float(np.std(iou_l)),
    }
    with open(SAM_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print("Wrote", SAM_JSON)
    return True


def recompute_stu() -> bool:
    with open(WEEK7_SPLIT) as f:
        pairs = json.load(f).get("test", [])
    mae_l, ssim_l, psnr_l, dice_l, iou_l = [], [], [], [], []
    for i, item in enumerate(pairs):
        pre_path = item.get("pre_path")
        if not pre_path or not os.path.isfile(pre_path):
            continue
        cid = "week7_test_%03d" % i
        pred_path = os.path.join(STU_OUTPUT, cid + ".nii.gz")
        inp_path = os.path.join(STU_INPUT, cid + "_0000.nii.gz")
        if not os.path.isfile(pred_path) or not os.path.isfile(inp_path):
            continue
        pred = load_nii(pred_path)
        pre_vol = load_nii(inp_path)
        pred_bin = (pred > 0).astype(np.float32)
        if pred_bin.shape != pre_vol.shape:
            pred_bin = align_pred_to_shape(pred_bin, pre_vol.shape)
        pred_bin = np.clip(pred_bin, 0, 1).astype(np.float32)
        m = mask_diagnostic_metrics(pred_bin)
        mae_l.append(m["mae"])
        ssim_l.append(m["ssim"])
        psnr_l.append(m["psnr"])
        dice_l.append(m["dice"])
        iou_l.append(m["iou"])
    if not mae_l:
        return False
    out = {
        "model": "STU-Net",
        "task": "mask_vs_brain_mask",
        "note": "Segmentation (105-class); foreground binarized vs Week7 brain mask on pre grid. Not CVR.",
        "primary_metrics": ["dice", "iou"],
        "pixel_image_metrics_note": PIXEL_METRICS_NOTE,
        "n_test": len(mae_l),
        "mae_mean": float(np.mean(mae_l)),
        "mae_std": float(np.std(mae_l)),
        "ssim_mean": float(np.mean(ssim_l)),
        "ssim_std": float(np.std(ssim_l)),
        "psnr_mean": float(np.mean(psnr_l)),
        "psnr_std": float(np.std(psnr_l)),
        "dice_mean": float(np.mean(dice_l)),
        "dice_std": float(np.std(dice_l)),
        "iou_mean": float(np.mean(iou_l)),
        "iou_std": float(np.std(iou_l)),
    }
    with open(STU_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print("Wrote", STU_JSON)
    return True


def main() -> None:
    ok_s = recompute_sam()
    ok_t = recompute_stu()
    if not ok_s:
        print("Skip SAM (no predictions in %s)" % SAM_PRED_DIR)
    if not ok_t:
        print("Skip STU (no predictions in %s)" % STU_OUTPUT)


if __name__ == "__main__":
    main()
