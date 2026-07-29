#!/usr/bin/env python3
"""
NOT pre→post ASL synthesis — STU-Net is run as **off-the-shelf segmentation** on pre scans only.

No training on post-ACZ targets; metrics vs brain mask are **not** comparable to CAE3D reconstruction.
There is no checked-in STU-Net **pre→post regression** finetuner in this repo.

See: third_party_foundation_3d/FOUNDATION_PRE_TO_POST_FINETUNING.md

---
Run STU-Net on full Week7 test set; overlap + legacy pixel metrics vs Week7 brain mask.
Saves third_party_foundation_3d/stunet_week7_results.json.
STU-Net: 105-class labels; binarize foreground (label > 0). Primary: Dice/IoU; not CBF.
"""
import os
import sys
import json
import shutil
import subprocess
import numpy as np
import nibabel as nib

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_REPO_ROOT, "pyproject.toml")):
    _parent = os.path.dirname(_REPO_ROOT)
    if _parent == _REPO_ROOT:
        raise RuntimeError("Could not locate repository root (pyproject.toml not found)")
    _REPO_ROOT = _parent

ROOT = os.path.dirname(os.path.abspath(__file__))
from foundation_mask_diagnostic_metrics import (  # noqa: E402
    PIXEL_METRICS_NOTE,
    align_pred_to_shape,
    mask_diagnostic_metrics,
)
WEEK7_SPLIT = os.path.join(_REPO_ROOT, "combined_subject_split.json")
INPUT_DIR = os.path.join(ROOT, "stunet_week7_test_input")
OUTPUT_DIR = os.path.join(ROOT, "stunet_week7_test_output")
RESULTS_JSON = os.path.join(ROOT, "stunet_week7_results.json")
NNUNET_PYTHON = os.environ.get("NNUNET_PYTHON", os.path.join(_REPO_ROOT, "miniconda3/envs/julih_monai/bin/python3"))


def load_nii(path):
    return np.asarray(nib.load(path).get_fdata()).squeeze().astype(np.float32)


def main():
    with open(WEEK7_SPLIT) as f:
        data = json.load(f)
    pairs = data.get("test", [])
    if not pairs:
        print("No Week7 test pairs.")
        sys.exit(1)

    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # Prepare input: nnUNet expects CASENAME_0000.nii.gz
    case_ids = []
    for i, item in enumerate(pairs):
        pre_path = item.get("pre_path")
        if not pre_path or not os.path.isfile(pre_path):
            continue
        cid = "week7_test_%03d" % i
        case_ids.append((cid, pre_path))
        shutil.copy2(pre_path, os.path.join(INPUT_DIR, cid + "_0000.nii.gz"))

    if not case_ids:
        print("No valid test paths.")
        sys.exit(1)

    # Run nnUNet predict (single call on folder)
    env = os.environ.copy()
    env["RESULTS_FOLDER"] = env.get("RESULTS_FOLDER", os.path.join(ROOT, "stunet_results"))
    env["nnUNet_raw_data_base"] = env.get("nnUNet_raw_data_base", os.path.join(ROOT, "nnunet_data", "raw"))
    env["nnUNet_preprocessed"] = env.get("nnUNet_preprocessed", os.path.join(ROOT, "nnunet_data", "preprocessed"))
    print("Running STU-Net on %d test volumes..." % len(case_ids))
    cmd = [
        NNUNET_PYTHON, "-m", "nnunet.inference.predict_simple",
        "-i", INPUT_DIR, "-o", OUTPUT_DIR,
        "-t", "101", "-m", "3d_fullres", "-f", "0",
        "-tr", "STUNetTrainer_base", "-chk", "base_ep4k",
        "--mode", "fast", "--disable_tta",
    ]
    subprocess.run(cmd, env=env, cwd=ROOT, check=False, capture_output=False)

    # Metrics: binarized seg on pre grid; brain mask via Week7 helper (same convention as CBF)
    mae_list, ssim_list, psnr_list, dice_list, iou_list = [], [], [], [], []
    for cid, _ in case_ids:
        pred_path = os.path.join(OUTPUT_DIR, cid + ".nii.gz")
        inp_path = os.path.join(INPUT_DIR, cid + "_0000.nii.gz")
        if not os.path.isfile(pred_path) or not os.path.isfile(inp_path):
            continue
        pred = load_nii(pred_path)
        pre_vol = load_nii(inp_path)
        pred_bin = (pred > 0).astype(np.float32)
        if pred_bin.shape != pre_vol.shape:
            pred_bin = align_pred_to_shape(pred_bin, pre_vol.shape)
        pred_bin = np.clip(pred_bin, 0, 1).astype(np.float32)
        m = mask_diagnostic_metrics(pred_bin)
        mae_list.append(m["mae"])
        ssim_list.append(m["ssim"])
        psnr_list.append(m["psnr"])
        dice_list.append(m["dice"])
        iou_list.append(m["iou"])

    if not mae_list:
        print("No predictions found.")
        sys.exit(1)

    results = {
        "model": "STU-Net",
        "task": "mask_vs_brain_mask",
        "note": "Segmentation (105-class); foreground binarized vs Week7 brain mask on pre grid. Not CVR.",
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
