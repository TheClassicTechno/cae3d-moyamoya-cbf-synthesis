#!/usr/bin/env python3
"""Merge foundation experiment JSONs into one file for supplementary / paper checks."""
from __future__ import annotations

import argparse
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(ROOT)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "foundation_supplement_metrics.json"))
    args = ap.parse_args()

    rows = []

    med_multi = os.path.join(ROOT, "med3dvlm_week7_cvr", "med3dvlm_week7_results_multiseed.json")
    med_single = os.path.join(ROOT, "med3dvlm_week7_cvr", "med3dvlm_week7_results.json")
    med = med_multi if os.path.isfile(med_multi) else med_single
    if os.path.isfile(med):
        d = json.load(open(med))
        rows.append(
            {
                "model": "Med3DVLM_DCFormer_decoder",
                "comparable_to_main_table1": True,
                "task": "pre_to_post_cbf_in_brain",
                "metrics_code": "week7_preprocess.metrics_in_brain",
                "source_json": med,
                "mae_mean": d.get("mae_mean"),
                "mae_std": d.get("mae_std"),
                "ssim_mean": d.get("ssim_mean"),
                "ssim_std": d.get("ssim_std"),
                "psnr_mean": d.get("psnr_mean"),
                "psnr_std": d.get("psnr_std"),
            }
        )

    sam_cbf = os.path.join(ROOT, "sam_med3d_week7_cbf", "sam_med3d_week7_cbf_results_multiseed.json")
    if os.path.isfile(sam_cbf):
        d = json.load(open(sam_cbf))
        rows.append(
            {
                "model": "SAM-Med3D_CBF_decoder",
                "comparable_to_main_table1": True,
                "task": "pre_to_post_cbf_in_brain",
                "metrics_code": "week7_preprocess.metrics_in_brain",
                "encoder": "SAM-Med3D_image_encoder_frozen",
                "source_json": sam_cbf,
                "mae_mean": d.get("mae_mean"),
                "mae_std": d.get("mae_std"),
                "ssim_mean": d.get("ssim_mean"),
                "ssim_std": d.get("ssim_std"),
                "psnr_mean": d.get("psnr_mean"),
                "psnr_std": d.get("psnr_std"),
            }
        )

    sam = os.path.join(ROOT, "sam_med3d_week7_results.json")
    if os.path.isfile(sam):
        d = json.load(open(sam))
        rows.append(
            {
                "model": "SAM-Med3D_mask_diagnostic",
                "comparable_to_main_table1": False,
                "task": d.get("task", "mask_vs_brain_mask"),
                "note": d.get("note"),
                "primary_metrics": d.get("primary_metrics"),
                "source_json": sam,
                "dice_mean": d.get("dice_mean"),
                "dice_std": d.get("dice_std"),
                "iou_mean": d.get("iou_mean"),
                "iou_std": d.get("iou_std"),
                "mae_mean": d.get("mae_mean"),
                "ssim_mean": d.get("ssim_mean"),
                "psnr_mean": d.get("psnr_mean"),
            }
        )

    stu = os.path.join(ROOT, "stunet_week7_results.json")
    if os.path.isfile(stu):
        d = json.load(open(stu))
        rows.append(
            {
                "model": "STU-Net",
                "comparable_to_main_table1": False,
                "task": d.get("task", "mask_vs_brain_mask"),
                "note": d.get("note"),
                "primary_metrics": d.get("primary_metrics"),
                "source_json": stu,
                "dice_mean": d.get("dice_mean"),
                "dice_std": d.get("dice_std"),
                "iou_mean": d.get("iou_mean"),
                "iou_std": d.get("iou_std"),
                "mae_mean": d.get("mae_mean"),
                "ssim_mean": d.get("ssim_mean"),
                "psnr_mean": d.get("psnr_mean"),
            }
        )

    out = {"rows": rows, "description": "See README_FOUNDATION_CBF_MATCH.md"}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print("Wrote", args.out, "(%d rows)" % len(rows))


if __name__ == "__main__":
    main()
