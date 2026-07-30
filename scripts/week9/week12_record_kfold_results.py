#!/usr/bin/env python3
"""
Record all K-fold results into week12 folder: copy fixed-32 (week11_kfold) and full (week11_kfold_full)
summaries and per-fold JSONs, then write an index (week12/README.md or week12_kfold_results.md).

Usage (from repo root):
  python scripts/week9/week12_record_kfold_results.py
  python scripts/week9/week12_record_kfold_results.py --week12_dir /data1/julih/week12
"""
import argparse
import json
import os
import shutil
from pathlib import Path

ROOT = Path("/data1/julih")
WEEK11_FIXED = ROOT / "week11_kfold"
WEEK11_FULL = ROOT / "week11_kfold_full"
WEEK12_DEFAULT = ROOT / "week12"
WEEK9_STATS = ROOT / "week9_stats"
GUIDED_BPROP_DIR = WEEK9_STATS / "guided_backprop"
RSM_CSV = WEEK9_STATS / "rsm_unet3d.csv"
RSM_PNG = WEEK9_STATS / "rsm_unet3d.png"
RSM_IDS = WEEK9_STATS / "rsm_unet3d_subject_ids.txt"


def main():
    ap = argparse.ArgumentParser(description="Record K-fold results into week12")
    ap.add_argument("--week12_dir", default=str(WEEK12_DEFAULT), help="Output directory for week12")
    args = ap.parse_args()
    out = Path(args.week12_dir)
    out.mkdir(parents=True, exist_ok=True)
    fixed_dir = out / "kfold_fixed32"
    full_dir = out / "kfold_full"
    fixed_dir.mkdir(exist_ok=True)
    full_dir.mkdir(exist_ok=True)

    lines = ["# Week 12: K-fold cross-validation results\n", "- Fixed-32: 220 train+val in K folds, 32 test fixed (primary).\n", "- Full: all 252 in K folds (secondary).\n\n"]

    # Copy fixed-32
    if WEEK11_FIXED.exists():
        for f in WEEK11_FIXED.glob("*"):
            if f.is_file():
                shutil.copy2(f, fixed_dir / f.name)
        summaries = list(fixed_dir.glob("*_kfold_summary.json"))
        lines.append("## Fixed-32 test (primary)\n")
        lines.append("Source: week11_kfold/\n\n")
        for s in sorted(summaries):
            try:
                with open(s) as f:
                    d = json.load(f)
                name = s.stem.replace("_kfold_summary", "")
                lines.append("- **%s**: MAE %.4f ± %.4f, SSIM %.4f ± %.4f, PSNR %.2f ± %.2f\n" % (
                    name, d.get("mae_mean", 0), d.get("mae_std", 0), d.get("ssim_mean", 0), d.get("ssim_std", 0),
                    d.get("psnr_mean", 0), d.get("psnr_std", 0)))
            except Exception:
                lines.append("- %s: (see %s)\n" % (s.name, s.name))
        lines.append("\n")
    else:
        lines.append("## Fixed-32: no week11_kfold/ found\n\n")

    # Copy full
    if WEEK11_FULL.exists():
        for f in WEEK11_FULL.glob("*"):
            if f.is_file():
                shutil.copy2(f, full_dir / f.name)
        summaries = list(full_dir.glob("*_kfold_summary.json"))
        lines.append("## Full K-fold (secondary)\n")
        lines.append("Source: week11_kfold_full/\n\n")
        for s in sorted(summaries):
            try:
                with open(s) as f:
                    d = json.load(f)
                name = s.stem.replace("_kfold_summary", "")
                lines.append("- **%s**: MAE %.4f ± %.4f, SSIM %.4f ± %.4f, PSNR %.2f ± %.2f\n" % (
                    name, d.get("mae_mean", 0), d.get("mae_std", 0), d.get("ssim_mean", 0), d.get("ssim_std", 0),
                    d.get("psnr_mean", 0), d.get("psnr_std", 0)))
            except Exception:
                lines.append("- %s: (see %s)\n" % (s.name, s.name))
    else:
        lines.append("## Full K-fold: no week11_kfold_full/ found\n\n")

    # Explainability: RSM and guided backprop
    lines.append("---\n\n## Explainability (Week 9)\n\n")
    lines.append("### Methods\n\n")
    lines.append("- **Guided backpropagation:** Gradient-based attribution for the Week 7 UNet 3D. ")
    lines.append("Backward hooks on ReLU/LeakyReLU propagate only positive gradients; gradient w.r.t. input is saved as attribution. ")
    lines.append("Inplace activations are disabled to avoid view/inplace conflicts. ")
    lines.append("Script: `scripts/week9/week9_guided_backprop.py`. ")
    lines.append("Output: PNG (middle-slice attribution) and optional NIfTI per test subject.\n\n")
    lines.append("- **Representational similarity matrix (RSM):** Encoder bottleneck features (smallest-spatial 5D activation) are extracted per test subject, flattened, and pairwise similarity (Pearson or cosine) is computed. ")
    lines.append("Script: `scripts/week9/week9_representational_similarity.py`. ")
    lines.append("Output: N×N similarity matrix (CSV), heatmap (PNG), and subject IDs.\n\n")
    lines.append("### Results and outputs\n\n")
    if GUIDED_BPROP_DIR.exists():
        pngs = list(GUIDED_BPROP_DIR.glob("guided_backprop_*.png"))
        niftis = list(GUIDED_BPROP_DIR.glob("guided_backprop_*.nii.gz"))
        lines.append("- **Guided backprop:** %d PNGs, %d NIfTI in `week9_stats/guided_backprop/`.\n" % (len(pngs), len(niftis)))
    else:
        lines.append("- **Guided backprop:** Output dir `week9_stats/guided_backprop/` (run script to generate).\n")
    if RSM_CSV.is_file() and RSM_PNG.is_file():
        lines.append("- **RSM:** `week9_stats/rsm_unet3d.csv`, `rsm_unet3d.png`, `rsm_unet3d_subject_ids.txt`.\n")
    else:
        lines.append("- **RSM:** Run `week9_representational_similarity.py` to generate CSV/PNG/IDs in `week9_stats/`.\n")

    readme = out / "week12_kfold_results.md"
    with open(readme, "w") as f:
        f.writelines(lines)
    print("Recorded K-fold results to %s" % out)
    print("  %s/ (fixed-32)" % fixed_dir)
    print("  %s/ (full)" % full_dir)
    print("  %s" % readme)


if __name__ == "__main__":
    main()
