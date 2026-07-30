#!/usr/bin/env python3
"""Emit LaTeX tabular rows from week11_kfold_full/*_kfold_summary.json (textbook rotating-test K-fold).

No GPU. Run from repo root:
  python3 scripts/week9/print_true_kfold_latex.py

Optional:
  JULIH_ROOT=/path python3 scripts/week9/print_true_kfold_latex.py
"""
from __future__ import annotations

import json
import os

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_REPO_ROOT, "pyproject.toml")):
    _parent = os.path.dirname(_REPO_ROOT)
    if _parent == _REPO_ROOT:
        break
    _REPO_ROOT = _parent

ROOT = os.environ.get("JULIH_ROOT", _REPO_ROOT)
KF = os.path.join(ROOT, "week11_kfold_full")

ROWS = [
    ("week7_unet3d_kfold_summary.json", "UNet3D"),
    ("UNet_3D_kfold_summary.json", "UNet\\_3D"),
    ("week7_resnet3d_kfold_summary.json", "ResNet\\_3D"),
    ("FNO_3D_kfold_summary.json", "FNO\\_3D"),
    ("Patch_3D_kfold_summary.json", "Patch\\_3D"),
    ("Cold_3D_kfold_summary.json", "Cold\\_3D"),
    ("DDPM_3D_kfold_summary.json", "DDPM\\_3D"),
    ("Hybrid_3D_kfold_summary.json", "Hybrid\\_3D"),
    ("Residual_3D_tips_kfold_summary.json", "Residual\\_3D\\_tips"),
]


def fmt(x: float, nd: int = 4) -> str:
    return f"{x:.{nd}f}"


def main() -> None:
    lines = []
    for fname, tex_name in ROWS:
        path = os.path.join(KF, fname)
        if not os.path.isfile(path):
            lines.append(f"% missing {fname}")
            continue
        d = json.load(open(path))
        mae = d["mae_mean"]
        ms = d["mae_std"]
        sm = d["ssim_mean"]
        ss = d["ssim_std"]
        pm = d["psnr_mean"]
        ps = d["psnr_std"]
        lines.append(
            f" {tex_name} & ${fmt(mae)} \\pm {fmt(ms)}$ & ${fmt(sm, 3)} \\pm {fmt(ss, 3)}$ & ${fmt(pm, 2)} \\pm {fmt(ps, 2)}$ \\\\"
        )
    print("% --- textbook K-fold (week11_kfold_full); paste into supplement table ---")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
