#!/usr/bin/env python3
"""Emit LaTeX tabular rows for newmlhcpaper Table tab:kfold_secondary from week11_kfold/*_kfold_summary.json.

No GPU. Run from repo root:
  python3 scripts/week9/print_secondary_kfold_latex.py

Copy output into week9_stats/newmlhcpaper.tex between \\midrule and \\bottomrule of tab:kfold_secondary.
"""
from __future__ import annotations

import json
import os

ROOT = os.environ.get("JULIH_ROOT", "/data1/julih")
KF = os.path.join(ROOT, "week11_kfold")

ROWS = [
    ("week7_unet3d_kfold_summary.json", "UNet3D"),
    ("UNet_3D_kfold_summary.json", "UNet\\_3D"),
    ("week7_resnet3d_kfold_summary.json", "ResNet\\_3D"),
    ("FNO_3D_kfold_summary.json", "FNO\\_3D"),
    ("Patch_3D_kfold_summary.json", "Patch\\_3D"),
]


def fmt(x: float, nd: int = 4) -> str:
    return f"{x:.{nd}f}"


def main() -> None:
    lines = []
    for fname, tex_name in ROWS:
        path = os.path.join(KF, fname)
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
    print("% --- paste below \\midrule in tab:kfold_secondary ---")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
