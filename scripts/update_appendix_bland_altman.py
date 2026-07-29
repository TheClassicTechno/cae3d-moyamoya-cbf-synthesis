#!/usr/bin/env python3
"""
Update the supplementary Bland-Altman table in julipaper.tex with values from
week8_stats/bland_altman_summary.md (after running week8_export and bland_altman script).
Maps JSON model names to paper names: week7_unet3d -> CAE_3D_s, UNet_3D -> CAE_3D, etc.
"""
import re
from pathlib import Path

ROOT = Path("<repo-root>")
MD_PATH = ROOT / "week8_stats" / "bland_altman_summary.md"
TEX_PATH = ROOT / "julipaper.tex"

MODEL_TO_PAPER = {
    "week7_unet3d": "CAE\\_3D\\_s (scripts)",
    "UNet_3D": "CAE\\_3D (UNet tips)",
    "week7_resnet3d": "ResNet\\_3D",
    "week7_unet2d": "CAE\\_2D",
    "Cold_3D": "Cold\\_3D",
    "Residual_3D": "Residual\\_3D",
    "DDPM_3D": "DDPM\\_3D",
}


def parse_md(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("|") and "---" in line or line.startswith("| Model |"):
                continue
            if line.startswith("|") and "Mean diff" not in line:
                parts = [p.strip() for p in line.split("|") if p.strip()]
                if len(parts) >= 5:
                    model, mean_diff, std_diff, loa_low, loa_high = parts[0], parts[1], parts[2], parts[3], parts[4]
                    rows.append((model, mean_diff, std_diff, loa_low, loa_high))
    return rows


def format_tex_row(paper_name, mean_diff, std_diff, loa_low, loa_high):
    # Format numbers for LaTeX (minus signs)
    def fmt(x):
        s = str(x).strip()
        if s.startswith("-") and s[1:].replace(".", "").isdigit():
            return "$-$" + s[1:]
        return s
    return f"    {paper_name} & {fmt(mean_diff)} & {fmt(std_diff)} & {fmt(loa_low)} & {fmt(loa_high)} \\\\"


def main():
    if not MD_PATH.is_file():
        print("Not found:", MD_PATH)
        return
    rows = parse_md(MD_PATH)
    if not rows:
        print("No data rows in", MD_PATH)
        return
    models_in_md = {r[0] for r in rows}
    if "Residual_3D" not in models_in_md:
        rows.append(("Residual_3D", "---", "---", "---", "---"))
    if "DDPM_3D" not in models_in_md:
        rows.append(("DDPM_3D", "---", "---", "---", "---"))

    tex = TEX_PATH.read_text()
    lines = []
    for model, mean_diff, std_diff, loa_low, loa_high in rows:
        paper_name = MODEL_TO_PAPER.get(model, model.replace("_", "\\_"))
        lines.append(format_tex_row(paper_name, mean_diff, std_diff, loa_low, loa_high))
    new_body = "\n".join(lines)

    idx = tex.find("\\label{tab:blandaltman_supp}")
    if idx == -1:
        print("Could not find tab:blandaltman_supp in julipaper.tex")
        return
    rest = tex[idx:]
    mid = rest.find("\\midrule")
    if mid == -1:
        print("Could not find \\midrule in appendix table")
        return
    start = idx + mid + len("\\midrule")
    rest2 = tex[start:]
    bottom = rest2.find("\\bottomrule")
    if bottom == -1:
        print("Could not find \\bottomrule in appendix table")
        return
    end = start + bottom
    new_tex = tex[:start] + "\n" + new_body + "\n  " + tex[end:]
    TEX_PATH.write_text(new_tex)
    print("Updated", TEX_PATH, "with", len(rows), "Bland-Altman rows from", MD_PATH)


if __name__ == "__main__":
    main()
