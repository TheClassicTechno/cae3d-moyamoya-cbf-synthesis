#!/usr/bin/env python3
"""
Build the final Week7 results table: all 2D and 3D CVR results from known JSONs.
Writes the table section into WEEK7_TABLE_RESULTS.md (replacing the existing table).
"""
import os
import json
import re

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_REPO_ROOT, "pyproject.toml")):
    _parent = os.path.dirname(_REPO_ROOT)
    if _parent == _REPO_ROOT:
        raise RuntimeError("Could not locate repository root (pyproject.toml not found)")
    _REPO_ROOT = _parent

ROOT = _REPO_ROOT
TABLE_PATH = os.path.join(ROOT, "WEEK7_TABLE_RESULTS.md")

# (path, display_name, "2D" | "3D") - order: 2D first, then 3D
RESULT_FILES = [
    # 2D
    (os.path.join(ROOT, "scripts/week7_results/week7_unet2d_results.json"), "UNet", "2D"),
    (os.path.join(ROOT, "Diffusion_baseline/runs_diffusion_post_from_pre/test_metrics_diffusion.json"), "DDPM", "2D"),
    (os.path.join(ROOT, "Diffusion_ColdDiffusion/runs_cold_diffusion/test_metrics.json"), "Cold Diffusion", "2D"),
    (os.path.join(ROOT, "Diffusion_ResidualDiffusion/runs_residual_diffusion/test_metrics.json"), "Residual Diffusion", "2D"),
    (os.path.join(ROOT, "Diffusion_Option1/runs_diffusion_post_from_pre/test_metrics_week7.json"), "DDPM Option1", "2D"),
    (os.path.join(ROOT, "Diffusion_Option2/runs_diffusion_post_from_pre/test_metrics_week7.json"), "DDPM Option2", "2D"),
    # 3D CVR
    (os.path.join(ROOT, "scripts/week7_results/week7_unet3d_results.json"), "UNet 3D (scripts)", "3D"),
    (os.path.join(ROOT, "UNet_3D/unet_3d_results_week7.json"), "UNet 3D (UNet_3D)", "3D"),
    (os.path.join(ROOT, "NeuralOperators/fno_3d_week7_best_results.json"), "FNO 3D", "3D"),
    (os.path.join(ROOT, "NeuralOperators/fno_3d_spectral_week7_best_results.json"), "FNO 3D Spectral", "3D"),
    (os.path.join(ROOT, "Diffusion_3D_PatchVolume/patch_diffusion_week7_results.json"), "Patch/Volume 3D", "3D"),
    (os.path.join(ROOT, "Diffusion_ResidualDiffusion_3D/residual_diffusion_3d_tips_week7_results.json"), "Residual Diffusion 3D (tips)", "3D"),
    (os.path.join(ROOT, "Diffusion_ResidualDiffusion_3D/residual_diffusion_3d_week7_results.json"), "Residual Diffusion 3D", "3D"),
    (os.path.join(ROOT, "Diffusion_ColdDiffusion_3D/cold_diffusion_3d_week7_results.json"), "Cold Diffusion 3D", "3D"),
    (os.path.join(ROOT, "Hybrid_UNet_Diffusion/hybrid_unet_diffusion_week7_results.json"), "Hybrid UNet+Diffusion", "3D"),
    (os.path.join(ROOT, "Diffusion_baseline_3D/ddpm_3d_week7_results.json"), "DDPM simple 3D", "3D"),
    (os.path.join(ROOT, "Diffusion_3D_Latent/cold_diffusion_latent_week7_results.json"), "v-prediction (latent) 3D", "3D"),
    (os.path.join(ROOT, "scripts/week7_results/week7_resnet3d_results.json"), "3D ResNet (MONAI)", "3D"),
    (os.path.join(ROOT, "scripts/week7_results/week7_medicalnet_transfer_results.json"), "MedicalNet R18 pretrained (Med3D transfer)", "3D"),
    (os.path.join(ROOT, "scripts/week7_results/week7_swin_unetr_ssl_results.json"), "Swin UNETR + SSL encoder (MONAI)", "3D"),
    (os.path.join(ROOT, "third_party_foundation_3d/med3dvlm_week7_cvr/med3dvlm_week7_results.json"), "Med3DVLM (DCFormer+CVR)", "3D"),
    (os.path.join(ROOT, "Diffusion_MAISI/run3d_week7/maisi_week7_results.json"), "MAISI 3D", "3D"),
]


def load_metrics(path):
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        data = json.load(f)
    if "test_results" in data:
        data = data["test_results"]
    mae = data.get("mae_mean") or data.get("MAE") or data.get("L1")
    ssim = data.get("ssim_mean") or data.get("SSIM")
    psnr = data.get("psnr_mean") or data.get("PSNR")
    if mae is None and ssim is None and psnr is None:
        return None
    return (
        f"{mae:.4f}" if mae is not None else "-",
        f"{ssim:.4f}" if ssim is not None else "-",
        f"{psnr:.2f}" if psnr is not None else "-",
    )


def main():
    rows = []
    for path, name, typ in RESULT_FILES:
        m = load_metrics(path)
        if m is None:
            continue
        mae_s, ssim_s, psnr_s = m
        rows.append((name, typ, mae_s, ssim_s, psnr_s))

    table_lines = [
        "## Results table (Week 7 pipeline)",
        "",
        "**2D baseline reference (paper, middle slice):** MAE 0.0497, SSIM 0.7886, PSNR 21.49 dB.",
        "",
        "| # | Model | Type | MAE | SSIM | PSNR (dB) | Status |",
        "|---|--------|------|-----|------|-----------|--------|",
    ]
    for i, (name, typ, mae_s, ssim_s, psnr_s) in enumerate(rows, 1):
        table_lines.append(f"| {i} | **{name}** | {typ} | **{mae_s}** | **{ssim_s}** | **{psnr_s}** | Done (Week7) |")
    table_lines.append("")
    table_lines.append("- **Done:** Trained and evaluated with the Week7 pipeline (91x109x91, brain mask, combined 2020-2023, same augmentations). Results in `scripts/week7_results/` and project-specific JSONs.")
    table_lines.append("- **Pending (same config):** DDPM Option1/Option2 2D, MAISI / other 3D - add result JSON path to `scripts/build_final_week7_table.py` when available.")
    table_lines.append("")
    table_lines.append("---")
    table_lines.append("")

    new_section = "\n".join(table_lines)

    with open(TABLE_PATH) as f:
        content = f.read()

    start = content.find("## Results table (Week 7 pipeline)")
    end = content.find("## How to run remaining models with same config")
    if end == -1:
        end = content.find("## How to run remaining models")
    if start == -1 or end == -1:
        print("Could not find table section; table unchanged. (start=%s end=%s path=%s)" % (start, end, TABLE_PATH))
        return
    content_new = content[:start] + new_section + "\n\n" + content[end:]
    if content_new != content:
        with open(TABLE_PATH, "w") as f:
            f.write(content_new)
        print("Updated WEEK7_TABLE_RESULTS.md with final 2D+3D CVR table (%d rows)." % len(rows))
    else:
        print("Table unchanged (%d rows; no new result JSONs)." % len(rows))


if __name__ == "__main__":
    main()
