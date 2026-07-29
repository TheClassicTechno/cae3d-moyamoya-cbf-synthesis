#!/usr/bin/env python3
"""
Build Week 7 Phase 2 / Phase 3 results table. Reads only from *_phase2* and *_phase2_phase3* JSONs.
Does not modify WEEK7_TABLE_RESULTS.md (Phase 1 table). Writes WEEK7_TABLE_PHASE2_PHASE3.md.
"""
import os
import json

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_REPO_ROOT, "pyproject.toml")):
    _parent = os.path.dirname(_REPO_ROOT)
    if _parent == _REPO_ROOT:
        raise RuntimeError("Could not locate repository root (pyproject.toml not found)")
    _REPO_ROOT = _parent

ROOT = _REPO_ROOT
TABLE_PATH = os.path.join(ROOT, "WEEK7_TABLE_PHASE2_PHASE3.md")

# (path, display_name, "2D" | "3D") - Phase 2 and/or Phase 3 result paths only
PHASE2_PHASE3_RESULT_FILES = [
    # 2D
    (os.path.join(ROOT, "scripts/week7_results/week7_unet2d_phase2_results.json"), "UNet", "2D"),
    (os.path.join(ROOT, "Diffusion_baseline/runs_diffusion_post_from_pre/test_metrics_diffusion_phase2.json"), "DDPM", "2D"),
    (os.path.join(ROOT, "Diffusion_ColdDiffusion/runs_cold_diffusion/test_metrics_phase2.json"), "Cold Diffusion", "2D"),
    (os.path.join(ROOT, "Diffusion_ResidualDiffusion/runs_residual_diffusion/test_metrics_phase2.json"), "Residual Diffusion", "2D"),
    (os.path.join(ROOT, "Diffusion_Option1/runs_diffusion_post_from_pre/test_metrics_week7_phase2.json"), "DDPM Option1", "2D"),
    (os.path.join(ROOT, "Diffusion_Option2/runs_diffusion_post_from_pre/test_metrics_week7_phase2.json"), "DDPM Option2", "2D"),
    # 3D
    (os.path.join(ROOT, "scripts/week7_results/week7_unet3d_phase2_phase3_results.json"), "UNet 3D (scripts)", "3D"),
    (os.path.join(ROOT, "UNet_3D/unet_3d_results_week7_phase2.json"), "UNet 3D (UNet_3D)", "3D"),
    (os.path.join(ROOT, "NeuralOperators/fno_3d_week7_phase2_results.json"), "FNO 3D", "3D"),
    (os.path.join(ROOT, "NeuralOperators/fno_3d_spectral_week7_phase2_results.json"), "FNO 3D Spectral", "3D"),
    (os.path.join(ROOT, "Diffusion_3D_PatchVolume/patch_diffusion_week7_phase2_results.json"), "Patch/Volume 3D", "3D"),
    (os.path.join(ROOT, "Diffusion_ResidualDiffusion_3D/residual_diffusion_3d_tips_week7_phase2_results.json"), "Residual Diffusion 3D (tips)", "3D"),
    (os.path.join(ROOT, "Diffusion_ResidualDiffusion_3D/residual_diffusion_3d_week7_phase2_results.json"), "Residual Diffusion 3D", "3D"),
    (os.path.join(ROOT, "Diffusion_ColdDiffusion_3D/cold_diffusion_3d_week7_phase2_results.json"), "Cold Diffusion 3D", "3D"),
    (os.path.join(ROOT, "Hybrid_UNet_Diffusion/hybrid_unet_diffusion_week7_phase2_results.json"), "Hybrid UNet+Diffusion", "3D"),
    (os.path.join(ROOT, "Diffusion_baseline_3D/ddpm_3d_week7_phase2_results.json"), "DDPM simple 3D", "3D"),
    (os.path.join(ROOT, "Diffusion_3D_Latent/cold_diffusion_latent_week7_phase2_results.json"), "v-prediction (latent) 3D", "3D"),
    (os.path.join(ROOT, "scripts/week7_results/week7_resnet3d_phase2_phase3_results.json"), "3D ResNet (MONAI)", "3D"),
    (os.path.join(ROOT, "third_party_foundation_3d/med3dvlm_week7_cvr/med3dvlm_week7_phase2_results.json"), "Med3DVLM (DCFormer+CVR)", "3D"),
    (os.path.join(ROOT, "Diffusion_MAISI/run3d_week7/maisi_week7_phase2_results.json"), "MAISI 3D", "3D"),
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
    for path, name, typ in PHASE2_PHASE3_RESULT_FILES:
        m = load_metrics(path)
        if m is None:
            rows.append((name, typ, "-", "-", "-", "Pending"))
            continue
        mae_s, ssim_s, psnr_s = m
        rows.append((name, typ, mae_s, ssim_s, psnr_s, "Done"))

    lines = [
        "# Week 7 - Phase 2 / Phase 3 results (separate from Phase 1)",
        "",
        "Metrics from runs with **region-weighted loss (Phase 2)** and, for 3D script trainers, **subject-specific mask-weighted loss (Phase 3)**.",
        "Phase 1 (brain-only eval) table remains in **WEEK7_TABLE_RESULTS.md**.",
        "",
        "| # | Model | Type | MAE | SSIM | PSNR (dB) | Status |",
        "|---|--------|------|-----|------|-----------|--------|",
    ]
    for i, r in enumerate(rows, 1):
        name, typ, mae_s, ssim_s, psnr_s, status = r
        lines.append(f"| {i} | **{name}** | {typ} | **{mae_s}** | **{ssim_s}** | **{psnr_s}** | {status} |")
    lines.extend([
        "",
        "---",
        "",
        "To regenerate: run each model with `WEEK7_REGION_WEIGHT=1` (and for 3D script trainers `WEEK7_SUBJECT_MASKS=1`), then:",
        "`python3 scripts/build_week7_phase2_phase3_table.py`",
        "",
    ])
    with open(TABLE_PATH, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {TABLE_PATH} ({len(rows)} rows).")


if __name__ == "__main__":
    main()
