#!/usr/bin/env python3
"""
Build Week7 3D model comparison table: MAE, SSIM, PSNR (and Dice for segmentation models).
Reads from known result JSONs and third_party_foundation_3d/sam_med3d_week7_results.json.
Writes third_party_foundation_3d/WEEK7_3D_COMPARISON.md and optionally updates WEEK7_TABLE_RESULTS.md.
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
FOUNDATION_DIR = os.path.join(ROOT, "third_party_foundation_3d")
OUT_MD = os.path.join(FOUNDATION_DIR, "WEEK7_3D_COMPARISON.md")

# (path, display_name, note)
RESULT_FILES = [
    (os.path.join(ROOT, "scripts/week7_results/week7_unet3d_results.json"), "UNet 3D (scripts)", "CVR"),
    (os.path.join(ROOT, "UNet_3D/unet_3d_results_week7.json"), "UNet 3D (UNet_3D)", "CVR"),
    (os.path.join(ROOT, "NeuralOperators/fno_3d_week7_best_results.json"), "FNO 3D", "CVR"),
    (os.path.join(ROOT, "NeuralOperators/fno_3d_spectral_week7_best_results.json"), "FNO 3D Spectral", "CVR"),
    (os.path.join(ROOT, "Diffusion_3D_PatchVolume/patch_diffusion_week7_results.json"), "Patch/Volume 3D", "CVR"),
    (os.path.join(ROOT, "Diffusion_ResidualDiffusion_3D/residual_diffusion_3d_tips_week7_results.json"), "Residual Diffusion 3D (tips)", "CVR"),
    (os.path.join(ROOT, "Diffusion_ResidualDiffusion_3D/residual_diffusion_3d_week7_results.json"), "Residual Diffusion 3D", "CVR"),
    (os.path.join(ROOT, "Diffusion_ColdDiffusion_3D/cold_diffusion_3d_week7_results.json"), "Cold Diffusion 3D", "CVR"),
    (os.path.join(ROOT, "Hybrid_UNet_Diffusion/hybrid_unet_diffusion_week7_results.json"), "Hybrid UNet+Diffusion", "CVR"),
    (os.path.join(ROOT, "Diffusion_baseline_3D/ddpm_3d_week7_results.json"), "DDPM 3D", "CVR"),
    (os.path.join(FOUNDATION_DIR, "sam_med3d_week7_results.json"), "SAM-Med3D", "mask vs mask (seg)"),
    (os.path.join(FOUNDATION_DIR, "stunet_week7_results.json"), "STU-Net", "mask vs mask (seg)"),
]


def load_row(path, name, note):
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        data = json.load(f)
    if "test_results" in data:
        data = data["test_results"]
    mae = data.get("mae_mean") or data.get("MAE")
    ssim = data.get("ssim_mean") or data.get("SSIM")
    psnr = data.get("psnr_mean") or data.get("PSNR")
    dice = data.get("dice_mean")
    return {
        "name": name,
        "note": note,
        "mae": mae,
        "ssim": ssim,
        "psnr": psnr,
        "dice": dice,
    }


def main():
    rows = []
    for path, name, note in RESULT_FILES:
        r = load_row(path, name, note)
        if r is None:
            continue
        rows.append(r)

    lines = [
        "# Week7 3D models — MAE, SSIM, PSNR comparison",
        "",
        "Same test set (32 subjects). CVR = pre→post prediction; seg = segmentation (mask vs brain mask).",
        "",
        "| Model | Task | MAE | SSIM | PSNR (dB) | Dice |",
        "|-------|------|-----|------|-----------|------|",
    ]
    for r in rows:
        mae_s = f"{r['mae']:.4f}" if r['mae'] is not None else "—"
        ssim_s = f"{r['ssim']:.4f}" if r['ssim'] is not None else "—"
        psnr_s = f"{r['psnr']:.2f}" if r['psnr'] is not None else "—"
        dice_s = f"{r['dice']:.4f}" if r.get('dice') is not None else "—"
        lines.append(f"| **{r['name']}** | {r['note']} | **{mae_s}** | **{ssim_s}** | **{psnr_s}** | {dice_s} |")
    lines.append("")
    lines.append("---")
    lines.append("- **CVR:** metrics are predicted post vs ground-truth post (full volume).")
    lines.append("- **SAM-Med3D:** segmentation model; MAE/SSIM/PSNR are predicted mask vs brain mask (not CVR). Dice = mask overlap.")
    with open(OUT_MD, "w") as f:
        f.write("\n".join(lines))
    print("Wrote", OUT_MD)
    for r in rows:
        print(" ", r["name"], "MAE:", r["mae"], "SSIM:", r["ssim"], "PSNR:", r["psnr"])


if __name__ == "__main__":
    main()
