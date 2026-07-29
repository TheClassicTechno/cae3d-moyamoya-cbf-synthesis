#!/usr/bin/env python3
"""Update WEEK7_TABLE_RESULTS.md with test metrics from JSON result files when they exist."""
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


def load_metrics(path):
    """Return (mae, ssim, psnr) from a result JSON, or None if file missing/invalid."""
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
    paths = [
        (os.path.join(ROOT, "Diffusion_baseline/runs_diffusion_post_from_pre/test_metrics_diffusion.json"), 3, "DDPM", "2D"),
        (os.path.join(ROOT, "Diffusion_ColdDiffusion/runs_cold_diffusion/test_metrics.json"), 4, "Cold Diffusion", "2D"),
        (os.path.join(ROOT, "Diffusion_ResidualDiffusion/runs_residual_diffusion/test_metrics.json"), 5, "Residual Diffusion", "2D"),
        (os.path.join(ROOT, "UNet_3D/unet_3d_results_week7.json"), None, "UNet (UNet_3D)", "3D"),  # insert row 4 or update
        (os.path.join(ROOT, "Diffusion_baseline_3D/ddpm_3d_week7_results.json"), 7, "DDPM simple", "3D"),
        (os.path.join(ROOT, "Diffusion_ColdDiffusion_3D/cold_diffusion_3d_week7_results.json"), 8, "Cold Diffusion", "3D"),
        (os.path.join(ROOT, "Diffusion_ResidualDiffusion_3D/residual_diffusion_3d_week7_results.json"), 9, "Residual Diffusion", "3D"),
        (os.path.join(ROOT, "NeuralOperators/fno_3d_week7_results.json"), 10, "FNO 3D", "3D"),
    ]
    with open(TABLE_PATH) as f:
        content = f.read()

    changed = False

    for path, row_id, model_name, typ in paths:
        metrics = load_metrics(path)
        if metrics is None:
            continue
        mae_s, ssim_s, psnr_s = metrics
        if row_id is not None:
            new_row = f"| {row_id} | **{model_name}** | {typ} | **{mae_s}** | **{ssim_s}** | **{psnr_s}** | Done (Week7) |"
            content = re.sub(rf"\|\s*{row_id}\s*\|\s*(\*\*)?[^|]*\|\s*{typ}\s\|[^\n]+", new_row, content, count=1)
            changed = True
        else:
            # UNet (UNet_3D): insert or update row 4
            new_line = f"| 4 | **{model_name}** | {typ} | **{mae_s}** | **{ssim_s}** | **{psnr_s}** | Done (Week7) |"
            if "UNet (UNet_3D)" in content:
                content = re.sub(r"\|\s*4\s*\|\s*\*\*UNet \(UNet_3D\)\*\*\s*\|\s*3D\s\|[^\n]+", new_line, content, count=1)
            else:
                content = content.replace("| 14 | MAISI / other | 3D |", "| 15 | MAISI / other | 3D |", 1)
                content = content.replace("| 13 | Patch/Volume 3D | 3D |", "| 14 | Patch/Volume 3D | 3D |", 1)
                content = content.replace("| 12 | Hybrid UNet+Diffusion | 3D |", "| 13 | Hybrid UNet+Diffusion | 3D |", 1)
                content = content.replace("| 11 | v-prediction (latent) | 3D |", "| 12 | v-prediction (latent) | 3D |", 1)
                content = content.replace("| 10 | FNO 3D | 3D |", "| 11 | FNO 3D | 3D |", 1)
                content = content.replace("| 9 | Residual Diffusion | 3D |", "| 10 | Residual Diffusion | 3D |", 1)
                content = content.replace("| 8 | Cold Diffusion | 3D |", "| 9 | Cold Diffusion | 3D |", 1)
                content = content.replace("| 7 | DDPM simple | 3D |", "| 8 | DDPM simple | 3D |", 1)
                content = content.replace("| 6 | DDPM (Option1/Option2) | 2D |", "| 7 | DDPM (Option1/Option2) | 2D |", 1)
                content = content.replace("| 5 | Residual Diffusion | 2D |", "| 6 | Residual Diffusion | 2D |", 1)
                content = content.replace("| 4 | Cold Diffusion | 2D |", new_line + "\n| 5 | Cold Diffusion | 2D |", 1)
            changed = True

    if changed:
        with open(TABLE_PATH, "w") as f:
            f.write(content)
        print("Updated WEEK7_TABLE_RESULTS.md")
    else:
        print("No result files found to update (DDPM or UNet_3D Week7). Table unchanged.")


if __name__ == "__main__":
    main()
