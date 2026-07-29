#!/usr/bin/env python3
"""
Build per-subject JSONs for week8_significance_and_bland_altman.py from regional eval outputs.

Each regional JSON has per_volume: [{ "id": subject_id, "full_mae", "full_ssim", "full_psnr", "per_region": { name: { mae, ssim, psnr, mean_perfusion_pred, mean_perfusion_gt, n_voxels } } }].
We write one JSON per (model, subject) with: model, mae, ssim, psnr, pred_mean, target_mean
(pred_mean/target_mean = voxel-weighted mean across regions), and per_region: { territory: { mean_pred, mean_gt, n_voxels } }
for full 3D regional reporting (Bland–Altman by territory). Regions with n_voxels < 10 are omitted from per_region.
"""
import json
import os
from pathlib import Path

ROOT = Path("<repo-root>")
REGIONAL_FILES = [
    ("week8_regional_unet3d.json", "week7_unet3d"),
    ("week8_regional_resnet3d.json", "week7_resnet3d"),
]
OUT_DIR = ROOT / "week8_per_subject_metrics"


def weighted_mean_per_volume(per_region: dict) -> tuple[float, float]:
    """Return (pred_mean, target_mean) weighted by n_voxels across regions."""
    total_n = 0
    sum_pred = 0.0
    sum_tgt = 0.0
    for name, r in per_region.items():
        n = r.get("n_voxels", 0)
        if n < 1:
            continue
        total_n += n
        sum_pred += (r.get("mean_perfusion_pred") or 0) * n
        sum_tgt += (r.get("mean_perfusion_gt") or 0) * n
    if total_n <= 0:
        return 0.0, 0.0
    return sum_pred / total_n, sum_tgt / total_n


MIN_VOXELS_FOR_PER_REGION = 10  # omit territory from per_region if n_voxels < this


def per_region_for_export(per_region: dict) -> dict:
    """Return { territory: { mean_pred, mean_gt, n_voxels } } for regions with n_voxels >= MIN_VOXELS_FOR_PER_REGION."""
    out = {}
    for name, r in per_region.items():
        n = r.get("n_voxels", 0)
        if n < MIN_VOXELS_FOR_PER_REGION:
            continue
        pred = r.get("mean_perfusion_pred")
        gt = r.get("mean_perfusion_gt")
        if pred is None and gt is None:
            continue
        out[name] = {
            "mean_pred": pred,
            "mean_gt": gt,
            "n_voxels": int(n),
        }
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for filename, model_name in REGIONAL_FILES:
        path = ROOT / filename
        if not path.is_file():
            print("Skip (not found):", path)
            continue
        with open(path) as f:
            data = json.load(f)
        for vol in data.get("per_volume", []):
            sid = vol.get("id", "unknown")
            mae = vol.get("full_mae")
            ssim = vol.get("full_ssim")
            psnr = vol.get("full_psnr")
            pr = vol.get("per_region", {})
            pred_mean, target_mean = weighted_mean_per_volume(pr)
            out = {
                "model": model_name,
                "subject_id": sid,
                "mae": mae,
                "ssim": ssim,
                "psnr": psnr,
                "pred_mean": pred_mean,
                "target_mean": target_mean,
                "per_region": per_region_for_export(pr),
            }
            out_path = OUT_DIR / f"{model_name}_{sid}.json"
            with open(out_path, "w") as f:
                json.dump(out, f, indent=0)
        print(f"Wrote {len(data['per_volume'])} JSONs for {model_name}")
    print("Output dir:", OUT_DIR)


if __name__ == "__main__":
    main()
