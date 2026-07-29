#!/usr/bin/env python3
"""Read combined result JSONs and print MAE / SSIM / PSNR for pasting into FINAL_METRICS_TABLE_WITH_2020.md.
Run after combined training finishes. Does not start or stop any training."""
import json
import os

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_REPO_ROOT, "pyproject.toml")):
    _parent = os.path.dirname(_REPO_ROOT)
    if _parent == _REPO_ROOT:
        raise RuntimeError("Could not locate repository root (pyproject.toml not found)")
    _REPO_ROOT = _parent

BASE = _REPO_ROOT
FILES = [
    ("3D DDPM (combined)", f"{BASE}/Diffusion_baseline_3D/ddpm_3d_combined_results.json"),
    ("UNet 3D (combined)", f"{BASE}/UNet_3D/unet_3d_results_combined.json"),
    ("FNO 3D (combined)", f"{BASE}/NeuralOperators/fno_3d_finetuned_slice_brain_combined_results.json"),
]

def get_metrics(path):
    with open(path) as f:
        d = json.load(f)
    if "test_results" in d:
        t = d["test_results"]
    else:
        t = d
    return t.get("mae_mean"), t.get("ssim_mean"), t.get("psnr_mean")

def main():
    print("Combined results (paste into Section 1 table; run after training finishes):\n")
    for name, path in FILES:
        if not os.path.isfile(path):
            print(f"{name}: (file not yet created) {path}")
            continue
        try:
            mae, ssim, psnr = get_metrics(path)
            if mae is None:
                print(f"{name}: no mae_mean in {path}")
            else:
                print(f"{name}: MAE {mae:.4f}  SSIM {ssim:.4f}  PSNR {psnr:.2f} dB")
        except Exception as e:
            print(f"{name}: error {e}")

if __name__ == "__main__":
    main()
