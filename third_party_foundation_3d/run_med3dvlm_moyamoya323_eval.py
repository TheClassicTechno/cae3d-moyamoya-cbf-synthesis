#!/usr/bin/env python3
"""
Med3DVLM inference on all 323 Moyamoya subjects (no training).
Reads /tmp/combined_all_323.json (pre_path, post_path pairs).
Saves results to third_party_foundation_3d/moyamoya323_eval/Med3DVLM/

Run from repo root:
  PYTHONNOUSERSITE=1 \\
  PYTHONPATH=/data1/julih/scripts:/data1/julih/third_party_foundation_3d/Med3DVLM \\
    /data1/julih/miniconda3/envs/julih_monai/bin/python \\
    third_party_foundation_3d/run_med3dvlm_moyamoya323_eval.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = "/data1/julih"
REPO_SCRIPTS = os.path.join(ROOT, "scripts")
MED3DVLM_REPO = os.path.join(ROOT, "third_party_foundation_3d", "Med3DVLM")
for p in (REPO_SCRIPTS, MED3DVLM_REPO):
    if p not in sys.path:
        sys.path.insert(0, p)

from week7_preprocess import TARGET_SHAPE, load_volume, get_brain_mask_for_shape, metrics_in_brain
from src.model.encoder.dcformer import decomp_small

SUBJECTS_JSON = "/tmp/combined_all_323.json"
CKPT_DIR = os.path.join(ROOT, "third_party_foundation_3d", "med3dvlm_week7_cvr")
OUT_DIR = os.path.join(ROOT, "third_party_foundation_3d", "moyamoya323_eval", "Med3DVLM")
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_SHAPE_3D = TARGET_SHAPE
ENC_SIZE = (128, 128, 128)

SEEDS_AND_CKPTS = [
    (42,  os.path.join(CKPT_DIR, "med3dvlm_cvr_best_seed42.pt")),
    (123, os.path.join(CKPT_DIR, "med3dvlm_cvr_best_seed123.pt")),
    (456, os.path.join(CKPT_DIR, "med3dvlm_cvr_best_seed456.pt")),
]


def load_subjects(json_path: str) -> list[dict]:
    with open(json_path) as f:
        data = json.load(f)
    return [s for s in data if os.path.isfile(s.get("pre_path", "")) and os.path.isfile(s.get("post_path", ""))]


class CVRDecoder3D(nn.Module):
    def __init__(self, in_ch=768):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose3d(in_ch, 384, 4, stride=2, padding=1), nn.BatchNorm3d(384), nn.GELU(),
            nn.ConvTranspose3d(384, 192, 4, stride=2, padding=1), nn.BatchNorm3d(192), nn.GELU(),
            nn.ConvTranspose3d(192, 96, 4, stride=2, padding=1), nn.BatchNorm3d(96), nn.GELU(),
            nn.ConvTranspose3d(96, 48, 4, stride=2, padding=1), nn.BatchNorm3d(48), nn.GELU(),
            nn.ConvTranspose3d(48, 24, 4, stride=2, padding=1), nn.BatchNorm3d(24), nn.GELU(),
            nn.ConvTranspose3d(24, 1, 4, stride=2, padding=1),
        )

    def forward(self, x):
        return self.up(x)


class DCFormerCVR(nn.Module):
    def __init__(self, input_size=ENC_SIZE):
        super().__init__()
        self.encoder = decomp_small(input_size=input_size)
        for p in self.encoder.parameters():
            p.requires_grad = False
        enc_ch = self.encoder.channels[-1]
        self.decoder = CVRDecoder3D(in_ch=enc_ch)

    def forward(self, x):
        feats = self.encoder(x)
        last = feats[-1]
        B, N, C = last.shape
        last = last.permute(0, 2, 1).view(B, C, 2, 2, 2)
        return self.decoder(last)


def eval_one_seed(seed, ckpt_path, subjects):
    print(f"  seed={seed} ({len(subjects)} subjects)")
    model = DCFormerCVR(input_size=ENC_SIZE).to(DEVICE)
    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()

    results = []
    for i, item in enumerate(subjects):
        pre_vol = load_volume(item["pre_path"], target_shape=TARGET_SHAPE_3D, apply_mask=True, minmax=True)
        post_vol = load_volume(item["post_path"], target_shape=TARGET_SHAPE_3D, apply_mask=True, minmax=True)

        pre_t = torch.from_numpy(pre_vol).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
        pre_128 = F.interpolate(pre_t, size=ENC_SIZE, mode="trilinear", align_corners=False)

        with torch.no_grad():
            pred_128 = model(pre_128)
            pred_t = F.interpolate(pred_128, size=TARGET_SHAPE_3D, mode="trilinear", align_corners=False).clamp(0.0, 1.0)

        pred_np = pred_t.squeeze().cpu().numpy()
        m = metrics_in_brain(pred_np, post_vol, data_range=1.0)
        results.append({"mae": m["mae_mean"], "ssim": m["ssim_mean"], "psnr": m["psnr_mean"]})

        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(subjects)}")

    return {
        "seed": seed,
        "n_subjects": len(results),
        "mae_mean": float(np.mean([r["mae"] for r in results])),
        "mae_std": float(np.std([r["mae"] for r in results])),
        "ssim_mean": float(np.mean([r["ssim"] for r in results])),
        "ssim_std": float(np.std([r["ssim"] for r in results])),
        "psnr_mean": float(np.mean([r["psnr"] for r in results])),
        "psnr_std": float(np.std([r["psnr"] for r in results])),
        "per_subject": results,
    }


def main():
    subjects = load_subjects(SUBJECTS_JSON)
    print(f"Loaded {len(subjects)} subjects from {SUBJECTS_JSON}")

    all_seed_results = []
    for seed, ckpt_path in SEEDS_AND_CKPTS:
        if not os.path.isfile(ckpt_path):
            print(f"  SKIP seed={seed}: {ckpt_path} not found")
            continue
        r = eval_one_seed(seed, ckpt_path, subjects)
        all_seed_results.append(r)
        with open(os.path.join(OUT_DIR, f"med3dvlm_323_seed{seed}.json"), "w") as f:
            json.dump(r, f, indent=2)
        print(f"  seed={seed}: MAE={r['mae_mean']:.4f} SSIM={r['ssim_mean']:.4f} PSNR={r['psnr_mean']:.2f}")

    if all_seed_results:
        agg = {
            "model": "Med3DVLM",
            "n_seeds": len(all_seed_results),
            "n_subjects": all_seed_results[0]["n_subjects"],
            "mae_mean": float(np.mean([r["mae_mean"] for r in all_seed_results])),
            "mae_std": float(np.std([r["mae_mean"] for r in all_seed_results])),
            "ssim_mean": float(np.mean([r["ssim_mean"] for r in all_seed_results])),
            "ssim_std": float(np.std([r["ssim_mean"] for r in all_seed_results])),
            "psnr_mean": float(np.mean([r["psnr_mean"] for r in all_seed_results])),
            "psnr_std": float(np.std([r["psnr_mean"] for r in all_seed_results])),
            "per_seed": all_seed_results,
        }
        out = os.path.join(OUT_DIR, "med3dvlm_323_multiseed.json")
        with open(out, "w") as f:
            json.dump(agg, f, indent=2)
        print(f"\nAggregate: MAE={agg['mae_mean']:.4f}±{agg['mae_std']:.4f} "
              f"SSIM={agg['ssim_mean']:.4f}±{agg['ssim_std']:.4f} "
              f"PSNR={agg['psnr_mean']:.2f}±{agg['psnr_std']:.2f}")
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
