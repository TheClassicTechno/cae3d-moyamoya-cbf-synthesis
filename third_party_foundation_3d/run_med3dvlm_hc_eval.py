#!/usr/bin/env python3
"""
Med3DVLM HC False-Positive Evaluation
======================================
Load the already-trained Med3DVLM DCFormer CVR model (frozen encoder +
trainable decoder) and run inference-only on the 64 healthy control subjects.

Mirrors the CAE3D HC FP analysis (scripts/hc_false_positive_eval.py) so that
results are directly comparable in the paper.

Checkpoints: third_party_foundation_3d/med3dvlm_week7_cvr/
  - med3dvlm_cvr_best_seed42.pt   (frozen encoder)
  - med3dvlm_cvr_best_seed123.pt
  - med3dvlm_cvr_best_seed456.pt

Run from repo root:
  PYTHONPATH=/data1/julih/scripts:/data1/julih/third_party_foundation_3d/Med3DVLM \\
    python3 third_party_foundation_3d/run_med3dvlm_hc_eval.py
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

HC_CBF_ROOT = os.path.join(ROOT, "healthy_controls_CBF")
CKPT_DIR = os.path.join(ROOT, "third_party_foundation_3d", "med3dvlm_week7_cvr")
OUT_DIR = os.path.join(ROOT, "third_party_foundation_3d", "hc_eval", "Med3DVLM")
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_SHAPE_3D = TARGET_SHAPE
ENC_SIZE = (128, 128, 128)
FP_THRESHOLD = -0.05

SEEDS_AND_CKPTS = [
    (42,  os.path.join(CKPT_DIR, "med3dvlm_cvr_best_seed42.pt")),
    (123, os.path.join(CKPT_DIR, "med3dvlm_cvr_best_seed123.pt")),
    (456, os.path.join(CKPT_DIR, "med3dvlm_cvr_best_seed456.pt")),
]


def list_hc_pairs() -> list[dict]:
    pairs = []
    if not os.path.isdir(HC_CBF_ROOT):
        raise FileNotFoundError(HC_CBF_ROOT)
    for subject in sorted(os.listdir(HC_CBF_ROOT)):
        bl = os.path.join(HC_CBF_ROOT, subject, "baseline", "CBF_oxford_asl_standard.nii.gz")
        dx = os.path.join(HC_CBF_ROOT, subject, "diamox", "CBF_oxford_asl_standard.nii.gz")
        if os.path.isfile(bl) and os.path.isfile(dx):
            pairs.append({"subject": subject, "pre_path": bl, "post_path": dx})
    return pairs


class CVRDecoder3D(nn.Module):
    def __init__(self, in_ch=768):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose3d(in_ch, 384, 4, stride=2, padding=1),
            nn.BatchNorm3d(384),
            nn.GELU(),
            nn.ConvTranspose3d(384, 192, 4, stride=2, padding=1),
            nn.BatchNorm3d(192),
            nn.GELU(),
            nn.ConvTranspose3d(192, 96, 4, stride=2, padding=1),
            nn.BatchNorm3d(96),
            nn.GELU(),
            nn.ConvTranspose3d(96, 48, 4, stride=2, padding=1),
            nn.BatchNorm3d(48),
            nn.GELU(),
            nn.ConvTranspose3d(48, 24, 4, stride=2, padding=1),
            nn.BatchNorm3d(24),
            nn.GELU(),
            nn.ConvTranspose3d(24, 1, 4, stride=2, padding=1),
        )

    def forward(self, x):
        return self.up(x)


class DCFormerCVR(nn.Module):
    def __init__(self, input_size=ENC_SIZE, freeze_encoder=True):
        super().__init__()
        self.encoder = decomp_small(input_size=input_size)
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
        enc_ch = self.encoder.channels[-1]
        self.decoder = CVRDecoder3D(in_ch=enc_ch)
        self.enc_size = input_size

    def forward(self, x):
        feats = self.encoder(x)
        last = feats[-1]
        B, N, C = last.shape
        s = 2
        last = last.permute(0, 2, 1).view(B, C, s, s, s)
        return self.decoder(last)


def eval_one_seed(seed: int, ckpt_path: str, pairs: list[dict], brain_mask: np.ndarray) -> dict:
    print(f"  seed={seed}, ckpt={os.path.basename(ckpt_path)}")
    model = DCFormerCVR(input_size=ENC_SIZE, freeze_encoder=True).to(DEVICE)
    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()

    mask_np = brain_mask
    results = []

    for item in pairs:
        subject = item["subject"]
        pre_vol = load_volume(item["pre_path"], target_shape=TARGET_SHAPE_3D, apply_mask=True, minmax=True)
        post_vol = load_volume(item["post_path"], target_shape=TARGET_SHAPE_3D, apply_mask=True, minmax=True)

        pre_t = torch.from_numpy(pre_vol).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
        pre_128 = F.interpolate(pre_t, size=ENC_SIZE, mode="trilinear", align_corners=False)

        with torch.no_grad():
            pred_128 = model(pre_128)
            pred_t = F.interpolate(pred_128, size=TARGET_SHAPE_3D, mode="trilinear", align_corners=False)
            pred_t = pred_t.clamp(0.0, 1.0)

        pred_np = pred_t.squeeze().cpu().numpy()
        m = metrics_in_brain(pred_np, post_vol, data_range=1.0)

        pre_mean = float((pre_vol * mask_np).sum() / (mask_np.sum() + 1e-8))
        pred_mean = float((pred_np * mask_np).sum() / (mask_np.sum() + 1e-8))
        delta_pred = (pred_mean - pre_mean) / (pre_mean + 1e-8)
        fp = int(delta_pred < FP_THRESHOLD)

        results.append({
            "subject": subject,
            "mae": m["mae_mean"],
            "ssim": m["ssim_mean"],
            "psnr": m["psnr_mean"],
            "delta_pred": delta_pred,
            "false_positive": fp,
        })

    mae_vals = [r["mae"] for r in results]
    ssim_vals = [r["ssim"] for r in results]
    psnr_vals = [r["psnr"] for r in results]
    delta_vals = [r["delta_pred"] for r in results]
    fp_rate = float(np.mean([r["false_positive"] for r in results]))

    return {
        "seed": seed,
        "n_subjects": len(results),
        "mae_mean": float(np.mean(mae_vals)),
        "mae_std": float(np.std(mae_vals)),
        "ssim_mean": float(np.mean(ssim_vals)),
        "ssim_std": float(np.std(ssim_vals)),
        "psnr_mean": float(np.mean(psnr_vals)),
        "psnr_std": float(np.std(psnr_vals)),
        "delta_pred_mean": float(np.mean(delta_vals)),
        "delta_pred_std": float(np.std(delta_vals)),
        "fp_rate": fp_rate,
        "per_subject": results,
    }


def main():
    pairs = list_hc_pairs()
    print(f"Found {len(pairs)} HC pairs")

    brain_mask = get_brain_mask_for_shape(TARGET_SHAPE_3D)

    all_seed_results = []
    for seed, ckpt_path in SEEDS_AND_CKPTS:
        if not os.path.isfile(ckpt_path):
            print(f"  SKIP seed={seed}: {ckpt_path} not found")
            continue
        r = eval_one_seed(seed, ckpt_path, pairs, brain_mask)
        all_seed_results.append(r)
        seed_json = os.path.join(OUT_DIR, f"med3dvlm_hc_eval_seed{seed}.json")
        with open(seed_json, "w") as f:
            json.dump(r, f, indent=2)
        print(f"  seed={seed}: MAE={r['mae_mean']:.4f} SSIM={r['ssim_mean']:.4f} PSNR={r['psnr_mean']:.2f} "
              f"delta_pred={r['delta_pred_mean']*100:.1f}% FP={r['fp_rate']*100:.1f}%")

    if all_seed_results:
        agg = {
            "model": "Med3DVLM DCFormer (frozen encoder + trained decoder)",
            "n_seeds": len(all_seed_results),
            "n_subjects": all_seed_results[0]["n_subjects"],
            "mae_mean": float(np.mean([r["mae_mean"] for r in all_seed_results])),
            "mae_std": float(np.std([r["mae_mean"] for r in all_seed_results])),
            "ssim_mean": float(np.mean([r["ssim_mean"] for r in all_seed_results])),
            "ssim_std": float(np.std([r["ssim_mean"] for r in all_seed_results])),
            "psnr_mean": float(np.mean([r["psnr_mean"] for r in all_seed_results])),
            "psnr_std": float(np.std([r["psnr_mean"] for r in all_seed_results])),
            "delta_pred_mean": float(np.mean([r["delta_pred_mean"] for r in all_seed_results])),
            "delta_pred_std": float(np.std([r["delta_pred_mean"] for r in all_seed_results])),
            "fp_rate_mean": float(np.mean([r["fp_rate"] for r in all_seed_results])),
            "fp_rate_std": float(np.std([r["fp_rate"] for r in all_seed_results])),
            "per_seed": all_seed_results,
        }
        out_json = os.path.join(OUT_DIR, "med3dvlm_hc_eval_multiseed.json")
        with open(out_json, "w") as f:
            json.dump(agg, f, indent=2)
        print(f"\nAggregate ({len(all_seed_results)} seeds):")
        print(f"  MAE={agg['mae_mean']:.4f}±{agg['mae_std']:.4f}")
        print(f"  SSIM={agg['ssim_mean']:.4f}±{agg['ssim_std']:.4f}")
        print(f"  PSNR={agg['psnr_mean']:.2f}±{agg['psnr_std']:.2f}")
        print(f"  delta_pred={agg['delta_pred_mean']*100:.1f}%±{agg['delta_pred_std']*100:.1f}%")
        print(f"  FP rate={agg['fp_rate_mean']*100:.1f}%±{agg['fp_rate_std']*100:.1f}%")
        print(f"Saved {out_json}")


if __name__ == "__main__":
    main()
