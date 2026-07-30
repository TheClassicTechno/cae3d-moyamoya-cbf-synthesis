#!/usr/bin/env python3
"""
SAM-Med3D HC False-Positive Evaluation
=======================================
Load the already-trained SAM-Med3D CBF regressor (frozen image encoder +
trainable decoder) and run inference-only on the 64 healthy control subjects.

Mirrors the CAE3D HC FP analysis (scripts/hc_false_positive_eval.py) so that
results are directly comparable in the paper.

Checkpoints: third_party_foundation_3d/sam_med3d_week7_cbf/
  - sam_med3d_cbf_decoder_best_seed42.pt   (frozen encoder, decoder only)
  - sam_med3d_cbf_decoder_best_seed123.pt
  - sam_med3d_cbf_decoder_best_seed456.pt

Run from repo root:
  PYTHONPATH=/data1/julih/scripts:/data1/julih/third_party_foundation_3d/SAM-Med3D \\
    python3 third_party_foundation_3d/run_sam_med3d_hc_eval.py
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
SAM_REPO = os.path.join(ROOT, "third_party_foundation_3d", "SAM-Med3D")
for p in (REPO_SCRIPTS, SAM_REPO):
    if p not in sys.path:
        sys.path.insert(0, p)

from week7_preprocess import TARGET_SHAPE, load_volume, get_brain_mask_for_shape, metrics_in_brain

HC_CBF_ROOT = os.path.join(ROOT, "healthy_controls_CBF")
CKPT_DIR = os.path.join(ROOT, "third_party_foundation_3d", "sam_med3d_week7_cbf")
OUT_DIR = os.path.join(ROOT, "third_party_foundation_3d", "hc_eval", "SAM_Med3D")
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_SHAPE_3D = TARGET_SHAPE
UP_SPATIAL = 128
FP_THRESHOLD = -0.05

# Frozen-encoder checkpoints (trained with run_sam_med3d_week7_cbf_regressor.py)
SEEDS_AND_CKPTS = [
    (42,  os.path.join(CKPT_DIR, "sam_med3d_cbf_decoder_best_seed42.pt")),
    (123, os.path.join(CKPT_DIR, "sam_med3d_cbf_decoder_best_seed123.pt")),
    (456, os.path.join(CKPT_DIR, "sam_med3d_cbf_decoder_best_seed456.pt")),
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


def load_sam_med3d():
    import medim
    ckpt_path = "https://huggingface.co/blueyo0/SAM-Med3D/blob/main/sam_med3d_turbo.pth"
    for loc in [
        os.path.join(ROOT, "third_party_foundation_3d", "SAM-Med3D", "ckpt", "sam_med3d_turbo.pth"),
        os.path.join(ROOT, "third_party_foundation_3d", "ckpt", "sam_med3d_turbo.pth"),
    ]:
        if os.path.isfile(loc):
            ckpt_path = loc
            break
    return medim.create_model("SAM-Med3D", pretrained=True, checkpoint_path=ckpt_path)


def _sam_norm_1ch(sam: nn.Module, x: torch.Tensor) -> torch.Tensor:
    pm = sam.pixel_mean.to(x.device).flatten()[0].view(1, 1, 1, 1, 1)
    ps = sam.pixel_std.to(x.device).flatten()[0].view(1, 1, 1, 1, 1)
    return (x - pm) / (ps + 1e-8)


@torch.no_grad()
def probe_encoder_out(sam: nn.Module) -> tuple[int, int, int, int]:
    isz = int(sam.image_encoder.img_size)
    dummy = torch.zeros(1, 1, isz, isz, isz, device=DEVICE)
    x = _sam_norm_1ch(sam, dummy * 255.0)
    emb = sam.image_encoder(x)
    _b, c, d, h, w = emb.shape
    return c, d, h, w


def make_decoder(in_ch: int, grid0: int) -> nn.Module:
    layers: list[nn.Module] = []
    c = in_ch
    d = grid0
    while d < UP_SPATIAL:
        out_c = max(c // 2, 16)
        layers += [
            nn.ConvTranspose3d(c, out_c, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(out_c),
            nn.GELU(),
        ]
        c = out_c
        d *= 2
    layers.append(nn.Conv3d(c, 1, kernel_size=1))
    return nn.Sequential(*layers)


def preprocess_for_sam(sam: nn.Module, vol_np: np.ndarray) -> torch.Tensor:
    """vol_np: (H,W,D) float32 in [0,1] -> SAM encoder input (1,1,S,S,S)."""
    t = torch.from_numpy(vol_np).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
    x = _sam_norm_1ch(sam, t * 255.0)
    isz = int(sam.image_encoder.img_size)
    return F.interpolate(x, size=(isz, isz, isz), mode="trilinear", align_corners=False)


def eval_one_seed(seed: int, ckpt_path: str, pairs: list[dict], sam: nn.Module, brain_mask: np.ndarray) -> dict:
    print(f"  seed={seed}, ckpt={os.path.basename(ckpt_path)}")
    c0, g, _, _ = probe_encoder_out(sam)
    decoder = make_decoder(c0, g).to(DEVICE)

    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    decoder.load_state_dict(state["decoder"])
    decoder.eval()
    sam.image_encoder.eval()

    mask_t = torch.from_numpy(brain_mask).float().to(DEVICE)
    mask_np = brain_mask

    results = []
    for item in pairs:
        subject = item["subject"]
        pre_vol = load_volume(item["pre_path"], target_shape=TARGET_SHAPE_3D, apply_mask=True, minmax=True)
        post_vol = load_volume(item["post_path"], target_shape=TARGET_SHAPE_3D, apply_mask=True, minmax=True)

        x_in = preprocess_for_sam(sam, pre_vol)
        with torch.no_grad():
            emb = sam.image_encoder(x_in)
            y = decoder(emb)
            pred_t = F.interpolate(y, size=TARGET_SHAPE_3D, mode="trilinear", align_corners=False)
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

    print("Loading SAM-Med3D base model...")
    sam = load_sam_med3d().to(DEVICE)
    for p in sam.image_encoder.parameters():
        p.requires_grad = False

    all_seed_results = []
    for seed, ckpt_path in SEEDS_AND_CKPTS:
        if not os.path.isfile(ckpt_path):
            print(f"  SKIP seed={seed}: {ckpt_path} not found")
            continue
        r = eval_one_seed(seed, ckpt_path, pairs, sam, brain_mask)
        all_seed_results.append(r)
        seed_json = os.path.join(OUT_DIR, f"sam_med3d_hc_eval_seed{seed}.json")
        with open(seed_json, "w") as f:
            json.dump(r, f, indent=2)
        print(f"  seed={seed}: MAE={r['mae_mean']:.4f} SSIM={r['ssim_mean']:.4f} PSNR={r['psnr_mean']:.2f} "
              f"delta_pred={r['delta_pred_mean']*100:.1f}% FP={r['fp_rate']*100:.1f}%")

    if all_seed_results:
        agg = {
            "model": "SAM-Med3D (frozen encoder + trained decoder)",
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
        out_json = os.path.join(OUT_DIR, "sam_med3d_hc_eval_multiseed.json")
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
