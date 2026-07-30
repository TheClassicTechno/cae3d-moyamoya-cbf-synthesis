#!/usr/bin/env python3
"""
SAM-Med3D inference on all 323 Moyamoya subjects (no training).
Reads /tmp/combined_all_323.json (pre_path, post_path pairs).
Saves results to third_party_foundation_3d/moyamoya323_eval/SAM_Med3D/

Run from repo root:
  PYTHONNOUSERSITE=1 \\
  PYTHONPATH=/data1/julih/scripts:/data1/julih/third_party_foundation_3d/SAM-Med3D \\
    /data1/julih/miniconda3/envs/julih_monai/bin/python \\
    third_party_foundation_3d/run_sam_med3d_moyamoya323_eval.py
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

SUBJECTS_JSON = "/tmp/combined_all_323.json"
CKPT_DIR = os.path.join(ROOT, "third_party_foundation_3d", "sam_med3d_week7_cbf")
OUT_DIR = os.path.join(ROOT, "third_party_foundation_3d", "moyamoya323_eval", "SAM_Med3D")
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_SHAPE_3D = TARGET_SHAPE
UP_SPATIAL = 128

SEEDS_AND_CKPTS = [
    (42,  os.path.join(CKPT_DIR, "sam_med3d_cbf_decoder_best_seed42.pt")),
    (123, os.path.join(CKPT_DIR, "sam_med3d_cbf_decoder_best_seed123.pt")),
    (456, os.path.join(CKPT_DIR, "sam_med3d_cbf_decoder_best_seed456.pt")),
]


def load_subjects(json_path: str) -> list[dict]:
    with open(json_path) as f:
        data = json.load(f)
    return [s for s in data if os.path.isfile(s.get("pre_path", "")) and os.path.isfile(s.get("post_path", ""))]


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


def _sam_norm_1ch(sam, x):
    pm = sam.pixel_mean.to(x.device).flatten()[0].view(1, 1, 1, 1, 1)
    ps = sam.pixel_std.to(x.device).flatten()[0].view(1, 1, 1, 1, 1)
    return (x - pm) / (ps + 1e-8)


@torch.no_grad()
def probe_encoder_out(sam):
    isz = int(sam.image_encoder.img_size)
    dummy = torch.zeros(1, 1, isz, isz, isz, device=DEVICE)
    x = _sam_norm_1ch(sam, dummy * 255.0)
    emb = sam.image_encoder(x)
    _b, c, d, h, w = emb.shape
    return c, d


def make_decoder(in_ch, grid0):
    layers = []
    c, d = in_ch, grid0
    while d < UP_SPATIAL:
        out_c = max(c // 2, 16)
        layers += [nn.ConvTranspose3d(c, out_c, 4, stride=2, padding=1), nn.BatchNorm3d(out_c), nn.GELU()]
        c, d = out_c, d * 2
    layers.append(nn.Conv3d(c, 1, 1))
    return nn.Sequential(*layers)


def eval_one_seed(seed, ckpt_path, subjects, sam, brain_mask):
    print(f"  seed={seed} ({len(subjects)} subjects)")
    c0, g = probe_encoder_out(sam)
    decoder = make_decoder(c0, g).to(DEVICE)
    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    decoder.load_state_dict(state["decoder"])
    decoder.eval()
    sam.image_encoder.eval()

    results = []
    for i, item in enumerate(subjects):
        pre_vol = load_volume(item["pre_path"], target_shape=TARGET_SHAPE_3D, apply_mask=True, minmax=True)
        post_vol = load_volume(item["post_path"], target_shape=TARGET_SHAPE_3D, apply_mask=True, minmax=True)

        t = torch.from_numpy(pre_vol).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
        x = _sam_norm_1ch(sam, t * 255.0)
        isz = int(sam.image_encoder.img_size)
        x_in = F.interpolate(x, size=(isz, isz, isz), mode="trilinear", align_corners=False)

        with torch.no_grad():
            emb = sam.image_encoder(x_in)
            y = decoder(emb)
            pred_t = F.interpolate(y, size=TARGET_SHAPE_3D, mode="trilinear", align_corners=False).clamp(0.0, 1.0)

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
    brain_mask = get_brain_mask_for_shape(TARGET_SHAPE_3D)

    print("Loading SAM-Med3D...")
    sam = load_sam_med3d().to(DEVICE)
    for p in sam.image_encoder.parameters():
        p.requires_grad = False

    all_seed_results = []
    for seed, ckpt_path in SEEDS_AND_CKPTS:
        if not os.path.isfile(ckpt_path):
            print(f"  SKIP seed={seed}: {ckpt_path} not found")
            continue
        r = eval_one_seed(seed, ckpt_path, subjects, sam, brain_mask)
        all_seed_results.append(r)
        with open(os.path.join(OUT_DIR, f"sam_med3d_323_seed{seed}.json"), "w") as f:
            json.dump(r, f, indent=2)
        print(f"  seed={seed}: MAE={r['mae_mean']:.4f} SSIM={r['ssim_mean']:.4f} PSNR={r['psnr_mean']:.2f}")

    if all_seed_results:
        agg = {
            "model": "SAM-Med3D",
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
        out = os.path.join(OUT_DIR, "sam_med3d_323_multiseed.json")
        with open(out, "w") as f:
            json.dump(agg, f, indent=2)
        print(f"\nAggregate: MAE={agg['mae_mean']:.4f}±{agg['mae_std']:.4f} "
              f"SSIM={agg['ssim_mean']:.4f}±{agg['ssim_std']:.4f} "
              f"PSNR={agg['psnr_mean']:.2f}±{agg['psnr_std']:.2f}")
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
