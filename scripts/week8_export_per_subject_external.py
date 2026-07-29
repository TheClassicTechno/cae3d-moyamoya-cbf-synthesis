#!/usr/bin/env python3
"""
Export per-subject metrics for UNet_3D, Cold_3D, Residual_3D, DDPM_3D, FNO_3D, Hybrid_3D, Patch_3D.

Default: test split only -> week8_per_subject_metrics/<model>_<subject_id>.json

Use --splits train,val,test --out-dir <path> for the full combined cohort (see
week8_export_per_subject_full_cohort.py). Each JSON may include "split": "train"|"val"|"test".

Each model needs its Week 7 checkpoint (e.g. unet_3d_week7_best.pt).
"""
from __future__ import annotations

import os

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_REPO_ROOT, "pyproject.toml")):
    _parent = os.path.dirname(_REPO_ROOT)
    if _parent == _REPO_ROOT:
        raise RuntimeError("Could not locate repository root (pyproject.toml not found)")
    _REPO_ROOT = _parent

import os
import sys
import json
import traceback

ROOT = _REPO_ROOT
OUT_DIR = os.path.join(ROOT, "week8_per_subject_metrics")
os.makedirs(OUT_DIR, exist_ok=True)

# Ensure scripts and week7 data/preprocess are importable
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from week7_data import get_week7_splits, Week7VolumePairs3D
from week7_preprocess import (
    load_volume,
    TARGET_SHAPE,
    metrics_in_brain,
    get_brain_mask,
    get_pre_post_pairs_with_subject_id,
    collect_pre_post_quads_by_splits,
)

import numpy as np
import torch

WEEK7_ORIGINAL = (91, 109, 91)
PAD_3D = (96, 112, 96)


def _resolve_quads(split_keys):
    """List of (subject_id, pre, post, split_name)."""
    return collect_pre_post_quads_by_splits(split_keys)


def _pad_3d(pre_t, post_t, target_shape):
    if (pre_t.shape[2], pre_t.shape[3], pre_t.shape[4]) == target_shape:
        return pre_t, post_t
    th, tw, td = target_shape
    import torch.nn.functional as F
    _, _, h, w, d = pre_t.shape
    ph, pw, pd = max(0, th - h), max(0, tw - w), max(0, td - d)
    if ph or pw or pd:
        pre_t = F.pad(pre_t, (0, pd, 0, pw, 0, ph), mode="constant", value=0)
        post_t = F.pad(post_t, (0, pd, 0, pw, 0, ph), mode="constant", value=0)
    return pre_t[:, :, :th, :tw, :td], post_t[:, :, :th, :tw, :td]


def _crop_to_91(pred_np, post_np):
    if pred_np.shape[-3:] != WEEK7_ORIGINAL:
        pred_np = pred_np[:, :, :WEEK7_ORIGINAL[0], :WEEK7_ORIGINAL[1], :WEEK7_ORIGINAL[2]]
        post_np = post_np[:, :, :WEEK7_ORIGINAL[0], :WEEK7_ORIGINAL[1], :WEEK7_ORIGINAL[2]]
    return pred_np, post_np


def _brain_mean(pred_vol, target_vol, mask=None):
    if mask is None:
        mask = get_brain_mask()
    if mask.shape != pred_vol.shape:
        from scipy.ndimage import zoom as _z
        factors = [pred_vol.shape[i] / mask.shape[i] for i in range(3)]
        mask = _z(mask.astype(np.float32), factors, order=0)
    mask = (mask > 0.5).astype(np.float32)
    n = mask.sum() + 1e-8
    return float((pred_vol * mask).sum() / n), float((target_vol * mask).sum() / n)


def export_unet_3d(out_dir=None, split_keys=None):
    """UNet_3D (tips): single forward pass."""
    out_dir = out_dir or OUT_DIR
    split_keys = split_keys or ["test"]
    os.makedirs(out_dir, exist_ok=True)
    sys.path.insert(0, os.path.join(ROOT, "UNet_3D"))
    try:
        from model_3d import make_unet_3d, _pad_3d_if_needed
    except Exception as e:
        print("Skip UNet_3D:", e)
        return 0
    ckpt_path = os.path.join(ROOT, "UNet_3D", "unet_3d_week7_best.pt")
    if not os.path.isfile(ckpt_path):
        print("Skip UNet_3D: checkpoint not found", ckpt_path)
        return 0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = make_unet_3d().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
    model.eval()
    quads = _resolve_quads(split_keys)
    pairs = [(p, q) for (_, p, q, _) in quads]
    test_ds = Week7VolumePairs3D(pairs, augment=False)
    n = 0
    with torch.no_grad():
        for i, (sid, _, _, split_name) in enumerate(quads):
            pre_t, post_t = test_ds[i]
            pre_t = pre_t.unsqueeze(0).to(device)
            post_t = post_t.unsqueeze(0).to(device)
            pre_t, post_t = _pad_3d_if_needed(pre_t, post_t, PAD_3D)
            pred_t = model(pre_t)
            pred_np = pred_t.cpu().numpy()
            post_np = post_t.cpu().numpy()
            pred_np, post_np = _crop_to_91(pred_np, post_np)
            pred_vol = pred_np[0, 0]
            post_vol = post_np[0, 0]
            met = metrics_in_brain(pred_vol, post_vol, data_range=1.0)
            pred_mean, target_mean = _brain_mean(pred_vol, post_vol)
            out = {
                "model": "UNet_3D",
                "subject_id": sid,
                "split": split_name,
                "mae": float(met["mae_mean"]),
                "ssim": float(met["ssim_mean"]),
                "psnr": float(met["psnr_mean"]),
                "pred_mean": pred_mean,
                "target_mean": target_mean,
            }
            with open(os.path.join(out_dir, f"UNet_3D_{sid}.json"), "w") as f:
                json.dump(out, f, indent=0)
            n += 1
    print("Wrote", n, "per-subject JSONs for UNet_3D ->", out_dir)
    return n


def export_cold_3d(out_dir=None, split_keys=None):
    """Cold Diffusion 3D: iterative sampling."""
    out_dir = out_dir or OUT_DIR
    split_keys = split_keys or ["test"]
    os.makedirs(out_dir, exist_ok=True)
    cold_dir = os.path.join(ROOT, "Diffusion_ColdDiffusion_3D")
    # Ensure Cold's model_3d is used (remove UNet_3D from path if present)
    while cold_dir in sys.path:
        sys.path.remove(cold_dir)
    sys.path.insert(0, cold_dir)
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("cold_model_3d", os.path.join(cold_dir, "model_3d.py"))
        cold_model_3d = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cold_model_3d)
        ColdDiffusionNet3D = cold_model_3d.ColdDiffusionNet3D
        cold_sample = cold_model_3d.cold_sample
        make_alpha_schedule = cold_model_3d.make_alpha_schedule
        _pad_3d = cold_model_3d._pad_3d
    except Exception as e:
        print("Skip Cold_3D:", e)
        return 0
    ckpt_path = os.path.join(ROOT, "Diffusion_ColdDiffusion_3D", "cold_diffusion_3d_week7_best.pt")
    if not os.path.isfile(ckpt_path):
        print("Skip Cold_3D: checkpoint not found", ckpt_path)
        return 0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ColdDiffusionNet3D().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
    model.eval()
    alpha = make_alpha_schedule(100)
    alpha_t = torch.from_numpy(alpha).float().to(device)
    quads = _resolve_quads(split_keys)
    pairs = [(p, q) for (_, p, q, _) in quads]
    test_ds = Week7VolumePairs3D(pairs, augment=False)
    n = 0
    with torch.no_grad():
        for i, (sid, _, _, split_name) in enumerate(quads):
            pre_t, post_t = test_ds[i]
            pre_t = pre_t.unsqueeze(0).to(device)
            post_t = post_t.unsqueeze(0).to(device)
            pre_t, post_t = _pad_3d(pre_t, post_t, PAD_3D)
            pred_t = cold_sample(model, pre_t, 100, alpha_t, device)
            pred_np = pred_t.cpu().numpy()
            post_np = post_t.cpu().numpy()
            pred_np = pred_np[:, :, :91, :109, :91]
            post_np = post_np[:, :, :91, :109, :91]
            pred_vol = pred_np[0, 0]
            post_vol = post_np[0, 0]
            met = metrics_in_brain(pred_vol, post_vol, data_range=1.0)
            pred_mean, target_mean = _brain_mean(pred_vol, post_vol)
            out = {
                "model": "Cold_3D",
                "subject_id": sid,
                "split": split_name,
                "mae": float(met["mae_mean"]),
                "ssim": float(met["ssim_mean"]),
                "psnr": float(met["psnr_mean"]),
                "pred_mean": pred_mean,
                "target_mean": target_mean,
            }
            with open(os.path.join(out_dir, f"Cold_3D_{sid}.json"), "w") as f:
                json.dump(out, f, indent=0)
            n += 1
    print("Wrote", n, "per-subject JSONs for Cold_3D ->", out_dir)
    return n


def export_residual_3d(out_dir=None, split_keys=None):
    """Residual Diffusion 3D: full sampling loop over test set."""
    out_dir = out_dir or OUT_DIR
    split_keys = split_keys or ["test"]
    os.makedirs(out_dir, exist_ok=True)
    res_dir = os.path.join(ROOT, "Diffusion_ResidualDiffusion_3D")
    ckpt_path = os.path.join(res_dir, "residual_diffusion_3d_week7_best.pt")
    if not os.path.isfile(ckpt_path):
        print("Skip Residual_3D: checkpoint not found", ckpt_path)
        return 0
    import importlib.util
    spec = importlib.util.spec_from_file_location("residual_model_3d", os.path.join(res_dir, "model_3d.py"))
    res_model_3d = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(res_model_3d)
    SimpleResidualDiffusion3D = res_model_3d.SimpleResidualDiffusion3D
    make_beta_schedule = res_model_3d.make_beta_schedule
    p_sample_loop_residual = res_model_3d.p_sample_loop_residual
    res_pad_3d = res_model_3d._pad_3d
    n_timesteps = 1000
    residual_scale = 0.2
    device = "cuda" if torch.cuda.is_available() else "cpu"
    betas = make_beta_schedule(n_timesteps)
    alphas = 1.0 - betas
    alphas_bar = np.cumprod(alphas)
    alphas_bar_sqrt = torch.from_numpy(np.sqrt(alphas_bar)).float().to(device)
    one_minus_alphas_bar_sqrt = torch.from_numpy(np.sqrt(1.0 - alphas_bar)).float().to(device)
    betas_t = torch.from_numpy(betas).float().to(device)
    alphas_t = torch.from_numpy(alphas).float().to(device)
    model = SimpleResidualDiffusion3D(ch=16).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
    model.eval()
    quads = _resolve_quads(split_keys)
    pairs = [(p, q) for (_, p, q, _) in quads]
    test_ds = Week7VolumePairs3D(pairs, augment=False)
    n = 0
    with torch.no_grad():
        for i, (sid, _, _, split_name) in enumerate(quads):
            pre_t, post_t = test_ds[i]
            pre_t = pre_t.unsqueeze(0).to(device)
            post_t = post_t.unsqueeze(0).to(device)
            pre_t, post_t = res_pad_3d(pre_t, post_t, PAD_3D)
            residual_pred = p_sample_loop_residual(
                model, pre_t, n_timesteps,
                betas_t, alphas_t, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, device,
            )
            residual_pred = residual_pred * residual_scale
            pred_t = torch.clamp(pre_t + residual_pred, 0.0, 1.0)
            pred_np = pred_t.cpu().numpy()
            post_np = post_t.cpu().numpy()
            pred_np, post_np = _crop_to_91(pred_np, post_np)
            pred_vol = pred_np[0, 0]
            post_vol = post_np[0, 0]
            met = metrics_in_brain(pred_vol, post_vol, data_range=1.0)
            pred_mean, target_mean = _brain_mean(pred_vol, post_vol)
            out = {
                "model": "Residual_3D",
                "subject_id": sid,
                "split": split_name,
                "mae": float(met["mae_mean"]),
                "ssim": float(met["ssim_mean"]),
                "psnr": float(met["psnr_mean"]),
                "pred_mean": pred_mean,
                "target_mean": target_mean,
            }
            with open(os.path.join(out_dir, f"Residual_3D_{sid}.json"), "w") as f:
                json.dump(out, f, indent=0)
            n += 1
    print("Wrote", n, "per-subject JSONs for Residual_3D ->", out_dir)
    return n


def export_ddpm_3d(out_dir=None, split_keys=None):
    """DDPM 3D: full sampling loop over test set."""
    out_dir = out_dir or OUT_DIR
    split_keys = split_keys or ["test"]
    os.makedirs(out_dir, exist_ok=True)
    ddpm_dir = os.path.join(ROOT, "Diffusion_baseline_3D")
    ckpt_path = os.path.join(ddpm_dir, "ddpm_3d_week7_best.pt")
    if not os.path.isfile(ckpt_path):
        print("Skip DDPM_3D: checkpoint not found", ckpt_path)
        return 0
    import importlib.util
    spec = importlib.util.spec_from_file_location("ddpm_model_3d", os.path.join(ddpm_dir, "diffusion_model_3d.py"))
    ddpm_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ddpm_module)
    SimpleCondDiffusion3D = ddpm_module.SimpleCondDiffusion3D
    make_beta_schedule = ddpm_module.make_beta_schedule
    p_sample_loop = ddpm_module.p_sample_loop
    ddpm_pad_3d = ddpm_module._pad_3d
    n_timesteps = 1000
    device = "cuda" if torch.cuda.is_available() else "cpu"
    betas = make_beta_schedule(n_timesteps)
    alphas = 1.0 - betas
    alphas_bar = np.cumprod(alphas)
    alphas_bar_sqrt = torch.from_numpy(np.sqrt(alphas_bar)).float().to(device)
    one_minus_alphas_bar_sqrt = torch.from_numpy(np.sqrt(1.0 - alphas_bar)).float().to(device)
    betas_t = torch.from_numpy(betas).float().to(device)
    alphas_t = torch.from_numpy(alphas).float().to(device)
    model = SimpleCondDiffusion3D(ch=16).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
    model.eval()
    quads = _resolve_quads(split_keys)
    pairs = [(p, q) for (_, p, q, _) in quads]
    test_ds = Week7VolumePairs3D(pairs, augment=False)
    n = 0
    with torch.no_grad():
        for i, (sid, _, _, split_name) in enumerate(quads):
            pre_t, post_t = test_ds[i]
            pre_t = pre_t.unsqueeze(0).to(device)
            post_t = post_t.unsqueeze(0).to(device)
            pre_t, post_t = ddpm_pad_3d(pre_t, post_t, PAD_3D)
            pred_t = p_sample_loop(
                model, pre_t, n_timesteps,
                betas_t, alphas_t, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, device,
            )
            pred_np = pred_t.cpu().numpy()
            post_np = post_t.cpu().numpy()
            pred_np, post_np = _crop_to_91(pred_np, post_np)
            pred_vol = pred_np[0, 0]
            post_vol = post_np[0, 0]
            met = metrics_in_brain(pred_vol, post_vol, data_range=1.0)
            pred_mean, target_mean = _brain_mean(pred_vol, post_vol)
            out = {
                "model": "DDPM_3D",
                "subject_id": sid,
                "split": split_name,
                "mae": float(met["mae_mean"]),
                "ssim": float(met["ssim_mean"]),
                "psnr": float(met["psnr_mean"]),
                "pred_mean": pred_mean,
                "target_mean": target_mean,
            }
            with open(os.path.join(out_dir, f"DDPM_3D_{sid}.json"), "w") as f:
                json.dump(out, f, indent=0)
            n += 1
    print("Wrote", n, "per-subject JSONs for DDPM_3D ->", out_dir)
    return n


def export_fno_3d(out_dir=None, split_keys=None):
    """FNO 3D (Week7): fno_3d_week7_best.pt, Week7 pairs, per-subject JSONs."""
    out_dir = out_dir or OUT_DIR
    split_keys = split_keys or ["test"]
    os.makedirs(out_dir, exist_ok=True)
    fno_dir = os.path.join(ROOT, "NeuralOperators")
    ckpt_path = os.path.join(fno_dir, "fno_3d_week7_best.pt")
    if not os.path.isfile(ckpt_path):
        print("Skip FNO_3D: checkpoint not found", ckpt_path)
        return 0
    sys.path.insert(0, fno_dir)
    try:
        from fno_3d_cvr import SimpleFNO3D, Week7VolumePairsFNO
    except Exception as e:
        print("Skip FNO_3D:", e)
        return 0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    width = ckpt["fc0.weight"].shape[0] if isinstance(ckpt, dict) else 32
    in_ch = 4
    modes = 12 if width == 64 else 8
    model = SimpleFNO3D(in_ch=in_ch, out_ch=1, modes=modes, width=width)
    model.load_state_dict(ckpt)
    model = model.to(device)
    model.eval()
    quads = _resolve_quads(split_keys)
    pairs = [(p, q) for (_, p, q, _) in quads]
    test_ds = Week7VolumePairsFNO(pairs)
    n = 0
    with torch.no_grad():
        for i, (sid, _, _, split_name) in enumerate(quads):
            pre_t, post_t = test_ds[i]
            pre_t = pre_t.unsqueeze(0).to(device)
            post_t = post_t.unsqueeze(0).to(device)
            pred_t = model(pre_t)
            pred_t = torch.clamp(pred_t, 0.0, 1.0)
            pred_np = pred_t.cpu().numpy()
            post_np = post_t.cpu().numpy()
            pred_np, post_np = _crop_to_91(pred_np, post_np)
            pred_vol = pred_np[0, 0]
            post_vol = post_np[0, 0]
            met = metrics_in_brain(pred_vol, post_vol, data_range=1.0)
            pred_mean, target_mean = _brain_mean(pred_vol, post_vol)
            out = {
                "model": "FNO_3D",
                "subject_id": sid,
                "split": split_name,
                "mae": float(met["mae_mean"]),
                "ssim": float(met["ssim_mean"]),
                "psnr": float(met["psnr_mean"]),
                "pred_mean": pred_mean,
                "target_mean": target_mean,
            }
            with open(os.path.join(out_dir, f"FNO_3D_{sid}.json"), "w") as f:
                json.dump(out, f, indent=0)
            n += 1
    print("Wrote", n, "per-subject JSONs for FNO_3D ->", out_dir)
    return n


def export_hybrid_3d(out_dir=None, split_keys=None):
    """Run standalone Hybrid_3D per-subject export script."""
    out_dir = out_dir or OUT_DIR
    split_keys = split_keys or ["test"]
    script = os.path.join(ROOT, "scripts", "week9", "week9_export_hybrid3d_per_subject.py")
    if not os.path.isfile(script):
        print("Skip Hybrid_3D: script not found", script)
        return 0
    import subprocess

    r = subprocess.run(
        [sys.executable, script, "--out-dir", out_dir, "--splits", ",".join(split_keys)],
        cwd=ROOT,
    )
    return 0 if r.returncode != 0 else len(_resolve_quads(split_keys))


def export_patch_3d(out_dir=None, split_keys=None):
    """Run standalone Patch_3D per-subject export script if present."""
    out_dir = out_dir or OUT_DIR
    split_keys = split_keys or ["test"]
    script = os.path.join(ROOT, "scripts", "week9", "week9_export_patch3d_per_subject.py")
    if not os.path.isfile(script):
        print("Skip Patch_3D: script not found", script)
        return 0
    import subprocess

    r = subprocess.run(
        [sys.executable, script, "--out-dir", out_dir, "--splits", ",".join(split_keys)],
        cwd=ROOT,
    )
    return 0 if r.returncode != 0 else len(_resolve_quads(split_keys))


def run_all_exports(out_dir=None, split_keys=None):
    """Run every in-tree exporter (same as main())."""
    out_dir = out_dir or OUT_DIR
    split_keys = split_keys or ["test"]
    print("Exporting per-subject metrics; splits=%s out_dir=%s" % (split_keys, out_dir))
    export_unet_3d(out_dir, split_keys)
    export_cold_3d(out_dir, split_keys)
    export_residual_3d(out_dir, split_keys)
    export_ddpm_3d(out_dir, split_keys)
    export_fno_3d(out_dir, split_keys)
    export_hybrid_3d(out_dir, split_keys)
    export_patch_3d(out_dir, split_keys)
    print("Output dir:", out_dir)


def main():
    run_all_exports(OUT_DIR, ["test"])


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Export per-subject metrics (Week 7 / combined split)")
    p.add_argument(
        "--splits",
        type=str,
        default="test",
        help="Comma-separated split keys: train,val,test (default: test only)",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="Directory for JSON files (default: week8_per_subject_metrics)",
    )
    p.add_argument("--only", type=str, default="", help="Run only this model (e.g. FNO_3D)")
    args = p.parse_args()
    split_keys = [x.strip() for x in args.splits.split(",") if x.strip()]
    out_dir = args.out_dir.strip() or OUT_DIR
    if args.only:
        name = args.only.strip()
        if name == "FNO_3D":
            export_fno_3d(out_dir, split_keys)
        elif name == "Hybrid_3D":
            export_hybrid_3d(out_dir, split_keys)
        elif name == "Patch_3D":
            export_patch_3d(out_dir, split_keys)
        elif name == "UNet_3D":
            export_unet_3d(out_dir, split_keys)
        elif name == "Cold_3D":
            export_cold_3d(out_dir, split_keys)
        elif name == "Residual_3D":
            export_residual_3d(out_dir, split_keys)
        elif name == "DDPM_3D":
            export_ddpm_3d(out_dir, split_keys)
        else:
            print("Unknown --only model.")
        print("Output dir:", out_dir)
    else:
        run_all_exports(out_dir, split_keys)
