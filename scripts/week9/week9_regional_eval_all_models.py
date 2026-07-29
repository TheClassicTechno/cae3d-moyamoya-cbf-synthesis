#!/usr/bin/env python3
"""
Regional evaluation (and thus ΔCBF) for all 3D models in the same way as unet3d/resnet3d.

For each model: run test-set inference, write prediction NIfTIs to week8_regional_preds/<Model>/,
then run week7_regional_eval.run_from_pred_dir to produce week8_regional_<Model>.json.
After this, run week9_delta_cbf_by_territory.py with all regional JSONs to get ΔCBF for every model.

Run from <repo-root>:
  python scripts/week9/week9_regional_eval_all_models.py
  python scripts/week9/week9_regional_eval_all_models.py --only FNO_3D

Then:
  python scripts/week9/week9_delta_cbf_by_territory.py --regional_json week8_regional_FNO_3D.json ... --output_dir week9_stats
  (or add the new JSONs to the default list in that script)
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
import argparse
import numpy as np
import nibabel as nib

ROOT = _REPO_ROOT
PRED_BASE = os.path.join(ROOT, "week8_regional_preds")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from week7_data import get_week7_splits, _subject_id_from_path, Week7VolumePairs3D
from week7_preprocess import load_volume, TARGET_SHAPE
from week7_regional_eval import run_from_pred_dir

WEEK7_ORIGINAL = (91, 109, 91)
PAD_3D = (96, 112, 96)


def _pad_3d(pre_t, post_t, target_shape):
    if (pre_t.shape[2], pre_t.shape[3], pre_t.shape[4]) == target_shape:
        return pre_t, post_t
    import torch.nn.functional as F
    th, tw, td = target_shape
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


def _write_pred_nifti(pred_vol: np.ndarray, post_path: str, out_path: str):
    """Write pred_vol (91,109,91) to out_path using affine from post_path."""
    aff = nib.load(post_path).affine if os.path.isfile(post_path) else np.eye(4)
    if pred_vol.shape != TARGET_SHAPE:
        from scipy.ndimage import zoom
        factors = [TARGET_SHAPE[i] / pred_vol.shape[i] for i in range(3)]
        pred_vol = zoom(pred_vol.astype(np.float32), factors, order=1)
    nib.save(nib.Nifti1Image(pred_vol.astype(np.float32), aff), out_path)


def write_pred_niftis_unet_3d(pred_dir: str) -> int:
    """UNet_3D: single forward pass."""
    import torch
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
    os.makedirs(pred_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = make_unet_3d().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
    model.eval()
    _, _, test_pairs = get_week7_splits()
    test_ds = Week7VolumePairs3D(test_pairs, augment=False)
    n = 0
    with torch.no_grad():
        for i in range(len(test_pairs)):
            pre_path, post_path = test_pairs[i]
            sid = _subject_id_from_path(pre_path)
            pre_t, post_t = test_ds[i]
            pre_t = pre_t.unsqueeze(0).to(device)
            post_t = post_t.unsqueeze(0).to(device)
            pre_t, post_t = _pad_3d_if_needed(pre_t, post_t, PAD_3D)
            pred_t = model(pre_t)
            pred_np = pred_t.cpu().numpy()
            post_np = post_t.cpu().numpy()
            pred_np, post_np = _crop_to_91(pred_np, post_np)
            pred_vol = pred_np[0, 0]
            _write_pred_nifti(pred_vol, post_path, os.path.join(pred_dir, f"post_{sid}_pred.nii.gz"))
            n += 1
    print("Wrote", n, "pred NIfTIs for UNet_3D to", pred_dir)
    return n


def write_pred_niftis_cold_3d(pred_dir: str) -> int:
    """Cold Diffusion 3D."""
    import torch
    cold_dir = os.path.join(ROOT, "Diffusion_ColdDiffusion_3D")
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
        _pad_3d_cold = cold_model_3d._pad_3d
    except Exception as e:
        print("Skip Cold_3D:", e)
        return 0
    ckpt_path = os.path.join(cold_dir, "cold_diffusion_3d_week7_best.pt")
    if not os.path.isfile(ckpt_path):
        print("Skip Cold_3D: checkpoint not found", ckpt_path)
        return 0
    os.makedirs(pred_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ColdDiffusionNet3D().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
    model.eval()
    alpha = make_alpha_schedule(100)
    alpha_t = torch.from_numpy(alpha).float().to(device)
    _, _, test_pairs = get_week7_splits()
    test_ds = Week7VolumePairs3D(test_pairs, augment=False)
    n = 0
    with torch.no_grad():
        for i in range(len(test_pairs)):
            pre_path, post_path = test_pairs[i]
            sid = _subject_id_from_path(pre_path)
            pre_t, post_t = test_ds[i]
            pre_t = pre_t.unsqueeze(0).to(device)
            post_t = post_t.unsqueeze(0).to(device)
            pre_t, post_t = _pad_3d_cold(pre_t, post_t, PAD_3D)
            pred_t = cold_sample(model, pre_t, 100, alpha_t, device)
            pred_np = pred_t.cpu().numpy()
            post_np = post_t.cpu().numpy()
            pred_np, post_np = _crop_to_91(pred_np, post_np)
            pred_vol = pred_np[0, 0]
            _write_pred_nifti(pred_vol, post_path, os.path.join(pred_dir, f"post_{sid}_pred.nii.gz"))
            n += 1
    print("Wrote", n, "pred NIfTIs for Cold_3D to", pred_dir)
    return n


def write_pred_niftis_residual_3d(pred_dir: str) -> int:
    """Residual Diffusion 3D."""
    import torch
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
    os.makedirs(pred_dir, exist_ok=True)
    _, _, test_pairs = get_week7_splits()
    test_ds = Week7VolumePairs3D(test_pairs, augment=False)
    n = 0
    with torch.no_grad():
        for i in range(len(test_pairs)):
            pre_path, post_path = test_pairs[i]
            sid = _subject_id_from_path(pre_path)
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
            _write_pred_nifti(pred_vol, post_path, os.path.join(pred_dir, f"post_{sid}_pred.nii.gz"))
            n += 1
    print("Wrote", n, "pred NIfTIs for Residual_3D to", pred_dir)
    return n


def write_pred_niftis_residual_3d_tips(pred_dir: str) -> int:
    """Residual Diffusion 3D with tips (DDIM + overlap; ImprovedResidualDiffusionNet3D). Week7 checkpoint."""
    import torch
    import importlib.util

    res_dir = os.path.join(ROOT, "Diffusion_ResidualDiffusion_3D")
    ckpt_path = os.path.join(res_dir, "residual_diffusion_3d_tips_week7_best.pt")
    if not os.path.isfile(ckpt_path):
        print("Skip Residual_3D_tips: checkpoint not found", ckpt_path)
        return 0
    tips_path = os.path.join(res_dir, "model_3d_with_tips.py")
    spec = importlib.util.spec_from_file_location("residual_model_3d_tips", tips_path)
    tips_m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tips_m)
    ImprovedResidualDiffusionNet3D = tips_m.ImprovedResidualDiffusionNet3D
    make_beta_schedule = tips_m.make_beta_schedule
    predict_full_volume_with_overlap = tips_m.predict_full_volume_with_overlap
    minmax_norm = tips_m.minmax_norm
    load_volume_week7 = tips_m.load_volume_week7

    n_timesteps_train = 500
    n_steps_ddim = 25
    residual_scale = 0.2
    schedule = "cosine"
    patch_size = (48, 48, 24)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    betas = make_beta_schedule(n_timesteps_train, schedule=schedule)
    alphas = 1.0 - betas
    alphas_bar = np.cumprod(alphas)
    alphas_bar_sqrt = np.sqrt(alphas_bar)
    one_minus_alphas_bar_sqrt = np.sqrt(1.0 - alphas_bar)
    betas_t = torch.from_numpy(betas).float().to(device)
    alphas_t = torch.from_numpy(alphas).float().to(device)
    alphas_bar_sqrt_t = torch.from_numpy(alphas_bar_sqrt).float().to(device)
    one_minus_alphas_bar_sqrt_t = torch.from_numpy(one_minus_alphas_bar_sqrt).float().to(device)

    model = ImprovedResidualDiffusionNet3D(channels=(64, 128, 256)).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
    model.eval()
    os.makedirs(pred_dir, exist_ok=True)
    _, _, test_pairs = get_week7_splits()
    n = 0
    with torch.no_grad():
        for pre_path, post_path in test_pairs:
            if not os.path.isfile(pre_path) or not os.path.isfile(post_path):
                continue
            sid = _subject_id_from_path(pre_path)
            try:
                pre_vol = minmax_norm(load_volume_week7(pre_path, pad_shape=(96, 112, 96)))
            except Exception as e:
                print("Skip", sid, e)
                continue
            pred_vol_np = predict_full_volume_with_overlap(
                model,
                pre_vol,
                n_timesteps_train,
                n_steps_ddim,
                betas_t,
                alphas_t,
                alphas_bar_sqrt_t,
                one_minus_alphas_bar_sqrt_t,
                residual_scale,
                patch_size=patch_size,
                overlap=0.5,
                device=device,
            )
            pred_vol_np = np.nan_to_num(pred_vol_np, nan=0.0, posinf=1.0, neginf=0.0)
            pred_vol_np = np.clip(pred_vol_np, 0.0, 1.0)
            pred_vol_np = pred_vol_np[: WEEK7_ORIGINAL[0], : WEEK7_ORIGINAL[1], : WEEK7_ORIGINAL[2]]
            _write_pred_nifti(pred_vol_np, post_path, os.path.join(pred_dir, f"post_{sid}_pred.nii.gz"))
            n += 1
    print("Wrote", n, "pred NIfTIs for Residual_3D_tips to", pred_dir)
    return n


def write_pred_niftis_ddpm_3d(pred_dir: str) -> int:
    """DDPM 3D."""
    import torch
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
    os.makedirs(pred_dir, exist_ok=True)
    _, _, test_pairs = get_week7_splits()
    test_ds = Week7VolumePairs3D(test_pairs, augment=False)
    n = 0
    with torch.no_grad():
        for i in range(len(test_pairs)):
            pre_path, post_path = test_pairs[i]
            sid = _subject_id_from_path(pre_path)
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
            _write_pred_nifti(pred_vol, post_path, os.path.join(pred_dir, f"post_{sid}_pred.nii.gz"))
            n += 1
    print("Wrote", n, "pred NIfTIs for DDPM_3D to", pred_dir)
    return n


def write_pred_niftis_fno_3d(pred_dir: str) -> int:
    """FNO 3D."""
    import torch
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
    os.makedirs(pred_dir, exist_ok=True)
    _, _, test_pairs = get_week7_splits()
    test_ds = Week7VolumePairsFNO(test_pairs)
    n = 0
    with torch.no_grad():
        for i in range(len(test_pairs)):
            pre_path, post_path = test_pairs[i]
            sid = _subject_id_from_path(pre_path)
            pre_t, post_t = test_ds[i]
            pre_t = pre_t.unsqueeze(0).to(device)
            post_t = post_t.unsqueeze(0).to(device)
            pred_t = model(pre_t)
            pred_t = torch.clamp(pred_t, 0.0, 1.0)
            pred_np = pred_t.cpu().numpy()
            post_np = post_t.cpu().numpy()
            pred_np, post_np = _crop_to_91(pred_np, post_np)
            pred_vol = pred_np[0, 0]
            _write_pred_nifti(pred_vol, post_path, os.path.join(pred_dir, f"post_{sid}_pred.nii.gz"))
            n += 1
    print("Wrote", n, "pred NIfTIs for FNO_3D to", pred_dir)
    return n


def write_pred_niftis_hybrid_3d(pred_dir: str) -> int:
    """Hybrid_3D: same inference as week9_export_hybrid3d_per_subject."""
    import torch
    hybrid_dir = os.path.join(ROOT, "Hybrid_UNet_Diffusion")
    sys.path.insert(0, hybrid_dir)
    sys.path.insert(0, os.path.join(ROOT, "Diffusion_3D_Latent"))
    try:
        from unet_diffusion_hybrid import load_volume_week7, strict_normalize_volume, p_sample_ddim_hybrid, make_beta_schedule, HybridDiffusionNet
        from vae_3d import VAE3D
        from monai.networks.nets import UNet
    except Exception as e:
        print("Skip Hybrid_3D:", e)
        return 0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    target_size = (96, 112, 96)
    n_timesteps_train, n_steps_ddim, residual_scale = 1000, 25, 0.5
    vae_path = os.path.join(ROOT, "Diffusion_3D_Latent", "vae_3d_week7_best.pt")
    if not os.path.isfile(vae_path):
        vae_path = os.path.join(ROOT, "Diffusion_3D_Latent", "vae_3d_best.pt")
    unet_path = os.path.join(ROOT, "UNet_3D", "unet_3d_week7_best.pt")
    ckpt_path = os.path.join(hybrid_dir, "hybrid_unet_diffusion_week7_best.pt")
    if not all(os.path.isfile(p) for p in (ckpt_path, unet_path, vae_path)):
        print("Skip Hybrid_3D: missing checkpoint", ckpt_path, unet_path, vae_path)
        return 0
    vae_model = VAE3D(in_channels=1, latent_channels=4).to(device)
    vae_model.load_state_dict(torch.load(vae_path, map_location=device))
    vae_model.eval()
    unet_model = UNet(spatial_dims=3, in_channels=1, out_channels=1, channels=(16, 32, 64, 128), strides=(2, 2, 2), num_res_units=2, act=("LeakyReLU", {"inplace": True}), norm="INSTANCE", dropout=0.0).to(device)
    unet_model.load_state_dict(torch.load(unet_path, map_location=device))
    unet_model.eval()
    model = HybridDiffusionNet(latent_channels=4, channels=(32, 64, 128), use_v_prediction=False).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    betas, alphas, alpha_bar = make_beta_schedule(n_timesteps=n_timesteps_train, schedule="cosine")
    alphas_bar_sqrt = torch.from_numpy(np.sqrt(alpha_bar)).float().to(device)
    one_minus_alphas_bar_sqrt = torch.from_numpy(np.sqrt(1.0 - alpha_bar)).float().to(device)
    betas_t = torch.from_numpy(betas).float().to(device)
    alphas_t = torch.from_numpy(alphas).float().to(device)
    _, _, test_pairs = get_week7_splits()
    os.makedirs(pred_dir, exist_ok=True)
    n = 0
    with torch.no_grad():
        for pre_p, post_p in test_pairs:
            if not os.path.isfile(pre_p) or not os.path.isfile(post_p):
                continue
            sid = _subject_id_from_path(pre_p)
            try:
                pre_vol = strict_normalize_volume(load_volume_week7(pre_p, pad_shape=target_size))
                post_vol = strict_normalize_volume(load_volume_week7(post_p, pad_shape=target_size))
            except Exception as e:
                continue
            try:
                pred_vol = p_sample_ddim_hybrid(model, vae_model, unet_model, pre_vol, n_timesteps_train, n_steps_ddim, betas_t, alphas_t, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, residual_scale, target_size, device)
                pred_np = pred_vol[0, 0].cpu().numpy()
            except Exception as e:
                continue
            if pred_np.shape[-3:] != WEEK7_ORIGINAL:
                pred_np = pred_np[:WEEK7_ORIGINAL[0], :WEEK7_ORIGINAL[1], :WEEK7_ORIGINAL[2]]
            _write_pred_nifti(pred_np, post_p, os.path.join(pred_dir, f"post_{sid}_pred.nii.gz"))
            n += 1
    print("Wrote", n, "pred NIfTIs for Hybrid_3D to", pred_dir)
    return n


def write_pred_niftis_patch_3d(pred_dir: str) -> int:
    """Patch_3D: same inference as week9_export_patch3d_per_subject."""
    import torch
    patch_dir = os.path.join(ROOT, "Diffusion_3D_PatchVolume")
    sys.path.insert(0, patch_dir)
    sys.path.insert(0, os.path.join(ROOT, "Diffusion_3D_Latent"))
    try:
        from utils import strict_normalize_volume
        from patch_volume_vae import PatchVolumeVAE, extract_patches, reconstruct_volume_from_patches
    except Exception as e:
        print("Skip Patch_3D:", e)
        return 0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    target_size = (96, 96, 96)
    patch_size = (24, 24, 24)
    stride = 12
    vae_ckpt = os.path.join(patch_dir, "patch_vae_week7_best.pt")
    if not os.path.isfile(vae_ckpt):
        print("Skip Patch_3D: checkpoint not found", vae_ckpt)
        return 0
    vae_model = PatchVolumeVAE(
        in_channels=1, latent_channels=4,
        encoder_channels=(32, 64, 128), decoder_channels=(128, 64, 32, 1),
        num_res_blocks=2, patch_size=patch_size,
    ).to(device)
    vae_model.load_state_dict(torch.load(vae_ckpt, map_location=device))
    vae_model.eval()

    def load_volume_week7(nii_path, pad_shape=(96, 96, 96)):
        vol = load_volume(nii_path, target_shape=TARGET_SHAPE, apply_mask=True, minmax=True)
        if vol.shape != pad_shape:
            out = np.zeros(pad_shape, dtype=vol.dtype)
            sh = [min(vol.shape[i], pad_shape[i]) for i in range(3)]
            out[:sh[0], :sh[1], :sh[2]] = vol[:sh[0], :sh[1], :sh[2]]
            return out.astype(np.float32)
        return vol.astype(np.float32)

    _, _, test_pairs = get_week7_splits()
    os.makedirs(pred_dir, exist_ok=True)
    n = 0
    with torch.no_grad():
        for pre_p, post_p in test_pairs:
            if not os.path.isfile(pre_p) or not os.path.isfile(post_p):
                continue
            sid = _subject_id_from_path(pre_p)
            try:
                pre_vol = strict_normalize_volume(load_volume_week7(pre_p, pad_shape=target_size))
                post_vol = strict_normalize_volume(load_volume_week7(post_p, pad_shape=target_size))
            except Exception as e:
                continue
            try:
                pre_patches, patch_coords = extract_patches(pre_vol, patch_size, stride)
                predicted_patches = []
                for pre_patch in pre_patches:
                    pre_t = torch.from_numpy(pre_patch).unsqueeze(0).unsqueeze(0).float().to(device)
                    pre_latent = vae_model.encode_to_latent(pre_t)
                    pred_patch = vae_model.decode_from_latent(pre_latent)
                    predicted_patches.append(pred_patch[0, 0].cpu().numpy())
                pred_vol = reconstruct_volume_from_patches(
                    predicted_patches, patch_coords, target_size, patch_size, stride
                )
            except Exception as e:
                continue
            if np.isnan(pred_vol).any() or np.isinf(pred_vol).any():
                continue
            if pred_vol.shape != WEEK7_ORIGINAL:
                pred_vol = pred_vol[:91, :109, :91].copy()
            _write_pred_nifti(pred_vol, post_p, os.path.join(pred_dir, f"post_{sid}_pred.nii.gz"))
            n += 1
    print("Wrote", n, "pred NIfTIs for Patch_3D to", pred_dir)
    return n


# Map model name -> (write_pred_niftis_fn, regional_json_name)
MODELS = {
    "UNet_3D": (write_pred_niftis_unet_3d, "UNet_3D"),
    "Cold_3D": (write_pred_niftis_cold_3d, "Cold_3D"),
    "Residual_3D": (write_pred_niftis_residual_3d, "Residual_3D"),
    "Residual_3D_tips": (write_pred_niftis_residual_3d_tips, "Residual_3D_tips"),
    "DDPM_3D": (write_pred_niftis_ddpm_3d, "DDPM_3D"),
    "FNO_3D": (write_pred_niftis_fno_3d, "FNO_3D"),
    "Hybrid_3D": (write_pred_niftis_hybrid_3d, "Hybrid_3D"),
    "Patch_3D": (write_pred_niftis_patch_3d, "Patch_3D"),
}


def main():
    ap = argparse.ArgumentParser(description="Regional eval (and ΔCBF) for all 3D models")
    ap.add_argument("--only", type=str, default="", help="Run only this model (e.g. FNO_3D)")
    ap.add_argument("--skip-nifti", action="store_true", help="Skip writing NIfTIs; only run regional eval from existing pred dirs")
    args = ap.parse_args()

    models_to_run = list(MODELS.keys()) if not args.only else [args.only.strip()]
    if args.only and args.only.strip() not in MODELS:
        print("Unknown --only. Choose from:", list(MODELS.keys()))
        return

    post_dir = os.path.join(ROOT, "post")
    masks_dir = os.path.join(ROOT, "Masks") if os.path.isdir(os.path.join(ROOT, "Masks")) else None

    for model_name in models_to_run:
        write_fn, json_name = MODELS[model_name]
        pred_dir = os.path.join(PRED_BASE, model_name)
        if not args.skip_nifti:
            count = write_fn(pred_dir)
            if count == 0:
                continue
        if not os.path.isdir(pred_dir):
            print("Skip regional eval for", model_name, "(no pred dir)")
            continue
        out_path = os.path.join(ROOT, f"week8_regional_{json_name}.json")
        run_from_pred_dir(pred_dir, post_dir, masks_dir, out_path)

    print("Done. Run week9_delta_cbf_by_territory.py with all week8_regional_*.json to update ΔCBF CSV.")


if __name__ == "__main__":
    main()
