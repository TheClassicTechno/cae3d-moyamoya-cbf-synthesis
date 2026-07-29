#!/usr/bin/env python3
"""
Hybrid UNet-Diffusion Model for 3D Moyamoya CVR Prediction
==========================================================
Combines the best 3D UNet (MAE: 0.0353, SSIM: 0.8252) with 
Residual Diffusion Latent (best diffusion: MAE: 0.0728, SSIM: 0.7116).

Strategy: Two-stage refinement
1. Stage 1: Use pre-trained 3D UNet to get initial prediction
2. Stage 2: Use Residual Diffusion Latent to refine UNet prediction

The diffusion model learns to predict the residual between:
- UNet prediction (initial guess)
- Ground truth (target)

This allows diffusion to focus on fine-grained refinement rather than
generating from scratch.
"""

import os
import sys
import glob
import json
import random
import time
from typing import List, Tuple, Optional

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_REPO_ROOT, "pyproject.toml")):
    _parent = os.path.dirname(_REPO_ROOT)
    if _parent == _REPO_ROOT:
        raise RuntimeError("Could not locate repository root (pyproject.toml not found)")
    _REPO_ROOT = _parent

import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from monai.networks.nets import UNet
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr

# Import VAE and utilities from latent diffusion
sys.path.append(os.path.join(_REPO_ROOT, 'Diffusion_3D_Latent'))
from vae_3d import VAE3D
from utils import EMA, strict_normalize_volume


###############################################################################
# Helper Functions
###############################################################################

def load_full_volume(nii_path: str, target_size=(128, 128, 64)) -> np.ndarray:
    """Load a NIfTI file and resize to target_size."""
    img = nib.load(nii_path)
    data = img.get_fdata(dtype=np.float32)
    if data.ndim == 4:
        data = data[..., 0]
    zoom_factors = [target_size[i] / data.shape[i] for i in range(3)]
    resized_data = zoom(data, zoom_factors, order=1)
    return resized_data.astype(np.float32)


def load_volume_week7(nii_path: str, pad_shape=(96, 112, 96)) -> np.ndarray:
    """Load with Week7 preprocessing (91×109×91, brain mask, minmax) then pad to pad_shape."""
    import sys
    sys.path.insert(0, os.path.join(_REPO_ROOT, 'scripts'))
    from week7_preprocess import load_volume, TARGET_SHAPE
    vol = load_volume(nii_path, target_shape=TARGET_SHAPE, apply_mask=True, minmax=True)
    if vol.shape != pad_shape:
        out = np.zeros(pad_shape, dtype=vol.dtype)
        sh = [min(vol.shape[i], pad_shape[i]) for i in range(3)]
        out[:sh[0], :sh[1], :sh[2]] = vol[:sh[0], :sh[1], :sh[2]]
        return out.astype(np.float32)
    return vol.astype(np.float32)


def pre_to_post_path(pre_path: str) -> str:
    """Convert pre-scan path to post-scan path."""
    basename = os.path.basename(pre_path).replace('pre_', 'post_')
    dirname = os.path.dirname(pre_path).replace('/pre', '/post')
    if dirname == '':
        dirname = 'post'
    return os.path.join(dirname, basename)


###############################################################################
# Dataset: UNet predictions + ground truth
###############################################################################

class HybridDataset(Dataset):
    """
    Dataset that:
    1. Loads pre/post volumes
    2. Gets UNet prediction for pre volume
    3. Encodes UNet prediction and ground truth to latent space
    4. Computes residual (ground_truth - unet_prediction) in latent space
    If load_fn is provided (e.g. for Week7), use it instead of load_full_volume(., target_size).
    """
    def __init__(self, pre_paths, unet_model, vae_model, target_size=(128, 128, 64), device='cuda', load_fn=None, post_paths=None):
        self.pre_paths = pre_paths
        self.unet_model = unet_model
        self.vae_model = vae_model
        self.target_size = target_size
        self.device = device
        self.load_fn = load_fn
        if post_paths is not None:
            self.items = [(pre, post) for pre, post in zip(pre_paths, post_paths) if os.path.isfile(pre) and os.path.isfile(post)]
        else:
            self.items = [(p, pre_to_post_path(p)) for p in pre_paths
                          if os.path.exists(pre_to_post_path(p))]
        print(f"   Loaded {len(self.items)} paired volumes")
        
        # Set models to eval mode
        unet_model.eval()
        vae_model.eval()
        
        # Pre-compute UNet predictions and encode to latent space
        print("   Computing UNet predictions and encoding to latent space...")
        self.latent_pairs = []
        
        with torch.no_grad():
            for i, (pre_p, post_p) in enumerate(self.items):
                if (i + 1) % 20 == 0:
                    print(f"    Processed {i+1}/{len(self.items)} volumes...")
                
                # Load and normalize volumes
                if self.load_fn is not None:
                    pre_vol = strict_normalize_volume(self.load_fn(pre_p))
                    post_vol = strict_normalize_volume(self.load_fn(post_p))
                else:
                    pre_vol = strict_normalize_volume(load_full_volume(pre_p, target_size))
                    post_vol = strict_normalize_volume(load_full_volume(post_p, target_size))
                
                # Get UNet prediction
                pre_t = torch.from_numpy(pre_vol).unsqueeze(0).unsqueeze(0).float().to(device)
                unet_pred = unet_model(pre_t)  # (1, 1, H, W, D)
                unet_pred_np = unet_pred[0, 0].cpu().numpy()
                unet_pred_np = np.clip(unet_pred_np, 0.0, 1.0)  # Ensure [0, 1]
                
                # Encode to latent space
                unet_pred_t = torch.from_numpy(unet_pred_np).unsqueeze(0).unsqueeze(0).float().to(device)
                post_t = torch.from_numpy(post_vol).unsqueeze(0).unsqueeze(0).float().to(device)
                
                unet_pred_latent = vae_model.encode_to_latent(unet_pred_t)
                post_latent = vae_model.encode_to_latent(post_t)
                
                # Store: (unet_pred_latent, post_latent)
                # The residual will be computed during training
                self.latent_pairs.append((unet_pred_latent.cpu(), post_latent.cpu()))
        
        print(f"   Pre-computed all UNet predictions and latent encodings")
    
    def __len__(self):
        return len(self.latent_pairs)
    
    def __getitem__(self, idx):
        unet_pred_latent, post_latent = self.latent_pairs[idx]
        return unet_pred_latent.squeeze(0), post_latent.squeeze(0)


###############################################################################
# Residual Diffusion Model (refines UNet predictions)
###############################################################################

class HybridDiffusionNet(nn.Module):
    """
    Residual Diffusion model that refines UNet predictions.
    
    Input: noisy_residual_latent + unet_pred_latent
    Output: predicted noise in residual space
    
    The residual is: post_latent - unet_pred_latent
    """
    def __init__(self, latent_channels=4, channels=(32, 64, 128), use_v_prediction=False):
        super().__init__()
        self.use_v_prediction = use_v_prediction
        self.unet = UNet(
            spatial_dims=3,
            in_channels=latent_channels * 2,  # noisy_residual + unet_pred_latent
            out_channels=latent_channels,  # Predicted noise in residual latent space
            channels=list(channels),
            strides=(2, 2),
            num_res_units=2,
            act=("LeakyReLU", {"inplace": True}),
            norm="INSTANCE",
            dropout=0.1,
        )
    
    def forward(self, noisy_residual_latent, unet_pred_latent, t):
        """Concatenate inputs for conditioning."""
        x = torch.cat([noisy_residual_latent, unet_pred_latent], dim=1)
        return self.unet(x)


###############################################################################
# Diffusion Process
###############################################################################

def make_beta_schedule(n_timesteps=1000, beta_start=0.0001, beta_end=0.02, schedule='cosine'):
    """Create beta schedule (cosine recommended)."""
    if schedule == 'cosine':
        steps = np.arange(n_timesteps + 1, dtype=np.float64) / n_timesteps
        s = 0.008
        alpha_bar = np.cos((steps + s) / (1 + s) * np.pi / 2) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]  # Normalize so alpha_bar[0] = 1
        alpha_bar = np.maximum.accumulate(alpha_bar[::-1])[::-1]  # Ensure monotonic
        alphas = np.diff(alpha_bar)
        alphas = np.clip(alphas, 1e-6, 1.0 - 1e-6)
        betas = 1.0 - alphas
    else:
        betas = np.linspace(beta_start, beta_end, n_timesteps)
        alphas = 1.0 - betas
        alpha_bar = np.cumprod(alphas)
    
    return betas.astype(np.float32), alphas.astype(np.float32), alpha_bar.astype(np.float32)


def q_sample_residual_latent(residual_latent, t, alphas_bar_sqrt, one_minus_alphas_bar_sqrt):
    """
    Add noise to residual in latent space.
    
    residual_t = sqrt(alpha_bar_t) * residual_0 + sqrt(1 - alpha_bar_t) * noise
    """
    noise = torch.randn_like(residual_latent)
    t_cpu = t.cpu()
    alpha_bar_t = alphas_bar_sqrt[t_cpu].to(residual_latent.device).view(-1, 1, 1, 1, 1)
    sqrt_one_minus_alpha_bar_t = one_minus_alphas_bar_sqrt[t_cpu].to(residual_latent.device).view(-1, 1, 1, 1, 1)
    
    residual_t = alpha_bar_t * residual_latent + sqrt_one_minus_alpha_bar_t * noise
    return residual_t, noise


def p_sample_ddim_hybrid(model, vae_model, unet_model, pre_vol, n_timesteps_train, n_steps_ddim,
                         betas, alphas, alphas_bar_sqrt, one_minus_alphas_bar_sqrt,
                         residual_scale, target_size, device):
    """
    DDIM sampling for hybrid model.
    
    Process:
    1. Get UNet prediction
    2. Encode UNet prediction to latent space
    3. Run diffusion to refine (predict residual)
    4. Add residual to UNet prediction
    5. Decode to volume space
    """
    unet_model.eval()
    vae_model.eval()
    model.eval()
    
    with torch.no_grad():
        # Normalize pre volume
        pre_vol_norm = strict_normalize_volume(pre_vol)
        pre_t = torch.from_numpy(pre_vol_norm).unsqueeze(0).unsqueeze(0).float().to(device)
        
        # Stage 1: Get UNet prediction
        unet_pred = unet_model(pre_t)  # (1, 1, H, W, D)
        unet_pred_np = unet_pred[0, 0].cpu().numpy()
        unet_pred_np = np.clip(unet_pred_np, 0.0, 1.0)
        
        # Encode UNet prediction to latent space
        unet_pred_t = torch.from_numpy(unet_pred_np).unsqueeze(0).unsqueeze(0).float().to(device)
        unet_pred_latent = vae_model.encode_to_latent(unet_pred_t)
        unet_pred_latent = torch.clamp(unet_pred_latent, -5.0, 5.0)
        
        # Encode pre volume to latent (for conditioning if needed)
        pre_latent = vae_model.encode_to_latent(pre_t)
        pre_latent = torch.clamp(pre_latent, -5.0, 5.0)
        
        # Stage 2: Initialize residual with noise
        residual_t_latent = torch.randn_like(unet_pred_latent) * 0.1  # Small initial noise
        
        # Create step schedule for DDIM
        step_indices = np.linspace(0, n_timesteps_train - 1, n_steps_ddim, dtype=int)
        
        # Reverse diffusion in latent space
        for i in range(n_steps_ddim - 1, -1, -1):
            t_val = step_indices[i]
            t_batch = torch.full((residual_t_latent.size(0),), t_val, dtype=torch.long, device=device)
            
            # Predict noise in residual latent space
            pred_noise = model(residual_t_latent, unet_pred_latent, t_batch)
            
            # Predict residual_0 in latent space
            alpha_t = alphas[t_val].to(device)
            alpha_bar_t = alphas_bar_sqrt[t_val].to(device) ** 2
            alpha_t = torch.clamp(alpha_t, min=1e-6, max=1.0-1e-6)
            alpha_bar_t = torch.clamp(alpha_bar_t, min=1e-6, max=1.0-1e-6)
            sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar_t)
            sqrt_one_minus_alpha_bar = torch.clamp(sqrt_one_minus_alpha_bar, min=1e-6)
            sqrt_alpha_t = torch.sqrt(alpha_t)
            
            residual_0_latent_pred = (residual_t_latent - (1 - alpha_t) / sqrt_one_minus_alpha_bar * pred_noise) / sqrt_alpha_t
            residual_0_latent_pred = torch.clamp(residual_0_latent_pred, -5.0, 5.0)
            
            # Check for NaN
            if torch.isnan(residual_0_latent_pred).any() or torch.isinf(residual_0_latent_pred).any():
                residual_0_latent_pred = torch.zeros_like(residual_0_latent_pred)
            
            # DDIM step: deterministic update
            if i > 0:
                t_next = step_indices[i-1]
                alpha_t_next = alphas_bar_sqrt[t_next].to(device) ** 2
                alpha_t_next = torch.clamp(alpha_t_next, min=1e-6, max=1.0-1e-6)
                sqrt_alpha_t_next = torch.sqrt(alpha_t_next)
                sqrt_one_minus_alpha_t_next = torch.sqrt(1.0 - alpha_t_next)
                residual_t_latent = sqrt_alpha_t_next * residual_0_latent_pred + sqrt_one_minus_alpha_t_next * pred_noise
                residual_t_latent = torch.clamp(residual_t_latent, -10.0, 10.0)
                
                # Check for NaN
                if torch.isnan(residual_t_latent).any() or torch.isinf(residual_t_latent).any():
                    residual_t_latent = torch.randn_like(residual_t_latent) * 0.1
        
        # Scale residual
        residual_0_latent = residual_0_latent_pred * residual_scale
        residual_0_latent = torch.clamp(residual_0_latent, -3.0, 3.0)
        
        # Add residual to UNet prediction and decode
        refined_latent = unet_pred_latent + residual_0_latent
        refined_latent = torch.clamp(refined_latent, -10.0, 10.0)
        
        # Check for NaN before decoding
        if torch.isnan(refined_latent).any() or torch.isinf(refined_latent).any():
            refined_latent = unet_pred_latent  # Fallback to UNet prediction
        
        refined_vol = vae_model.decode_from_latent(refined_latent)
        
        # Check for NaN in decoded volume
        if torch.isnan(refined_vol).any() or torch.isinf(refined_vol).any():
            refined_vol = unet_pred  # Fallback to UNet prediction
        
        return refined_vol.clamp(0.0, 1.0)


###############################################################################
# Training
###############################################################################

def train_one_epoch(model, loader, optimizer, n_timesteps, alphas_bar_sqrt, one_minus_alphas_bar_sqrt,
                    residual_scale, device, ema=None):
    model.train()
    epoch_loss = 0.0
    criterion = nn.MSELoss()
    n_batches = 0
    
    for unet_pred_latent, post_latent in loader:
        unet_pred_latent = unet_pred_latent.to(device)
        post_latent = post_latent.to(device)
        
        # Clamp latent values
        unet_pred_latent = torch.clamp(unet_pred_latent, -5.0, 5.0)
        post_latent = torch.clamp(post_latent, -5.0, 5.0)
        
        # Compute residual: ground_truth - unet_prediction
        residual_latent = post_latent - unet_pred_latent
        residual_latent = torch.clamp(residual_latent, -3.0, 3.0)
        residual_latent_scaled = residual_latent / residual_scale
        
        B = residual_latent_scaled.shape[0]
        t_batch = torch.randint(0, n_timesteps, (B,), dtype=torch.long, device=device)
        
        # Add noise to residual
        residual_t_latent, noise = q_sample_residual_latent(
            residual_latent_scaled, t_batch, alphas_bar_sqrt, one_minus_alphas_bar_sqrt
        )
        
        # Check for NaN/Inf
        if torch.isnan(residual_t_latent).any() or torch.isinf(residual_t_latent).any():
            continue
        
        residual_t_latent = torch.clamp(residual_t_latent, -10.0, 10.0)
        
        optimizer.zero_grad()
        
        # Predict noise (epsilon-prediction)
        pred_noise = model(residual_t_latent, unet_pred_latent, t_batch)
        loss = criterion(pred_noise, noise)
        
        # Check for NaN loss
        if torch.isnan(loss) or torch.isinf(loss):
            optimizer.zero_grad()
            continue
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Update EMA
        if ema is not None:
            ema.update()
        
        epoch_loss += loss.item()
        n_batches += 1
    
    return epoch_loss / max(n_batches, 1)


###############################################################################
# Evaluation
###############################################################################

def evaluate_model(model, vae_model, unet_model, test_items, n_timesteps_train, n_steps_ddim,
                  betas, alphas, alphas_bar_sqrt, one_minus_alphas_bar_sqrt,
                  residual_scale, target_size, device, load_fn=None):
    """Evaluate hybrid model on test set. If load_fn (e.g. Week7) given, use it to load volumes."""
    import sys  # local import so sys is always bound (Bland–Altman block uses sys.path)
    model.eval()
    vae_model.eval()
    unet_model.eval()
    mae_list, ssim_list, psnr_list = [], [], []
    all_predicted = []
    all_ground_truth = []
    
    with torch.no_grad():
        for pre_p, post_p in test_items:
            if load_fn is not None:
                pre_vol = strict_normalize_volume(load_fn(pre_p))
                post_vol = strict_normalize_volume(load_fn(post_p))
            else:
                pre_vol = strict_normalize_volume(load_full_volume(pre_p, target_size))
                post_vol = strict_normalize_volume(load_full_volume(post_p, target_size))
            
            # Predict using hybrid model
            try:
                pred_vol = p_sample_ddim_hybrid(
                    model, vae_model, unet_model, pre_vol, n_timesteps_train, n_steps_ddim,
                    betas, alphas, alphas_bar_sqrt, one_minus_alphas_bar_sqrt,
                    residual_scale, target_size, device
                )
                pred_vol_np = pred_vol[0, 0].cpu().numpy()
                
                # Check for NaN
                if np.isnan(pred_vol_np).any() or np.isinf(pred_vol_np).any():
                    print(f"Warning: NaN/Inf in prediction for {pre_p}, skipping")
                    continue
                
                # Collect for Bland-Altman analysis
                all_predicted.append(pred_vol_np.flatten())
                all_ground_truth.append(post_vol.flatten())
                
                if load_fn is not None:
                    if os.path.join(_REPO_ROOT, "scripts") not in sys.path:
                        sys.path.insert(0, os.path.join(_REPO_ROOT, "scripts"))
                    from week7_preprocess import metrics_in_brain
                    m = metrics_in_brain(pred_vol_np, post_vol, data_range=1.0)
                    mae_list.append(m["mae_mean"])
                    if not (np.isnan(m["ssim_mean"]) or np.isinf(m["ssim_mean"])):
                        ssim_list.append(m["ssim_mean"])
                    if not (np.isnan(m["psnr_mean"]) or np.isinf(m["psnr_mean"])):
                        psnr_list.append(m["psnr_mean"])
                else:
                    mae_list.append(np.abs(pred_vol_np - post_vol).mean())
                    ssim_val = ssim(post_vol, pred_vol_np, data_range=1.0)
                    psnr_val = psnr(post_vol, pred_vol_np, data_range=1.0)
                    if not (np.isnan(ssim_val) or np.isinf(ssim_val)):
                        ssim_list.append(ssim_val)
                    if not (np.isnan(psnr_val) or np.isinf(psnr_val)):
                        psnr_list.append(psnr_val)
            except Exception as e:
                print(f"Error evaluating {pre_p}: {e}, skipping")
                continue
    
    # Perform Bland-Altman analysis
    if len(all_predicted) > 0:
        all_predicted_flat = np.concatenate(all_predicted)
        all_ground_truth_flat = np.concatenate(all_ground_truth)
        sys.path.append(os.path.join(_REPO_ROOT, 'Diffusion_3D_Latent'))
        from utils import bland_altman_analysis
        ba_results = bland_altman_analysis(all_predicted_flat, all_ground_truth_flat)
    else:
        ba_results = {
            'mean_bias': np.nan, 'std_diff': np.nan, 'upper_loa': np.nan,
            'lower_loa': np.nan, 'loa_upper_ci': np.nan, 'loa_lower_ci': np.nan,
            'bias_ci_upper': np.nan, 'bias_ci_lower': np.nan, 'n_samples': 0
        }
    
    return {
        'mae_mean': float(np.mean(mae_list)),
        'mae_std': float(np.std(mae_list)),
        'ssim_mean': float(np.mean(ssim_list)),
        'ssim_std': float(np.std(ssim_list)),
        'psnr_mean': float(np.mean(psnr_list)),
        'psnr_std': float(np.std(psnr_list)),
        'bland_altman': ba_results,
    }


###############################################################################
# Main
###############################################################################

def main():
    print("="*70)
    print("HYBRID UNET-DIFFUSION MODEL FOR 3D MOYAMOYA CVR PREDICTION")
    print("="*70)
    
    # Configuration
    CONFIG = {
        'target_size': (128, 128, 64),
        'batch_size': 8,  # Larger batch size due to latent space
        'lr': 5e-4,
        'epochs': int(os.environ.get('WEEK9_QUICK_EPOCHS', 100)),
        'early_stop_patience': 15,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'seed': int(os.environ.get('SEED', 1337)),  # Week8: use SEED env for 3-seed runs
        'n_timesteps_train': 1000,
        'n_steps_ddim': 25,
        'residual_scale': 0.5,
        'ema_decay': 0.9999,
        'latent_channels': 4,
        'channels': (32, 64, 128),
        'use_v_prediction': False,
    }
    
    print(f"\nConfiguration:")
    for k, v in CONFIG.items():
        print(f"  {k}: {v}")
    
    # Set seed
    random.seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])
    torch.manual_seed(CONFIG['seed'])
    
    sys.path.insert(0, os.path.join(_REPO_ROOT, 'scripts'))
    from week7_preprocess import is_env_flag, is_week7_kfold
    use_week7 = '--week7' in sys.argv or is_env_flag('WEEK7') or is_week7_kfold()
    if use_week7:
        CONFIG['target_size'] = (96, 112, 96)
        use_phase2 = is_env_flag('WEEK7_REGION_WEIGHT')
        CONFIG['ckpt_name'] = 'hybrid_unet_diffusion_week7_best.pt'
        CONFIG['results_name'] = 'hybrid_unet_diffusion_week7_phase2_results.json' if use_phase2 else 'hybrid_unet_diffusion_week7_results.json'
        print("  Week7: 91×109×91 brain mask, pad 96×112×96, combined 2020-2023 split")
    else:
        CONFIG['ckpt_name'] = 'hybrid_unet_diffusion_best.pt'
        CONFIG['results_name'] = 'hybrid_unet_diffusion_results.json'
    
    # Load pre-trained models
    print(f"\n Loading pre-trained models...")
    
    # Load VAE (Week7: prefer vae_3d_week7_best.pt if exists, else fallback to vae_3d_best.pt)
    vae_path_week7 = os.path.join(_REPO_ROOT, 'Diffusion_3D_Latent/vae_3d_week7_best.pt')
    vae_path = vae_path_week7 if use_week7 and os.path.exists(vae_path_week7) else '<repo-root>/Diffusion_3D_Latent/vae_3d_best.pt'
    if use_week7 and vae_path == os.path.join(_REPO_ROOT, 'Diffusion_3D_Latent/vae_3d_best.pt'):
        print("   Note: Using vae_3d_best.pt (Week7 VAE not found)")
    if not os.path.exists(vae_path):
        raise FileNotFoundError(f"VAE model not found: {vae_path}")
    vae_model = VAE3D(in_channels=1, latent_channels=CONFIG['latent_channels']).to(CONFIG['device'])
    vae_model.load_state_dict(torch.load(vae_path, map_location=CONFIG['device']))
    vae_model.eval()
    for param in vae_model.parameters():
        param.requires_grad = False
    print(f"   Loaded VAE from {vae_path}")
    
    # Load UNet (Week7: use week7 checkpoint)
    unet_path = os.path.join(_REPO_ROOT, 'UNet_3D/unet_3d_week7_best.pt') if use_week7 else os.path.join(_REPO_ROOT, 'UNet_3D/unet_3d_best.pt')
    if not os.path.exists(unet_path):
        raise FileNotFoundError(f"UNet model not found: {unet_path}")
    unet_model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128),
        strides=(2, 2, 2),
        num_res_units=2,
        act=("LeakyReLU", {"inplace": True}),
        norm="INSTANCE",
        dropout=0.0,
    ).to(CONFIG['device'])
    unet_model.load_state_dict(torch.load(unet_path, map_location=CONFIG['device']))
    unet_model.eval()
    for param in unet_model.parameters():
        param.requires_grad = False
    print(f"   Loaded UNet from {unet_path}")
    
    # Load data
    print(f"\n Loading data...")
    if use_week7:
        sys.path.insert(0, os.path.join(_REPO_ROOT, 'scripts'))
        from week7_data import get_week7_splits
        train_pairs, val_pairs, test_pairs = get_week7_splits()
        train_pre = [p[0] for p in train_pairs]
        train_post = [p[1] for p in train_pairs]
        val_pre = [p[0] for p in val_pairs]
        val_post = [p[1] for p in val_pairs]
        test_pre = [p[0] for p in test_pairs]
        test_post = [p[1] for p in test_pairs]
        print(f"  Week7: {len(train_pairs)} train / {len(val_pairs)} val / {len(test_pairs)} test (combined 2020-2023)")
        load_fn = load_volume_week7
    else:
        load_fn = None
    data_dir = _REPO_ROOT
    if not use_week7:
        all_pre = sorted(glob.glob(f"{data_dir}/pre/pre_*.nii.gz"))
        print(f"  Found {len(all_pre)} pre-scans")
        all_pre_paired = [p for p in all_pre if os.path.exists(pre_to_post_path(p))]
        print(f"  Found {len(all_pre_paired)} with matching post-scans")
        trainval_pre, test_pre = train_test_split(
            all_pre_paired, test_size=0.25, random_state=CONFIG['seed'], shuffle=True
        )
        train_pre, val_pre = train_test_split(
            trainval_pre, test_size=0.125/0.75, random_state=CONFIG['seed'], shuffle=True
        )
        train_post = [pre_to_post_path(p) for p in train_pre]
        val_post = [pre_to_post_path(p) for p in val_pre]
        test_post = [pre_to_post_path(p) for p in test_pre]
        print(f"\nSplit: {len(train_pre)} train / {len(val_pre)} val / {len(test_pre)} test")
    
    if use_week7:
        print(f"\nSplit: {len(train_pre)} train / {len(val_pre)} val / {len(test_pre)} test")
    
    # Create datasets
    print(f"\n Creating datasets...")
    train_dataset = HybridDataset(
        train_pre, unet_model, vae_model, CONFIG['target_size'], CONFIG['device'],
        load_fn=load_fn, post_paths=train_post if use_week7 else None
    )
    val_dataset = HybridDataset(
        val_pre, unet_model, vae_model, CONFIG['target_size'], CONFIG['device'],
        load_fn=load_fn, post_paths=val_post if use_week7 else None
    )
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=2)
    
    # Create diffusion model
    print(f"\n  Creating hybrid diffusion model...")
    model = HybridDiffusionNet(
        latent_channels=CONFIG['latent_channels'],
        channels=CONFIG['channels'],
        use_v_prediction=CONFIG['use_v_prediction']
    ).to(CONFIG['device'])
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    
    # Create diffusion schedule
    betas, alphas, alpha_bar = make_beta_schedule(
        n_timesteps=CONFIG['n_timesteps_train'],
        schedule='cosine'
    )
    alphas_bar_sqrt = torch.from_numpy(np.sqrt(alpha_bar)).float().to(CONFIG['device'])
    one_minus_alphas_bar_sqrt = torch.from_numpy(np.sqrt(1.0 - alpha_bar)).float().to(CONFIG['device'])
    betas = torch.from_numpy(betas).float().to(CONFIG['device'])
    alphas = torch.from_numpy(alphas).float().to(CONFIG['device'])
    
    # Setup training
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    ema = EMA(model, decay=CONFIG['ema_decay'])
    
    ckpt_name = CONFIG.get('ckpt_name', 'hybrid_unet_diffusion_best.pt')
    # Training loop
    print(f"\n Training hybrid model...")
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(1, CONFIG['epochs'] + 1):
        start_time = time.time()
        
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, CONFIG['n_timesteps_train'],
            alphas_bar_sqrt, one_minus_alphas_bar_sqrt,
            CONFIG['residual_scale'], CONFIG['device'], ema
        )
        
        # Validate (no optimizer, no EMA)
        model.eval()
        val_loss = 0.0
        criterion = nn.MSELoss()
        n_batches = 0
        
        with torch.no_grad():
            for unet_pred_latent, post_latent in val_loader:
                unet_pred_latent = unet_pred_latent.to(CONFIG['device'])
                post_latent = post_latent.to(CONFIG['device'])
                
                unet_pred_latent = torch.clamp(unet_pred_latent, -5.0, 5.0)
                post_latent = torch.clamp(post_latent, -5.0, 5.0)
                
                residual_latent = post_latent - unet_pred_latent
                residual_latent = torch.clamp(residual_latent, -3.0, 3.0)
                residual_latent_scaled = residual_latent / CONFIG['residual_scale']
                
                B = residual_latent_scaled.shape[0]
                t_batch = torch.randint(0, CONFIG['n_timesteps_train'], (B,), dtype=torch.long, device=CONFIG['device'])
                
                residual_t_latent, noise = q_sample_residual_latent(
                    residual_latent_scaled, t_batch, alphas_bar_sqrt, one_minus_alphas_bar_sqrt
                )
                
                if torch.isnan(residual_t_latent).any() or torch.isinf(residual_t_latent).any():
                    continue
                
                residual_t_latent = torch.clamp(residual_t_latent, -10.0, 10.0)
                pred_noise = model(residual_t_latent, unet_pred_latent, t_batch)
                loss = criterion(pred_noise, noise)
                
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    val_loss += loss.item()
                    n_batches += 1
        
        val_loss = val_loss / max(n_batches, 1)
        
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        elapsed = time.time() - start_time
        
        # Check for improvement
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), CONFIG.get('ckpt_name', 'hybrid_unet_diffusion_best.pt'))
            ema.apply_shadow()
            torch.save(model.state_dict(), ckpt_name.replace('_best.pt', '_ema_best.pt'))
            ema.restore()
            print(f"Epoch {epoch:3d}/{CONFIG['epochs']} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - LR: {current_lr:.6f} - Time: {elapsed:.1f}s")
            print(f"   Saved new best model")
        else:
            patience_counter += 1
            print(f"Epoch {epoch:3d}/{CONFIG['epochs']} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - LR: {current_lr:.6f} - Time: {elapsed:.1f}s")
        
        if patience_counter >= CONFIG['early_stop_patience']:
            print(f"\nEarly stopping at epoch {epoch}")
            break
    
    # Load best model (use ckpt_name so Week7 loads week7_best.pt)
    model.load_state_dict(torch.load(ckpt_name, map_location=CONFIG['device']))
    ema_ckpt = ckpt_name.replace('_best.pt', '_ema_best.pt')
    if os.path.exists(ema_ckpt):
        ema.apply_shadow()
        model.load_state_dict(torch.load(ema_ckpt, map_location=CONFIG['device']))
    
    # Evaluate on test set
    print(f"\n Final Evaluation on Test Set...")
    test_items = test_pairs if use_week7 else [(p, pre_to_post_path(p)) for p in test_pre]
    # Only evaluate on pairs where both files exist
    test_items = [(pre, post) for pre, post in test_items if os.path.isfile(pre) and os.path.isfile(post)]
    if not test_items:
        print("  No test pairs with existing files; skipping evaluation.")
        results = {'mae_mean': float('nan'), 'ssim_mean': float('nan'), 'psnr_mean': float('nan'), 'bland_altman': {'n_samples': 0}}
    else:
        results = evaluate_model(
            model, vae_model, unet_model, test_items,
            CONFIG['n_timesteps_train'], CONFIG['n_steps_ddim'],
            betas, alphas, alphas_bar_sqrt, one_minus_alphas_bar_sqrt,
            CONFIG['residual_scale'], CONFIG['target_size'], CONFIG['device'],
            load_fn=load_fn
        )
    
    print(f"\nHybrid UNet-Diffusion Results:")
    print(f"  MAE:  {results['mae_mean']:.4f} ± {results['mae_std']:.4f}")
    print(f"  SSIM: {results['ssim_mean']:.4f} ± {results['ssim_std']:.4f}")
    print(f"  PSNR: {results['psnr_mean']:.2f} ± {results['psnr_std']:.2f} dB")
    
    # Print Bland-Altman analysis
    ba = results.get('bland_altman', {})
    if ba.get('n_samples', 0) > 0:
        print(f"\nBland-Altman Analysis (Clinical Agreement):")
        print(f"  Mean Bias: {ba['mean_bias']:.6f} (95% CI: [{ba['bias_ci_lower']:.6f}, {ba['bias_ci_upper']:.6f}])")
        print(f"  Upper LOA: {ba['upper_loa']:.6f} (95% CI: [{ba['loa_upper_ci']:.6f}, ...])")
        print(f"  Lower LOA: {ba['lower_loa']:.6f} (95% CI: [..., {ba['loa_lower_ci']:.6f}])")
        print(f"  SD of Differences: {ba['std_diff']:.6f}")
        print(f"  N Samples: {ba['n_samples']:,}")
    
    # Save results to JSON
    with open(CONFIG.get('results_name', 'hybrid_unet_diffusion_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Format Bland-Altman text for results file
    if ba.get('n_samples', 0) > 0:
        ba_text = f"""
Mean Bias: {ba['mean_bias']:.6f} (95% CI: [{ba['bias_ci_lower']:.6f}, {ba['bias_ci_upper']:.6f}])
  The mean difference between predicted and ground truth measurements.
  A bias close to 0 indicates minimal systematic error.

Upper Limit of Agreement (LOA): {ba['upper_loa']:.6f} (95% CI: [{ba['loa_upper_ci']:.6f}, ...])
  Upper bound of the 95% agreement interval. 95% of differences should fall below this value.

Lower Limit of Agreement (LOA): {ba['lower_loa']:.6f} (95% CI: [..., {ba['loa_lower_ci']:.6f}])
  Lower bound of the 95% agreement interval. 95% of differences should fall above this value.

Standard Deviation of Differences: {ba['std_diff']:.6f}
  Measures the variability in the differences between measurements.

Number of Samples: {ba['n_samples']:,}
  Total voxels analyzed for agreement assessment.

Clinical Interpretation:
  - If the limits of agreement fall within clinically acceptable ranges,
    the model predictions are suitable for clinical decision-making.
  - The confidence intervals provide uncertainty estimates for the bias and LOA.
"""
    else:
        ba_text = "\nBland-Altman analysis not available (insufficient samples)."
    
    # Save comprehensive results to text file
    import datetime
    results_txt = f"""======================================================================
HYBRID UNET-DIFFUSION - FINAL RESULTS
======================================================================

Training Configuration:
  - Model: Hybrid UNet + Residual Diffusion Latent
  - VAE Latent Channels: {CONFIG['latent_channels']}
  - Target Size: {CONFIG['target_size']}
  - Diffusion Timesteps: {CONFIG['n_timesteps_train']}
  - DDIM Steps: {CONFIG['n_steps_ddim']}
  - Residual Scale: {CONFIG['residual_scale']}
  - EMA Decay: {CONFIG['ema_decay']}
  - Epochs: {CONFIG['epochs']}
  - Best Val Loss: {best_val_loss:.6f}

======================================================================
TEST SET EVALUATION METRICS
======================================================================

Mean Absolute Error (MAE):
  Mean: {results['mae_mean']:.6f}
  Std:  {results['mae_std']:.6f}
  Lower is better

Structural Similarity Index (SSIM):
  Mean: {results['ssim_mean']:.6f}
  Std:  {results['ssim_std']:.6f}
  Higher is better (range: 0-1)

Peak Signal-to-Noise Ratio (PSNR):
  Mean: {results['psnr_mean']:.2f} dB
  Std:  {results['psnr_std']:.2f} dB
  Higher is better

======================================================================
BLAND-ALTMAN ANALYSIS (Clinical Agreement Assessment)
======================================================================

Bland-Altman analysis evaluates agreement between predicted and ground truth
measurements. Critical for clinical decision-making to determine if predictions
are within acceptable clinical intervals for reproducibility/repeatability.
{ba_text}
======================================================================
COMPARISON WITH BASELINES
======================================================================

2D Paper Baseline:
  MAE:  0.0497
  SSIM: 0.7886
  PSNR: 21.49 dB

3D UNet Baseline:
  MAE:  ~0.0450 (estimated)
  SSIM: ~0.8000 (estimated)
  PSNR: ~22.00 dB (estimated)

Hybrid UNet-Diffusion:
  MAE:  {results['mae_mean']:.6f} ({'+' if results['mae_mean'] > 0.0497 else ''}{((results['mae_mean'] / 0.0497 - 1) * 100):.1f}% vs 2D paper)
  SSIM: {results['ssim_mean']:.6f} ({'+' if results['ssim_mean'] > 0.7886 else ''}{((results['ssim_mean'] / 0.7886 - 1) * 100):.1f}% vs 2D paper)
  PSNR: {results['psnr_mean']:.2f} dB ({'+' if results['psnr_mean'] > 21.49 else ''}{((results['psnr_mean'] / 21.49 - 1) * 100):.1f}% vs 2D paper)

======================================================================
MODEL FILES
======================================================================

  Model: hybrid_unet_diffusion_best.pt
  EMA Model: hybrid_unet_diffusion_ema_best.pt
  Results JSON: hybrid_unet_diffusion_results.json
  Results TXT: HYBRID_UNET_DIFFUSION_RESULTS.txt

======================================================================
Training completed: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
======================================================================
"""
    
    with open('HYBRID_UNET_DIFFUSION_RESULTS.txt', 'w') as f:
        f.write(results_txt)
    
    print(f"\n Training Complete!")
    print(f"  Model saved: hybrid_unet_diffusion_best.pt")
    print(f"  EMA model saved: hybrid_unet_diffusion_ema_best.pt")
    print(f"  Results JSON: hybrid_unet_diffusion_results.json")
    print(f"  Results TXT: HYBRID_UNET_DIFFUSION_RESULTS.txt")
    print("="*70)


if __name__ == '__main__':
    main()
