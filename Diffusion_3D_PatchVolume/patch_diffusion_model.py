#!/usr/bin/env python3
"""
Patch-Volume Diffusion Model (3D MedDiffusion Style)
=====================================================
Implements diffusion in patch-based latent space with specialized noise estimator.

Based on research: "3D MedDiffusion: A 3D Medical Diffusion Model for 
Controllable and High-quality Medical Image Generation" (2024)
"""

import os
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

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr

import sys
sys.path.append(os.path.join(_REPO_ROOT, 'Diffusion_3D_Latent'))
from utils import EMA, strict_normalize_volume

from patch_volume_vae import PatchVolumeVAE, extract_patches, reconstruct_volume_from_patches
from specialized_noise_estimator import MultiScaleUNet3D

def load_full_volume(nii_path: str, target_size=(128, 128, 64)) -> np.ndarray:
    """Load a NIfTI file and resize to target_size."""
    img = nib.load(nii_path)
    data = img.get_fdata(dtype=np.float32)
    if data.ndim == 4:
        data = data[..., 0]
    zoom_factors = [target_size[i] / data.shape[i] for i in range(3)]
    resized_data = zoom(data, zoom_factors, order=1)
    return resized_data.astype(np.float32)


def pre_to_post_path(pre_path: str) -> str:
    """Convert pre-scan path to post-scan path."""
    basename = os.path.basename(pre_path).replace('pre_', 'post_')
    dirname = os.path.dirname(pre_path).replace('/pre', '/post')
    if dirname == '':
        dirname = 'post'
    return os.path.join(dirname, basename)


###############################################################################
# Patch-Based Dataset
###############################################################################

class PatchDiffusionDataset(Dataset):
    """
    Dataset that loads volumes, extracts patches, and encodes to latent space.
    If load_fn (e.g. Week7) is given, use it instead of load_full_volume(., target_size).
    """
    def __init__(
        self,
        pre_paths,
        vae_model,
        target_size=(128, 128, 64),
        patch_size=(32, 32, 16),
        stride=16,  # 50% overlap
        device='cuda',
        load_fn=None
    ):
        self.pre_paths = pre_paths
        self.vae_model = vae_model
        self.target_size = target_size
        self.patch_size = patch_size
        self.stride = stride
        self.device = device
        self.load_fn = load_fn
        
        self.items = [(p, pre_to_post_path(p)) for p in pre_paths 
                      if os.path.exists(pre_to_post_path(p))]
        print(f"   Loaded {len(self.items)} paired volumes")
        
        # Pre-extract patches and encode to latent space
        print("   Extracting patches and encoding to latent space...")
        self.patch_pairs = []
        
        vae_model.eval()
        with torch.no_grad():
            for i, (pre_p, post_p) in enumerate(self.items):
                if (i + 1) % 20 == 0:
                    print(f"    Processed {i+1}/{len(self.items)} volumes...")
                
                # Load and normalize volumes
                if load_fn is not None:
                    pre_vol = strict_normalize_volume(load_fn(pre_p))
                    post_vol = strict_normalize_volume(load_fn(post_p))
                else:
                    pre_vol = strict_normalize_volume(load_full_volume(pre_p, target_size))
                    post_vol = strict_normalize_volume(load_full_volume(post_p, target_size))
                
                # Extract patches
                pre_patches, pre_coords = extract_patches(pre_vol, patch_size, stride)
                post_patches, post_coords = extract_patches(post_vol, patch_size, stride)
                
                # Match patches by coordinates
                pre_dict = {coord: patch for coord, patch in zip(pre_coords, pre_patches)}
                post_dict = {coord: patch for coord, patch in zip(post_coords, post_patches)}
                
                # Encode matching patches to latent space
                for coord in pre_dict.keys():
                    if coord in post_dict:
                        pre_patch = pre_dict[coord]
                        post_patch = post_dict[coord]
                        
                        # Convert to tensor and encode
                        pre_t = torch.from_numpy(pre_patch).unsqueeze(0).unsqueeze(0).float().to(device)
                        post_t = torch.from_numpy(post_patch).unsqueeze(0).unsqueeze(0).float().to(device)
                        
                        pre_latent = vae_model.encode_to_latent(pre_t)
                        post_latent = vae_model.encode_to_latent(post_t)
                        
                        self.patch_pairs.append((
                            pre_latent.squeeze(0).cpu(),
                            post_latent.squeeze(0).cpu(),
                            coord  # Store coordinate for reconstruction
                        ))
        
        print(f"   Extracted and encoded {len(self.patch_pairs)} patch pairs")
    
    def __len__(self):
        return len(self.patch_pairs)
    
    def __getitem__(self, idx):
        pre_latent, post_latent, coord = self.patch_pairs[idx]
        return pre_latent, post_latent, coord


###############################################################################
# Diffusion Process
###############################################################################

def make_beta_schedule(n_timesteps=1000, schedule='cosine'):
    """Create cosine beta schedule."""
    if schedule == 'cosine':
        steps = np.arange(n_timesteps + 1, dtype=np.float64) / n_timesteps
        s = 0.008
        alpha_bar = np.cos((steps + s) / (1 + s) * np.pi / 2) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        alpha_bar = np.maximum.accumulate(alpha_bar[::-1])[::-1]
        alphas = np.diff(alpha_bar)
        alphas = np.clip(alphas, 1e-6, 1.0 - 1e-6)
        betas = 1.0 - alphas
    else:
        betas = np.linspace(0.0001, 0.02, n_timesteps)
        alphas = 1.0 - betas
        alpha_bar = np.cumprod(alphas)
    
    return betas.astype(np.float32), alphas.astype(np.float32), alpha_bar.astype(np.float32)


def q_sample_patch_latent(latent, t, alphas_bar_sqrt, one_minus_alphas_bar_sqrt):
    """Add noise to patch latent."""
    noise = torch.randn_like(latent)
    t_cpu = t.cpu()
    alpha_bar_t = alphas_bar_sqrt[t_cpu].to(latent.device).view(-1, 1, 1, 1, 1)
    sqrt_one_minus_alpha_bar_t = one_minus_alphas_bar_sqrt[t_cpu].to(latent.device).view(-1, 1, 1, 1, 1)
    
    noisy_latent = alpha_bar_t * latent + sqrt_one_minus_alpha_bar_t * noise
    return noisy_latent, noise


def p_sample_ddim_patch(
    model, vae_model, pre_patches, n_timesteps_train, n_steps_ddim,
    betas, alphas, alphas_bar_sqrt, one_minus_alphas_bar_sqrt,
    patch_size, stride, volume_shape, device
):
    """
    DDIM sampling for patch-based diffusion.
    Reconstructs full volume from denoised patches.
    """
    model.eval()
    vae_model.eval()
    
    with torch.no_grad():
        # Encode all pre patches to latent space
        pre_latents = []
        for pre_patch in pre_patches:
            pre_t = torch.from_numpy(pre_patch).unsqueeze(0).unsqueeze(0).float().to(device)
            pre_latent = vae_model.encode_to_latent(pre_t)
            pre_latents.append(pre_latent.squeeze(0))
        
        # Extract patches from pre volume and get coordinates
        pre_vol = np.zeros(volume_shape, dtype=np.float32)
        # For simplicity, we'll use the first pre patch's volume
        # In practice, you'd reconstruct the full pre volume first
        pre_patches_list, patch_coords = extract_patches(pre_vol, patch_size, stride)
        
        # Initialize with noise
        post_latents = []
        for pre_latent in pre_latents:
            post_latent = torch.randn_like(pre_latent) * 0.1
            post_latents.append(post_latent)
        
        # Create step schedule for DDIM
        step_indices = np.linspace(0, n_timesteps_train - 1, n_steps_ddim, dtype=int)
        
        # Reverse diffusion for each patch
        for i in range(n_steps_ddim - 1, -1, -1):
            t_val = step_indices[i]
            t_batch = torch.full((1,), t_val, dtype=torch.long, device=device)
            
            # Process each patch
            new_post_latents = []
            for noisy_latent, pre_latent in zip(post_latents, pre_latents):
                # Time embedding
                t_emb = t_batch.view(1, 1, 1, 1, 1).expand(1, 1, *noisy_latent.shape[2:])
                
                # Concatenate inputs
                x = torch.cat([noisy_latent.unsqueeze(0), pre_latent.unsqueeze(0), t_emb], dim=1)
                
                # Predict noise
                pred_noise = model(x)
                
                # Predict x0
                alpha_t = alphas[t_val].to(device)
                alpha_bar_t = alphas_bar_sqrt[t_val].to(device) ** 2
                alpha_t = torch.clamp(alpha_t, min=1e-6, max=1.0-1e-6)
                alpha_bar_t = torch.clamp(alpha_bar_t, min=1e-6, max=1.0-1e-6)
                sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar_t)
                sqrt_one_minus_alpha_bar = torch.clamp(sqrt_one_minus_alpha_bar, min=1e-6)
                sqrt_alpha_t = torch.sqrt(alpha_t)
                
                pred_x0 = (noisy_latent.unsqueeze(0) - (1 - alpha_t) / sqrt_one_minus_alpha_bar * pred_noise) / sqrt_alpha_t
                pred_x0 = torch.clamp(pred_x0, -5.0, 5.0)
                
                # DDIM step
                if i > 0:
                    t_next = step_indices[i-1]
                    alpha_t_next = alphas_bar_sqrt[t_next].to(device) ** 2
                    alpha_t_next = torch.clamp(alpha_t_next, min=1e-6, max=1.0-1e-6)
                    sqrt_alpha_t_next = torch.sqrt(alpha_t_next)
                    sqrt_one_minus_alpha_t_next = torch.sqrt(1.0 - alpha_t_next)
                    
                    noisy_latent = sqrt_alpha_t_next * pred_x0 + sqrt_one_minus_alpha_t_next * pred_noise
                    noisy_latent = torch.clamp(noisy_latent, -10.0, 10.0)
                    new_post_latents.append(noisy_latent.squeeze(0))
                else:
                    new_post_latents.append(pred_x0.squeeze(0))
            
            post_latents = new_post_latents
        
        # Decode patches and reconstruct volume
        decoded_patches = []
        for post_latent in post_latents:
            decoded_patch = vae_model.decode_from_latent(post_latent.unsqueeze(0))
            decoded_patch_np = decoded_patch[0, 0].cpu().numpy()
            decoded_patches.append(decoded_patch_np)
        
        # Reconstruct full volume with overlap blending
        reconstructed_vol = reconstruct_volume_from_patches(
            decoded_patches, patch_coords, volume_shape, patch_size, stride
        )
        
        return torch.from_numpy(reconstructed_vol).unsqueeze(0).unsqueeze(0).float().to(device).clamp(0.0, 1.0)


###############################################################################
# Training
###############################################################################

def train_one_epoch(
    model, loader, optimizer, n_timesteps, alphas_bar_sqrt, one_minus_alphas_bar_sqrt,
    device, ema=None
):
    model.train()
    epoch_loss = 0.0
    criterion = nn.MSELoss()
    n_batches = 0
    
    for pre_latent, post_latent, coord in loader:
        pre_latent = pre_latent.to(device)
        post_latent = post_latent.to(device)
        
        # Clamp latent values
        pre_latent = torch.clamp(pre_latent, -5.0, 5.0)
        post_latent = torch.clamp(post_latent, -5.0, 5.0)
        
        B = post_latent.shape[0]
        t_batch = torch.randint(0, n_timesteps, (B,), dtype=torch.long, device=device)
        
        # Add noise
        noisy_latent, noise = q_sample_patch_latent(
            post_latent, t_batch, alphas_bar_sqrt, one_minus_alphas_bar_sqrt
        )
        
        if torch.isnan(noisy_latent).any() or torch.isinf(noisy_latent).any():
            continue
        
        noisy_latent = torch.clamp(noisy_latent, -10.0, 10.0)
        
        # Time embedding
        t_emb = t_batch.view(-1, 1, 1, 1, 1).expand(-1, 1, *noisy_latent.shape[2:])
        
        # Concatenate inputs
        x = torch.cat([noisy_latent, pre_latent, t_emb], dim=1)
        
        optimizer.zero_grad()
        pred_noise = model(x)
        loss = criterion(pred_noise, noise)
        
        if torch.isnan(loss) or torch.isinf(loss):
            optimizer.zero_grad()
            continue
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if ema is not None:
            ema.update()
        
        epoch_loss += loss.item()
        n_batches += 1
    
    return epoch_loss / max(n_batches, 1)
