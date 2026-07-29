#!/usr/bin/env python3
"""
3D DDPM (Denoising Diffusion Probabilistic Model) for Moyamoya CVR
===================================================================
Full volumetric (3D) implementation of standard diffusion baseline.

Key Changes from 2D:
- spatial_dims=3 in DiffusionModelUNet
- Input shape: (B, 1, H, W, D)
- Time embedding reshape: .view(-1,1,1,1,1) for 3D
- All convolutions are 3D

Reference: Ho et al. "Denoising Diffusion Probabilistic Models" (2020)
"""

import os
import os, sys, glob, json, random, time
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_REPO_ROOT, "pyproject.toml")):
    _parent = os.path.dirname(_REPO_ROOT)
    if _parent == _REPO_ROOT:
        raise RuntimeError("Could not locate repository root (pyproject.toml not found)")
    _REPO_ROOT = _parent

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from monai.networks.nets import DiffusionModelUNet
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr


###############################################################################
# Helper Functions (same as UNet_3D)
###############################################################################

def load_full_volume(nii_path: str, target_size=(128, 128, 64)) -> np.ndarray:
    img = nib.load(nii_path)
    data = img.get_fdata(dtype=np.float32)
    if data.ndim == 4:
        data = data[..., 0]
    zoom_factors = [target_size[i] / data.shape[i] for i in range(3)]
    resized_data = zoom(data, zoom_factors, order=1)
    return resized_data.astype(np.float32)

def minmax_norm(x: np.ndarray) -> np.ndarray:
    mn, mx = x.min(), x.max()
    return (x - mn) / (mx - mn) if (mx - mn) > 1e-6 else np.zeros_like(x)

# Week7: 91x109x91 -> pad to 96x112x96 for model; metrics/save at 91x109x91
WEEK7_ORIGINAL_SHAPE = (91, 109, 91)
PAD_3D_WEEK7 = (96, 112, 96)


def _pad_3d(pre_t, post_t, target_shape):
    """Pad (B,1,H,W,D) to target_shape. Used for Week7."""
    if target_shape is None or (pre_t.shape[2], pre_t.shape[3], pre_t.shape[4]) == target_shape:
        return pre_t, post_t
    th, tw, td = target_shape
    _, _, h, w, d = pre_t.shape
    ph = max(0, th - h)
    pw = max(0, tw - w)
    pd = max(0, td - d)
    if ph or pw or pd:
        pre_t = torch.nn.functional.pad(pre_t, (0, pd, 0, pw, 0, ph), mode="constant", value=0)
        post_t = torch.nn.functional.pad(post_t, (0, pd, 0, pw, 0, ph), mode="constant", value=0)
    return pre_t[:, :, :th, :tw, :td], post_t[:, :, :th, :tw, :td]


def pre_to_post_path(pre_path: str) -> str:
    basename = os.path.basename(pre_path).replace('pre_', 'post_')
    dirname = os.path.dirname(pre_path).replace('/pre', '/post')
    if dirname == '':
        dirname = 'post'
    return os.path.join(dirname, basename)


###############################################################################
# Dataset
###############################################################################

class FullVolumePairs(Dataset):
    def __init__(self, pre_paths, target_size=(128, 128, 64), augment=False):
        self.target_size = target_size
        self.augment = augment
        self.items = [(p, pre_to_post_path(p)) for p in pre_paths 
                      if os.path.exists(pre_to_post_path(p))]
        print(f"  → Loaded {len(self.items)} paired volumes" + (" (augment=True)" if augment else ""))
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        pre_p, post_p = self.items[idx]
        pre_vol = minmax_norm(load_full_volume(pre_p, self.target_size))
        post_vol = minmax_norm(load_full_volume(post_p, self.target_size))
        if self.augment:
            # 3D augmentation: random flips, small intensity scale
            if random.random() > 0.5:
                pre_vol = np.flip(pre_vol, axis=0).copy()
                post_vol = np.flip(post_vol, axis=0).copy()
            if random.random() > 0.5:
                pre_vol = np.flip(pre_vol, axis=1).copy()
                post_vol = np.flip(post_vol, axis=1).copy()
            if random.random() > 0.5:
                pre_vol = np.flip(pre_vol, axis=2).copy()
                post_vol = np.flip(post_vol, axis=2).copy()
            scale = 0.9 + 0.2 * random.random()
            pre_vol = np.clip(pre_vol * scale, 0, 1).astype(np.float32)
            post_vol = np.clip(post_vol * scale, 0, 1).astype(np.float32)
        pre_t = torch.from_numpy(pre_vol).unsqueeze(0).float()
        post_t = torch.from_numpy(post_vol).unsqueeze(0).float()
        return pre_t, post_t


###############################################################################
# 3D Diffusion Model
###############################################################################

class SimpleCondDiffusion3D(nn.Module):
    """
    Lightweight 3D noise-prediction model (no attention) to avoid MONAI OOM.
    Same interface as CondDiffusionUNet3D: forward(x_t, pre_t, t) -> noise.
    """
    def __init__(self, ch=16):
        super().__init__()
        self.t_embed = nn.Embedding(1001, ch)  # t in [0, 1000]
        self.conv_in = nn.Conv3d(2 + ch, ch, 3, padding=1)  # 2 + 1 for t broadcast
        self.d1 = nn.Sequential(
            nn.Conv3d(ch, ch * 2, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(ch * 2, ch * 2, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.d2 = nn.Sequential(
            nn.Conv3d(ch * 2, ch * 4, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(ch * 4, ch * 4, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.mid = nn.Sequential(
            nn.Conv3d(ch * 4, ch * 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(ch * 4, ch * 4, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.u2 = nn.Sequential(
            nn.ConvTranspose3d(ch * 4, ch * 2, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(ch * 2, ch * 2, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.u1 = nn.Sequential(
            nn.ConvTranspose3d(ch * 2, ch, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(ch, ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.conv_out = nn.Conv3d(ch, 1, 1)
        self.ch = ch

    def forward(self, x_t, pre_t, t):
        B, _, H, W, D = x_t.shape
        cond = torch.cat([x_t, pre_t], dim=1)  # (B, 2, H, W, D)
        t_emb = self.t_embed(t.clamp(0, 1000))  # (B, ch)
        t_broadcast = t_emb.view(B, self.ch, 1, 1, 1).expand(B, self.ch, H, W, D)
        x = torch.cat([cond, t_broadcast], dim=1)  # (B, 2+ch, H, W, D)
        x = self.conv_in(x)
        d1 = self.d1(x)
        d2 = self.d2(d1)
        m = self.mid(d2)
        u2 = self.u2(m) + d1
        u1 = self.u1(u2) + x
        return self.conv_out(u1)


class CondDiffusionUNet3D(nn.Module):
    """
    3D conditional diffusion model.
    Concatenates noisy post (x_t) with pre-scan conditioning.
    """
    def __init__(self, model_channels=32, num_res_blocks=2):  # Reduced from 64
        super().__init__()
        self.model = DiffusionModelUNet(
            spatial_dims=3,          # KEY: 3D diffusion
            in_channels=2,           # x_t (1) + pre_t (1)
            out_channels=1,          # Predicted noise
            channels=(model_channels, model_channels*2),  # Reduced depth: was (64, 128, 256)
            attention_levels=(False, False),  # Disabled attention to save memory
            num_res_blocks=num_res_blocks,
            num_head_channels=8,  # Reduced from 32
        )
    
    def forward(self, x_t, pre_t, t):
        """
        Args:
            x_t: Noisy post-scan (B, 1, H, W, D)
            pre_t: Pre-scan conditioning (B, 1, H, W, D)
            t: Timesteps (B,)
        Returns:
            Predicted noise (B, 1, H, W, D)
        """
        cond_input = torch.cat([x_t, pre_t], dim=1)  # (B, 2, H, W, D)
        return self.model(cond_input, timesteps=t)


###############################################################################
# Diffusion Process
###############################################################################

def make_beta_schedule(n_timesteps=1000, beta_start=0.0001, beta_end=0.02, schedule='linear'):
    """Beta schedule: 'linear' (default) or 'cosine' (Nichol & Dhariwal)."""
    if schedule == 'cosine':
        # alpha_bar_t = cos((t/T + s) / (1+s) * pi/2)^2, s=0.01; then beta_t from alpha_bar
        s = 0.008
        t = np.arange(n_timesteps + 1, dtype=np.float64) / n_timesteps
        alpha_bar = np.cos((t + s) / (1 + s) * (np.pi / 2)) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
        betas = np.clip(betas, 1e-4, 0.999).astype(np.float32)
        return betas
    return np.linspace(beta_start, beta_end, n_timesteps, dtype=np.float32)

def q_sample(x_0, t_batch, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, noise=None):
    """
    Forward diffusion: add noise to x_0.
    x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
    """
    if noise is None:
        noise = torch.randn_like(x_0)
    
    B = x_0.shape[0]
    coef1 = alphas_bar_sqrt[t_batch].view(B, 1, 1, 1, 1)
    coef2 = one_minus_alphas_bar_sqrt[t_batch].view(B, 1, 1, 1, 1)
    
    return coef1 * x_0 + coef2 * noise, noise

def p_sample_step(model, x_t, pre_t, t_val, betas, alphas, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, device):
    """Single reverse diffusion step."""
    B = x_t.shape[0]
    t_batch = torch.full((B,), t_val, dtype=torch.long, device=device)
    
    with torch.no_grad():
        pred_noise = model(x_t, pre_t, t_batch)
    
    alpha_t = alphas[t_val].item() if hasattr(alphas[t_val], 'item') else float(alphas[t_val])
    alpha_bar_t = (alphas_bar_sqrt[t_val] ** 2).item() if hasattr(alphas_bar_sqrt[t_val], 'item') else float(alphas_bar_sqrt[t_val] ** 2)
    beta_t = betas[t_val].item() if hasattr(betas[t_val], 'item') else float(betas[t_val])
    
    # Predict x_0
    x_0_pred = (x_t - (1 - alpha_t) / np.sqrt(1 - alpha_bar_t) * pred_noise) / np.sqrt(alpha_t)
    x_0_pred = torch.clamp(x_0_pred, 0, 1)
    
    if t_val > 0:
        noise = torch.randn_like(x_t)
        sigma_t = np.sqrt(beta_t)
        x_t_prev = (1 / np.sqrt(alpha_t)) * (x_t - ((1 - alpha_t) / np.sqrt(1 - alpha_bar_t)) * pred_noise) + sigma_t * noise
    else:
        x_t_prev = x_0_pred
    
    return x_t_prev

def p_sample_loop(model, pre_t, n_timesteps, betas, alphas, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, device, ddpm_steps=None):
    """Full reverse diffusion from T to 0. If ddpm_steps is set (e.g. 100), subsample timesteps."""
    x_t = torch.randn_like(pre_t).to(device)
    if ddpm_steps is not None and ddpm_steps < n_timesteps:
        steps = np.linspace(n_timesteps - 1, 0, ddpm_steps, dtype=int).tolist()
    else:
        steps = list(reversed(range(n_timesteps)))
    for t_val in steps:
        x_t = p_sample_step(model, x_t, pre_t, t_val, betas, alphas, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, device)
    return x_t


def p_sample_step_ddim(model, x_t, pre_t, t_val, alphas, alphas_bar_sqrt, device):
    """Deterministic reverse step (DDIM): no noise."""
    B = x_t.shape[0]
    t_batch = torch.full((B,), t_val, dtype=torch.long, device=device)
    with torch.no_grad():
        pred_noise = model(x_t, pre_t, t_batch)
    alpha_t = alphas[t_val].item() if hasattr(alphas[t_val], 'item') else float(alphas[t_val])
    alpha_bar_t = (alphas_bar_sqrt[t_val] ** 2).item() if hasattr(alphas_bar_sqrt[t_val], 'item') else float(alphas_bar_sqrt[t_val] ** 2)
    x_0_pred = (x_t - (1 - alpha_t) / np.sqrt(1 - alpha_bar_t) * pred_noise) / np.sqrt(alpha_t)
    x_0_pred = torch.clamp(x_0_pred, 0, 1)
    if t_val > 0:
        alpha_bar_prev = (alphas_bar_sqrt[t_val - 1] ** 2).item() if hasattr(alphas_bar_sqrt[t_val - 1], 'item') else float(alphas_bar_sqrt[t_val - 1] ** 2)
        c1 = np.sqrt(alpha_bar_prev)
        c2 = np.sqrt(1.0 - alpha_bar_prev)
        x_t_prev = c1 * x_0_pred + c2 * pred_noise
        x_t_prev = torch.clamp(x_t_prev, 0, 1)
    else:
        x_t_prev = x_0_pred
    return x_t_prev


def p_sample_loop_ddim(model, pre_t, n_timesteps, alphas, alphas_bar_sqrt, device):
    """Full reverse with DDIM (deterministic)."""
    x_t = torch.randn_like(pre_t).to(device)
    for t_val in reversed(range(n_timesteps)):
        x_t = p_sample_step_ddim(model, x_t, pre_t, t_val, alphas, alphas_bar_sqrt, device)
    return x_t


def p_sample_loop_ddim_steps(model, pre_t, step_indices, alphas, alphas_bar_sqrt, device):
    """DDIM with a subset of timesteps. step_indices e.g. [999, 979, ..., 19, 0] (50 steps)."""
    x_t = torch.randn_like(pre_t).to(device)
    for i in range(len(step_indices) - 1):
        t_val = step_indices[i]
        t_prev = step_indices[i + 1]
        B = x_t.shape[0]
        t_batch = torch.full((B,), t_val, dtype=torch.long, device=device)
        with torch.no_grad():
            pred_noise = model(x_t, pre_t, t_batch)
        alpha_t = alphas[t_val].item() if hasattr(alphas[t_val], 'item') else float(alphas[t_val])
        alpha_bar_t = (alphas_bar_sqrt[t_val] ** 2).item() if hasattr(alphas_bar_sqrt[t_val], 'item') else float(alphas_bar_sqrt[t_val] ** 2)
        x_0_pred = (x_t - (1 - alpha_t) / np.sqrt(1 - alpha_bar_t) * pred_noise) / np.sqrt(alpha_t)
        x_0_pred = torch.clamp(x_0_pred, 0, 1)
        alpha_bar_prev = (alphas_bar_sqrt[t_prev] ** 2).item() if hasattr(alphas_bar_sqrt[t_prev], 'item') else float(alphas_bar_sqrt[t_prev] ** 2)
        c1 = np.sqrt(alpha_bar_prev)
        c2 = np.sqrt(1.0 - alpha_bar_prev)
        x_t = c1 * x_0_pred + c2 * pred_noise
        x_t = torch.clamp(x_t, 0, 1)
    t_val = step_indices[-1]
    t_batch = torch.full((x_t.shape[0],), t_val, dtype=torch.long, device=device)
    with torch.no_grad():
        pred_noise = model(x_t, pre_t, t_batch)
    alpha_t = alphas[t_val].item() if hasattr(alphas[t_val], 'item') else float(alphas[t_val])
    alpha_bar_t = (alphas_bar_sqrt[t_val] ** 2).item() if hasattr(alphas_bar_sqrt[t_val], 'item') else float(alphas_bar_sqrt[t_val] ** 2)
    x_0_pred = (x_t - (1 - alpha_t) / np.sqrt(1 - alpha_bar_t) * pred_noise) / np.sqrt(alpha_t)
    return torch.clamp(x_0_pred, 0, 1)


###############################################################################
# Training
###############################################################################

def train_one_epoch(model, loader, optimizer, n_timesteps, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, device, use_brain_mask=False, pad_3d=None):
    model.train()
    epoch_loss = 0.0
    
    for pre_vol, post_vol in loader:
        pre_vol = pre_vol.to(device)
        post_vol = post_vol.to(device)
        if pad_3d is not None:
            pre_vol, post_vol = _pad_3d(pre_vol, post_vol, pad_3d)
        
        B = post_vol.shape[0]
        t_batch = torch.randint(0, n_timesteps, (B,), dtype=torch.long, device=device)
        
        x_t, noise = q_sample(post_vol, t_batch, alphas_bar_sqrt, one_minus_alphas_bar_sqrt)
        
        optimizer.zero_grad()
        pred_noise = model(x_t, pre_vol, t_batch)
        if use_brain_mask:
            # Brain = pre > 0.05; only compute loss on brain voxels
            mask = (pre_vol > 0.05).float()
            err = (pred_noise - noise) ** 2
            loss = (err * mask).sum() / (mask.sum() + 1e-8)
        else:
            loss = ((pred_noise - noise) ** 2).mean()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    return epoch_loss / len(loader)


###############################################################################
# Evaluation
###############################################################################

def evaluate_model(model, loader, n_timesteps, betas, alphas, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, device, pad_3d=None, use_ddim=False, ddim_steps=50, ddpm_steps=None):
    model.eval()
    mae_list, ssim_list, psnr_list = [], [], []
    oh, ow, od = WEEK7_ORIGINAL_SHAPE if pad_3d else (None, None, None)
    step_indices = np.linspace(n_timesteps - 1, 0, ddim_steps, dtype=int).tolist() if use_ddim else None

    with torch.no_grad():
        for pre_vol, post_vol in loader:
            pre_vol = pre_vol.to(device)
            post_vol = post_vol.to(device)
            if pad_3d is not None:
                pre_vol, post_vol = _pad_3d(pre_vol, post_vol, pad_3d)

            if use_ddim and step_indices is not None:
                pred_vol = p_sample_loop_ddim_steps(model, pre_vol, step_indices, alphas, alphas_bar_sqrt, device)
            else:
                pred_vol = p_sample_loop(model, pre_vol, n_timesteps, betas, alphas, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, device, ddpm_steps=ddpm_steps)
            
            pred_np = pred_vol.cpu().numpy()
            post_np = post_vol.cpu().numpy()
            if pad_3d is not None and oh is not None:
                pred_np = pred_np[:, :, :oh, :ow, :od]
                post_np = post_np[:, :, :oh, :ow, :od]
            
            for i in range(pred_np.shape[0]):
                pred_i = pred_np[i, 0]
                post_i = post_np[i, 0]
                if pad_3d is not None:
                    import sys
                    if os.path.join(_REPO_ROOT, "scripts") not in sys.path:
                        sys.path.insert(0, os.path.join(_REPO_ROOT, "scripts"))
                    from week7_preprocess import metrics_in_brain
                    m = metrics_in_brain(pred_i, post_i, data_range=1.0)
                    mae_list.append(m["mae_mean"])
                    ssim_list.append(m["ssim_mean"])
                    psnr_list.append(m["psnr_mean"])
                else:
                    mae_list.append(np.abs(pred_i - post_i).mean())
                    ssim_list.append(ssim(post_i, pred_i, data_range=1.0))
                    psnr_list.append(psnr(post_i, pred_i, data_range=1.0))
    
    return {
        'mae_mean': float(np.mean(mae_list)),
        'mae_std': float(np.std(mae_list)),
        'ssim_mean': float(np.mean(ssim_list)),
        'ssim_std': float(np.std(ssim_list)),
        'psnr_mean': float(np.mean(psnr_list)),
        'psnr_std': float(np.std(psnr_list)),
    }


###############################################################################
# Main
###############################################################################

def main():
    print("="*70)
    print("3D DDPM BASELINE FOR MOYAMOYA CVR PREDICTION")
    print("="*70)
    
    CONFIG = {
        'target_size': (64, 64, 32),  # Reduced to avoid OOM; Week7 uses (96,112,96)
        'batch_size': 1,
        'lr': 1e-4,
        'epochs': 30,
        'early_stop_patience': 10,
        'n_timesteps': 1000,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'seed': int(os.environ.get('SEED', 42)),
        'use_brain_mask': True,
        'use_simple_model': True,
        'use_augment': True,
        'use_2020': False,
        'use_week7': os.environ.get('WEEK7', '').lower() in ('1', 'true', 'yes') or '--week7' in sys.argv,
        'beta_schedule': 'cosine' if '--cosine' in sys.argv else os.environ.get('BETA_SCHEDULE', 'linear'),
        'split_2020_json': '<repo-root>/2020_single_delay_split.json',
        'combined_split_json': '<repo-root>/combined_subject_split.json',
    }
    
    print(f"\nConfiguration:")
    for k, v in CONFIG.items():
        print(f"  {k}: {v}")
    
    random.seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])
    torch.manual_seed(CONFIG['seed'])
    
    # Beta schedule
    betas = make_beta_schedule(CONFIG['n_timesteps'], schedule=CONFIG.get('beta_schedule', 'linear'))
    alphas = 1.0 - betas
    alphas_bar = np.cumprod(alphas)
    alphas_bar_sqrt = np.sqrt(alphas_bar)
    one_minus_alphas_bar_sqrt = np.sqrt(1.0 - alphas_bar)
    
    # Convert to torch tensors
    betas = torch.from_numpy(betas).float().to(CONFIG['device'])
    alphas = torch.from_numpy(alphas).float().to(CONFIG['device'])
    alphas_bar_sqrt = torch.from_numpy(alphas_bar_sqrt).float().to(CONFIG['device'])
    one_minus_alphas_bar_sqrt = torch.from_numpy(one_minus_alphas_bar_sqrt).float().to(CONFIG['device'])
    
    # Load data
    print(f"\n📂 Loading data...")
    use_json_split = False
    split_json = None
    ckpt_base = 'ddpm_3d_best'
    pad_3d = None
    if CONFIG.get('use_week7'):
        sys.path.insert(0, os.path.join(_REPO_ROOT, 'scripts'))
        from week7_data import get_week7_splits, Week7VolumePairs3D
        from week7_preprocess import TARGET_SHAPE
        train_pairs, val_pairs, test_pairs = get_week7_splits()
        CONFIG['target_size'] = PAD_3D_WEEK7
        pad_3d = PAD_3D_WEEK7
        use_phase2_3d = os.environ.get('WEEK7_REGION_WEIGHT', '').lower() in ('1', 'true', 'yes')
        ckpt_base = 'ddpm_3d_week7_best'
        CONFIG['results_name_week7'] = 'ddpm_3d_week7_phase2_results.json' if use_phase2_3d else 'ddpm_3d_week7_results.json'
        print(f"  Week7: 91x109x91 + brain mask: {len(train_pairs)} train / {len(val_pairs)} val / {len(test_pairs)} test")
        train_dataset = Week7VolumePairs3D(train_pairs, augment=True, target_shape=TARGET_SHAPE)
        val_dataset = Week7VolumePairs3D(val_pairs, augment=False, target_shape=TARGET_SHAPE)
        test_dataset = Week7VolumePairs3D(test_pairs, augment=False, target_shape=TARGET_SHAPE)
    elif CONFIG.get('combined_split_json') and os.path.isfile(CONFIG['combined_split_json']):
        split_json = CONFIG['combined_split_json']
        ckpt_base = 'ddpm_3d_combined_best'
        use_json_split = True
    elif CONFIG.get('use_2020'):
        split_json = CONFIG.get('split_2020_json', '<repo-root>/2020_single_delay_split.json')
        use_json_split = True
    if use_json_split and split_json:
        sys.path.insert(0, os.path.join(_REPO_ROOT, 'scripts'))
        from data_2020_single_delay import Dataset2020SingleDelay
        print(f"\n📊 Creating datasets from {split_json}...")
        train_dataset = Dataset2020SingleDelay(split_json, 'train', CONFIG['target_size'])
        val_dataset = Dataset2020SingleDelay(split_json, 'val', CONFIG['target_size'])
        test_dataset = Dataset2020SingleDelay(split_json, 'test', CONFIG['target_size'])
        print(f"\nSplit: {len(train_dataset)} train / {len(val_dataset)} val / {len(test_dataset)} test (subject-level)")
    elif not CONFIG.get('use_week7'):
        data_dir = _REPO_ROOT
        all_pre = sorted(glob.glob(f"{data_dir}/pre/pre_*.nii.gz"))
        all_pre_paired = [p for p in all_pre if os.path.exists(pre_to_post_path(p))]
        print(f"  Found {len(all_pre_paired)} paired volumes")
        random.shuffle(all_pre_paired)
        n = len(all_pre_paired)
        n_train = int(0.75 * n)
        n_val = int(0.125 * n)
        train_pre = all_pre_paired[:n_train]
        val_pre = all_pre_paired[n_train:n_train+n_val]
        test_pre = all_pre_paired[n_train+n_val:]
        print(f"\nSplit: {len(train_pre)} train / {len(val_pre)} val / {len(test_pre)} test")
        print(f"\n📊 Creating datasets...")
        train_dataset = FullVolumePairs(train_pre, CONFIG['target_size'], augment=CONFIG.get('use_augment', False))
        val_dataset = FullVolumePairs(val_pre, CONFIG['target_size'])
        test_dataset = FullVolumePairs(test_pre, CONFIG['target_size'])
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=2)
    
    # Model (use SimpleCondDiffusion3D to avoid MONAI middle-block attention OOM)
    use_simple = CONFIG.get('use_simple_model', True)
    print(f"\n🏗️  Creating 3D Diffusion Model (simple={use_simple})...")
    model = (SimpleCondDiffusion3D(ch=16) if use_simple else CondDiffusionUNet3D()).to(CONFIG['device'])
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['lr'])

    eval_only = '--eval-only' in sys.argv
    eval_ddim = '--eval-ddim' in sys.argv
    eval_steps = 50
    ddpm_steps_arg = None
    for i, a in enumerate(sys.argv):
        if a == '--eval-steps' and i + 1 < len(sys.argv):
            eval_steps = int(sys.argv[i + 1])
        if a == '--ddpm-steps' and i + 1 < len(sys.argv):
            ddpm_steps_arg = int(sys.argv[i + 1])

    if eval_only:
        ckpt_path = ckpt_base + '.pt'
        if not os.path.isfile(ckpt_path):
            print("Eval-only: checkpoint not found:", ckpt_path)
            return
        if eval_ddim:
            label = " (DDIM %d steps)" % eval_steps
        elif ddpm_steps_arg is not None:
            label = " (DDPM %d steps)" % ddpm_steps_arg
        else:
            label = " (DDPM %d steps)" % CONFIG['n_timesteps']
        print(f"\n📂 Eval-only: loading {ckpt_path}" + label)
        model.load_state_dict(torch.load(ckpt_path, map_location=CONFIG['device']))
        test_results = evaluate_model(model, test_loader, CONFIG['n_timesteps'], betas, alphas,
                                       alphas_bar_sqrt, one_minus_alphas_bar_sqrt, CONFIG['device'],
                                       pad_3d=pad_3d, use_ddim=eval_ddim, ddim_steps=eval_steps, ddpm_steps=ddpm_steps_arg)
        print(f"\nTest Results:")
        print(f"  MAE:  {test_results['mae_mean']:.4f} ± {test_results['mae_std']:.4f}")
        print(f"  SSIM: {test_results['ssim_mean']:.4f} ± {test_results['ssim_std']:.4f}")
        print(f"  PSNR: {test_results['psnr_mean']:.2f} ± {test_results['psnr_std']:.2f} dB")
        if eval_ddim:
            suffix = "_ddim%d" % eval_steps
        elif ddpm_steps_arg is not None:
            suffix = "_ddpm%d" % ddpm_steps_arg
        else:
            suffix = ""
        results_name = ckpt_base.replace('_best', '_results') + suffix + '.json'
        with open(results_name, 'w') as f:
            json.dump({'config': CONFIG, 'test_results': test_results, 'eval_ddim': eval_ddim, 'eval_steps': eval_steps if eval_ddim else None, 'ddpm_steps': ddpm_steps_arg}, f, indent=2)
        print(f"\n✅ Eval complete. Saved: {results_name}")
        return

    # Training
    print(f"\n🚀 Training...")
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(1, CONFIG['epochs'] + 1):
        start_time = time.time()
        
        train_loss = train_one_epoch(model, train_loader, optimizer, CONFIG['n_timesteps'],
                                      alphas_bar_sqrt, one_minus_alphas_bar_sqrt, CONFIG['device'],
                                      use_brain_mask=CONFIG.get('use_brain_mask', False), pad_3d=pad_3d)
        
        # Validation (sample 2 examples to save time)
        model.eval()
        val_samples = 0
        val_loss_sum = 0
        criterion = nn.MSELoss()
        with torch.no_grad():
            for pre_vol, post_vol in val_loader:
                if val_samples >= 4:
                    break
                pre_vol = pre_vol.to(CONFIG['device'])
                post_vol = post_vol.to(CONFIG['device'])
                if pad_3d is not None:
                    pre_vol, post_vol = _pad_3d(pre_vol, post_vol, pad_3d)
                B = post_vol.shape[0]
                t_batch = torch.randint(0, CONFIG['n_timesteps'], (B,), dtype=torch.long, device=CONFIG['device'])
                x_t, noise = q_sample(post_vol, t_batch, alphas_bar_sqrt, one_minus_alphas_bar_sqrt)
                pred_noise = model(x_t, pre_vol, t_batch)
                val_loss_sum += criterion(pred_noise, noise).item()
                val_samples += 1
        
        val_loss = val_loss_sum / max(val_samples, 1)
        epoch_time = time.time() - start_time
        
        print(f"Epoch {epoch:3d}/{CONFIG['epochs']} - Train: {train_loss:.4f} - Val: {val_loss:.4f} - Time: {epoch_time:.1f}s")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), ckpt_base + '.pt')
            print(f"  ✓ Saved new best model")
        else:
            patience_counter += 1
            if patience_counter >= CONFIG['early_stop_patience']:
                print(f"\n⚠️  Early stopping triggered")
                break
    
    # Evaluation
    print(f"\n📊 Final Evaluation on Test Set...")
    model.load_state_dict(torch.load(ckpt_base + '.pt'))
    test_results = evaluate_model(model, test_loader, CONFIG['n_timesteps'], betas, alphas,
                                   alphas_bar_sqrt, one_minus_alphas_bar_sqrt, CONFIG['device'], pad_3d=pad_3d)
    
    print(f"\nTest Results:")
    print(f"  MAE:  {test_results['mae_mean']:.4f} ± {test_results['mae_std']:.4f}")
    print(f"  SSIM: {test_results['ssim_mean']:.4f} ± {test_results['ssim_std']:.4f}")
    print(f"  PSNR: {test_results['psnr_mean']:.2f} ± {test_results['psnr_std']:.2f} dB")
    
    results_name = CONFIG.get('results_name_week7') or (ckpt_base.replace('_best', '_results') + '.json')
    config_name = ckpt_base.replace('_best', '_config') + '.json'
    results = {'config': CONFIG, 'test_results': test_results}
    with open(results_name, 'w') as f:
        json.dump(results, f, indent=2)
    if use_simple:
        with open(config_name, 'w') as f:
            json.dump({'model': 'simple', 'target_size': list(CONFIG['target_size'])}, f, indent=2)
    print(f"\n✅ Complete! Saved: {ckpt_base}.pt, {results_name}" + (f", {config_name}" if use_simple else ""))
    print("="*70)


if __name__ == "__main__":
    main()
