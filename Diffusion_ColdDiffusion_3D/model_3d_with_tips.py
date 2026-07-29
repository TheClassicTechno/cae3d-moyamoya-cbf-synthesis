#!/usr/bin/env python3
"""
IMPROVED 3D Cold Diffusion with Tips from difusion3dtips.txt
=============================================================
Improvements:
1. DDIM sampling (25 steps instead of 200)
2. Patch-based training (96×96×48 patches)
3. Overlap blending at inference
4. Better conditioning
5. Cosine degradation schedule (already had this)

Based on: difusion3dtips.txt recommendations
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

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from monai.networks.nets import UNet
from monai.losses import SSIMLoss
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr


###############################################################################
# Helper Functions
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

def pre_to_post_path(pre_path: str) -> str:
    basename = os.path.basename(pre_path).replace('pre_', 'post_')
    dirname = os.path.dirname(pre_path).replace('/pre', '/post')
    if dirname == '':
        dirname = 'post'
    return os.path.join(dirname, basename)


def load_volume_week7(nii_path: str, pad_shape=(96, 112, 96)) -> np.ndarray:
    sys.path.insert(0, os.path.join(_REPO_ROOT, 'scripts'))
    from week7_preprocess import load_volume, TARGET_SHAPE
    vol = load_volume(nii_path, target_shape=TARGET_SHAPE, apply_mask=True, minmax=True)
    if vol.shape != pad_shape:
        out = np.zeros(pad_shape, dtype=vol.dtype)
        sh = [min(vol.shape[i], pad_shape[i]) for i in range(3)]
        out[:sh[0], :sh[1], :sh[2]] = vol[:sh[0], :sh[1], :sh[2]]
        return out.astype(np.float32)
    return vol.astype(np.float32)


###############################################################################
# Patch-based Dataset (Tip #2)
###############################################################################

class Patch3DDataset(Dataset):
    """Patch-based dataset. Optional load_fn and items for Week7."""
    def __init__(self, pre_paths, patch_size=(96, 96, 48), num_patches=8, load_fn=None, items=None):
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.load_fn = load_fn
        self.items = items if items is not None else [(p, pre_to_post_path(p)) for p in pre_paths
                          if os.path.exists(pre_to_post_path(p))]
        print(f"  → Loaded {len(self.items)} paired volumes")
    
    def __len__(self):
        return len(self.items) * self.num_patches
    
    def _load_vol(self, path):
        if self.load_fn is not None:
            return minmax_norm(self.load_fn(path))
        return minmax_norm(load_full_volume(path))
    
    def extract_random_patch(self, vol):
        """Extract one random patch."""
        D, H, W = vol.shape
        pd, ph, pw = self.patch_size
        
        if D < pd or H < ph or W < pw:
            # Volume too small, pad or crop
            vol = zoom(vol, [max(1, pd/D), max(1, ph/H), max(1, pw/W)], order=1)
            D, H, W = vol.shape
        
        z = random.randint(0, max(0, D - pd))
        y = random.randint(0, max(0, H - ph))
        x = random.randint(0, max(0, W - pw))
        
        patch = vol[z:z+pd, y:y+ph, x:x+pw]
        
        # Ensure exact size (pad if needed)
        if patch.shape != self.patch_size:
            pad_d = max(0, pd - patch.shape[0])
            pad_h = max(0, ph - patch.shape[1])
            pad_w = max(0, pw - patch.shape[2])
            patch = np.pad(patch, ((0, pad_d), (0, pad_h), (0, pad_w)), mode='edge')
            patch = patch[:pd, :ph, :pw]
        
        return patch
    
    def __getitem__(self, idx):
        vol_idx = idx // self.num_patches
        pre_p, post_p = self.items[vol_idx]
        pre_vol = self._load_vol(pre_p)
        post_vol = self._load_vol(post_p)
        pre_patch = self.extract_random_patch(pre_vol)
        post_patch = self.extract_random_patch(post_vol)
        pre_t = torch.from_numpy(pre_patch).unsqueeze(0).float()
        post_t = torch.from_numpy(post_patch).unsqueeze(0).float()
        return pre_t, post_t


###############################################################################
# Improved 3D Cold Diffusion Model
###############################################################################

class ImprovedColdDiffusionNet3D(nn.Module):
    """
    Improved 3D Cold Diffusion with better conditioning.
    """
    def __init__(self, channels=(64, 128, 256)):
        super().__init__()
        self.unet = UNet(
            spatial_dims=3,
            in_channels=2,  # x_t + pre (better conditioning via concatenation)
            out_channels=1,  # Predicted x_0
            channels=list(channels),
            strides=(2, 2),
            num_res_units=3,
            act=("LeakyReLU", {"inplace": True}),
            norm="INSTANCE",
            dropout=0.1,
        )
    
    def forward(self, x_t, pre_t, t):
        # Concatenate condition (Tip #5: Better conditioning)
        cond_input = torch.cat([x_t, pre_t], dim=1)
        return self.unet(cond_input)


###############################################################################
# Cold Diffusion Process with Cosine Schedule
###############################################################################

def make_cosine_schedule(n_timesteps=200):
    """Cosine degradation schedule - smoother than linear."""
    steps = np.arange(n_timesteps + 1, dtype=np.float64) / n_timesteps
    alpha_bar = np.cos((steps + 0.008) / 1.008 * np.pi / 2) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    alpha = 1 - alpha_bar
    return alpha.astype(np.float32)

def cold_degrade(x_0, pre_t, t_batch, alpha_schedule):
    """Deterministic degradation with cosine schedule."""
    B = x_0.shape[0]
    alpha_t = alpha_schedule[t_batch].view(-1, 1, 1, 1, 1)
    x_t = alpha_t * x_0 + (1 - alpha_t) * pre_t
    return x_t


###############################################################################
# DDIM Sampling (Tip #3: 20-50 steps instead of 1000)
###############################################################################

def cold_sample_ddim(model, pre_t, n_timesteps_train=200, n_steps_ddim=25, device='cuda'):
    """
    DDIM-style sampling for Cold Diffusion.
    Uses fewer steps (25) for faster, more stable inference.
    """
    model.eval()
    with torch.no_grad():
        # Start from pre (most degraded state)
        x_t = pre_t.clone().to(device)
        
        # Create step schedule for DDIM
        step_indices = np.linspace(0, n_timesteps_train - 1, n_steps_ddim, dtype=int)
        alpha_schedule = make_cosine_schedule(n_timesteps_train)
        alpha_schedule = torch.from_numpy(alpha_schedule).float().to(device)
        
        # Reverse diffusion (from pre towards post)
        for i in range(n_steps_ddim - 1, -1, -1):
            t_val = step_indices[i]
            t_batch = torch.full((x_t.size(0),), t_val, dtype=torch.long, device=device)
            
            # Predict x_0
            x_0_pred = model(x_t, pre_t, t_batch)
            
            # DDIM step: deterministic update
            if i > 0:
                t_next = step_indices[i-1]
                alpha_t = alpha_schedule[t_val]
                alpha_t_next = alpha_schedule[t_next]
                
                # Deterministic update (DDIM)
                x_t = alpha_t_next * x_0_pred + (1 - alpha_t_next) * pre_t
        
        return x_0_pred.clamp(0, 1)


###############################################################################
# Overlap Blending for Full Volume Inference (Tip #2)
###############################################################################

def predict_full_volume_with_overlap(model, pre_vol, patch_size=(96, 96, 48), 
                                     overlap=0.5, n_steps_ddim=25, device='cuda'):
    """
    Predict full volume using patch-based inference with overlap blending.
    Uses Gaussian blending to avoid seams.
    """
    D, H, W = pre_vol.shape
    pd, ph, pw = patch_size
    stride_d = int(pd * (1 - overlap))
    stride_h = int(ph * (1 - overlap))
    stride_w = int(pw * (1 - overlap))
    
    # Create output volume and weight map
    output = np.zeros_like(pre_vol)
    weights = np.zeros_like(pre_vol)
    
    # Create Gaussian weight window
    def gaussian_window(size):
        center = size / 2
        sigma = size / 4
        x = np.arange(size)
        return np.exp(-0.5 * ((x - center) / sigma) ** 2)
    
    w_d = gaussian_window(pd)
    w_h = gaussian_window(ph)
    w_w = gaussian_window(pw)
    weight_window = np.outer(np.outer(w_d, w_h), w_w).reshape(pd, ph, pw)
    
    # Sliding window inference
    pre_t = torch.from_numpy(pre_vol).unsqueeze(0).unsqueeze(0).float().to(device)
    
    for z in range(0, D, stride_d):
        for y in range(0, H, stride_h):
            for x in range(0, W, stride_w):
                z_end = min(z + pd, D)
                y_end = min(y + ph, H)
                x_end = min(x + pw, W)
                
                # Extract patch
                pre_patch = pre_vol[z:z_end, y:y_end, x:x_end]
                if pre_patch.shape != (pd, ph, pw):
                    # Pad if needed
                    pad_d = pd - pre_patch.shape[0]
                    pad_h = ph - pre_patch.shape[1]
                    pad_w = pw - pre_patch.shape[2]
                    pre_patch = np.pad(pre_patch, ((0, pad_d), (0, pad_h), (0, pad_w)), mode='edge')
                    pre_patch = pre_patch[:pd, :ph, :pw]
                
                pre_patch_t = torch.from_numpy(pre_patch).unsqueeze(0).unsqueeze(0).float().to(device)
                
                # Predict patch
                pred_patch = cold_sample_ddim(model, pre_patch_t, n_steps_ddim=n_steps_ddim, device=device)
                pred_patch_np = pred_patch[0, 0].cpu().numpy()
                
                # Crop to actual size
                pred_patch_np = pred_patch_np[:z_end-z, :y_end-y, :x_end-x]
                weight_patch = weight_window[:z_end-z, :y_end-y, :x_end-x]
                
                # Blend
                output[z:z_end, y:y_end, x:x_end] += pred_patch_np * weight_patch
                weights[z:z_end, y:y_end, x:x_end] += weight_patch
    
    # Normalize by weights
    output = output / (weights + 1e-8)
    return output


###############################################################################
# Training
###############################################################################

def train_one_epoch(model, loader, optimizer, n_timesteps, alpha_schedule, device):
    model.train()
    criterion_l1 = nn.L1Loss()
    criterion_ssim = SSIMLoss(spatial_dims=3)
    
    total_loss = 0.0
    for pre_patch, post_patch in loader:
        pre_patch = pre_patch.to(device)
        post_patch = post_patch.to(device)
        B = pre_patch.size(0)
        
        # Sample timestep (beta-weighted sampling - Tip from improved version)
        t_batch = torch.randint(0, n_timesteps, (B,), dtype=torch.long, device=device)
        
        # Degrade
        x_t = cold_degrade(post_patch, pre_patch, t_batch, alpha_schedule)
        
        # Predict
        x_0_pred = model(x_t, pre_patch, t_batch)
        
        # Loss (0.6 L1 + 0.4 SSIM)
        loss_l1 = criterion_l1(x_0_pred, post_patch)
        loss_ssim = 1 - criterion_ssim(x_0_pred, post_patch)
        loss = 0.6 * loss_l1 + 0.4 * loss_ssim
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item() * B
    
    return total_loss / len(loader.dataset)


###############################################################################
# Evaluation
###############################################################################

def evaluate_model(model, loader, n_timesteps_train, n_steps_ddim, device, patch_size=(96, 96, 48), use_week7=False):
    """Evaluate on full volumes using overlap blending. If use_week7, metrics are brain-only."""
    model.eval()
    mae_list, ssim_list, psnr_list = [], [], []
    if use_week7:
        import sys
        if os.path.join(_REPO_ROOT, "scripts") not in sys.path:
            sys.path.insert(0, os.path.join(_REPO_ROOT, "scripts"))
        from week7_preprocess import metrics_in_brain

    with torch.no_grad():
        for pre_vol, post_vol in loader:
            pre_vol_np = pre_vol[0, 0].cpu().numpy()
            post_vol_np = post_vol[0, 0].cpu().numpy()

            # Predict with overlap blending (use same patch_size as training)
            pred_vol_np = predict_full_volume_with_overlap(
                model, pre_vol_np, patch_size=patch_size, n_steps_ddim=n_steps_ddim, device=device
            )

            if use_week7:
                m = metrics_in_brain(pred_vol_np, post_vol_np, data_range=1.0)
                mae_list.append(m["mae_mean"])
                ssim_list.append(m["ssim_mean"])
                psnr_list.append(m["psnr_mean"])
            else:
                mae_list.append(np.abs(pred_vol_np - post_vol_np).mean())
                ssim_list.append(ssim(post_vol_np, pred_vol_np, data_range=1.0))
                psnr_list.append(psnr(post_vol_np, pred_vol_np, data_range=1.0))
    
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
    print("IMPROVED 3D COLD DIFFUSION WITH TIPS")
    print("="*70)
    
    use_week7 = os.environ.get('WEEK7', '').lower() in ('1', 'true', 'yes') or '--week7' in sys.argv
    CONFIG = {
        'target_size': (96, 112, 96) if use_week7 else (128, 128, 64),
        'patch_size': (48, 48, 24) if use_week7 else (96, 96, 48),  # Week7: smaller patch to avoid OOM
        'num_patches': 8,
        'batch_size': 2 if use_week7 else 4,  # Week7: smaller batch to avoid OOM
        'lr': 1e-3,
        'epochs': 100,
        'early_stop_patience': 15,
        'n_timesteps_train': 200,
        'n_steps_ddim': 25,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'seed': int(os.environ.get('SEED', 42)),
        'use_week7': use_week7,
    }
    use_phase2 = use_week7 and os.environ.get('WEEK7_REGION_WEIGHT', '').lower() in ('1', 'true', 'yes')
    ckpt_name = 'cold_diffusion_3d_tips_week7_best.pt' if use_week7 else 'cold_diffusion_3d_tips_best.pt'
    results_name = 'cold_diffusion_3d_tips_week7_phase2_results.json' if use_phase2 else ('cold_diffusion_3d_tips_week7_results.json' if use_week7 else 'cold_diffusion_3d_tips_results.json')
    
    print(f"\nTips Implemented:")
    print(f"  ✅ DDIM sampling (25 steps instead of 200)")
    print(f"  ✅ Patch-based training (96×96×48 patches)")
    print(f"  ✅ Overlap blending at inference")
    print(f"  ✅ Better conditioning (concatenation)")
    print(f"  ✅ Cosine degradation schedule")
    if use_week7:
        print(f"  ✅ Week7: 91×109×91 brain mask, pad 96×112×96")
    
    print(f"\nConfiguration:")
    for k, v in CONFIG.items():
        print(f"  {k}: {v}")
    
    random.seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])
    torch.manual_seed(CONFIG['seed'])
    
    alpha_np = make_cosine_schedule(CONFIG['n_timesteps_train'])
    alpha_schedule = torch.from_numpy(alpha_np).float().to(CONFIG['device'])
    
    # Load data
    print(f"\n📂 Loading data...")
    if use_week7:
        sys.path.insert(0, os.path.join(_REPO_ROOT, 'scripts'))
        from week7_data import get_week7_splits
        train_pairs, val_pairs, test_pairs = get_week7_splits()
        load_fn = lambda p: load_volume_week7(p, pad_shape=(96, 112, 96))
        train_items = [(a,b) for a,b in zip([p[0] for p in train_pairs], [p[1] for p in train_pairs]) if os.path.isfile(a) and os.path.isfile(b)]
        val_items = [(a,b) for a,b in zip([p[0] for p in val_pairs], [p[1] for p in val_pairs]) if os.path.isfile(a) and os.path.isfile(b)]
        test_items = [(a,b) for a,b in zip([p[0] for p in test_pairs], [p[1] for p in test_pairs]) if os.path.isfile(a) and os.path.isfile(b)]
        print(f"  Week7: {len(train_items)} train / {len(val_items)} val / {len(test_items)} test")
    else:
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
        train_items = [(p, pre_to_post_path(p)) for p in train_pre]
        val_items = [(p, pre_to_post_path(p)) for p in val_pre]
        test_items = [(p, pre_to_post_path(p)) for p in test_pre]
        load_fn = None
        print(f"\nSplit: {len(train_pre)} train / {len(val_pre)} val / {len(test_pre)} test")
    
    print(f"\n📊 Creating patch-based datasets...")
    train_dataset = Patch3DDataset(None, patch_size=CONFIG['patch_size'], num_patches=CONFIG['num_patches'], load_fn=load_fn, items=train_items)
    val_dataset = Patch3DDataset(None, patch_size=CONFIG['patch_size'], num_patches=CONFIG['num_patches'], load_fn=load_fn, items=val_items)
    
    class FullVolumePairs(Dataset):
        def __init__(self, items, target_size=(128, 128, 64), load_fn=None):
            self.items = items
            self.target_size = target_size
            self.load_fn = load_fn
            print(f"  → Test: {len(self.items)} paired volumes")
        
        def __len__(self):
            return len(self.items)
        
        def __getitem__(self, idx):
            pre_p, post_p = self.items[idx]
            if self.load_fn is not None:
                pre_vol = minmax_norm(self.load_fn(pre_p))
                post_vol = minmax_norm(self.load_fn(post_p))
            else:
                pre_vol = minmax_norm(load_full_volume(pre_p, self.target_size))
                post_vol = minmax_norm(load_full_volume(post_p, self.target_size))
            pre_t = torch.from_numpy(pre_vol).unsqueeze(0).float()
            post_t = torch.from_numpy(post_vol).unsqueeze(0).float()
            return pre_t, post_t
    
    test_dataset = FullVolumePairs(test_items, target_size=CONFIG['target_size'], load_fn=load_fn)
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], 
                             shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], 
                           shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # Model
    print(f"\n🏗️  Creating Improved 3D Cold Diffusion Model...")
    model = ImprovedColdDiffusionNet3D(channels=(64, 128, 256)).to(CONFIG['device'])
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training
    print(f"\n🚀 Training...")
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(1, CONFIG['epochs'] + 1):
        start_time = time.time()
        
        train_loss = train_one_epoch(model, train_loader, optimizer, 
                                     CONFIG['n_timesteps_train'], alpha_schedule, 
                                     CONFIG['device'])
        
        # Validation (on patches)
        model.eval()
        val_loss_sum = 0
        val_samples = 0
        criterion_l1 = nn.L1Loss()
        with torch.no_grad():
            for pre_patch, post_patch in val_loader:
                if val_samples >= 20:  # Limit validation samples
                    break
                pre_patch = pre_patch.to(CONFIG['device'])
                post_patch = post_patch.to(CONFIG['device'])
                B = pre_patch.size(0)
                t_batch = torch.randint(0, CONFIG['n_timesteps_train'], (B,), 
                                       dtype=torch.long, device=CONFIG['device'])
                x_t = cold_degrade(post_patch, pre_patch, t_batch, alpha_schedule)
                x_0_pred = model(x_t, pre_patch, t_batch)
                val_loss_sum += criterion_l1(x_0_pred, post_patch).item() * B
                val_samples += B
        
        val_loss = val_loss_sum / max(val_samples, 1)
        scheduler.step(val_loss)
        
        epoch_time = time.time() - start_time
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch:3d}/{CONFIG['epochs']} - Train: {train_loss:.4f} - Val: {val_loss:.4f} - LR: {current_lr:.6f} - Time: {epoch_time:.1f}s")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), ckpt_name)
            print(f"  ✓ Saved new best model")
        else:
            patience_counter += 1
            if patience_counter >= CONFIG['early_stop_patience']:
                print(f"  Early stopping at epoch {epoch}")
                break
    
    # Final evaluation on test set
    print(f"\n📊 Final Evaluation on Test Set...")
    model.load_state_dict(torch.load(ckpt_name))
    test_results = evaluate_model(model, test_loader, CONFIG['n_timesteps_train'],
                                  CONFIG['n_steps_ddim'], CONFIG['device'], patch_size=CONFIG['patch_size'], use_week7=CONFIG['use_week7'])
    
    print(f"\nTest Results:")
    print(f"  MAE:  {test_results['mae_mean']:.4f} ± {test_results['mae_std']:.4f}")
    print(f"  SSIM: {test_results['ssim_mean']:.4f} ± {test_results['ssim_std']:.4f}")
    print(f"  PSNR: {test_results['psnr_mean']:.2f} ± {test_results['psnr_std']:.2f} dB")
    
    # Save results
    with open(results_name, 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print(f"\n✅ Complete! Saved: {ckpt_name}, {results_name}")
    print("="*70)


if __name__ == "__main__":
    main()
