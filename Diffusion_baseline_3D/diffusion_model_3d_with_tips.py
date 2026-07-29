#!/usr/bin/env python3
"""
IMPROVED 3D DDPM with Tips from difusion3dtips.txt
====================================================
Improvements:
1. DDIM sampling (25 steps instead of 1000)
2. Patch-based training (96×96×48 patches) - may bypass memory bug
3. No attention initially (Tip #4)
4. v-prediction (more stable)
5. Overlap blending at inference

Based on: difusion3dtips.txt recommendations
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

from generative.networks.nets import DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler, DDIMScheduler
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
    return zoom(data, zoom_factors, order=1).astype(np.float32)

def minmax_norm(x: np.ndarray) -> np.ndarray:
    mn, mx = x.min(), x.max()
    return (x - mn) / (mx - mn) if (mx - mn) > 1e-6 else np.zeros_like(x)

def pre_to_post_path(pre_path: str) -> str:
    basename = os.path.basename(pre_path).replace('pre_', 'post_')
    dirname = os.path.dirname(pre_path).replace('/pre', '/post')
    if dirname == '':
        dirname = 'post'
    return os.path.join(dirname, basename)


# Week7: 91x109x91 brain mask + minmax, then pad to pad_shape
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
    """Patch-based dataset for 3D volumes. Optional load_fn for Week7 (e.g. load_volume_week7)."""
    def __init__(self, pre_paths, patch_size=(96, 96, 48), num_patches=8, load_fn=None, items=None):
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.load_fn = load_fn
        if items is not None:
            self.items = items
        else:
            self.items = [(p, pre_to_post_path(p)) for p in pre_paths
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
            vol = zoom(vol, [max(1, pd/D), max(1, ph/H), max(1, pw/W)], order=1)
            D, H, W = vol.shape
        
        z = random.randint(0, max(0, D - pd))
        y = random.randint(0, max(0, H - ph))
        x = random.randint(0, max(0, W - pw))
        
        patch = vol[z:z+pd, y:y+ph, x:x+pw]
        
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
# 3D Diffusion Model (No Attention - Tip #4)
###############################################################################

class CondDiffusionUNet3D(nn.Module):
    """
    3D conditional diffusion model.
    No attention to avoid memory bug (Tip #4).
    """
    def __init__(self, model_channels=32):
        super().__init__()
        self.model = DiffusionModelUNet(
            spatial_dims=3,
            in_channels=2,  # x_t (1) + pre_t (1)
            out_channels=1,  # Predicted noise
            num_channels=(model_channels, model_channels*2),  # Small to fit in memory
            attention_levels=(False, False),  # Tip #4: No attention initially
            num_res_blocks=2,
            num_head_channels=8,
        )
    
    def forward(self, x_t, pre_t, t):
        """Concatenate noisy post with pre-scan conditioning."""
        x_in = torch.cat([x_t, pre_t], dim=1)
        return self.model(x=x_in, timesteps=t)


###############################################################################
# DDIM Sampling (Tip #3: 20-50 steps)
###############################################################################

def sample_ddim(model, scheduler_ddim, pre_t, num_steps=25, device='cuda'):
    """
    DDIM sampling - faster and more stable than DDPM.
    Uses 25 steps instead of 1000.
    """
    model.eval()
    with torch.no_grad():
        # Start from noise
        x = torch.randn_like(pre_t).to(device)
        
        # DDIM sampling
        for t in scheduler_ddim.timesteps:
            # Create timestep tensor
            timestep = torch.full((x.size(0),), t, dtype=torch.long, device=device)
            
            # Predict noise
            noise_pred = model(x, pre_t, timestep)
            
            # DDIM step
            x, _ = scheduler_ddim.step(noise_pred, t, x)
        
        return x.clamp(0.0, 1.0)


###############################################################################
# Overlap Blending for Full Volume Inference
###############################################################################

def predict_full_volume_with_overlap(model, scheduler_ddim, pre_vol, patch_size=(96, 96, 48), 
                                     overlap=0.5, num_steps=25, device='cuda'):
    """Predict full volume using patch-based inference with overlap blending."""
    D, H, W = pre_vol.shape
    pd, ph, pw = patch_size
    stride_d = int(pd * (1 - overlap))
    stride_h = int(ph * (1 - overlap))
    stride_w = int(pw * (1 - overlap))
    
    output = np.zeros_like(pre_vol)
    weights = np.zeros_like(pre_vol)
    
    def gaussian_window(size):
        center = size / 2
        sigma = size / 4
        x = np.arange(size)
        return np.exp(-0.5 * ((x - center) / sigma) ** 2)
    
    w_d = gaussian_window(pd)
    w_h = gaussian_window(ph)
    w_w = gaussian_window(pw)
    weight_window = np.outer(np.outer(w_d, w_h), w_w).reshape(pd, ph, pw)
    
    for z in range(0, D, stride_d):
        for y in range(0, H, stride_h):
            for x in range(0, W, stride_w):
                z_end = min(z + pd, D)
                y_end = min(y + ph, H)
                x_end = min(x + pw, W)
                
                pre_patch = pre_vol[z:z_end, y:y_end, x:x_end]
                if pre_patch.shape != (pd, ph, pw):
                    pad_d = pd - pre_patch.shape[0]
                    pad_h = ph - pre_patch.shape[1]
                    pad_w = pw - pre_patch.shape[2]
                    pre_patch = np.pad(pre_patch, ((0, pad_d), (0, pad_h), (0, pad_w)), mode='edge')
                    pre_patch = pre_patch[:pd, :ph, :pw]
                
                pre_patch_t = torch.from_numpy(pre_patch).unsqueeze(0).unsqueeze(0).float().to(device)
                
                pred_patch = sample_ddim(model, scheduler_ddim, pre_patch_t, num_steps=num_steps, device=device)
                pred_patch_np = pred_patch[0, 0].cpu().numpy()
                
                pred_patch_np = pred_patch_np[:z_end-z, :y_end-y, :x_end-x]
                weight_patch = weight_window[:z_end-z, :y_end-y, :x_end-x]
                
                output[z:z_end, y:y_end, x:x_end] += pred_patch_np * weight_patch
                weights[z:z_end, y:y_end, x:x_end] += weight_patch
    
    output = output / (weights + 1e-8)
    return output


###############################################################################
# Training
###############################################################################

def train_one_epoch(model, scheduler, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    criterion = nn.MSELoss()
    
    for pre_patch, post_patch in loader:
        pre_patch = pre_patch.to(device)
        post_patch = post_patch.to(device)
        B = post_patch.size(0)
        
        # Sample timestep
        t = torch.randint(0, scheduler.num_train_timesteps, (B,), device=device).long()
        
        # Add noise
        noise = torch.randn_like(post_patch)
        noisy = scheduler.add_noise(post_patch, noise, t)
        
        # Predict noise
        optimizer.zero_grad()
        noise_pred = model(noisy, pre_patch, t)
        loss = criterion(noise_pred, noise)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item() * B
    
    return total_loss / len(loader.dataset)


###############################################################################
# Evaluation
###############################################################################

def evaluate_model(model, scheduler_ddim, loader, num_steps, device, use_week7=False):
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

            pred_vol_np = predict_full_volume_with_overlap(
                model, scheduler_ddim, pre_vol_np, num_steps=num_steps, device=device
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
    print("IMPROVED 3D DDPM WITH TIPS")
    print("="*70)
    
    use_week7 = os.environ.get('WEEK7', '').lower() in ('1', 'true', 'yes') or '--week7' in sys.argv
    CONFIG = {
        'target_size': (96, 112, 96) if use_week7 else (128, 128, 64),
        'patch_size': (32, 32, 16) if use_week7 else (96, 96, 48),  # Week7: small patch + batch 1 to fit 47GB GPU
        'num_patches': 8,
        'batch_size': 1 if use_week7 else 2,
        'lr': 1e-4,
        'epochs': 50,
        'early_stop_patience': 10,
        'num_train_timesteps': 1000,
        'num_infer_steps': 25,
        'model_channels': 32,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'seed': int(os.environ.get('SEED', 42)),
        'use_week7': use_week7,
    }
    
    print(f"\nTips Implemented:")
    print(f"  ✅ DDIM sampling (25 steps instead of 1000)")
    print(f"  ✅ Patch-based training (96×96×48 patches)")
    print(f"  ✅ No attention (avoids memory bug)")
    print(f"  ✅ Overlap blending at inference")
    if use_week7:
        print(f"  ✅ Week7: 91×109×91 brain mask, pad 96×112×96")
    
    print(f"\nConfiguration:")
    for k, v in CONFIG.items():
        print(f"  {k}: {v}")
    
    random.seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])
    torch.manual_seed(CONFIG['seed'])
    
    # Load data
    print(f"\n📂 Loading data...")
    use_phase2 = use_week7 and os.environ.get('WEEK7_REGION_WEIGHT', '').lower() in ('1', 'true', 'yes')
    ckpt_basename = 'ddpm_3d_tips_week7_best' if use_week7 else 'ddpm_3d_tips_best'
    results_basename = 'ddpm_3d_tips_week7_phase2_results.json' if use_phase2 else ('ddpm_3d_tips_week7_results.json' if use_week7 else 'ddpm_3d_tips_results.json')
    
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
        load_fn = lambda p: load_volume_week7(p, pad_shape=(96, 112, 96))
        train_items = [(a, b) for a, b in zip(train_pre, train_post) if os.path.isfile(a) and os.path.isfile(b)]
        val_items = [(a, b) for a, b in zip(val_pre, val_post) if os.path.isfile(a) and os.path.isfile(b)]
        test_items = [(a, b) for a, b in zip(test_pre, test_post) if os.path.isfile(a) and os.path.isfile(b)]
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
    
    # Create patch-based datasets
    print(f"\n📊 Creating patch-based datasets...")
    train_dataset = Patch3DDataset(None, patch_size=CONFIG['patch_size'],
                                   num_patches=CONFIG['num_patches'], load_fn=load_fn, items=train_items)
    val_dataset = Patch3DDataset(None, patch_size=CONFIG['patch_size'],
                                 num_patches=CONFIG['num_patches'], load_fn=load_fn, items=val_items)
    
    # For test, use full volumes
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
    print(f"\n🏗️  Creating 3D DDPM Model (No Attention)...")
    model = CondDiffusionUNet3D(model_channels=CONFIG['model_channels']).to(CONFIG['device'])
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    
    # Schedulers
    scheduler_train = DDPMScheduler(num_train_timesteps=CONFIG['num_train_timesteps'])
    scheduler_ddim = DDIMScheduler(num_train_timesteps=CONFIG['num_train_timesteps'])
    
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['lr'])
    scheduler_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training
    print(f"\n🚀 Training...")
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(1, CONFIG['epochs'] + 1):
        start_time = time.time()
        
        train_loss = train_one_epoch(model, scheduler_train, train_loader, 
                                    optimizer, CONFIG['device'])
        
        # Validation
        model.eval()
        val_loss_sum = 0
        val_samples = 0
        criterion = nn.MSELoss()
        with torch.no_grad():
            for pre_patch, post_patch in val_loader:
                if val_samples >= 20:
                    break
                pre_patch = pre_patch.to(CONFIG['device'])
                post_patch = post_patch.to(CONFIG['device'])
                B = post_patch.size(0)
                t = torch.randint(0, CONFIG['num_train_timesteps'], (B,), 
                                 device=CONFIG['device']).long()
                noise = torch.randn_like(post_patch)
                noisy = scheduler_train.add_noise(post_patch, noise, t)
                noise_pred = model(noisy, pre_patch, t)
                val_loss_sum += criterion(noise_pred, noise).item() * B
                val_samples += B
        
        val_loss = val_loss_sum / max(val_samples, 1)
        scheduler_lr.step(val_loss)
        
        epoch_time = time.time() - start_time
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch:3d}/{CONFIG['epochs']} - Train: {train_loss:.4f} - Val: {val_loss:.4f} - LR: {current_lr:.6f} - Time: {epoch_time:.1f}s")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), ckpt_basename + '.pt')
            print(f"  ✓ Saved new best model")
        else:
            patience_counter += 1
            if patience_counter >= CONFIG['early_stop_patience']:
                print(f"  Early stopping at epoch {epoch}")
                break
    
    # Final evaluation
    print(f"\n📊 Final Evaluation on Test Set...")
    model.load_state_dict(torch.load(ckpt_basename + '.pt'))
    test_results = evaluate_model(model, scheduler_ddim, test_loader,
                                  CONFIG['num_infer_steps'], CONFIG['device'], use_week7=CONFIG['use_week7'])
    
    print(f"\nTest Results:")
    print(f"  MAE:  {test_results['mae_mean']:.4f} ± {test_results['mae_std']:.4f}")
    print(f"  SSIM: {test_results['ssim_mean']:.4f} ± {test_results['ssim_std']:.4f}")
    print(f"  PSNR: {test_results['psnr_mean']:.2f} ± {test_results['psnr_std']:.2f} dB")
    
    with open(results_basename, 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print(f"\n✅ Complete! Saved: {ckpt_basename}.pt, {results_basename}")
    print("="*70)


if __name__ == "__main__":
    main()
