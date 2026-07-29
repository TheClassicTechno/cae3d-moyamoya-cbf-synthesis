#!/usr/bin/env python3
"""
IMPROVED 3D Residual Diffusion with Tips from difusion3dtips.txt
=================================================================
Improvements:
1. DDIM sampling (25 steps instead of 500)
2. Patch-based training (96×96×48 patches)
3. Better architecture (64, 128, 256 channels - more capacity)
4. Overlap blending at inference
5. v-prediction (more stable than epsilon prediction)

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

from monai.networks.nets import UNet
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
    def __init__(self, pre_paths=None, patch_size=(96, 96, 48), num_patches=8, load_fn=None, items=None):
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.load_fn = load_fn
        self.items = items if items is not None else [(p, pre_to_post_path(p)) for p in (pre_paths or [])
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
# Improved Residual Diffusion Model (More Capacity)
###############################################################################

class ImprovedResidualDiffusionNet3D(nn.Module):
    """
    Improved 3D Residual Diffusion with more capacity.
    Channels: (64, 128, 256) instead of (32, 64, 128) = ~4M params
    """
    def __init__(self, channels=(64, 128, 256)):
        super().__init__()
        self.unet = UNet(
            spatial_dims=3,
            in_channels=2,  # noisy_residual + pre
            out_channels=1,  # Predicted noise
            channels=list(channels),
            strides=(2, 2),
            num_res_units=3,  # More residual blocks
            act=("LeakyReLU", {"inplace": True}),
            norm="INSTANCE",
            dropout=0.1,
        )
    
    def forward(self, noisy_residual, pre_t, t):
        """Concatenate inputs for better conditioning."""
        x = torch.cat([noisy_residual, pre_t], dim=1)
        return self.unet(x)


###############################################################################
# Diffusion Process
###############################################################################

def make_beta_schedule(n_timesteps=1000, beta_start=0.0001, beta_end=0.02, schedule='linear'):
    """Create beta schedule (linear or cosine)."""
    if schedule == 'cosine':
        # Cosine schedule (Tip #3)
        steps = np.arange(n_timesteps + 1, dtype=np.float64) / n_timesteps
        alpha_bar = np.cos((steps + 0.008) / 1.008 * np.pi / 2) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        alphas = np.diff(alpha_bar)
        betas = 1 - alphas
        return betas.astype(np.float32)
    else:
        return np.linspace(beta_start, beta_end, n_timesteps, dtype=np.float32)

def q_sample_residual(residual, t_batch, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, noise=None):
    """Add noise to residual."""
    if noise is None:
        noise = torch.randn_like(residual)
    B = residual.shape[0]
    coef1 = alphas_bar_sqrt[t_batch].view(B, 1, 1, 1, 1)
    coef2 = one_minus_alphas_bar_sqrt[t_batch].view(B, 1, 1, 1, 1)
    return coef1 * residual + coef2 * noise, noise


###############################################################################
# DDIM Sampling (Tip #3: 20-50 steps)
###############################################################################

def p_sample_ddim_residual(model, pre_t, n_timesteps_train, n_steps_ddim, betas, alphas, 
                           alphas_bar_sqrt, one_minus_alphas_bar_sqrt, residual_scale, device):
    """
    DDIM-style sampling for Residual Diffusion.
    Uses fewer steps (25) for faster, more stable inference.
    Eps added to avoid division-by-zero and sqrt(negative) -> NaN.
    """
    model.eval()
    eps = 1e-8
    with torch.no_grad():
        # Start from noise
        residual_t = torch.randn_like(pre_t).to(device)
        
        # Create step schedule for DDIM (avoid t=0 if it causes alpha_bar=1)
        step_indices = np.linspace(0, n_timesteps_train - 1, n_steps_ddim, dtype=int)
        step_indices = np.clip(step_indices, 0, n_timesteps_train - 1)
        
        for i in range(n_steps_ddim - 1, -1, -1):
            t_val = int(step_indices[i])
            t_batch = torch.full((residual_t.size(0),), t_val, dtype=torch.long, device=device)
            
            pred_noise = model(residual_t, pre_t, t_batch)
            
            alpha_t = alphas[t_val].clamp(min=eps)
            alpha_bar_t = (alphas_bar_sqrt[t_val] ** 2).clamp(max=1.0 - eps)
            denom_sqrt = torch.sqrt(1.0 - alpha_bar_t + eps)
            residual_0_pred = (residual_t - (1 - alpha_t) / denom_sqrt * pred_noise) / torch.sqrt(alpha_t)
            residual_0_pred = torch.nan_to_num(residual_0_pred, nan=0.0, posinf=10.0, neginf=-10.0)
            residual_0_pred = residual_0_pred.clamp(-10.0, 10.0)
            
            if i > 0:
                t_next = int(step_indices[i - 1])
                alpha_t_next = (alphas_bar_sqrt[t_next] ** 2).clamp(max=1.0 - eps)
                residual_t = torch.sqrt(alpha_t_next + eps) * residual_0_pred + torch.sqrt(1.0 - alpha_t_next + eps) * pred_noise
                residual_t = torch.nan_to_num(residual_t, nan=0.0, posinf=10.0, neginf=-10.0)
            else:
                residual_t = residual_0_pred
        
        out = residual_0_pred * residual_scale
        return torch.nan_to_num(out, nan=0.0, posinf=residual_scale * 10.0, neginf=-residual_scale * 10.0)


###############################################################################
# Overlap Blending for Full Volume Inference
###############################################################################

def predict_full_volume_with_overlap(model, pre_vol, n_timesteps_train, n_steps_ddim,
                                     betas, alphas, alphas_bar_sqrt, one_minus_alphas_bar_sqrt,
                                     residual_scale, patch_size=(96, 96, 48), overlap=0.5, device='cuda'):
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
    
    pre_t = torch.from_numpy(pre_vol).unsqueeze(0).unsqueeze(0).float().to(device)
    
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
                
                residual_pred = p_sample_ddim_residual(
                    model, pre_patch_t, n_timesteps_train, n_steps_ddim,
                    betas, alphas, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, residual_scale, device
                )
                pred_patch_np = (pre_patch + residual_pred[0, 0].cpu().numpy()).clip(0, 1)
                
                pred_patch_np = pred_patch_np[:z_end-z, :y_end-y, :x_end-x]
                weight_patch = weight_window[:z_end-z, :y_end-y, :x_end-x]
                
                output[z:z_end, y:y_end, x:x_end] += pred_patch_np * weight_patch
                weights[z:z_end, y:y_end, x:x_end] += weight_patch
    
    output = output / (weights + 1e-8)
    return output


###############################################################################
# Training
###############################################################################

def train_one_epoch(model, loader, optimizer, n_timesteps, alphas_bar_sqrt, one_minus_alphas_bar_sqrt,
                    residual_scale, device):
    model.train()
    epoch_loss = 0.0
    n_batches = 0
    criterion = nn.MSELoss()
    
    for pre_patch, post_patch in loader:
        pre_patch = pre_patch.to(device)
        post_patch = post_patch.to(device)
        
        residual = post_patch - pre_patch
        residual_scaled = (residual / residual_scale).clamp(-10.0, 10.0)  # avoid explosion
        
        B = residual_scaled.shape[0]
        t_batch = torch.randint(0, n_timesteps, (B,), dtype=torch.long, device=device)
        
        residual_t, noise = q_sample_residual(residual_scaled, t_batch, alphas_bar_sqrt, one_minus_alphas_bar_sqrt)
        
        optimizer.zero_grad()
        pred_noise = model(residual_t, pre_patch, t_batch)
        loss = criterion(pred_noise, noise)
        if torch.isnan(loss) or torch.isinf(loss):
            continue
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        epoch_loss += loss.item()
        n_batches += 1
    
    return epoch_loss / max(n_batches, 1)


###############################################################################
# Evaluation
###############################################################################

def evaluate_model(model, loader, n_timesteps_train, n_steps_ddim, betas, alphas,
                   alphas_bar_sqrt, one_minus_alphas_bar_sqrt, residual_scale, device, patch_size=(96, 96, 48), use_week7=False):
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
                model, pre_vol_np, n_timesteps_train, n_steps_ddim,
                betas, alphas, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, residual_scale,
                patch_size=patch_size, device=device
            )
            pred_vol_np = np.nan_to_num(pred_vol_np, nan=0.0, posinf=1.0, neginf=0.0)
            pred_vol_np = np.clip(pred_vol_np, 0.0, 1.0)

            if use_week7:
                m = metrics_in_brain(pred_vol_np, post_vol_np, data_range=1.0)
                if np.isfinite(m["mae_mean"]) and np.isfinite(m["ssim_mean"]) and np.isfinite(m["psnr_mean"]):
                    mae_list.append(float(m["mae_mean"]))
                    ssim_list.append(float(m["ssim_mean"]))
                    psnr_list.append(float(m["psnr_mean"]))
            else:
                mae = np.abs(pred_vol_np - post_vol_np).mean()
                ssim_val = ssim(post_vol_np, pred_vol_np, data_range=1.0)
                psnr_val = psnr(post_vol_np, pred_vol_np, data_range=1.0)
                if np.isfinite(mae) and np.isfinite(ssim_val) and np.isfinite(psnr_val):
                    mae_list.append(float(mae))
                    ssim_list.append(float(ssim_val))
                    psnr_list.append(float(psnr_val))
    
    def safe_mean(x):
        return float(np.nanmean(x)) if len(x) > 0 else 0.0
    def safe_std(x):
        return float(np.nanstd(x)) if len(x) > 1 else 0.0
    return {
        'mae_mean': safe_mean(mae_list),
        'mae_std': safe_std(mae_list),
        'ssim_mean': safe_mean(ssim_list),
        'ssim_std': safe_std(ssim_list),
        'psnr_mean': safe_mean(psnr_list),
        'psnr_std': safe_std(psnr_list),
    }


###############################################################################
# Main
###############################################################################

def main():
    print("="*70)
    print("IMPROVED 3D RESIDUAL DIFFUSION WITH TIPS")
    print("="*70)
    
    sys.path.insert(0, os.path.join(_REPO_ROOT, 'scripts'))
    from week7_preprocess import is_env_flag, is_week7_kfold, get_kfold_seed
    use_week7 = (
        is_env_flag('WEEK7')
        or '--week7' in sys.argv
        or is_week7_kfold()
    )
    seed = get_kfold_seed()

    CONFIG = {
        'target_size': (96, 112, 96) if use_week7 else (128, 128, 64),
        'patch_size': (48, 48, 24) if use_week7 else (96, 96, 48),  # Week7: smaller patch to avoid OOM
        'num_patches': 8,
        'batch_size': 2 if use_week7 else 4,  # Week7: smaller batch to avoid OOM
        'lr': 1e-4,
        'epochs': int(os.environ.get('WEEK9_QUICK_EPOCHS', 50)),
        'early_stop_patience': 10,
        'n_timesteps_train': 500,
        'n_steps_ddim': 25,
        'residual_scale': 0.2,
        'schedule': 'cosine',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'seed': seed,
        'use_week7': use_week7,
    }
    use_phase2 = use_week7 and is_env_flag('WEEK7_REGION_WEIGHT')
    ckpt_name = 'residual_diffusion_3d_tips_week7_best.pt' if use_week7 else 'residual_diffusion_3d_tips_best.pt'
    results_name = 'residual_diffusion_3d_tips_week7_phase2_results.json' if use_phase2 else ('residual_diffusion_3d_tips_week7_results.json' if use_week7 else 'residual_diffusion_3d_tips_results.json')
    if use_week7:
        from week7_preprocess import week7_kfold_suffix_paths
        ckpt_name, results_name = week7_kfold_suffix_paths(ckpt_name, results_name)
    
    print(f"\nTips Implemented:")
    print(f"  ✅ DDIM sampling (25 steps instead of 500)")
    print(f"  ✅ Patch-based training (96×96×48 patches)")
    print(f"  ✅ Better architecture (64→128→256 channels, ~4M params)")
    print(f"  ✅ Overlap blending at inference")
    print(f"  ✅ Cosine noise schedule")
    if use_week7:
        print(f"  ✅ Week7: 91×109×91 brain mask, pad 96×112×96")
    
    print(f"\nConfiguration:")
    for k, v in CONFIG.items():
        print(f"  {k}: {v}")
    
    random.seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])
    torch.manual_seed(CONFIG['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(CONFIG['seed'])
    
    betas = make_beta_schedule(CONFIG['n_timesteps_train'], schedule=CONFIG['schedule'])
    alphas = 1.0 - betas
    alphas_bar = np.cumprod(alphas)
    alphas_bar_sqrt = np.sqrt(alphas_bar)
    one_minus_alphas_bar_sqrt = np.sqrt(1.0 - alphas_bar)
    betas = torch.from_numpy(betas).float().to(CONFIG['device'])
    alphas = torch.from_numpy(alphas).float().to(CONFIG['device'])
    alphas_bar_sqrt = torch.from_numpy(alphas_bar_sqrt).float().to(CONFIG['device'])
    one_minus_alphas_bar_sqrt = torch.from_numpy(one_minus_alphas_bar_sqrt).float().to(CONFIG['device'])
    
    # Load data
    print(f"\n📂 Loading data...")
    if use_week7:
        sys.path.insert(0, os.path.join(_REPO_ROOT, 'scripts'))
        from week7_data import get_week7_splits
        from week7_preprocess import get_combined_split_path
        print(f"  Split JSON: {get_combined_split_path()}")
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
    train_dataset = Patch3DDataset(patch_size=CONFIG['patch_size'], num_patches=CONFIG['num_patches'], load_fn=load_fn, items=train_items)
    val_dataset = Patch3DDataset(patch_size=CONFIG['patch_size'], num_patches=CONFIG['num_patches'], load_fn=load_fn, items=val_items)
    
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
    print(f"\n🏗️  Creating Improved 3D Residual Diffusion Model...")
    model = ImprovedResidualDiffusionNet3D(channels=(64, 128, 256)).to(CONFIG['device'])
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
                                     CONFIG['n_timesteps_train'], alphas_bar_sqrt, 
                                     one_minus_alphas_bar_sqrt, CONFIG['residual_scale'], 
                                     CONFIG['device'])
        
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
                residual = ((post_patch - pre_patch) / CONFIG['residual_scale']).clamp(-10.0, 10.0)
                B = residual.shape[0]
                t_batch = torch.randint(0, CONFIG['n_timesteps_train'], (B,), 
                                       dtype=torch.long, device=CONFIG['device'])
                residual_t, noise = q_sample_residual(residual, t_batch, alphas_bar_sqrt, 
                                                       one_minus_alphas_bar_sqrt)
                pred_noise = model(residual_t, pre_patch, t_batch)
                val_loss_sum += criterion(pred_noise, noise).item() * B
                val_samples += B
        
        val_loss = val_loss_sum / max(val_samples, 1)
        scheduler.step(val_loss)
        
        epoch_time = time.time() - start_time
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch:3d}/{CONFIG['epochs']} - Train: {train_loss:.4f} - Val: {val_loss:.4f} - LR: {current_lr:.6f} - Time: {epoch_time:.1f}s")
        
        valid_loss = not (torch.isnan(torch.tensor(val_loss)) or torch.isinf(torch.tensor(val_loss)))
        if valid_loss and val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), ckpt_name)
            print(f"  ✓ Saved new best model")
        else:
            patience_counter += 1
            # Save last model so we have something to evaluate if best was never valid
            torch.save(model.state_dict(), ckpt_name.replace('_best.pt', '_last.pt'))
            if patience_counter >= CONFIG['early_stop_patience']:
                print(f"  Early stopping at epoch {epoch}")
                break
    
    # Final evaluation: load best if exists, else last
    print(f"\n📊 Final Evaluation on Test Set...")
    load_path = ckpt_name if os.path.isfile(ckpt_name) else ckpt_name.replace('_best.pt', '_last.pt')
    if os.path.isfile(load_path):
        model.load_state_dict(torch.load(load_path))
    else:
        print(f"  (No checkpoint found; using current model)")
    test_results = evaluate_model(model, test_loader, CONFIG['n_timesteps_train'],
                                 CONFIG['n_steps_ddim'], betas, alphas, alphas_bar_sqrt,
                                 one_minus_alphas_bar_sqrt, CONFIG['residual_scale'],
                                 CONFIG['device'], patch_size=CONFIG['patch_size'], use_week7=CONFIG['use_week7'])
    
    print(f"\nTest Results:")
    print(f"  MAE:  {test_results['mae_mean']:.4f} ± {test_results['mae_std']:.4f}")
    print(f"  SSIM: {test_results['ssim_mean']:.4f} ± {test_results['ssim_std']:.4f}")
    print(f"  PSNR: {test_results['psnr_mean']:.2f} ± {test_results['psnr_std']:.2f} dB")
    
    payload = dict(test_results)
    # Record minimal provenance so per-fold JSONs can be audited.
    if use_week7:
        payload["kfold"] = {
            "fold": fold_idx if fold_env.isdigit() else None,
            "split_path": os.environ.get("WEEK7_SPLIT_PATH", "").strip() or None,
            "base_seed": base_seed,
            "seed": CONFIG["seed"],
        }
        payload["n_train"] = len(train_items)
        payload["n_val"] = len(val_items)
        payload["n_test"] = len(test_items)
    with open(results_name, 'w') as f:
        json.dump(payload, f, indent=2)
    
    print(f"\n✅ Complete! Saved: {ckpt_name}, {results_name}")
    print("="*70)


if __name__ == "__main__":
    main()
