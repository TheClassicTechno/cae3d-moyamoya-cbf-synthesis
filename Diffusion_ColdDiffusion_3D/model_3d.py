#!/usr/bin/env python3
"""
3D Cold Diffusion for Moyamoya CVR Prediction
==============================================
Full volumetric (3D) implementation using deterministic degradation.

Key Differences from 2D:
- spatial_dims=3 in UNet
- Degradation: x_t = (1-alpha)*post + alpha*pre (3D volumes)
- All operations on (B, 1, H, W, D) tensors

Reference: Bansal et al. "Cold Diffusion" (2022)
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

# Week7: 91x109x91 -> pad 96x112x96; metrics/save at 91x109x91
WEEK7_ORIGINAL_SHAPE = (91, 109, 91)
PAD_3D_WEEK7 = (96, 112, 96)


def _pad_3d(pre_t, post_t, target_shape):
    if target_shape is None or (pre_t.shape[2], pre_t.shape[3], pre_t.shape[4]) == target_shape:
        return pre_t, post_t
    th, tw, td = target_shape
    _, _, h, w, d = pre_t.shape
    ph, pw, pd = max(0, th - h), max(0, tw - w), max(0, td - d)
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
    def __init__(self, pre_paths, target_size=(128, 128, 64)):
        self.target_size = target_size
        self.items = [(p, pre_to_post_path(p)) for p in pre_paths 
                      if os.path.exists(pre_to_post_path(p))]
        print(f"  → Loaded {len(self.items)} paired volumes")
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        pre_p, post_p = self.items[idx]
        pre_vol = minmax_norm(load_full_volume(pre_p, self.target_size))
        post_vol = minmax_norm(load_full_volume(post_p, self.target_size))
        pre_t = torch.from_numpy(pre_vol).unsqueeze(0).float()
        post_t = torch.from_numpy(post_vol).unsqueeze(0).float()
        return pre_t, post_t


###############################################################################
# 3D Cold Diffusion Model
###############################################################################

class ColdDiffusionNet3D(nn.Module):
    """
    3D Cold Diffusion restoration network.
    Predicts clean x_0 from degraded x_t.
    """
    def __init__(self, channels=(32, 64, 128)):
        super().__init__()
        self.unet = UNet(
            spatial_dims=3,
            in_channels=2,  # x_t + pre (concatenated)
            out_channels=1,  # Predicted x_0 (clean post)
            channels=list(channels),
            strides=(2, 2),
            num_res_units=2,
            act=("LeakyReLU", {"inplace": True}),
            norm="INSTANCE",
            dropout=0.0,
        )
    
    def forward(self, x_t, pre_t, t):
        """
        Args:
            x_t: Degraded post-scan (B, 1, H, W, D)
            pre_t: Pre-scan conditioning (B, 1, H, W, D)
            t: Timesteps (B,) - not used in deterministic cold diffusion
        Returns:
            Predicted clean post (B, 1, H, W, D)
        """
        cond_input = torch.cat([x_t, pre_t], dim=1)  # (B, 2, H, W, D)
        return self.unet(cond_input)


###############################################################################
# Cold Diffusion Process
###############################################################################

def make_alpha_schedule(n_timesteps=100):
    """Linear alpha schedule from 0 to 1."""
    return np.linspace(0.0, 1.0, n_timesteps, dtype=np.float32)

def cold_degrade(x_0, pre_t, t_batch, alpha_schedule):
    """
    Deterministic degradation: x_t = (1 - alpha_t) * x_0 + alpha_t * pre_t
    At t=0: x_t = x_0 (clean post)
    At t=T: x_t = pre_t (fully degraded to pre)
    """
    B = x_0.shape[0]
    alpha_t = alpha_schedule[t_batch].view(B, 1, 1, 1, 1)
    return (1 - alpha_t) * x_0 + alpha_t * pre_t

def cold_restore_step(model, x_t, pre_t, t_val, alpha_schedule, device):
    """Single restoration step."""
    B = x_t.shape[0]
    t_batch = torch.full((B,), t_val, dtype=torch.long, device=device)
    
    with torch.no_grad():
        x_0_pred = model(x_t, pre_t, t_batch)
        x_0_pred = torch.clamp(x_0_pred, 0, 1)
    
    if t_val > 0:
        # Move one step closer to x_0
        alpha_t_prev = alpha_schedule[t_val - 1]
        x_t_prev = (1 - alpha_t_prev) * x_0_pred + alpha_t_prev * pre_t
    else:
        x_t_prev = x_0_pred
    
    return x_t_prev

def cold_sample(model, pre_t, n_timesteps, alpha_schedule, device):
    """Full restoration from t=T to t=0."""
    x_t = pre_t.clone()  # Start from degraded state (pre-scan)
    
    for t_val in reversed(range(n_timesteps)):
        x_t = cold_restore_step(model, x_t, pre_t, t_val, alpha_schedule, device)
    
    return x_t


###############################################################################
# Training
###############################################################################

def train_one_epoch(model, loader, optimizer, criterion_l1, criterion_ssim, n_timesteps, alpha_schedule, device, pad_3d=None):
    model.train()
    epoch_loss = 0.0
    
    for pre_vol, post_vol in loader:
        pre_vol = pre_vol.to(device)
        post_vol = post_vol.to(device)
        if pad_3d is not None:
            pre_vol, post_vol = _pad_3d(pre_vol, post_vol, pad_3d)
        
        B = post_vol.shape[0]
        t_batch = torch.randint(0, n_timesteps, (B,), dtype=torch.long, device=device)
        
        # Degrade post → x_t
        x_t = cold_degrade(post_vol, pre_vol, t_batch, alpha_schedule)
        
        # Predict x_0 (clean post)
        optimizer.zero_grad()
        x_0_pred = model(x_t, pre_vol, t_batch)
        
        loss_l1 = criterion_l1(x_0_pred, post_vol)
        loss_ssim = criterion_ssim(x_0_pred, post_vol)
        loss = loss_l1 + loss_ssim
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    return epoch_loss / len(loader)


###############################################################################
# Evaluation
###############################################################################

def evaluate_model(model, loader, n_timesteps, alpha_schedule, device, pad_3d=None):
    model.eval()
    mae_list, ssim_list, psnr_list = [], [], []
    oh, ow, od = WEEK7_ORIGINAL_SHAPE if pad_3d else (None, None, None)
    
    with torch.no_grad():
        for pre_vol, post_vol in loader:
            pre_vol = pre_vol.to(device)
            post_vol = post_vol.to(device)
            if pad_3d is not None:
                pre_vol, post_vol = _pad_3d(pre_vol, post_vol, pad_3d)
            
            pred_vol = cold_sample(model, pre_vol, n_timesteps, alpha_schedule, device)
            
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
    print("3D COLD DIFFUSION FOR MOYAMOYA CVR PREDICTION")
    print("="*70)
    
    CONFIG = {
        'target_size': (128, 128, 64),
        'batch_size': 2,
        'lr': 1e-3,
        'epochs': 50,
        'early_stop_patience': 10,
        'n_timesteps': 100,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'seed': int(os.environ.get('SEED', 42)),
    }
    
    print(f"\nConfiguration:")
    for k, v in CONFIG.items():
        print(f"  {k}: {v}")
    
    random.seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])
    torch.manual_seed(CONFIG['seed'])
    
    # Alpha schedule
    alpha_np = make_alpha_schedule(CONFIG['n_timesteps'])
    alpha_schedule = torch.from_numpy(alpha_np).float().to(CONFIG['device'])
    
    # Load data
    print(f"\n📂 Loading data...")
    sys.path.insert(0, os.path.join(_REPO_ROOT, 'scripts'))
    from week7_preprocess import is_env_flag, is_week7_kfold, TARGET_SHAPE, get_combined_split_path, week7_kfold_suffix_paths
    use_week7 = (
        is_env_flag('WEEK7')
        or '--week7' in sys.argv
        or is_week7_kfold()
    )
    pad_3d = None
    ckpt_name = 'cold_diffusion_3d_best.pt'
    results_name = 'cold_diffusion_3d_results.json'
    if use_week7:
        from week7_data import get_week7_splits, Week7VolumePairs3D
        print(f"  Split JSON: {get_combined_split_path()}")
        train_pairs, val_pairs, test_pairs = get_week7_splits()
        CONFIG['target_size'] = PAD_3D_WEEK7
        pad_3d = PAD_3D_WEEK7
        use_phase2 = is_env_flag('WEEK7_REGION_WEIGHT')
        ckpt_name = 'cold_diffusion_3d_week7_best.pt'
        results_name = 'cold_diffusion_3d_week7_phase2_results.json' if use_phase2 else 'cold_diffusion_3d_week7_results.json'
        ckpt_name, results_name = week7_kfold_suffix_paths(ckpt_name, results_name)
        print(f"  Week7: 91x109x91 + brain mask: {len(train_pairs)} train / {len(val_pairs)} val / {len(test_pairs)} test")
        train_dataset = Week7VolumePairs3D(train_pairs, augment=True, target_shape=TARGET_SHAPE)
        val_dataset = Week7VolumePairs3D(val_pairs, augment=False, target_shape=TARGET_SHAPE)
        test_dataset = Week7VolumePairs3D(test_pairs, augment=False, target_shape=TARGET_SHAPE)
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
        print(f"\nSplit: {len(train_pre)} train / {len(val_pre)} val / {len(test_pre)} test")
        train_dataset = FullVolumePairs(train_pre, CONFIG['target_size'])
        val_dataset = FullVolumePairs(val_pre, CONFIG['target_size'])
        test_dataset = FullVolumePairs(test_pre, CONFIG['target_size'])
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=2)
    
    # Model
    print(f"\n🏗️  Creating 3D Cold Diffusion Model...")
    model = ColdDiffusionNet3D().to(CONFIG['device'])
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    
    criterion_l1 = nn.L1Loss()
    criterion_ssim = SSIMLoss(spatial_dims=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['lr'])
    
    # Training
    print(f"\n🚀 Training...")
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(1, CONFIG['epochs'] + 1):
        start_time = time.time()
        
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion_l1, criterion_ssim,
                                      CONFIG['n_timesteps'], alpha_schedule, CONFIG['device'], pad_3d=pad_3d)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for pre_vol, post_vol in val_loader:
                pre_vol = pre_vol.to(CONFIG['device'])
                post_vol = post_vol.to(CONFIG['device'])
                if pad_3d is not None:
                    pre_vol, post_vol = _pad_3d(pre_vol, post_vol, pad_3d)
                B = post_vol.shape[0]
                t_batch = torch.randint(0, CONFIG['n_timesteps'], (B,), dtype=torch.long, device=CONFIG['device'])
                x_t = cold_degrade(post_vol, pre_vol, t_batch, alpha_schedule)
                x_0_pred = model(x_t, pre_vol, t_batch)
                loss = criterion_l1(x_0_pred, post_vol) + criterion_ssim(x_0_pred, post_vol)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        
        epoch_time = time.time() - start_time
        
        print(f"Epoch {epoch:3d}/{CONFIG['epochs']} - Train: {train_loss:.4f} - Val: {val_loss:.4f} - Time: {epoch_time:.1f}s")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), ckpt_name)
            print(f"  ✓ Saved new best model")
        else:
            patience_counter += 1
            if patience_counter >= CONFIG['early_stop_patience']:
                print(f"\n⚠️  Early stopping triggered")
                break
    
    # Evaluation
    print(f"\n📊 Final Evaluation on Test Set...")
    model.load_state_dict(torch.load(ckpt_name))
    test_results = evaluate_model(model, test_loader, CONFIG['n_timesteps'], alpha_schedule, CONFIG['device'], pad_3d=pad_3d)
    
    print(f"\nTest Results:")
    print(f"  MAE:  {test_results['mae_mean']:.4f} ± {test_results['mae_std']:.4f}")
    print(f"  SSIM: {test_results['ssim_mean']:.4f} ± {test_results['ssim_std']:.4f}")
    print(f"  PSNR: {test_results['psnr_mean']:.2f} ± {test_results['psnr_std']:.2f} dB")
    
    results = {'config': CONFIG, 'test_results': test_results}
    with open(results_name, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Complete! Saved: {ckpt_name}, {results_name}")
    print("="*70)


if __name__ == "__main__":
    main()
