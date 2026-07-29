#!/usr/bin/env python3
"""
3D Residual Diffusion for Moyamoya CVR Prediction
==================================================
Full volumetric (3D) implementation that models the residual (post - pre).

Key Differences from 2D:
- spatial_dims=3 in DiffusionModelUNet
- All operations on (B, 1, H, W, D) tensors
- Residual = post - pre (3D volumes)
- Final prediction = pre + predicted_residual

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
# 3D Residual Diffusion Model
###############################################################################

class SimpleResidualDiffusion3D(nn.Module):
    """Lightweight 3D residual noise-prediction model (no attention) for Week7 to avoid OOM."""
    def __init__(self, ch=16):
        super().__init__()
        self.t_embed = nn.Embedding(1001, ch)
        self.conv_in = nn.Conv3d(2 + ch, ch, 3, padding=1)
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

    def forward(self, noisy_residual, pre_t, t):
        B, _, H, W, D = noisy_residual.shape
        cond = torch.cat([noisy_residual, pre_t], dim=1)
        t_emb = self.t_embed(t.clamp(0, 1000))
        t_broadcast = t_emb.view(B, self.ch, 1, 1, 1).expand(B, self.ch, H, W, D)
        x = torch.cat([cond, t_broadcast], dim=1)
        x = self.conv_in(x)
        d1 = self.d1(x)
        d2 = self.d2(d1)
        m = self.mid(d2)
        u2 = self.u2(m) + d1
        u1 = self.u1(u2) + x
        return self.conv_out(u1)


class ResidualDiffusionNet3D(nn.Module):
    """
    3D diffusion model that operates on residuals (post - pre).
    """
    def __init__(self, model_channels=32, num_res_blocks=2):  # Reduced from 64
        super().__init__()
        self.model = DiffusionModelUNet(
            spatial_dims=3,
            in_channels=2,  # noisy_residual (1) + pre (1)
            out_channels=1,  # Predicted noise
            channels=(model_channels, model_channels*2),  # Reduced depth
            attention_levels=(False, False),  # Disabled attention to save memory
            num_res_blocks=num_res_blocks,
            num_head_channels=8,  # Reduced from 32
        )
    
    def forward(self, noisy_residual, pre_t, t):
        """
        Args:
            noisy_residual: Noisy residual (B, 1, H, W, D)
            pre_t: Pre-scan conditioning (B, 1, H, W, D)
            t: Timesteps (B,)
        Returns:
            Predicted noise (B, 1, H, W, D)
        """
        cond_input = torch.cat([noisy_residual, pre_t], dim=1)
        return self.model(cond_input, timesteps=t)


###############################################################################
# Diffusion Process
###############################################################################

def make_beta_schedule(n_timesteps=1000, beta_start=0.0001, beta_end=0.02):
    return np.linspace(beta_start, beta_end, n_timesteps, dtype=np.float32)

def q_sample_residual(residual, t_batch, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, noise=None):
    """Add noise to residual."""
    if noise is None:
        noise = torch.randn_like(residual)
    
    B = residual.shape[0]
    coef1 = alphas_bar_sqrt[t_batch].view(B, 1, 1, 1, 1)
    coef2 = one_minus_alphas_bar_sqrt[t_batch].view(B, 1, 1, 1, 1)
    
    return coef1 * residual + coef2 * noise, noise

def p_sample_step_residual(model, residual_t, pre_t, t_val, betas, alphas, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, device):
    """Single reverse diffusion step for residual."""
    B = residual_t.shape[0]
    t_batch = torch.full((B,), t_val, dtype=torch.long, device=device)
    
    with torch.no_grad():
        pred_noise = model(residual_t, pre_t, t_batch)
    
    alpha_t = alphas[t_val].view(1, 1, 1, 1, 1)
    alpha_bar_t = (alphas_bar_sqrt[t_val] ** 2).view(1, 1, 1, 1, 1)
    beta_t = betas[t_val].view(1, 1, 1, 1, 1)
    
    residual_0_pred = (residual_t - (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_t)
    
    if t_val > 0:
        noise = torch.randn_like(residual_t)
        sigma_t = torch.sqrt(beta_t)
        residual_t_prev = (1 / torch.sqrt(alpha_t)) * (residual_t - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * pred_noise) + sigma_t * noise
    else:
        residual_t_prev = residual_0_pred
    
    return residual_t_prev

def p_sample_loop_residual(model, pre_t, n_timesteps, betas, alphas, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, device):
    """Generate residual from noise."""
    residual_t = torch.randn_like(pre_t).to(device)
    
    for t_val in reversed(range(n_timesteps)):
        residual_t = p_sample_step_residual(model, residual_t, pre_t, t_val, betas, alphas,
                                            alphas_bar_sqrt, one_minus_alphas_bar_sqrt, device)
    
    return residual_t


###############################################################################
# Training
###############################################################################

def train_one_epoch(model, loader, optimizer, n_timesteps, alphas_bar_sqrt, one_minus_alphas_bar_sqrt,
                    residual_scale, device, pad_3d=None):
    model.train()
    epoch_loss = 0.0
    criterion = nn.MSELoss()
    
    for pre_vol, post_vol in loader:
        pre_vol = pre_vol.to(device)
        post_vol = post_vol.to(device)
        if pad_3d is not None:
            pre_vol, post_vol = _pad_3d(pre_vol, post_vol, pad_3d)
        
        # Compute residual and scale
        residual = post_vol - pre_vol
        residual_scaled = residual / residual_scale
        
        B = residual_scaled.shape[0]
        t_batch = torch.randint(0, n_timesteps, (B,), dtype=torch.long, device=device)
        
        # Add noise to residual
        residual_t, noise = q_sample_residual(residual_scaled, t_batch, alphas_bar_sqrt, one_minus_alphas_bar_sqrt)
        
        # Predict noise
        optimizer.zero_grad()
        pred_noise = model(residual_t, pre_vol, t_batch)
        loss = criterion(pred_noise, noise)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    return epoch_loss / len(loader)


###############################################################################
# Evaluation
###############################################################################

def evaluate_model(model, loader, n_timesteps, betas, alphas, alphas_bar_sqrt, one_minus_alphas_bar_sqrt,
                   residual_scale, device, pad_3d=None):
    model.eval()
    mae_list, ssim_list, psnr_list = [], [], []
    oh, ow, od = WEEK7_ORIGINAL_SHAPE if pad_3d else (None, None, None)
    
    with torch.no_grad():
        for pre_vol, post_vol in loader:
            pre_vol = pre_vol.to(device)
            post_vol = post_vol.to(device)
            if pad_3d is not None:
                pre_vol, post_vol = _pad_3d(pre_vol, post_vol, pad_3d)
            
            residual_pred = p_sample_loop_residual(model, pre_vol, n_timesteps, betas, alphas,
                                                   alphas_bar_sqrt, one_minus_alphas_bar_sqrt, device)
            residual_pred = residual_pred * residual_scale
            pred_vol = torch.clamp(pre_vol + residual_pred, 0, 1)
            
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
    print("3D RESIDUAL DIFFUSION FOR MOYAMOYA CVR PREDICTION")
    print("="*70)
    
    CONFIG = {
        'target_size': (96, 96, 48),  # Reduced from 128x128x64 to fit in memory
        'batch_size': 1,
        'lr': 1e-4,
        'epochs': 50,
        'early_stop_patience': 10,
        'n_timesteps': 1000,
        'residual_scale': 0.2,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'seed': int(os.environ.get('SEED', 42)),
    }
    
    print(f"\nConfiguration:")
    for k, v in CONFIG.items():
        print(f"  {k}: {v}")
    
    random.seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])
    torch.manual_seed(CONFIG['seed'])
    
    # Beta schedule
    betas = make_beta_schedule(CONFIG['n_timesteps'])
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
    use_week7 = os.environ.get('WEEK7', '').lower() in ('1', 'true', 'yes') or '--week7' in sys.argv
    pad_3d = None
    ckpt_name = 'residual_diffusion_3d_best.pt'
    results_name = 'residual_diffusion_3d_results.json'
    if use_week7:
        sys.path.insert(0, os.path.join(_REPO_ROOT, 'scripts'))
        from week7_data import get_week7_splits, Week7VolumePairs3D
        from week7_preprocess import TARGET_SHAPE
        train_pairs, val_pairs, test_pairs = get_week7_splits()
        CONFIG['target_size'] = PAD_3D_WEEK7
        pad_3d = PAD_3D_WEEK7
        use_phase2 = os.environ.get('WEEK7_REGION_WEIGHT', '').lower() in ('1', 'true', 'yes')
        ckpt_name = 'residual_diffusion_3d_week7_best.pt'
        results_name = 'residual_diffusion_3d_week7_phase2_results.json' if use_phase2 else 'residual_diffusion_3d_week7_results.json'
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
    
    # Model (use simple no-attention model for Week7 to avoid OOM)
    print(f"\n🏗️  Creating 3D Residual Diffusion Model...")
    if use_week7:
        model = SimpleResidualDiffusion3D(ch=16).to(CONFIG['device'])
        print(f"  Using SimpleResidualDiffusion3D (no attention) for Week7 to avoid OOM")
    else:
        model = ResidualDiffusionNet3D().to(CONFIG['device'])
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['lr'])
    
    eval_only = '--eval-only' in sys.argv or '--eval_only' in sys.argv
    if eval_only:
        print(f"\n📂 Eval-only: loading {ckpt_name} and running test evaluation...")
        if not os.path.isfile(ckpt_name):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_name}. Train first or run without --eval-only.")
        model.load_state_dict(torch.load(ckpt_name, map_location=CONFIG['device']))
        test_results = evaluate_model(model, test_loader, CONFIG['n_timesteps'], betas, alphas,
                                       alphas_bar_sqrt, one_minus_alphas_bar_sqrt,
                                       CONFIG['residual_scale'], CONFIG['device'], pad_3d=pad_3d)
        print(f"\nTest Results:")
        print(f"  MAE:  {test_results['mae_mean']:.4f} ± {test_results['mae_std']:.4f}")
        print(f"  SSIM: {test_results['ssim_mean']:.4f} ± {test_results['ssim_std']:.4f}")
        print(f"  PSNR: {test_results['psnr_mean']:.2f} ± {test_results['psnr_std']:.2f} dB")
        results = {'config': CONFIG, 'test_results': test_results}
        with open(results_name, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✅ Eval complete! Saved: {results_name}")
        print("="*70)
        return
    
    # Training
    print(f"\n🚀 Training...")
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(1, CONFIG['epochs'] + 1):
        start_time = time.time()
        
        train_loss = train_one_epoch(model, train_loader, optimizer, CONFIG['n_timesteps'],
                                      alphas_bar_sqrt, one_minus_alphas_bar_sqrt,
                                      CONFIG['residual_scale'], CONFIG['device'], pad_3d=pad_3d)
        
        # Validation (sample subset)
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
                residual = (post_vol - pre_vol) / CONFIG['residual_scale']
                B = residual.shape[0]
                t_batch = torch.randint(0, CONFIG['n_timesteps'], (B,), dtype=torch.long, device=CONFIG['device'])
                residual_t, noise = q_sample_residual(residual, t_batch, alphas_bar_sqrt, one_minus_alphas_bar_sqrt)
                pred_noise = model(residual_t, pre_vol, t_batch)
                val_loss_sum += criterion(pred_noise, noise).item()
                val_samples += 1
        
        val_loss = val_loss_sum / max(val_samples, 1)
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
    test_results = evaluate_model(model, test_loader, CONFIG['n_timesteps'], betas, alphas,
                                   alphas_bar_sqrt, one_minus_alphas_bar_sqrt,
                                   CONFIG['residual_scale'], CONFIG['device'], pad_3d=pad_3d)
    
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
