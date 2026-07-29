#!/usr/bin/env python3
"""
3D UNet Baseline for Moyamoya CVR Prediction
============================================
Full volumetric (3D) version of the winning model from the paper.

Key Differences from 2D:
- spatial_dims=3 in UNet
- Input shape: (B, 1, H, W, D) instead of (B, 1, H, W)
- Loads full volumes instead of middle slices
- Uses scipy.ndimage.zoom for 3D resizing
- SSIMLoss with spatial_dims=3

Author: Verified implementation based on GUIDE_3D_IMPLEMENTATION.txt
Date: 2026-01-22
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
# 1) Helper functions
###############################################################################

def load_full_volume(nii_path: str, target_size=(128, 128, 64)) -> np.ndarray:
    """
    Load a NIfTI file and resize to target_size.
    
    Args:
        nii_path: Path to .nii.gz file
        target_size: Desired (H, W, D) size
    
    Returns:
        3D numpy array of shape target_size
    """
    img = nib.load(nii_path)
    data = img.get_fdata(dtype=np.float32)
    
    # Handle 4D data (e.g., time series)
    if data.ndim == 4:
        data = data[..., 0]
    
    # Compute zoom factors
    zoom_factors = [target_size[i] / data.shape[i] for i in range(3)]
    
    # Resize with trilinear interpolation (order=1)
    resized_data = zoom(data, zoom_factors, order=1)
    
    return resized_data.astype(np.float32)


def minmax_norm(x: np.ndarray) -> np.ndarray:
    """Min-max normalization to [0,1]."""
    mn, mx = x.min(), x.max()
    if (mx - mn) > 1e-6:
        return (x - mn) / (mx - mn)
    else:
        return np.zeros_like(x)


def pre_to_post_path(pre_path: str) -> str:
    """Convert pre-scan path to post-scan path."""
    basename = os.path.basename(pre_path).replace('pre_', 'post_')
    dirname = os.path.dirname(pre_path).replace('/pre', '/post')
    if dirname == '':
        dirname = 'post'
    return os.path.join(dirname, basename)


###############################################################################
# 2) Dataset
###############################################################################

class FullVolumePairs(Dataset):
    """
    Dataset for loading paired 3D volumes (pre and post scans).
    
    Returns:
        pre_t: (1, H, W, D) tensor, normalized to [0,1]
        post_t: (1, H, W, D) tensor, normalized to [0,1]
    """
    
    def __init__(self, pre_paths: List[str], target_size=(128, 128, 64)):
        self.target_size = target_size
        self.items = []
        
        for p in pre_paths:
            post_p = pre_to_post_path(p)
            if os.path.exists(post_p):
                self.items.append((p, post_p))
        
        print(f"  → Loaded {len(self.items)} paired volumes")
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        pre_p, post_p = self.items[idx]
        
        pre_vol = minmax_norm(load_full_volume(pre_p, self.target_size))
        post_vol = minmax_norm(load_full_volume(post_p, self.target_size))
        
        pre_t = torch.from_numpy(pre_vol).unsqueeze(0).float()   # (1, H, W, D)
        post_t = torch.from_numpy(post_vol).unsqueeze(0).float() # (1, H, W, D)
        
        return pre_t, post_t


class PairsFromPaths(Dataset):
    """Dataset from explicit (pre_path, post_path) pairs (e.g. 2020 split JSON)."""
    def __init__(self, pairs: List[tuple], target_size=(128, 128, 64)):
        self.target_size = target_size
        self.items = list(pairs)
        print(f"  → Loaded {len(self.items)} paired volumes (from paths)")

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
# 3) Model
###############################################################################

def make_unet_3d(channels=(16, 32, 64, 128)) -> nn.Module:
    """
    Create 3D UNet using MONAI.
    
    Architecture:
    - spatial_dims=3 (KEY for 3D convolutions)
    - 1 input channel (pre-scan)
    - 1 output channel (predicted post-scan)
    - Instance normalization
    - LeakyReLU activation
    """
    model = UNet(
        spatial_dims=3,  # 3D convolutions
        in_channels=1,
        out_channels=1,
        channels=list(channels),
        strides=(2, 2, 2),
        num_res_units=2,
        act=("LeakyReLU", {"inplace": True}),
        norm="INSTANCE",
        dropout=0.0,
    )
    return model


###############################################################################
# 4) Training
###############################################################################

def _pad_3d_if_needed(pre_vol, post_vol, target_size):
    """Pad (B,1,H,W,D) to target_size for UNet divisibility (e.g. Week7 91,109,91 -> 96,112,96)."""
    if target_size is None or pre_vol.shape[2:] == target_size:
        return pre_vol, post_vol
    import torch.nn.functional as F
    _, _, h, w, d = pre_vol.shape
    th, tw, td = target_size
    if h < th or w < tw or d < td:
        pd = (0, max(0, td - d), 0, max(0, tw - w), 0, max(0, th - h))
        pre_vol = F.pad(pre_vol, pd, mode='constant', value=0)
        post_vol = F.pad(post_vol, pd, mode='constant', value=0)
    return pre_vol[:, :, :th, :tw, :td], post_vol[:, :, :th, :tw, :td]


def _pad_3d_mask(mask_vol, target_size):
    """Pad mask (B,1,H,W,D) to target_size to match pre/post."""
    import torch.nn.functional as F
    if mask_vol.shape[2:] == target_size:
        return mask_vol
    _, _, h, w, d = mask_vol.shape
    th, tw, td = target_size
    if h < th or w < tw or d < td:
        pd = (0, max(0, td - d), 0, max(0, tw - w), 0, max(0, th - h))
        mask_vol = F.pad(mask_vol, pd, mode='constant', value=0)
    return mask_vol[:, :, :th, :tw, :td]


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion_l1: nn.Module,
    criterion_ssim: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    pad_target_3d=None,
    static_region_mask_t=None,
) -> float:
    """Train for one epoch. If loader yields (pre, post, mask), use batch mask; else if static_region_mask_t set, use it; else standard L1+SSIM."""
    model.train()
    epoch_loss = 0.0

    for batch in loader:
        if len(batch) == 3:
            pre_vol, post_vol, mask_vol = batch
            mask_vol = mask_vol.to(device)
            if pad_target_3d is not None:
                mask_vol = _pad_3d_mask(mask_vol, pad_target_3d)
        else:
            pre_vol, post_vol = batch[0], batch[1]
            mask_vol = None
        if pad_target_3d is not None:
            pre_vol, post_vol = _pad_3d_if_needed(pre_vol, post_vol, pad_target_3d)
        pre_vol = pre_vol.to(device)
        post_vol = post_vol.to(device)

        mask_batch = mask_vol if mask_vol is not None else static_region_mask_t
        if static_region_mask_t is not None and mask_vol is None:
            mask_batch = static_region_mask_t.expand(pre_vol.size(0), -1, -1, -1, -1)

        optimizer.zero_grad()
        output = model(pre_vol)

        if mask_batch is not None:
            loss_l1 = (torch.abs(output - post_vol) * mask_batch).sum() / (mask_batch.sum() + 1e-8)
        else:
            loss_l1 = criterion_l1(output, post_vol)
        loss_ssim = criterion_ssim(output, post_vol)
        loss = loss_l1 + loss_ssim

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(loader)


###############################################################################
# 5) Evaluation
###############################################################################

def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    pad_target_3d=None,
    use_week7_brain_only: bool = False,
) -> dict:
    """
    Evaluate model on validation/test set.
    If use_week7_brain_only, MAE/SSIM/PSNR are computed only inside brain mask.
    Returns:
        Dictionary with MAE, SSIM, PSNR (mean ± std)
    """
    model.eval()
    if use_week7_brain_only:
        import sys
        sys.path.insert(0, os.path.join(_REPO_ROOT, "scripts"))
        from week7_preprocess import metrics_in_brain

    mae_list = []
    ssim_list = []
    psnr_list = []

    with torch.no_grad():
        for batch in loader:
            pre_vol, post_vol = batch[0], batch[1]
            if pad_target_3d is not None:
                pre_vol, post_vol = _pad_3d_if_needed(pre_vol, post_vol, pad_target_3d)
            pre_vol = pre_vol.to(device)
            post_vol = post_vol.to(device)

            pred_vol = model(pre_vol)

            # Convert to numpy for metric calculation
            pred_np = pred_vol.cpu().numpy()
            post_np = post_vol.cpu().numpy()

            batch_size = pred_np.shape[0]
            for i in range(batch_size):
                pred_i = pred_np[i, 0]  # (H, W, D)
                post_i = post_np[i, 0]

                if use_week7_brain_only:
                    m = metrics_in_brain(pred_i, post_i, data_range=1.0)
                    mae_list.append(m["mae_mean"])
                    ssim_list.append(m["ssim_mean"])
                    psnr_list.append(m["psnr_mean"])
                else:
                    mae = np.abs(pred_i - post_i).mean()
                    mae_list.append(mae)
                    ssim_val = ssim(post_i, pred_i, data_range=1.0)
                    ssim_list.append(ssim_val)
                    psnr_val = psnr(post_i, pred_i, data_range=1.0)
                    psnr_list.append(psnr_val)

    return {
        'mae_mean': float(np.mean(mae_list)),
        'mae_std': float(np.std(mae_list)),
        'ssim_mean': float(np.mean(ssim_list)),
        'ssim_std': float(np.std(ssim_list)),
        'psnr_mean': float(np.mean(psnr_list)),
        'psnr_std': float(np.std(psnr_list)),
    }


###############################################################################
# 6) Main
###############################################################################

def main():
    """
    Full training pipeline for 3D UNet.
    
    Steps:
    1. Load all paired volumes
    2. Split into train/val/test
    3. Train model
    4. Evaluate and save results
    """
    print("="*70)
    print("3D UNET BASELINE FOR MOYAMOYA CVR PREDICTION")
    print("="*70)
    
    # Configuration
    CONFIG = {
        'target_size': (128, 128, 64),
        'batch_size': 2,
        'lr': 1e-3,
        'epochs': int(os.environ.get('WEEK9_QUICK_EPOCHS', 50)),
        'early_stop_patience': 10,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'seed': int(os.environ.get('SEED', 42)),
        'use_2020': False,
        'split_2020_json': '<repo-root>/2020_single_delay_split.json',
        'combined_split_json': '<repo-root>/combined_subject_split.json',
        'use_week7': (
            '--week7' in sys.argv
            or is_env_flag('WEEK7')
        ),
    }
    
    print(f"\nConfiguration:")
    for k, v in CONFIG.items():
        print(f"  {k}: {v}")
    
    # Set seed
    random.seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])
    torch.manual_seed(CONFIG['seed'])
    
    # Load data
    print(f"\n📂 Loading data...")
    # Per-fold K-fold: force Week7 pipeline so get_week7_splits() uses WEEK7_SPLIT_PATH (not combined JSON only).
    _kf_split = os.environ.get("WEEK7_SPLIT_PATH", "").strip()
    if _kf_split and os.path.isfile(_kf_split):
        CONFIG['use_week7'] = True

    split_json = None
    use_pairs = False
    ckpt_name = 'unet_3d_best.pt'
    results_name = 'unet_3d_results.json'
    use_phase2_3d = False
    static_region_mask_t = None

    if CONFIG.get('use_week7'):
        # Week7 standard: 91x109x91, brain mask; split from get_combined_split_path() (K-fold or combined)
        sys.path.insert(0, os.path.join(_REPO_ROOT, 'scripts'))
        from week7_data import get_week7_splits, Week7VolumePairs3D, Week7VolumePairs3DWithMasks
        from week7_preprocess import TARGET_SHAPE, get_region_weight_mask_for_shape, get_combined_split_path, week7_kfold_suffix_paths, is_env_flag
        print(f"  Split JSON: {get_combined_split_path()}")
        train_pairs, val_pairs, test_pairs = get_week7_splits()
        CONFIG['target_size'] = (96, 112, 96)  # pad for UNet divisibility
        use_region_weight = is_env_flag('WEEK7_REGION_WEIGHT')
        use_subject_masks = is_env_flag('WEEK7_SUBJECT_MASKS')
        use_phase2_3d = use_region_weight or use_subject_masks
        print(f"  Week7: 91x109x91 + brain mask, combined 2020-2023: {len(train_pairs)} train / {len(val_pairs)} val / {len(test_pairs)} test")
        if use_subject_masks:
            print(f"  Phase 3: subject-specific masks")
            train_dataset = Week7VolumePairs3DWithMasks(train_pairs, augment=True, target_shape=TARGET_SHAPE, pad_shape=CONFIG['target_size'])
            val_dataset = Week7VolumePairs3DWithMasks(val_pairs, augment=False, target_shape=TARGET_SHAPE, pad_shape=CONFIG['target_size'])
            test_dataset = Week7VolumePairs3DWithMasks(test_pairs, augment=False, target_shape=TARGET_SHAPE, pad_shape=CONFIG['target_size'])
        else:
            print(f"\n📊 Creating datasets (Week7VolumePairs3D)...")
            train_dataset = Week7VolumePairs3D(train_pairs, augment=True, target_shape=TARGET_SHAPE)
            val_dataset = Week7VolumePairs3D(val_pairs, augment=False, target_shape=TARGET_SHAPE)
            test_dataset = Week7VolumePairs3D(test_pairs, augment=False, target_shape=TARGET_SHAPE)
        if use_region_weight and not use_subject_masks:
            mask_np = get_region_weight_mask_for_shape(CONFIG['target_size'], vascular_weight=1.5)
            static_region_mask_t = torch.from_numpy(mask_np).float().to(CONFIG['device']).unsqueeze(0).unsqueeze(0)
        ckpt_name = 'unet_3d_week7_best.pt'
        results_name = 'unet_3d_results_week7_phase2.json' if use_phase2_3d else 'unet_3d_results_week7.json'
        ckpt_name, results_name = week7_kfold_suffix_paths(ckpt_name, results_name)
    elif CONFIG.get('combined_split_json') and os.path.isfile(CONFIG['combined_split_json']):
        split_json = CONFIG['combined_split_json']
        ckpt_name = 'unet_3d_combined_best.pt'
        results_name = 'unet_3d_results_combined.json'
        with open(split_json) as f:
            data = json.load(f)
        train_pairs = [(x['pre_path'], x['post_path']) for x in data['train']]
        val_pairs = [(x['pre_path'], x['post_path']) for x in data['val']]
        test_pairs = [(x['pre_path'], x['post_path']) for x in data['test']]
        print(f"  From {split_json}: {len(train_pairs)} train / {len(val_pairs)} val / {len(test_pairs)} test")
        print(f"\n📊 Creating datasets...")
        train_dataset = PairsFromPaths(train_pairs, CONFIG['target_size'])
        val_dataset = PairsFromPaths(val_pairs, CONFIG['target_size'])
        test_dataset = PairsFromPaths(test_pairs, CONFIG['target_size'])
    elif CONFIG.get('use_2020') and os.path.isfile(CONFIG.get('split_2020_json', '')):
        split_json = CONFIG['split_2020_json']
        ckpt_name = 'unet_3d_2020_best.pt'
        results_name = 'unet_3d_results_2020.json'
        with open(split_json) as f:
            data = json.load(f)
        train_pairs = [(x['pre_path'], x['post_path']) for x in data['train']]
        val_pairs = [(x['pre_path'], x['post_path']) for x in data['val']]
        test_pairs = [(x['pre_path'], x['post_path']) for x in data['test']]
        print(f"  From {split_json}: {len(train_pairs)} train / {len(val_pairs)} val / {len(test_pairs)} test")
        print(f"\n📊 Creating datasets...")
        train_dataset = PairsFromPaths(train_pairs, CONFIG['target_size'])
        val_dataset = PairsFromPaths(val_pairs, CONFIG['target_size'])
        test_dataset = PairsFromPaths(test_pairs, CONFIG['target_size'])
    else:
        data_dir = _REPO_ROOT
        all_pre = sorted(glob.glob(f"{data_dir}/pre/pre_*.nii.gz"))
        print(f"  Found {len(all_pre)} pre-scans")
        all_pre_paired = [p for p in all_pre if os.path.exists(pre_to_post_path(p))]
        print(f"  Found {len(all_pre_paired)} with matching post-scans")
        random.shuffle(all_pre_paired)
        n = len(all_pre_paired)
        n_train = int(0.75 * n)
        n_val = int(0.125 * n)
        train_pre = all_pre_paired[:n_train]
        val_pre = all_pre_paired[n_train:n_train+n_val]
        test_pre = all_pre_paired[n_train+n_val:]
        print(f"\nSplit: {len(train_pre)} train / {len(val_pre)} val / {len(test_pre)} test")
        print(f"\n📊 Creating datasets...")
        train_dataset = FullVolumePairs(train_pre, CONFIG['target_size'])
        val_dataset = FullVolumePairs(val_pre, CONFIG['target_size'])
        test_dataset = FullVolumePairs(test_pre, CONFIG['target_size'])
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=2)
    
    # Create model
    print(f"\n🏗️  Creating 3D UNet...")
    model = make_unet_3d().to(CONFIG['device'])
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    
    # Loss and optimizer
    criterion_l1 = nn.L1Loss()
    criterion_ssim = SSIMLoss(spatial_dims=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['lr'])
    
    # Training
    print(f"\n🚀 Training...")
    best_val_loss = float('inf')
    patience_counter = 0
    
    pad_target = CONFIG['target_size'] if CONFIG.get('use_week7') else None
    for epoch in range(1, CONFIG['epochs'] + 1):
        start_time = time.time()
        
        train_loss = train_one_epoch(
            model, train_loader, criterion_l1, criterion_ssim, optimizer, CONFIG['device'],
            pad_target_3d=pad_target,
            static_region_mask_t=static_region_mask_t if CONFIG.get('use_week7') else None,
        )
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                pre_vol, post_vol = batch[0], batch[1]
                if pad_target is not None:
                    pre_vol, post_vol = _pad_3d_if_needed(pre_vol, post_vol, pad_target)
                pre_vol = pre_vol.to(CONFIG['device'])
                post_vol = post_vol.to(CONFIG['device'])
                output = model(pre_vol)
                loss = criterion_l1(output, post_vol) + criterion_ssim(output, post_vol)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        
        epoch_time = time.time() - start_time
        
        print(f"Epoch {epoch:3d}/{CONFIG['epochs']} - Train: {train_loss:.4f} - Val: {val_loss:.4f} - Time: {epoch_time:.1f}s")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), ckpt_name)
            print(f"  ✓ Saved new best model (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= CONFIG['early_stop_patience']:
                print(f"\n⚠️  Early stopping triggered (patience={CONFIG['early_stop_patience']})")
                break
    
    # Final evaluation
    print(f"\n📊 Final Evaluation on Test Set...")
    model.load_state_dict(torch.load(ckpt_name, map_location=CONFIG['device']))
    test_results = evaluate(model, test_loader, CONFIG['device'], pad_target_3d=pad_target, use_week7_brain_only=CONFIG['use_week7'])
    
    print(f"\nTest Results:")
    print(f"  MAE:  {test_results['mae_mean']:.4f} ± {test_results['mae_std']:.4f}")
    print(f"  SSIM: {test_results['ssim_mean']:.4f} ± {test_results['ssim_std']:.4f}")
    print(f"  PSNR: {test_results['psnr_mean']:.2f} ± {test_results['psnr_std']:.2f} dB")
    
    # Save results
    results = {
        'config': CONFIG,
        'test_results': test_results,
        'n_train': len(train_dataset),
        'n_val': len(val_dataset),
        'n_test': len(test_dataset),
    }
    
    with open(results_name, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Training complete!")
    print(f"  Model saved: {ckpt_name}")
    print(f"  Results saved: {results_name}")
    print("="*70)


if __name__ == "__main__":
    main()
