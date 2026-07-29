#!/usr/bin/env python3
"""
Train Cold Diffusion in latent space with Week 7 pipeline.
- Data: get_week7_splits(), load_volume_week7 (91×109×91 brain mask, pad 96×112×96).
- Step 1: Train VAE on post volumes (skip if vae_3d_week7_best.pt exists).
- Step 2: Train cold diffusion in latent (post_latent noised, conditioned on pre_latent + t).
Saves: vae_3d_week7_best.pt, cold_diffusion_latent_week7_best.pt, cold_diffusion_latent_week7_ema_best.pt.
"""
import os
import sys
import json
import random
import time

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_REPO_ROOT, "pyproject.toml")):
    _parent = os.path.dirname(_REPO_ROOT)
    if _parent == _REPO_ROOT:
        raise RuntimeError("Could not locate repository root (pyproject.toml not found)")
    _REPO_ROOT = _parent

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.join(_REPO_ROOT, 'scripts'))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'Diffusion_3D_Latent'))
from week7_data import get_week7_splits
from week7_loader import load_volume_week7
from vae_3d import VAE3D, vae_loss
from utils import EMA, strict_normalize_volume
from cold_diffusion_latent import make_cosine_schedule, evaluate_model

from monai.networks.nets import UNet


# (96, 128, 96) so latent (24, 32, 24) is divisible by 8 for MONAI UNet strides
TARGET_SIZE = (96, 128, 96)
LATENT_CHANNELS = 4
VAE_CKPT = 'vae_3d_week7_best.pt'
DIFF_CKPT = 'cold_diffusion_latent_week7_best.pt'
DIFF_EMA_CKPT = 'cold_diffusion_latent_week7_ema_best.pt'
RESULTS_JSON = 'cold_diffusion_latent_week7_phase2_results.json' if os.environ.get('WEEK7_REGION_WEIGHT', '').lower() in ('1', 'true', 'yes') else 'cold_diffusion_latent_week7_results.json'


class Week7VolumePairs(Dataset):
    """(pre_path, post_path) -> load with load_volume_week7, return (pre_vol, post_vol) normalized."""
    def __init__(self, pairs, pad_shape=(96, 112, 96)):
        self.pairs = [(p, q) for p, q in pairs if os.path.isfile(p) and os.path.isfile(q)]
        self.pad_shape = pad_shape

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        pre_p, post_p = self.pairs[i]
        pre_vol = strict_normalize_volume(load_volume_week7(pre_p, pad_shape=self.pad_shape))
        post_vol = strict_normalize_volume(load_volume_week7(post_p, pad_shape=self.pad_shape))
        return (
            torch.from_numpy(pre_vol).float().unsqueeze(0),
            torch.from_numpy(post_vol).float().unsqueeze(0),
        )


class LatentDiffusionDataset(Dataset):
    """Pre-compute (pre_latent, post_latent) for diffusion training."""
    def __init__(self, pairs, vae_model, pad_shape, device):
        self.pad_shape = pad_shape
        self.vae_model = vae_model
        self.vae_model.eval()
        self.latent_pairs = []
        with torch.no_grad():
            for pre_p, post_p in pairs:
                if not (os.path.isfile(pre_p) and os.path.isfile(post_p)):
                    continue
                pre_vol = strict_normalize_volume(load_volume_week7(pre_p, pad_shape=pad_shape))
                post_vol = strict_normalize_volume(load_volume_week7(post_p, pad_shape=pad_shape))
                pre_t = torch.from_numpy(pre_vol).float().unsqueeze(0).unsqueeze(0).to(device)
                post_t = torch.from_numpy(post_vol).float().unsqueeze(0).unsqueeze(0).to(device)
                pre_latent = vae_model.encode_to_latent(pre_t)
                post_latent = vae_model.encode_to_latent(post_t)
                self.latent_pairs.append((pre_latent.cpu(), post_latent.cpu()))
        print(f"   Encoded {len(self.latent_pairs)} volume pairs to latent")

    def __len__(self):
        return len(self.latent_pairs)

    def __getitem__(self, i):
        # Squeeze(0) so batch collate gives (B, C, D, H, W) not (B, 1, C, D, H, W)
        return self.latent_pairs[i][0].squeeze(0), self.latent_pairs[i][1].squeeze(0)


def q_sample_latent(post_latent, t, alpha_bar_sqrt, one_minus_alpha_bar_sqrt, device):
    """Noise post_latent at step t: x_t = sqrt(alpha_bar_t)*post + sqrt(1-alpha_bar_t)*noise. Returns (x_t, noise)."""
    B = post_latent.shape[0]
    noise = torch.randn_like(post_latent, device=device)
    sqrt_ab = alpha_bar_sqrt[t].view(-1, 1, 1, 1, 1)
    sqrt_1_ab = one_minus_alpha_bar_sqrt[t].view(-1, 1, 1, 1, 1)
    x_t = sqrt_ab * post_latent + sqrt_1_ab * noise
    return x_t, noise


def train_vae_epoch(model, loader, optimizer, device, kl_weight=0.0001):
    model.train()
    total_loss = 0.0
    for pre_vol, post_vol in loader:
        post_vol = post_vol.to(device)
        optimizer.zero_grad()
        recon, mu, logvar = model(post_vol)
        loss_dict = vae_loss(recon, post_vol, mu, logvar, recon_weight=1.0, kl_weight=kl_weight)
        loss_dict['total'].backward()
        optimizer.step()
        total_loss += loss_dict['total'].item()
    return total_loss / len(loader)


def train_diffusion_epoch(model, loader, optimizer, n_timesteps, alpha_bar_sqrt, one_minus_alpha_bar_sqrt, device, ema=None):
    model.train()
    total_loss = 0.0
    criterion = nn.MSELoss()
    for pre_latent, post_latent in loader:
        pre_latent = pre_latent.to(device)
        post_latent = post_latent.to(device)
        pre_latent = torch.clamp(pre_latent, -5.0, 5.0)
        post_latent = torch.clamp(post_latent, -5.0, 5.0)
        B = post_latent.shape[0]
        t_batch = torch.randint(0, n_timesteps, (B,), dtype=torch.long, device=device)
        x_t, noise = q_sample_latent(post_latent, t_batch, alpha_bar_sqrt, one_minus_alpha_bar_sqrt, device)
        x_t = torch.clamp(x_t, -10.0, 10.0)
        time_emb = (t_batch.float() / max(n_timesteps - 1, 1)).view(-1, 1, 1, 1, 1).expand(-1, 1, *x_t.shape[2:])
        model_in = torch.cat([x_t, pre_latent, time_emb], dim=1)
        pred_noise = model(model_in)  # MONAI UNet takes only x; time is in model_in
        loss = criterion(pred_noise, noise)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if ema is not None:
            ema.update()
        total_loss += loss.item()
    return total_loss / len(loader)


def main():
    print("="*70)
    print("COLD DIFFUSION LATENT - WEEK 7 TRAINING")
    print("="*70)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    train_pairs, val_pairs, test_pairs = get_week7_splits()
    print(f"  Week7: {len(train_pairs)} train / {len(val_pairs)} val / {len(test_pairs)} test")

    # ---------- Step 1: VAE ----------
    vae = VAE3D(
        in_channels=1,
        latent_channels=LATENT_CHANNELS,
        channels=(32, 64, 128, 256),
        num_res_blocks=2,
        downsample_factor=4,
    ).to(device)

    if os.path.exists(VAE_CKPT):
        print(f"\n  Loading existing VAE from {VAE_CKPT} (skip VAE training)")
        vae.load_state_dict(torch.load(VAE_CKPT, map_location=device))
    else:
        print("\n  Training VAE on post volumes...")
        train_ds = Week7VolumePairs(train_pairs, pad_shape=TARGET_SIZE)
        train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=0)
        opt_vae = torch.optim.Adam(vae.parameters(), lr=1e-4)
        best_val_loss = float('inf')
        for epoch in range(1, 51):
            train_loss = train_vae_epoch(vae, train_loader, opt_vae, device, kl_weight=0.0001)
            print(f"  VAE Epoch {epoch}/50 - Train Loss: {train_loss:.4f}")
            if train_loss < best_val_loss:
                best_val_loss = train_loss
                torch.save(vae.state_dict(), VAE_CKPT)
        vae.load_state_dict(torch.load(VAE_CKPT, map_location=device))
    vae.eval()

    # ---------- Step 2: Cold diffusion ----------
    diff_model = UNet(
        spatial_dims=3,
        in_channels=LATENT_CHANNELS * 2 + 1,
        out_channels=LATENT_CHANNELS,
        channels=(32, 64, 128, 256),
        strides=(2, 2, 2),
        num_res_units=2,
    ).to(device)

    n_timesteps = 200
    alpha_bar = make_cosine_schedule(n_timesteps)
    alpha_bar = 1.0 - alpha_bar  # for noising: x_t = sqrt(alpha_bar)*x0 + sqrt(1-alpha_bar)*noise
    alpha_bar_sqrt = torch.from_numpy(np.sqrt(alpha_bar)).float().to(device)
    one_minus_alpha_bar_sqrt = torch.from_numpy(np.sqrt(1.0 - alpha_bar)).float().to(device)

    train_lds = LatentDiffusionDataset(train_pairs, vae, TARGET_SIZE, device)
    val_lds = LatentDiffusionDataset(val_pairs, vae, TARGET_SIZE, device)
    train_loader_d = DataLoader(train_lds, batch_size=4, shuffle=True, num_workers=0)
    val_loader_d = DataLoader(val_lds, batch_size=4, shuffle=False, num_workers=0)

    opt_diff = torch.optim.AdamW(diff_model.parameters(), lr=5e-4)
    ema = EMA(diff_model, decay=0.9999)
    best_val = float('inf')
    patience = 0

    print("\n  Training cold diffusion in latent...")
    for epoch in range(1, 101):
        t0 = time.time()
        train_loss = train_diffusion_epoch(
            diff_model, train_loader_d, opt_diff, n_timesteps,
            alpha_bar_sqrt, one_minus_alpha_bar_sqrt, device, ema
        )
        diff_model.eval()
        val_loss = 0.0
        criterion = nn.MSELoss()
        with torch.no_grad():
            for pre_latent, post_latent in val_loader_d:
                pre_latent = pre_latent.to(device)
                post_latent = post_latent.to(device)
                pre_latent = torch.clamp(pre_latent, -5.0, 5.0)
                post_latent = torch.clamp(post_latent, -5.0, 5.0)
                B = post_latent.shape[0]
                t_batch = torch.randint(0, n_timesteps, (B,), dtype=torch.long, device=device)
                x_t, noise = q_sample_latent(post_latent, t_batch, alpha_bar_sqrt, one_minus_alpha_bar_sqrt, device)
                x_t = torch.clamp(x_t, -10.0, 10.0)
                time_emb = (t_batch.float() / max(n_timesteps - 1, 1)).view(-1, 1, 1, 1, 1).expand(-1, 1, *x_t.shape[2:])
                model_in = torch.cat([x_t, pre_latent, time_emb], dim=1)
                pred_noise = diff_model(model_in)
                val_loss += criterion(pred_noise, noise).item()
        val_loss /= max(len(val_loader_d), 1)
        elapsed = time.time() - t0
        if val_loss < best_val:
            best_val = val_loss
            patience = 0
            torch.save(diff_model.state_dict(), DIFF_CKPT)
            ema.apply_shadow()
            torch.save(diff_model.state_dict(), DIFF_EMA_CKPT)
            ema.restore()
            print(f"  Epoch {epoch}/100 - Train: {train_loss:.4f} - Val: {val_loss:.4f} - Time: {elapsed:.1f}s - Saved")
        else:
            patience += 1
            print(f"  Epoch {epoch}/100 - Train: {train_loss:.4f} - Val: {val_loss:.4f} - Time: {elapsed:.1f}s")
        if patience >= 15:
            print("  Early stopping")
            break

    # Load best and run evaluation
    diff_model.load_state_dict(torch.load(DIFF_EMA_CKPT, map_location=device))
    from cold_diffusion_latent import evaluate_model
    test_items = [(p, q) for p, q in test_pairs if os.path.isfile(p) and os.path.isfile(q)]
    load_fn = lambda p: load_volume_week7(p, pad_shape=TARGET_SIZE)  # (96,128,96) -> latent (24,32,24)
    alpha_np = make_cosine_schedule(n_timesteps)
    results = evaluate_model(
        diff_model, vae, test_items, n_timesteps, 25, TARGET_SIZE, device, load_fn=load_fn
    )
    with open(RESULTS_JSON, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results: MAE {results['mae_mean']:.4f}  SSIM {results['ssim_mean']:.4f}  PSNR {results['psnr_mean']:.2f}")
    print(f"  Saved: {VAE_CKPT}, {DIFF_CKPT}, {DIFF_EMA_CKPT}, {RESULTS_JSON}")
    print("="*70)


if __name__ == '__main__':
    main()
