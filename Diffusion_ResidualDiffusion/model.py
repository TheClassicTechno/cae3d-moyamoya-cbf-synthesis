#!/usr/bin/env python3
"""
Residual Diffusion for Pre-to-Post MRI Translation

Instead of trying to generate the full post image, this approach models
only the RESIDUAL (difference) between pre and post scans:

    residual = post - pre
    
The diffusion model learns to generate this residual conditioned on pre,
then the final prediction is:
    
    predicted_post = pre + predicted_residual

Benefits:
1. Residual is smaller/sparser than full image -> easier to model
2. Model focuses on learning treatment effect, not reconstruction
3. Strong structural prior from pre image
4. Works well with limited data

The diffusion process adds noise to the residual, and the model learns
to denoise it conditioned on the pre image.
"""

import os
import sys
import re
import json
import random
from glob import glob
from typing import List, Tuple

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_REPO_ROOT, "pyproject.toml")):
    _parent = os.path.dirname(_REPO_ROOT)
    if _parent == _REPO_ROOT:
        raise RuntimeError("Could not locate repository root (pyproject.toml not found)")
    _REPO_ROOT = _parent

import numpy as np
import nibabel as nib
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

from monai.networks.nets import UNet
from monai.losses import SSIMLoss

from skimage.metrics import structural_similarity as sk_ssim
from skimage.metrics import peak_signal_noise_ratio as sk_psnr

import matplotlib.pyplot as plt

# ============================================================
# Logging
# ============================================================
LOG_F = None  # Will be opened in main()


def log(msg: str):
    """Print to stdout and also write to log file."""
    print(msg)
    if LOG_F is not None:
        LOG_F.write(msg + "\n")
        LOG_F.flush()


# ============================================================
# Config
# ============================================================
SEED = 1337
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Paths - script is in Diffusion_ResidualDiffusion/, data is one level up
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

PRE_DIR = os.path.join(ROOT_DIR, "pre_scans")
POST_DIR = os.path.join(ROOT_DIR, "post_scans")

RUN_DIR = os.path.join(BASE_DIR, "runs_residual_diffusion")
PRED_DIR = os.path.join(RUN_DIR, "pred_samples")
os.makedirs(PRED_DIR, exist_ok=True)

MODEL_PATH = os.path.join(RUN_DIR, "best_residual_diffusion.pt")
HISTORY_PATH = os.path.join(RUN_DIR, "training_history.json")

TARGET_SIZE = (128, 128)
BATCH_SIZE = 8
EPOCHS = 50
LR = 1e-3

# Diffusion params for residual
T = 100  # Number of diffusion steps (smaller than typical DDPM)
BETA_START = 1e-4
BETA_END = 2e-2

# Residual scaling - scale residuals to roughly [-1, 1] range
RESIDUAL_SCALE = 2.0  # Since images are in [0,1], residuals are in [-1,1]


# ============================================================
# Helpers: pairing & IO
# ============================================================
ID_RE = re.compile(r"(pre)_(\d{4})_(\d{3})\.nii\.gz$", re.IGNORECASE)


def id_from_pre_path(p: str) -> Tuple[str, str] | None:
    m = ID_RE.search(os.path.basename(p))
    if not m:
        return None
    return m.group(2), m.group(3)


def pre_to_post(pre_path: str) -> str | None:
    ids = id_from_pre_path(pre_path)
    if not ids:
        return None
    y, n = ids
    return os.path.join(POST_DIR, f"post_{y}_{n}.nii.gz")


def load_middle_axial_slice(nii_path: str) -> np.ndarray:
    img = nib.load(nii_path)
    data = img.get_fdata(dtype=np.float32)
    if data.ndim == 4:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"Unexpected shape {data.shape} for {nii_path}")
    H, W, D = data.shape
    return data[:, :, D // 2]


def minmax_norm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mn, mx = float(np.min(x)), float(np.max(x))
    if mx - mn < eps:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)


def to_tensor_resized(x2d: np.ndarray, target_hw=TARGET_SIZE) -> torch.Tensor:
    t = torch.from_numpy(x2d).unsqueeze(0).unsqueeze(0).float()
    t = F.interpolate(t, size=target_hw, mode="bilinear", align_corners=False)
    return t.squeeze(0)  # (1, H, W)


def ssim_np(a: np.ndarray, b: np.ndarray) -> float:
    dr = b.max() - b.min()
    if dr == 0:
        dr = 1.0
    return float(sk_ssim(a, b, data_range=dr))


def psnr_np(a: np.ndarray, b: np.ndarray) -> float:
    dr = b.max() - b.min()
    if dr == 0:
        dr = 1.0
    return float(sk_psnr(b, a, data_range=dr))


# ============================================================
# Dataset
# ============================================================
class AxialSlicePairs(Dataset):
    def __init__(self, pre_paths: List[str], target_size=TARGET_SIZE):
        self.items: List[Tuple[str, str]] = []
        self.target_size = target_size
        missing = 0
        for p in pre_paths:
            q = pre_to_post(p)
            if q and os.path.exists(q):
                self.items.append((p, q))
            else:
                missing += 1
        if not self.items:
            raise RuntimeError("No paired scans found.")
        if missing > 0:
            print(f"[WARN] Skipped {missing} pre scans with no matching post.")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        pre_p, post_p = self.items[idx]
        pre_sl = minmax_norm(load_middle_axial_slice(pre_p))
        post_sl = minmax_norm(load_middle_axial_slice(post_p))

        pre_t = to_tensor_resized(pre_sl, self.target_size)
        post_t = to_tensor_resized(post_sl, self.target_size)

        return pre_t.float(), post_t.float()


# ============================================================
# Diffusion Utilities for Residual
# ============================================================

# Linear beta schedule
betas = torch.linspace(BETA_START, BETA_END, T, dtype=torch.float32)
alphas = 1.0 - betas
alphas_bar = torch.cumprod(alphas, dim=0)

# Move to device later when needed
betas_cpu = betas
alphas_cpu = alphas
alphas_bar_cpu = alphas_bar


def get_schedules(device):
    """Get diffusion schedules on the correct device."""
    return (
        betas_cpu.to(device),
        alphas_cpu.to(device),
        alphas_bar_cpu.to(device),
    )


def q_sample_residual(residual: torch.Tensor, t: torch.Tensor, noise: torch.Tensor, alphas_bar: torch.Tensor) -> torch.Tensor:
    """
    Forward diffusion on residual: q(r_t | r_0).
    
    residual: (B, 1, H, W) - the clean residual (post - pre) scaled
    t: (B,) - timesteps
    noise: (B, 1, H, W) - random noise
    """
    a_bar = alphas_bar[t].view(-1, 1, 1, 1)
    return torch.sqrt(a_bar) * residual + torch.sqrt(1.0 - a_bar) * noise


def p_sample_step_residual(model, r_t: torch.Tensor, pre: torch.Tensor, t: torch.Tensor, 
                           betas: torch.Tensor, alphas: torch.Tensor, alphas_bar: torch.Tensor) -> torch.Tensor:
    """
    One reverse step: p(r_{t-1} | r_t, pre).
    
    The model predicts the noise epsilon, and we use DDPM formula to get r_{t-1}.
    """
    # Predict noise
    eps_pred = model(r_t, pre, t)

    a_t = alphas[t].view(-1, 1, 1, 1)
    a_bar_t = alphas_bar[t].view(-1, 1, 1, 1)
    
    # Handle t=0 case
    t_prev = torch.clamp(t - 1, min=0)
    a_bar_prev = alphas_bar[t_prev].view(-1, 1, 1, 1)
    a_bar_prev = torch.where(
        (t == 0).view(-1, 1, 1, 1),
        torch.ones_like(a_bar_prev),
        a_bar_prev
    )

    beta_t = betas[t].view(-1, 1, 1, 1)

    # DDPM mean formula
    sqrt_one_minus_a_bar = torch.sqrt(1.0 - a_bar_t)
    mean = (1.0 / torch.sqrt(a_t)) * (r_t - beta_t / sqrt_one_minus_a_bar * eps_pred)

    # Posterior variance
    var = beta_t * (1.0 - a_bar_prev) / (1.0 - a_bar_t + 1e-8)
    
    # Add noise except for last step
    noise = torch.randn_like(r_t)
    nonzero_mask = (t > 0).float().view(-1, 1, 1, 1)
    r_prev = mean + nonzero_mask * torch.sqrt(var) * noise
    
    return r_prev


def p_sample_loop_residual(model, pre: torch.Tensor) -> torch.Tensor:
    """
    Full reverse diffusion to generate residual from noise.
    
    Returns the predicted residual (scaled).
    """
    model.eval()
    B = pre.size(0)
    device = pre.device
    
    betas, alphas, alphas_bar = get_schedules(device)
    
    # Start from random noise
    r_t = torch.randn(B, 1, pre.size(2), pre.size(3), device=device)
    
    with torch.no_grad():
        for step in reversed(range(T)):
            t = torch.full((B,), step, device=device, dtype=torch.long)
            r_t = p_sample_step_residual(model, r_t, pre, t, betas, alphas, alphas_bar)
    
    return r_t


# ============================================================
# Model: Residual Diffusion Network
# ============================================================
class ResidualDiffusionNet(nn.Module):
    """
    Network that takes (noisy_residual, pre, t) and predicts the noise.
    
    Input: concatenate noisy_residual and pre -> 2 channels
    Output: predicted noise (1 channel)
    """
    def __init__(self):
        super().__init__()
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
        )
        
        # Main UNet backbone
        self.unet = UNet(
            spatial_dims=2,
            in_channels=2,  # noisy_residual + pre
            out_channels=1,  # predict noise
            channels=(64, 128, 256, 512),
            strides=(2, 2, 2),
            num_res_units=2,
            act=("LeakyReLU", {"inplace": True}),
            norm="INSTANCE",
            dropout=0.0,
        )
    
    def forward(self, r_t: torch.Tensor, pre: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        r_t: (B, 1, H, W) - noisy residual at timestep t
        pre: (B, 1, H, W) - conditioning pre image
        t: (B,) - timesteps
        """
        # Concatenate noisy residual and pre
        x_in = torch.cat([r_t, pre], dim=1)  # (B, 2, H, W)
        
        # Forward through UNet
        noise_pred = self.unet(x_in)
        
        return noise_pred


# ============================================================
# Training / Evaluation
# ============================================================
def train_one_epoch(model, loader, optimizer, pad_2d=None):
    model.train()
    total_loss = 0.0
    n = 0
    mask_t = None
    if pad_2d is not None and os.environ.get("WEEK7_REGION_WEIGHT", "").lower() in ("1", "true", "yes"):
        if os.path.join(_REPO_ROOT, "scripts") not in sys.path:
            sys.path.insert(0, os.path.join(_REPO_ROOT, "scripts"))
        from week7_preprocess import get_region_weight_mask_for_shape
        mask_np = get_region_weight_mask_for_shape(tuple(pad_2d), vascular_weight=1.5)
        mask_t = torch.from_numpy(mask_np).float().to(DEVICE).unsqueeze(0).unsqueeze(0)
    
    betas, alphas, alphas_bar = get_schedules(DEVICE)

    for pre_t, post_t in loader:
        pre_t = pre_t.to(DEVICE)
        post_t = post_t.to(DEVICE)
        if pad_2d is not None:
            pre_t, post_t = _pad_2d_if_needed(pre_t, post_t, pad_2d)
        B = pre_t.size(0)

        # Compute residual and scale
        residual = (post_t - pre_t) * RESIDUAL_SCALE  # Scale to roughly [-1, 1]

        # Sample random timesteps
        t = torch.randint(0, T, (B,), device=DEVICE, dtype=torch.long)
        
        # Add noise to residual
        noise = torch.randn_like(residual)
        r_t = q_sample_residual(residual, t, noise, alphas_bar)
        
        # Predict noise
        noise_pred = model(r_t, pre_t, t)
        
        # Loss: MSE (optionally region-weighted)
        if mask_t is not None:
            loss = (mask_t * (noise_pred - noise) ** 2).sum() / (mask_t.sum() * B + 1e-8)
        else:
            loss = F.mse_loss(noise_pred, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * B
        n += B

    return total_loss / n


@torch.no_grad()
def evaluate(model, loader, pad_2d=None):
    model.eval()
    total_L1 = 0.0
    total_MAE = 0.0
    total_SSIM = 0.0
    total_PSNR = 0.0
    n = 0

    for pre_t, post_t in loader:
        pre_t = pre_t.to(DEVICE)
        post_t = post_t.to(DEVICE)
        if pad_2d is not None:
            pre_t, post_t = _pad_2d_if_needed(pre_t, post_t, pad_2d)

        # Generate residual via diffusion
        residual_pred = p_sample_loop_residual(model, pre_t)
        
        # Unscale and add to pre to get predicted post
        residual_pred = residual_pred / RESIDUAL_SCALE
        pred = (pre_t + residual_pred).clamp(0, 1)

        B = pre_t.size(0)
        L1 = F.l1_loss(pred, post_t, reduction="mean")
        MAE = torch.mean(torch.abs(pred - post_t))

        total_L1 += L1.item() * B

        pred_np = pred.cpu().numpy()[:, 0]
        tgt_np = post_t.cpu().numpy()[:, 0]
        if pad_2d is not None:
            import sys
            if os.path.join(_REPO_ROOT, "scripts") not in sys.path:
                sys.path.insert(0, os.path.join(_REPO_ROOT, "scripts"))
            from week7_preprocess import metrics_in_brain_2d
            for i in range(B):
                m = metrics_in_brain_2d(pred_np[i], tgt_np[i], data_range=1.0)
                total_MAE += m["mae_mean"]
                total_SSIM += m["ssim_mean"]
                total_PSNR += m["psnr_mean"]
        else:
            total_MAE += MAE.item() * B
            for i in range(B):
                total_SSIM += ssim_np(pred_np[i], tgt_np[i])
                total_PSNR += psnr_np(pred_np[i], tgt_np[i])

        n += B

    if n == 0:
        return {"L1": None, "MAE": None, "SSIM": None, "PSNR": None}

    return {
        "L1": total_L1 / n,
        "MAE": total_MAE / n,
        "SSIM": total_SSIM / n,
        "PSNR": total_PSNR / n,
    }


# ============================================================
# Plotting
# ============================================================
def plot_history(history, out_dir):
    epochs = list(range(1, len(history["train_loss"]) + 1))

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, history["train_loss"], label="Train Loss (noise MSE)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Residual Diffusion Training Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "train_loss.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, history["val_SSIM"], label="Val SSIM")
    plt.xlabel("Epoch")
    plt.ylabel("SSIM")
    plt.title("Validation SSIM")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "val_SSIM.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, history["val_PSNR"], label="Val PSNR")
    plt.xlabel("Epoch")
    plt.ylabel("PSNR (dB)")
    plt.title("Validation PSNR")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "val_PSNR.png"), dpi=150)
    plt.close()


# Week7: pad 91x109 -> 96x112; save at 91x109
WEEK7_ORIGINAL_HW = (91, 109)
PAD_2D_WEEK7 = (96, 112)


def _pad_2d_if_needed(pre_t, post_t, target_hw):
    if target_hw is None or (pre_t.shape[2], pre_t.shape[3]) == target_hw:
        return pre_t, post_t
    th, tw = target_hw
    _, _, h, w = pre_t.shape
    if h < th or w < tw:
        pd = (0, max(0, tw - w), 0, max(0, th - h))
        pre_t = F.pad(pre_t, pd, mode="constant", value=0)
        post_t = F.pad(post_t, pd, mode="constant", value=0)
    return pre_t[:, :, :th, :tw], post_t[:, :, :th, :tw]


# ============================================================
# Main
# ============================================================
def main():
    global LOG_F
    LOG_F = open(os.path.join(RUN_DIR, "training_log.txt"), "w")

    log("=" * 60)
    log("Residual Diffusion: Pre-to-Post MRI Translation")
    log("=" * 60)

    week7 = "--week7" in sys.argv
    pad_2d = None
    if week7:
        sys.path.insert(0, os.path.join(_REPO_ROOT, "scripts"))
        from week7_data import get_week7_splits, Week7SlicePairs2D
        train_pairs, val_pairs, test_pairs = get_week7_splits()
        log(f"Week7: 91x109 + brain mask, combined 2020-2023: {len(train_pairs)} train / {len(val_pairs)} val / {len(test_pairs)} test")
        pad_2d = PAD_2D_WEEK7
        train_ds = Week7SlicePairs2D(train_pairs, augment=True)
        val_ds = Week7SlicePairs2D(val_pairs, augment=False)
        test_ds = Week7SlicePairs2D(test_pairs, augment=False)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    else:
        # Gather paired files
        all_pre = sorted(glob(os.path.join(PRE_DIR, "pre_*.nii.gz")))
        paired_pre = []
        for p in all_pre:
            q = pre_to_post(p)
            if q and os.path.exists(q):
                paired_pre.append(p)

        if not paired_pre:
            LOG_F.close()
            raise RuntimeError("No matched pre/post pairs found.")

        log(f"Total paired scans: {len(paired_pre)}")

        # Train/Test split (25% test)
        trainval_pre, test_pre = train_test_split(
            paired_pre, test_size=0.25, random_state=SEED, shuffle=True
        )
        log(f"Train+Val: {len(trainval_pre)} | Test: {len(test_pre)}")

        # Train/Val split (12% of trainval for validation)
        ds_trainval = AxialSlicePairs(trainval_pre, target_size=TARGET_SIZE)
        idxs = np.arange(len(ds_trainval))
        tr_idx, va_idx = train_test_split(
            idxs, test_size=0.12, random_state=SEED, shuffle=True
        )

        ds_train = Subset(ds_trainval, tr_idx)
        ds_val = Subset(ds_trainval, va_idx)

        log(f"Train: {len(ds_train)} | Val: {len(ds_val)}")

        train_loader = DataLoader(
            ds_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True
        )
        val_loader = DataLoader(
            ds_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
        )
        test_ds = AxialSlicePairs(test_pre, target_size=TARGET_SIZE)
        test_loader = DataLoader(
            test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
        )

    # Model / Optimizer
    model = ResidualDiffusionNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    history = {
        "train_loss": [],
        "val_L1": [],
        "val_MAE": [],
        "val_SSIM": [],
        "val_PSNR": [],
    }

    best_ssim = -1.0
    best_state = None

    log("\n=== Training Residual Diffusion ===")
    for ep in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, pad_2d=pad_2d)
        val_metrics = evaluate(model, val_loader, pad_2d=pad_2d)

        history["train_loss"].append(train_loss)
        for k in ["L1", "MAE", "SSIM", "PSNR"]:
            history[f"val_{k}"].append(val_metrics[k])

        log(
            f"[EP {ep:02d}/{EPOCHS}] "
            f"Train loss {train_loss:.4f} | "
            f"Val L1 {val_metrics['L1']:.4f}, "
            f"MAE {val_metrics['MAE']:.4f}, "
            f"SSIM {val_metrics['SSIM']:.4f}, "
            f"PSNR {val_metrics['PSNR']:.2f}"
        )

        if val_metrics["SSIM"] is not None and val_metrics["SSIM"] > best_ssim:
            best_ssim = val_metrics["SSIM"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # Save best model + history
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), MODEL_PATH)
    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)

    log(f"\nSaved best model to: {MODEL_PATH}")
    log(f"Training history saved to: {HISTORY_PATH}")

    # Plot curves
    plot_history(history, RUN_DIR)

    # Test evaluation
    log("\n=== Test Evaluation ===")
    if not week7:
        test_ds = AxialSlicePairs(test_pre, target_size=TARGET_SIZE)
        test_loader = DataLoader(
            test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
        )
    test_metrics = evaluate(model, test_loader, pad_2d=pad_2d)
    log(
        f"[TEST] L1 {test_metrics['L1']:.4f}, "
        f"MAE {test_metrics['MAE']:.4f}, "
        f"SSIM {test_metrics['SSIM']:.4f}, "
        f"PSNR {test_metrics['PSNR']:.2f}"
    )

    # Save test metrics (Phase 1 vs Phase 2 separate)
    phase2 = os.environ.get("WEEK7_REGION_WEIGHT", "").lower() in ("1", "true", "yes")
    out_name = "test_metrics_phase2.json" if (week7 and phase2) else "test_metrics.json"
    with open(os.path.join(RUN_DIR, out_name), "w") as f:
        json.dump(test_metrics, f, indent=2)

    # Save predictions (Week7: original dimensions 91x109)
    log("\n=== Saving Test Predictions ===")
    model.eval()
    count = 0
    save_hw = WEEK7_ORIGINAL_HW if pad_2d is not None else None
    with torch.no_grad():
        for pre_t, post_t in test_loader:
            pre_t = pre_t.to(DEVICE)
            post_t = post_t.to(DEVICE)
            if pad_2d is not None:
                pre_t, post_t = _pad_2d_if_needed(pre_t, post_t, pad_2d)

            residual_pred = p_sample_loop_residual(model, pre_t)
            residual_pred = residual_pred / RESIDUAL_SCALE
            pred = (pre_t + residual_pred).clamp(0, 1)

            pred_np = pred.cpu().numpy()
            pre_np = pre_t.cpu().numpy()
            post_np = post_t.cpu().numpy()
            if save_hw is not None:
                h, w = save_hw
                pred_np = pred_np[:, :, :h, :w]
                pre_np = pre_np[:, :, :h, :w]
                post_np = post_np[:, :, :h, :w]

            B = pred_np.shape[0]
            for b in range(B):
                np.save(os.path.join(PRED_DIR, f"in_{count:04d}.npy"), pre_np[b, 0])
                np.save(os.path.join(PRED_DIR, f"out_{count:04d}.npy"), pred_np[b, 0])
                np.save(os.path.join(PRED_DIR, f"tgt_{count:04d}.npy"), post_np[b, 0])
                count += 1

    log(f"Saved {count} prediction triplets to {PRED_DIR}")
    log("\nDone.")
    LOG_F.close()


if __name__ == "__main__":
    main()
