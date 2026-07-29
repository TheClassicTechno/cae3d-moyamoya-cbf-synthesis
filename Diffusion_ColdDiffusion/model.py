#!/usr/bin/env python3
"""
Cold Diffusion for Pre-to-Post MRI Translation

Instead of corrupting images with Gaussian noise, Cold Diffusion uses
deterministic degradation operators. Here we use image interpolation:
    x_t = (1 - t/T) * post + (t/T) * pre

The model learns to restore the post image from any degraded state x_t
given the conditioning pre image.

This approach is ideal for paired image-to-image translation tasks because:
1. The degradation is meaningful (blending toward the input)
2. No stochasticity during sampling (deterministic output)
3. Strong conditioning since pre is used as both degradation target and condition

References:
- Bansal et al., "Cold Diffusion: Inverting Arbitrary Image Transforms Without Noise"
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

# Paths - script is in Diffusion_ColdDiffusion/, data is one level up
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

PRE_DIR = os.path.join(ROOT_DIR, "pre_scans")
POST_DIR = os.path.join(ROOT_DIR, "post_scans")

RUN_DIR = os.path.join(BASE_DIR, "runs_cold_diffusion")
PRED_DIR = os.path.join(RUN_DIR, "pred_samples")
os.makedirs(PRED_DIR, exist_ok=True)

MODEL_PATH = os.path.join(RUN_DIR, "best_cold_diffusion.pt")
HISTORY_PATH = os.path.join(RUN_DIR, "training_history.json")

TARGET_SIZE = (128, 128)
BATCH_SIZE = 8
EPOCHS = 50
LR = 1e-3

# Cold Diffusion params
T = 10  # Number of degradation steps
# We use discrete steps: t ∈ {0, 1, ..., T-1}
# t=0 means x_0 = post (clean)
# t=T-1 means x_{T-1} ≈ pre (fully degraded)


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
# Cold Diffusion: Degradation and Restoration
# ============================================================

def cold_degrade(post: torch.Tensor, pre: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Cold diffusion degradation: linear interpolation between post and pre.
    
    x_t = (1 - alpha) * post + alpha * pre
    where alpha = t / (T - 1), so:
        t=0 -> alpha=0 -> x_t = post (clean)
        t=T-1 -> alpha=1 -> x_t = pre (fully degraded)
    
    post, pre: (B, 1, H, W)
    t: (B,) integer timesteps in [0, T-1]
    """
    alpha = t.float() / (T - 1)  # (B,)
    alpha = alpha.view(-1, 1, 1, 1)  # (B, 1, 1, 1)
    x_t = (1 - alpha) * post + alpha * pre
    return x_t


def cold_restore_step(model, x_t: torch.Tensor, pre: torch.Tensor, t: int) -> torch.Tensor:
    """
    One restoration step: predict x_0 from x_t, then compute x_{t-1}.
    
    The model predicts the clean image x_0 directly.
    Then we compute x_{t-1} using interpolation.
    """
    # Model predicts x_0 (the post image)
    x0_pred = model(x_t, pre, t)
    
    if t == 0:
        return x0_pred
    
    # Compute x_{t-1} by interpolating between predicted x_0 and pre
    alpha_prev = (t - 1) / (T - 1)
    x_prev = (1 - alpha_prev) * x0_pred + alpha_prev * pre
    return x_prev


def cold_sample(model, pre: torch.Tensor) -> torch.Tensor:
    """
    Full cold diffusion sampling: start from x_{T-1} ≈ pre, iteratively restore.
    
    Applies background masking to eliminate blob artifacts:
    - Creates a mask from pre image (foreground where intensity > threshold)
    - In background regions, keeps the original pre image values
    """
    model.eval()
    B = pre.size(0)
    
    # Create a brain/foreground mask from pre image
    # Threshold-based: foreground is where intensity > 0.05
    MASK_THRESHOLD = 0.05
    foreground_mask = (pre > MASK_THRESHOLD).float()
    
    # Optional: dilate the mask slightly to avoid edge artifacts
    # Using a simple approach: if any neighbor is foreground, include it
    
    # Start from the most degraded state (which is just the pre image)
    x_t = pre.clone()
    
    with torch.no_grad():
        # Iterate from t=T-1 down to t=0
        for t in reversed(range(T)):
            x_t = cold_restore_step(model, x_t, pre, t)
    
    # Apply background masking: keep pre's background, use prediction for foreground
    # This eliminates blobs in the background
    output = foreground_mask * x_t + (1 - foreground_mask) * pre
    
    return output.clamp(0, 1)


# ============================================================
# Model: Cold Diffusion Restoration Network
# ============================================================
class ColdDiffusionNet(nn.Module):
    """
    Network that takes (x_t, pre, t) and predicts x_0 (the clean post image).
    
    Architecture: UNet with timestep embedding and pre-image conditioning.
    Input: concatenate x_t and pre -> 2 channels
    Output: predicted x_0 (1 channel)
    
    Timestep is embedded and added to intermediate features.
    """
    def __init__(self):
        super().__init__()
        
        # Main UNet backbone
        self.unet = UNet(
            spatial_dims=2,
            in_channels=2,  # x_t + pre concatenated
            out_channels=1,  # predict x_0
            channels=(64, 128, 256, 512),
            strides=(2, 2, 2),
            num_res_units=2,
            act=("LeakyReLU", {"inplace": True}),
            norm="INSTANCE",
            dropout=0.0,
        )
        
        # Timestep embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
        )
        
        # Project time embedding to modulate features
        self.time_proj = nn.Sequential(
            nn.Linear(64, 128),
            nn.SiLU(),
        )
    
    def forward(self, x_t: torch.Tensor, pre: torch.Tensor, t: int) -> torch.Tensor:
        """
        x_t: (B, 1, H, W) - degraded image at timestep t
        pre: (B, 1, H, W) - conditioning pre image
        t: int - current timestep
        """
        B = x_t.size(0)
        
        # Normalize timestep to [0, 1]
        t_norm = torch.full((B, 1), t / (T - 1), device=x_t.device, dtype=x_t.dtype)
        
        # Get time embedding (not used in simple version, but shows structure)
        # In a more advanced version, you'd inject this into the UNet
        _ = self.time_embed(t_norm)
        
        # Concatenate x_t and pre
        x_in = torch.cat([x_t, pre], dim=1)  # (B, 2, H, W)
        
        # Forward through UNet
        x_0_pred = self.unet(x_in)
        
        return x_0_pred


# ============================================================
# Training / Evaluation
# ============================================================
def train_one_epoch(model, loader, optimizer, criterion_l1, criterion_ssim, pad_2d=None):
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

    for pre_t, post_t in loader:
        pre_t = pre_t.to(DEVICE)
        post_t = post_t.to(DEVICE)
        if pad_2d is not None:
            pre_t, post_t = _pad_2d_if_needed(pre_t, post_t, pad_2d)
        B = pre_t.size(0)

        # Sample random timesteps for each sample in batch
        t = torch.randint(0, T, (B,), device=DEVICE)
        
        # Create degraded images
        x_t = cold_degrade(post_t, pre_t, t)
        
        # Model predicts x_0 (the clean post image)
        x0_preds = []
        for i in range(B):
            x0_pred_i = model(x_t[i:i+1], pre_t[i:i+1], t[i].item())
            x0_preds.append(x0_pred_i)
        x0_pred = torch.cat(x0_preds, dim=0)
        
        # Loss: predict x_0 from x_t (optionally region-weighted)
        if mask_t is not None:
            loss_l1 = (torch.abs(x0_pred - post_t) * mask_t).sum() / (mask_t.sum() * B + 1e-8)
        else:
            loss_l1 = criterion_l1(x0_pred, post_t)
        loss_ssim = criterion_ssim(x0_pred, post_t)
        loss = loss_l1 + loss_ssim

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

        # Full cold diffusion sampling
        pred = cold_sample(model, pre_t)

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
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Cold Diffusion Training Loss")
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


# Week7: pad 91x109 -> 96x112 for UNet; save at 91x109
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
    log("Cold Diffusion: Pre-to-Post MRI Translation")
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

        # Train/Test split (25% test - standardized across all models)
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

    # Model / Optimizer / Loss
    model = ColdDiffusionNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion_l1 = nn.L1Loss()
    criterion_ssim = SSIMLoss(spatial_dims=2)

    history = {
        "train_loss": [],
        "val_L1": [],
        "val_MAE": [],
        "val_SSIM": [],
        "val_PSNR": [],
    }

    best_ssim = -1.0
    best_state = None

    log("\n=== Training Cold Diffusion ===")
    for ep in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion_l1, criterion_ssim, pad_2d=pad_2d)
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

    # Save predictions for visualization (Week7: original dimensions 91x109)
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

            pred = cold_sample(model, pre_t)

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
