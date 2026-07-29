#!/usr/bin/env python3
import os
import os, re, math, random, json, sys
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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split

from skimage.metrics import structural_similarity as sk_ssim
from skimage.metrics import peak_signal_noise_ratio as sk_psnr

import matplotlib.pyplot as plt  # for saving training curves

# MONAI Generative
# pip install monai-generative
from generative.networks.nets import DiffusionModelUNet

# ============================================================
# Config
# ============================================================
SEED = 1337
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# IMPORTANT: script is in Diffusion_baseline/, data is one level up
PRE_DIR = os.path.join("..", "pre_scans")
POST_DIR = os.path.join("..", "post_scans")

OUT_ROOT = "runs_diffusion_post_from_pre"
os.makedirs(OUT_ROOT, exist_ok=True)
MODEL_PATH = os.path.join(OUT_ROOT, "best_diffusion_unet.pt")
PRED_DIR = os.path.join(OUT_ROOT, "pred_samples")
os.makedirs(PRED_DIR, exist_ok=True)

TARGET_SIZE = (128, 128)
BATCH_SIZE = 8

# We keep 50 epochs:
# - Diffusion training is much heavier than UNet regression.
# - On small medical datasets, 40–50 epochs is usually enough to reach
#   a plateau; going far beyond tends to give diminishing returns and
#   long runtimes.
EPOCHS = 50
LR = 2e-4

# --- Diffusion hyperparams ---
T = 200  # number of diffusion steps (kept smaller than 1000 for speed)
BETA_START = 1e-4
BETA_END = 2e-2

# ============================================================
# Helpers: pairing, IO, metrics
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

def to_tensor_1chw(x2d: np.ndarray, target_hw=TARGET_SIZE) -> torch.Tensor:
    t = torch.from_numpy(x2d).unsqueeze(0).unsqueeze(0).float()  # (1,1,H,W)
    t = F.interpolate(t, size=target_hw, mode="bilinear", align_corners=False)
    return t.squeeze(0)  # (1,H,W)

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
        if len(self.items) == 0:
            raise RuntimeError("No paired scans found.")
        if missing > 0:
            print(f"[WARN] Skipped {missing} pre scans with no matching post.")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        pre_p, post_p = self.items[idx]
        pre_sl = minmax_norm(load_middle_axial_slice(pre_p))
        post_sl = minmax_norm(load_middle_axial_slice(post_p))

        pre_t = to_tensor_1chw(pre_sl, self.target_size)   # (1,H,W)
        post_t = to_tensor_1chw(post_sl, self.target_size) # (1,H,W)

        return pre_t.float(), post_t.float()

# ============================================================
# Diffusion utilities
# ============================================================

# beta schedule (linear)
betas = torch.linspace(BETA_START, BETA_END, T, dtype=torch.float32)
alphas = 1.0 - betas
alphas_bar = torch.cumprod(alphas, dim=0)

betas = betas.to(DEVICE)
alphas = alphas.to(DEVICE)
alphas_bar = alphas_bar.to(DEVICE)

def sample_timesteps(batch_size: int) -> torch.Tensor:
    return torch.randint(low=0, high=T, size=(batch_size,), device=DEVICE)

def q_sample(x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    """
    Forward diffusion: q(x_t | x_0).
    x0, noise: (B,1,H,W)
    t: (B,)
    """
    a_bar = alphas_bar[t].view(-1, 1, 1, 1)
    return torch.sqrt(a_bar) * x0 + torch.sqrt(1.0 - a_bar) * noise

def p_sample_step(model, x_t, pre_t, t: torch.Tensor) -> torch.Tensor:
    """
    One reverse step p(x_{t-1} | x_t, pre).
    """
    # predict noise epsilon_theta(x_t, t, pre)
    eps_theta = model(x_t, pre_t, t)

    a_t = alphas[t].view(-1, 1, 1, 1)
    a_bar_t = alphas_bar[t].view(-1, 1, 1, 1)
    if (t == 0).all():
        a_bar_prev = torch.ones_like(a_bar_t)
    else:
        t_prev = torch.clamp(t - 1, min=0)
        a_bar_prev = alphas_bar[t_prev].view(-1, 1, 1, 1)

    beta_t = betas[t].view(-1, 1, 1, 1)

    # DDPM mean formula
    sqrt_one_minus_a_bar = torch.sqrt(1.0 - a_bar_t)
    mean = (1.0 / torch.sqrt(a_t)) * (x_t - beta_t / sqrt_one_minus_a_bar * eps_theta)

    # variance (posterior)
    var = beta_t * (1.0 - a_bar_prev) / (1.0 - a_bar_t)
    noise = torch.randn_like(x_t)
    nonzero_mask = (t > 0).float().view(-1, 1, 1, 1)
    x_prev = mean + nonzero_mask * torch.sqrt(var) * noise
    return x_prev

def p_sample_loop(model, pre_t: torch.Tensor) -> torch.Tensor:
    """
    Run full reverse diffusion given conditioning pre_t.
    pre_t: (B,1,H,W)
    returns: generated post (B,1,H,W)
    """
    model.eval()
    B, _, H, W = pre_t.shape
    x_t = torch.randn(B, 1, H, W, device=DEVICE)

    with torch.no_grad():
        for step in reversed(range(T)):
            t = torch.full((B,), step, device=DEVICE, dtype=torch.long)
            x_t = p_sample_step(model, x_t, pre_t, t)
    return x_t

# ============================================================
# Conditional Diffusion Model wrapper
# ============================================================
class CondDiffusionUNet(nn.Module):
    """
    Wrap MONAI's DiffusionModelUNet to take (x_t, pre, t).
    in_channels = 2 : [noisy post, pre]
    out_channels = 1 : predicted noise epsilon on post channel.
    """
    def __init__(self):
        super().__init__()
        self.unet = DiffusionModelUNet(
            spatial_dims=2,
            in_channels=2,
            out_channels=1,
            num_channels=(64, 128, 256),
            attention_levels=(False, True, True),
            num_res_blocks=2,
            num_head_channels=(0, 32, 32),
            norm_num_groups=8,
        )

    def forward(self, x_t: torch.Tensor, pre_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # concat condition along channel: (B,2,H,W)
        x_in = torch.cat([x_t, pre_t], dim=1)
        return self.unet(x=x_in, timesteps=t)

# ============================================================
# Train / Eval
# ============================================================
def _pad_2d_if_needed(pre_t, post_t, target_hw):
    """Pad (B,1,H,W) to target_hw for Week7 (e.g. 91,109 -> 96,112)."""
    if target_hw is None or (pre_t.shape[2], pre_t.shape[3]) == target_hw:
        return pre_t, post_t
    th, tw = target_hw
    _, _, h, w = pre_t.shape
    if h < th or w < tw:
        pd = (0, max(0, tw - w), 0, max(0, th - h))
        pre_t = F.pad(pre_t, pd, mode="constant", value=0)
        post_t = F.pad(post_t, pd, mode="constant", value=0)
    return pre_t[:, :, :th, :tw], post_t[:, :, :th, :tw]


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

    for pre_t, post_t in loader:
        if pad_2d is not None:
            pre_t, post_t = _pad_2d_if_needed(pre_t, post_t, pad_2d)
        pre_t = pre_t.to(DEVICE)
        post_t = post_t.to(DEVICE)

        B = pre_t.size(0)
        t = sample_timesteps(B)
        noise = torch.randn_like(post_t)
        x_t = q_sample(post_t, t, noise)

        noise_pred = model(x_t, pre_t, t)
        if mask_t is not None:
            loss = ((mask_t * (noise_pred - noise) ** 2).sum() / (mask_t.sum() * B + 1e-8))
        else:
            loss = F.mse_loss(noise_pred, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * B
        n += B

    return total_loss / n

@torch.no_grad()
def evaluate_model(model, loader, pad_2d=None):
    """
    Evaluation using full sampling on the given loader.
    Returns mean L1, MAE, SSIM, PSNR on the generated post.
    """
    model.eval()
    total_l1 = 0.0
    total_mae = 0.0
    total_ssim = 0.0
    total_psnr = 0.0
    n = 0

    for pre_t, post_t in loader:
        if pad_2d is not None:
            pre_t, post_t = _pad_2d_if_needed(pre_t, post_t, pad_2d)
        pre_t = pre_t.to(DEVICE)
        post_t = post_t.to(DEVICE)

        gen = p_sample_loop(model, pre_t)  # (B,1,H,W)

        l1 = torch.mean(torch.abs(gen - post_t))
        mae = l1  # same as L1 over [0,1] images

        gen_np = gen.cpu().numpy()[:, 0]
        tgt_np = post_t.cpu().numpy()[:, 0]

        batch_ssim = 0.0
        batch_psnr = 0.0
        batch_mae_custom = 0.0
        for i in range(gen_np.shape[0]):
            if pad_2d is not None:
                import sys
                if os.path.join(_REPO_ROOT, "scripts") not in sys.path:
                    sys.path.insert(0, os.path.join(_REPO_ROOT, "scripts"))
                from week7_preprocess import metrics_in_brain_2d
                m = metrics_in_brain_2d(gen_np[i], tgt_np[i], data_range=1.0)
                batch_mae_custom += m["mae_mean"]
                batch_ssim += m["ssim_mean"]
                batch_psnr += m["psnr_mean"]
            else:
                batch_ssim += ssim_np(gen_np[i], tgt_np[i])
                batch_psnr += psnr_np(gen_np[i], tgt_np[i])

        B = pre_t.size(0)
        total_l1 += l1.item() * B
        if pad_2d is not None:
            total_mae += batch_mae_custom
        else:
            total_mae += mae.item() * B
        total_ssim += batch_ssim
        total_psnr += batch_psnr
        n += B

    metrics = {
        "L1": total_l1 / n,
        "MAE": total_mae / n,
        "SSIM": total_ssim / n,
        "PSNR": total_psnr / n,
    }
    return metrics

# ============================================================
# Plotting helper
# ============================================================
def plot_history(history, out_dir):
    epochs = list(range(1, len(history["train_loss"]) + 1))

    # 1) Train loss
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, history["train_loss"], label="Train diffusion loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE noise loss")
    plt.title("Training loss (diffusion)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "train_loss_diffusion.png"), dpi=150)
    plt.close()

    # 2) Val L1 & MAE
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, history["val_L1"], label="Val L1")
    plt.plot(epochs, history["val_MAE"], label="Val MAE", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.title("Validation L1 / MAE (generated vs post)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "val_L1_MAE_diffusion.png"), dpi=150)
    plt.close()

    # 3) Val SSIM & PSNR
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, history["val_SSIM"], label="Val SSIM")
    plt.xlabel("Epoch")
    plt.ylabel("SSIM (sum over batch)")
    plt.title("Validation SSIM (generated vs post)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "val_SSIM_diffusion.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, history["val_PSNR"], label="Val PSNR")
    plt.xlabel("Epoch")
    plt.ylabel("PSNR (dB, summed over batch)")
    plt.title("Validation PSNR (generated vs post)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "val_PSNR_diffusion.png"), dpi=150)
    plt.close()

# ============================================================
# Main
# ============================================================
def main():
    week7 = "--week7" in sys.argv
    save_predictions_only = "--save-predictions-only" in sys.argv
    if save_predictions_only and week7:
        # Load best model and run only test eval + save predictions (91x109)
        if not os.path.isfile(MODEL_PATH):
            raise FileNotFoundError(f"No checkpoint at {MODEL_PATH}; train first with --week7")
        sys.path.insert(0, os.path.join(_REPO_ROOT, "scripts"))
        from week7_data import get_week7_splits, Week7SlicePairs2D
        _, _, test_pairs = get_week7_splits()
        test_ds = Week7SlicePairs2D(test_pairs, augment=False)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
        pad_2d = (96, 112)
        model = CondDiffusionUNet().to(DEVICE)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
        print("=== Test evaluation (loaded checkpoint) ===")
        test_metrics = evaluate_model(model, test_loader, pad_2d=pad_2d)
        print(f"[TEST] MAE {test_metrics['MAE']:.4f}, SSIM {test_metrics['SSIM']:.4f}, PSNR {test_metrics['PSNR']:.2f}")
        with open(os.path.join(OUT_ROOT, "test_metrics_diffusion.json"), "w") as f:
            json.dump(test_metrics, f, indent=2)
        # Save predictions at original dimensions 91x109
        WEEK7_ORIGINAL_HW = (91, 109)
        count = 0
        with torch.no_grad():
            for pre_t, post_t in test_loader:
                pre_t = pre_t.to(DEVICE)
                post_t = post_t.to(DEVICE)
                pre_t, post_t = _pad_2d_if_needed(pre_t, post_t, pad_2d)
                gen = p_sample_loop(model, pre_t)
                pre_np = pre_t.cpu().numpy()[:, :, :91, :109]
                gen_np = gen.cpu().numpy()[:, :, :91, :109]
                tgt_np = post_t.cpu().numpy()[:, :, :91, :109]
                for i in range(pre_np.shape[0]):
                    np.save(os.path.join(PRED_DIR, f"in_{count:04d}.npy"), pre_np[i, 0])
                    np.save(os.path.join(PRED_DIR, f"out_{count:04d}.npy"), gen_np[i, 0])
                    np.save(os.path.join(PRED_DIR, f"tgt_{count:04d}.npy"), tgt_np[i, 0])
                    count += 1
        print(f"Saved {count} triplets (91x109) to {PRED_DIR}")
        print("Done.")
        return

    # ------------- Week7: same data/preprocess (91x109x91, brain mask, combined 2020-2023) -------------
    pad_2d = None
    if week7:
        sys.path.insert(0, os.path.join(_REPO_ROOT, "scripts"))
        from week7_data import get_week7_splits, Week7SlicePairs2D
        train_pairs, val_pairs, test_pairs = get_week7_splits()
        pad_2d = (96, 112)  # pad 91x109 to 96x112 for UNet
        print(f"Week7: 91x109 + brain mask, combined 2020-2023: {len(train_pairs)} train / {len(val_pairs)} val / {len(test_pairs)} test")
        train_ds = Week7SlicePairs2D(train_pairs, augment=True)
        val_ds = Week7SlicePairs2D(val_pairs, augment=False)
        test_ds = Week7SlicePairs2D(test_pairs, augment=False)
    else:
        # ------------- collect paired scans -------------
        all_pre = sorted(glob(os.path.join(PRE_DIR, "pre_*.nii.gz")))
        paired = []
        for p in all_pre:
            q = pre_to_post(p)
            if q and os.path.exists(q):
                paired.append(p)

        if len(paired) == 0:
            raise RuntimeError("No matched pre/post pairs found.")

        print(f"Total paired scans: {len(paired)}")

        # Train/test split with 25% test
        train_pre, test_pre = train_test_split(
            paired, test_size=0.25, random_state=SEED, shuffle=True
        )
        print(f"Train: {len(train_pre)} | Test: {len(test_pre)}")

        # Small internal val from train (10% of train)
        if len(train_pre) > 10:
            tr_pre, val_pre = train_test_split(
                train_pre, test_size=0.10, random_state=SEED, shuffle=True
            )
        else:
            tr_pre = train_pre
            val_pre = train_pre

        print(f"Train used: {len(tr_pre)} | Val: {len(val_pre)}")

        train_ds = AxialSlicePairs(tr_pre)
        val_ds = AxialSlicePairs(val_pre)
        test_ds = AxialSlicePairs(test_pre)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=2, pin_memory=True)

    # ------------- model / optim -------------
    model = CondDiffusionUNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val = float("inf")
    best_state = None
    history = {"train_loss": [], "val_L1": [], "val_MAE": [], "val_SSIM": [], "val_PSNR": []}

    print("\n=== Training diffusion model ===")
    for ep in range(1, EPOCHS + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, pad_2d=pad_2d)
        val_metrics = evaluate_model(model, val_loader, pad_2d=pad_2d)

        history["train_loss"].append(tr_loss)
        history["val_L1"].append(val_metrics["L1"])
        history["val_MAE"].append(val_metrics["MAE"])
        history["val_SSIM"].append(val_metrics["SSIM"])
        history["val_PSNR"].append(val_metrics["PSNR"])

        print(
            f"[EP {ep:02d}/{EPOCHS}] "
            f"Train loss {tr_loss:.4f} | "
            f"Val L1 {val_metrics['L1']:.4f}, "
            f"MAE {val_metrics['MAE']:.4f}, "
            f"SSIM {val_metrics['SSIM']:.4f}, "
            f"PSNR {val_metrics['PSNR']:.2f}"
        )

        # keep best by L1 (could also use SSIM)
        if val_metrics["L1"] < best_val:
            best_val = val_metrics["L1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), MODEL_PATH)
    with open(os.path.join(OUT_ROOT, "training_history_diffusion.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nSaved best diffusion model to {MODEL_PATH}")

    # Plot training curves for the paper
    plot_history(history, OUT_ROOT)

    # ------------- test evaluation -------------
    print("\n=== Test evaluation (diffusion) ===")
    test_metrics = evaluate_model(model, test_loader, pad_2d=pad_2d)
    print(
        f"[TEST] L1 {test_metrics['L1']:.4f}, "
        f"MAE {test_metrics['MAE']:.4f}, "
        f"SSIM {test_metrics['SSIM']:.4f}, "
        f"PSNR {test_metrics['PSNR']:.2f}"
    )

    # Save test metrics as JSON (Phase 1 vs Phase 2 separate paths)
    phase2 = os.environ.get("WEEK7_REGION_WEIGHT", "").lower() in ("1", "true", "yes")
    out_name = "test_metrics_diffusion_phase2.json" if (week7 and phase2) else "test_metrics_diffusion.json"
    with open(os.path.join(OUT_ROOT, out_name), "w") as f:
        json.dump(test_metrics, f, indent=2)

    # ------------- save predictions for test set (original dimensions: 91x109 for Week7) -------------
    print("\n=== Saving test predictions for analysis ===")
    # Week7: keep original image dimensions 91x109 (2D slice from 91x109x91)
    WEEK7_ORIGINAL_HW = (91, 109)
    save_hw = WEEK7_ORIGINAL_HW if pad_2d is not None else None
    model.eval()
    count = 0
    with torch.no_grad():
        for pre_t, post_t in test_loader:
            pre_t = pre_t.to(DEVICE)
            post_t = post_t.to(DEVICE)
            if pad_2d is not None:
                pre_t, post_t = _pad_2d_if_needed(pre_t, post_t, pad_2d)

            gen = p_sample_loop(model, pre_t)  # (B,1,H,W)

            pre_np = pre_t.cpu().numpy()
            gen_np = gen.cpu().numpy()
            tgt_np = post_t.cpu().numpy()
            if save_hw is not None:
                h, w = save_hw
                pre_np = pre_np[:, :, :h, :w]
                gen_np = gen_np[:, :, :h, :w]
                tgt_np = tgt_np[:, :, :h, :w]

            B = pre_np.shape[0]
            for i in range(B):
                np.save(os.path.join(PRED_DIR, f"in_{count:04d}.npy"),  pre_np[i, 0])
                np.save(os.path.join(PRED_DIR, f"out_{count:04d}.npy"), gen_np[i, 0])
                np.save(os.path.join(PRED_DIR, f"tgt_{count:04d}.npy"), tgt_np[i, 0])
                count += 1

    print(f"Saved {count} triplets (in/out/tgt) to {PRED_DIR}")
    print("Done.")

if __name__ == "__main__":
    main()
