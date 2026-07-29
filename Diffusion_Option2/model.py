#!/usr/bin/env python3
"""
Diffusion Option 2: Fine-tuning a pretrained MONAI DiffusionModelUNet
to generate post-acetazolamide scans from pre scans (middle axial slice).

- Backbone: DiffusionModelUNet from monai.generative
- Conditioning: concatenate noisy post slice with pre slice (2-channel input)
- Loss: MSE between predicted noise and true noise
- Split:
    * 20 scans held out as TEST
    * remaining -> Train+Val with 12% internal val split

Outputs (all INSIDE this Diffusion_Option2 folder):

    runs_diffusion_post_from_pre/
        best_diffusion_model.pt
        train_history.json
        pred_samples/
            in_XXXX.npy   (pre slice, 128x128, [0,1])
            out_XXXX.npy  (diffusion prediction)
            tgt_XXXX.npy  (post slice, ground truth)
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
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

from generative.networks.nets import DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler

from skimage.metrics import structural_similarity as sk_ssim
from skimage.metrics import peak_signal_noise_ratio as sk_psnr

# ============================================================
# Paths / Config
# ============================================================

SEED = 1337
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# This script is inside Diffusion_Option2
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)  # parent: where pre_scans / post_scans live

PRE_DIR = os.path.join(ROOT_DIR, "pre_scans")
POST_DIR = os.path.join(ROOT_DIR, "post_scans")

RUN_DIR = os.path.join(BASE_DIR, "runs_diffusion_post_from_pre")
PRED_DIR = os.path.join(RUN_DIR, "pred_samples")
os.makedirs(PRED_DIR, exist_ok=True)

MODEL_PATH = os.path.join(RUN_DIR, "best_diffusion_model.pt")
HISTORY_PATH = os.path.join(RUN_DIR, "train_history.json")

TARGET_SIZE = (128, 128)
BATCH_SIZE = 8
EPOCHS = 50

# Diffusion config
NUM_TRAIN_TIMESTEPS = 1000     # training timesteps
NUM_INFER_STEPS = 80           # sampling timesteps (fewer for speed)
LR = 5e-5                      # a bit smaller for fine-tuning

# === IMPORTANT ===
# Set this to your actual pretrained DiffusionModelUNet checkpoint if you have one.
# If the file does not exist, the script will print a warning and train from scratch.
PRETRAINED_CKPT = os.path.join(ROOT_DIR, "pretrained", "diffusion_unet_pretrained.pt")

VAL_FRACTION = 0.12            # internal validation fraction


# ============================================================
# Helpers: pairing & I/O
# ============================================================

ID_RE = re.compile(r"(pre)_(\d{4})_(\d{3})\.nii\.gz$", re.IGNORECASE)


def id_from_pre_path(p: str) -> Tuple[str, str] | None:
    """
    From '.../pre_YYYY_NNN.nii.gz' → ('YYYY', 'NNN')
    """
    m = ID_RE.search(os.path.basename(p))
    if not m:
        return None
    return m.group(2), m.group(3)


def pre_to_post(pre_path: str) -> str | None:
    """
    Map pre_YYYY_NNN → post_YYYY_NNN in POST_DIR.
    """
    ids = id_from_pre_path(pre_path)
    if not ids:
        return None
    y, n = ids
    return os.path.join(POST_DIR, f"post_{y}_{n}.nii.gz")


def load_middle_axial_slice(nii_path: str) -> np.ndarray:
    """
    Load NIfTI and return the middle axial slice (H,W).
    If 4D, take the first volume.
    """
    img = nib.load(nii_path)
    data = img.get_fdata(dtype=np.float32)

    if data.ndim == 4:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"Unexpected shape {data.shape} for {nii_path}")

    H, W, D = data.shape
    return data[:, :, D // 2]


def minmax_norm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Per-slice min-max normalization to [0,1].
    """
    mn, mx = float(np.min(x)), float(np.max(x))
    if mx - mn < eps:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)


def to_tensor_resized_1chw(x2d: np.ndarray, target_hw=TARGET_SIZE) -> torch.Tensor:
    """
    (H,W) numpy → (1,Ht,Wt) tensor, resized by bilinear interpolation.
    """
    t = torch.from_numpy(x2d).unsqueeze(0).unsqueeze(0).float()  # (1,1,H,W)
    t = F.interpolate(t, size=target_hw, mode="bilinear", align_corners=False)
    return t.squeeze(0)  # (1,Ht,Wt)


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
    """
    Paired pre/post middle axial slices → tensors (1,H,W), normalized & resized.
    """
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
            raise RuntimeError("No paired pre/post scans found.")
        if missing > 0:
            print(f"[WARN] Skipped {missing} pre scans with no matching post.")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        pre_p, post_p = self.items[idx]

        pre_sl = minmax_norm(load_middle_axial_slice(pre_p))
        post_sl = minmax_norm(load_middle_axial_slice(post_p))

        pre_t = to_tensor_resized_1chw(pre_sl, self.target_size)   # (1,H,W)
        post_t = to_tensor_resized_1chw(post_sl, self.target_size) # (1,H,W)

        return pre_t.float(), post_t.float()


# ============================================================
# Model / Diffusion utilities
# ============================================================

def make_diffusion_unet():
    """
    2D UNet-like diffusion backbone.
    in_channels=2: [noisy_post, pre]
    out_channels=1: predicted noise
    """
    return DiffusionModelUNet(
        spatial_dims=2,
        in_channels=2,
        out_channels=1,
        num_channels=(32, 64, 128, 256),
        attention_levels=(False, False, True, True),
        num_res_blocks=2,
        norm_num_groups=8,
        resblock_updown=True,
    )


def sample_ddpm(model, scheduler, pre_batch, num_steps=NUM_INFER_STEPS):
    """
    DDPM sampling conditioned on pre_batch.

    pre_batch: (B,1,H,W) in [0,1]
    returns : (B,1,H,W) predicted post images
    """
    model.eval()
    B, _, H, W = pre_batch.shape
    device = pre_batch.device

    # Start from pure noise
    x = torch.randn((B, 1, H, W), device=device)

    scheduler.set_timesteps(num_steps)

    with torch.no_grad():
        for t in scheduler.timesteps:
            # t is scalar or 0-D tensor → turn into python int
            t_int = int(t)
            # batch timesteps for the model
            t_batch = torch.full((B,), t_int, device=device, dtype=torch.long)

            # conditional input = [noisy post, pre]
            model_in = torch.cat([x, pre_batch], dim=1)  # (B,2,H,W)

            noise_pred = model(model_in, t_batch)

            # DDPMScheduler in MONAI returns (prev_sample, pred_x0)
            x, _ = scheduler.step(noise_pred, t_int, x)

    return x.clamp(0.0, 1.0)


# ============================================================
# Training / Evaluation
# ============================================================

def train_one_epoch(model, scheduler, loader, optimizer, pad_2d=None):
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

        # Random timesteps per sample
        t = torch.randint(
            0,
            scheduler.num_train_timesteps,
            (B,),
            device=DEVICE,
            dtype=torch.long,
        )

        # Add noise to post image
        noise = torch.randn_like(post_t)
        noisy_post = scheduler.add_noise(post_t, noise, t)

        # conditional input
        model_in = torch.cat([noisy_post, pre_t], dim=1)  # (B,2,H,W)

        # predict noise
        noise_pred = model(model_in, t)

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
def evaluate(model, scheduler, loader, pad_2d=None):
    """
    Sampling-based evaluation on val set.
    """
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

        pred = sample_ddpm(model, scheduler, pre_t)  # (B,1,H,W)

        B = pre_t.size(0)
        L1 = F.l1_loss(pred, post_t, reduction="mean")
        MAE = torch.mean(torch.abs(pred - post_t))

        total_L1 += L1.item() * B
        total_MAE += MAE.item() * B

        pred_np = pred.detach().cpu().numpy()[:, 0]
        tgt_np = post_t.detach().cpu().numpy()[:, 0]
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
    week7 = "--week7" in sys.argv
    pad_2d = None
    if week7:
        sys.path.insert(0, os.path.join(_REPO_ROOT, "scripts"))
        from week7_data import get_week7_splits, Week7SlicePairs2D
        from week7_preprocess import metrics_in_brain_2d
        train_pairs, val_pairs, test_pairs = get_week7_splits()
        print(f"Week7: 91x109 + brain mask, combined 2020-2023: {len(train_pairs)} train / {len(val_pairs)} val / {len(test_pairs)} test")
        pad_2d = PAD_2D_WEEK7
        train_ds = Week7SlicePairs2D(train_pairs, augment=True)
        val_ds = Week7SlicePairs2D(val_pairs, augment=False)
        test_ds = Week7SlicePairs2D(test_pairs, augment=False)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    else:
        # ---------- Gather paired files ----------
        all_pre = sorted(glob(os.path.join(PRE_DIR, "pre_*.nii.gz")))
        paired_pre = []
        for p in all_pre:
            q = pre_to_post(p)
            if q and os.path.exists(q):
                paired_pre.append(p)

        if not paired_pre:
            raise RuntimeError("No matched pre/post pairs found.")

        print(f"Total paired scans: {len(paired_pre)}")

        # ---------- Train / Test split (20 test scans) ----------
        trainval_pre, test_pre = train_test_split(
            paired_pre, test_size=20, random_state=SEED, shuffle=True
        )
        print(f"Train+Val: {len(trainval_pre)} | Test: {len(test_pre)}")

        # ---------- Train / Val datasets ----------
        ds_trainval = AxialSlicePairs(trainval_pre, target_size=TARGET_SIZE)

        idxs = np.arange(len(ds_trainval))
        tr_idx, va_idx = train_test_split(
            idxs, test_size=VAL_FRACTION, random_state=SEED, shuffle=True
        )

        ds_train = Subset(ds_trainval, tr_idx)
        ds_val = Subset(ds_trainval, va_idx)

        print(f"Train used: {len(ds_train)} | Val: {len(ds_val)}")

        train_loader = DataLoader(
            ds_train, batch_size=BATCH_SIZE, shuffle=True,
            num_workers=2, pin_memory=True
        )
        val_loader = DataLoader(
            ds_val, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=2, pin_memory=True
        )
        test_ds = AxialSlicePairs(test_pre, target_size=TARGET_SIZE)
        test_loader = DataLoader(
            test_ds, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=2, pin_memory=True
        )

    # ---------- Model / Scheduler / Optimizer ----------
    model = make_diffusion_unet().to(DEVICE)
    noise_scheduler = DDPMScheduler(num_train_timesteps=NUM_TRAIN_TIMESTEPS)

    # Load pretrained weights if available
    if PRETRAINED_CKPT is not None and os.path.exists(PRETRAINED_CKPT):
        print(f"[INFO] Loading pretrained weights from {PRETRAINED_CKPT}")
        state = torch.load(PRETRAINED_CKPT, map_location="cpu")
        model.load_state_dict(state, strict=False)
    else:
        print("[WARN] PRETRAINED_CKPT not found; training from scratch.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    history = {
        "train_loss": [],
        "val_L1": [],
        "val_MAE": [],
        "val_SSIM": [],
        "val_PSNR": [],
    }

    best_ssim = -1.0
    best_state = None

    print("\n=== Training conditional diffusion model (fine-tuning) ===")
    for ep in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, noise_scheduler, train_loader, optimizer, pad_2d=pad_2d)
        val_metrics = evaluate(model, noise_scheduler, val_loader, pad_2d=pad_2d)

        history["train_loss"].append(train_loss)
        for k in ["L1", "MAE", "SSIM", "PSNR"]:
            history[f"val_{k}"].append(val_metrics[k])

        print(
            f"[EP {ep:02d}/{EPOCHS}] "
            f"Train loss {train_loss:.4f} | "
            f"Val L1 {val_metrics['L1']:.4f}, "
            f"MAE {val_metrics['MAE']:.4f}, "
            f"SSIM {val_metrics['SSIM']:.4f}, "
            f"PSNR {val_metrics['PSNR']:.2f}"
        )

        if val_metrics["SSIM"] is not None and val_metrics["SSIM"] > best_ssim:
            best_ssim = val_metrics["SSIM"]
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

    # Save best model + history
    os.makedirs(RUN_DIR, exist_ok=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), MODEL_PATH)
    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nSaved best diffusion model to: {MODEL_PATH}")
    print(f"Training history saved to: {HISTORY_PATH}")

    # ---------- Inference on TEST set (Week7: 91x109) ----------
    print("\n=== Inference on TEST set (saving prediction triplets) ===")
    if not week7:
        test_ds = AxialSlicePairs(test_pre, target_size=TARGET_SIZE)
        test_loader = DataLoader(
            test_ds, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=2, pin_memory=True
        )

    model.eval()
    noise_scheduler.set_timesteps(NUM_INFER_STEPS)

    count = 0
    save_hw = WEEK7_ORIGINAL_HW if pad_2d is not None else None
    test_mae_list, test_ssim_list, test_psnr_list = [], [], []
    with torch.no_grad():
        for pre_t, post_t in test_loader:
            pre_t = pre_t.to(DEVICE)
            post_t = post_t.to(DEVICE)
            if pad_2d is not None:
                pre_t, post_t = _pad_2d_if_needed(pre_t, post_t, pad_2d)

            pred = sample_ddpm(model, noise_scheduler, pre_t)

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
                p = pred_np[b, 0]
                t = post_np[b, 0]
                m = metrics_in_brain_2d(p, t, data_range=1.0)
                test_mae_list.append(m["mae_mean"])
                test_ssim_list.append(m["ssim_mean"])
                test_psnr_list.append(m["psnr_mean"])
                np.save(os.path.join(PRED_DIR, f"in_{count:04d}.npy"), pre_np[b, 0])
                np.save(os.path.join(PRED_DIR, f"out_{count:04d}.npy"), p)
                np.save(os.path.join(PRED_DIR, f"tgt_{count:04d}.npy"), t)
                count += 1

    print(f"Saved {count} prediction triplets to {PRED_DIR}")
    if week7 and test_mae_list:
        test_metrics = {
            "mae_mean": float(np.mean(test_mae_list)),
            "ssim_mean": float(np.mean(test_ssim_list)),
            "psnr_mean": float(np.mean(test_psnr_list)),
        }
        phase2 = os.environ.get("WEEK7_REGION_WEIGHT", "").lower() in ("1", "true", "yes")
        test_json = os.path.join(RUN_DIR, "test_metrics_week7_phase2.json" if phase2 else "test_metrics_week7.json")
        with open(test_json, "w") as f:
            json.dump(test_metrics, f, indent=2)
        print(f"Week7 test metrics saved to {test_json}")
    print("\nDone.")


if __name__ == "__main__":
    main()
