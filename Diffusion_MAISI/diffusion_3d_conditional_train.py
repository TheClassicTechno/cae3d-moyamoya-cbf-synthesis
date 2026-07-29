import os
import sys
import json
import time
import logging
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import matplotlib.pyplot as plt

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_REPO_ROOT, "pyproject.toml")):
    _parent = os.path.dirname(_REPO_ROOT)
    if _parent == _REPO_ROOT:
        raise RuntimeError("Could not locate repository root (pyproject.toml not found)")
    _REPO_ROOT = _parent

from monai.networks.nets import DiffusionModelUNet, UNet
from monai.networks.schedulers import DDIMScheduler

# RAN THIS SCRIPT

# ============================================================
# ---------------------- Logging Setup ------------------------
# ============================================================

def setup_logger(save_dir):
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, "train.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("diff3d")


# ============================================================
# ---------------------- Dataset Class ------------------------
# ============================================================

def load_volume_week7(nii_path, pad_shape=(96, 112, 96)):
    """Week7: 91×109×91 brain mask + minmax, then pad to pad_shape."""
    import sys
    sys.path.insert(0, os.path.join(_REPO_ROOT, 'scripts'))
    from week7_preprocess import load_volume, TARGET_SHAPE
    vol = load_volume(nii_path, target_shape=TARGET_SHAPE, apply_mask=True, minmax=True)
    if vol.shape != pad_shape:
        out = np.zeros(pad_shape, dtype=vol.dtype)
        sh = [min(vol.shape[i], pad_shape[i]) for i in range(3)]
        out[:sh[0], :sh[1], :sh[2]] = vol[:sh[0], :sh[1], :sh[2]]
        return out.astype(np.float32)
    return vol.astype(np.float32)


class Patch3DDataset(Dataset):
    def __init__(self, pre_files, post_files, patch_size=64, num_patches=12, load_fn=None):
        self.pre_files = pre_files
        self.post_files = post_files
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.load_fn = load_fn

    def __len__(self):
        return len(self.pre_files)

    def extract_random_patches(self, vol):
        """Extract N random patches of size 64³."""
        D, H, W = vol.shape
        ps = self.patch_size
        patches = []

        for _ in range(self.num_patches):
            z = random.randint(0, max(0, D - ps))
            y = random.randint(0, max(0, H - ps))
            x = random.randint(0, max(0, W - ps))
            patch = vol[z:z+ps, y:y+ps, x:x+ps]
            patches.append(patch)

        return np.stack(patches)  # [N, 64, 64, 64]

    def __getitem__(self, idx):
        if self.load_fn is not None:
            pre = self.load_fn(self.pre_files[idx])
            post = self.load_fn(self.post_files[idx])
        else:
            pre = nib.load(self.pre_files[idx]).get_fdata().astype(np.float32)
            post = nib.load(self.post_files[idx]).get_fdata().astype(np.float32)

        # normalize global volumes
        pre = (pre - pre.mean()) / (pre.std() + 1e-6)
        post = (post - post.mean()) / (post.std() + 1e-6)

        # extract 64³ patches
        pre_patches = self.extract_random_patches(pre)     # [N,64³]
        post_patches = self.extract_random_patches(post)

        # add channels
        pre_patches = pre_patches[:, None]     # [N,1,64³]
        post_patches = post_patches[:, None]

        return torch.tensor(pre_patches), torch.tensor(post_patches)


# ============================================================
# ---------------------- Model Setup --------------------------
# ============================================================

def build_model(device, use_simple_unet=False):
    """Build noise predictor. use_simple_unet=True avoids MONAI DiffusionModelUNet (forward/temb issues on some versions)."""
    if use_simple_unet:
        model = UNet(
            spatial_dims=3,
            in_channels=2,
            out_channels=1,
            channels=(32, 64, 128, 256),
            strides=(2, 2, 2),
            num_res_units=2,
        ).to(device)
        # Forward for training: (x, t) -> model(x) only; we ignore t for this UNet (no time embedding).
        class _NoisePredictor(torch.nn.Module):
            def __init__(self, unet):
                super().__init__()
                self.unet = unet
            def forward(self, x, timesteps=None):
                return self.unet(x)
        model = _NoisePredictor(model)
    else:
        model = DiffusionModelUNet(
            spatial_dims=3,
            in_channels=2,
            out_channels=1,
            num_res_blocks=2,
            attention_levels=(False, False, False, False),
            norm_num_groups=8,
            resblock_updown=True,
        ).to(device)
    noise_scheduler = DDIMScheduler(num_train_timesteps=1000)
    return model, noise_scheduler


# ============================================================
# ---------------------- Training Loop ------------------------
# ============================================================

def train_patch_epoch(model, scheduler, loader, optimizer, device):
    model.train()
    total_loss = 0
    mse = nn.MSELoss()

    for pre_patches, post_patches in loader:
        B, N, C, D, H, W = post_patches.shape
        pre_patches = pre_patches.to(device)
        post_patches = post_patches.to(device)

        # flatten batch: treat patches independently
        pre_patches = pre_patches.view(B*N, 1, D, H, W)
        post_patches = post_patches.view(B*N, 1, D, H, W)

        optimizer.zero_grad()

        # diffusion timestep t
        t = torch.randint(0, scheduler.num_train_timesteps, (B*N,), device=device).long()

        # add noise to target (post)
        noise = torch.randn_like(post_patches)
        noisy = scheduler.add_noise(post_patches, noise, t)

        model_in = torch.cat([noisy, pre_patches], dim=1)   # [BN,2,64³]

        noise_pred = model(model_in, t)
        loss = mse(noise_pred, noise)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate(model, scheduler, loader, device):
    """Compute SSIM-like structural metric (simple correlation proxy)."""
    model.eval()
    total_corr = 0

    with torch.no_grad():
        for pre_patches, post_patches in loader:
            B, N, _, D, H, W = post_patches.shape
            pre_patches = pre_patches.to(device)
            post_patches = post_patches.to(device)

            pre_patches = pre_patches.view(B*N, 1, D, H, W)
            post_patches = post_patches.view(B*N, 1, D, H, W)

            t = torch.randint(0, scheduler.num_train_timesteps, (B*N,), device=device).long()
            noise = torch.randn_like(post_patches)
            noisy = scheduler.add_noise(post_patches, noise, t)
            model_in = torch.cat([noisy, pre_patches], dim=1)

            noise_pred = model(model_in, t)
            recon = noise - noise_pred  # crude reconstruction proxy

            # correlation = proxy for similarity
            corr = torch.mean(recon * post_patches).item()
            total_corr += corr

    return total_corr / len(loader)


# ============================================================
# ---------------------- Main Training ------------------------
# ============================================================

def main():
    use_week7 = "--week7" in sys.argv or os.environ.get("WEEK7") == "1"
    if use_week7:
        sys.path.insert(0, os.path.join(_REPO_ROOT, 'scripts'))
        from week7_data import get_week7_splits
        train_pairs, val_pairs, test_pairs = get_week7_splits()
        train_pre = [p[0] for p in train_pairs]
        train_post = [p[1] for p in train_pairs]
        val_pre = [p[0] for p in val_pairs]
        val_post = [p[1] for p in val_pairs]
        # Pad to 128³ so 64³ patches match MONAI DiffusionModelUNet spatial expectations
        load_fn = lambda p: load_volume_week7(p, pad_shape=(128, 128, 128))
        RUN_DIR = os.path.join(_REPO_ROOT, "Diffusion_MAISI/run3d_week7")
    else:
        PRE_DIR = os.environ.get("MOYAMOYA_LEGACY_PRE_DIR", "pre_scans")
        POST_DIR = os.environ.get("MOYAMOYA_LEGACY_POST_DIR", "post_scans")
        pre_files = sorted([os.path.join(PRE_DIR, f) for f in os.listdir(PRE_DIR) if f.endswith(".nii.gz")])
        post_files = sorted([os.path.join(POST_DIR, f) for f in os.listdir(POST_DIR) if f.endswith(".nii.gz")])
        assert len(pre_files) == len(post_files)
        N = len(pre_files)
        idxs = list(range(N))
        random.shuffle(idxs)
        test_idx, val_idx, train_idx = idxs[:20], idxs[20:40], idxs[40:]
        train_pre = [pre_files[i] for i in train_idx]
        train_post = [post_files[i] for i in train_idx]
        val_pre = [pre_files[i] for i in val_idx]
        val_post = [post_files[i] for i in val_idx]
        load_fn = None
        RUN_DIR = os.environ.get("MOYAMOYA_LEGACY_RUN_DIR", "Diffusion_MAISI/run3d_patch64")

    os.makedirs(RUN_DIR, exist_ok=True)
    logger = setup_logger(RUN_DIR)
    logger.info("==== 3D PATCH DIFFUSION (pre→post)" + (" [Week7]" if use_week7 else "") + " ====")
    logger.info(f"Train {len(train_pre)} | Val {len(val_pre)}")

    patch_size = 64
    patches_per_volume = 12

    train_ds = Patch3DDataset(train_pre, train_post, patch_size, patches_per_volume, load_fn=load_fn)
    val_ds = Patch3DDataset(val_pre, val_post, patch_size, patches_per_volume, load_fn=load_fn)

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4)

    device = torch.device("cuda")
    use_simple_unet = use_week7  # Avoid MONAI DiffusionModelUNet forward/temb issues on Week7 64^3 patches
    model, scheduler = build_model(device, use_simple_unet=use_simple_unet)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # -----------------------------------------
    # Training
    # -----------------------------------------
    EPOCHS = 200
    train_losses = []
    val_corrs = []
    best_corr = -999
    best_path = os.path.join(RUN_DIR, "best_model.pt")

    for epoch in range(1, EPOCHS+1):
        t0 = time.time()

        train_loss = train_patch_epoch(model, scheduler, train_loader, optimizer, device)
        val_corr = validate(model, scheduler, val_loader, device)

        train_losses.append(train_loss)
        val_corrs.append(val_corr)

        logger.info(f"[Epoch {epoch}/{EPOCHS}] Train Loss={train_loss:.4f} | ValCorr={val_corr:.4f} | time={time.time()-t0:.1f}s")

        # Save best
        if val_corr > best_corr:
            best_corr = val_corr
            torch.save(model.state_dict(), best_path)
            logger.info("   ✓ Saved BEST checkpoint")

        # Save every epoch
        torch.save(model.state_dict(), os.path.join(RUN_DIR, f"model_epoch{epoch}.pt"))

    # -----------------------------------------
    # Save learning curves
    # -----------------------------------------
    plt.figure(figsize=(8,4))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_corrs, label="Val Corr")
    plt.legend()
    plt.title("Training Curves")
    plt.savefig(os.path.join(RUN_DIR, "training_curves.png"))
    plt.close()

    logger.info("=== Training Completed ===")

    # Week7: run test set eval (center 64^3 patch per volume), save maisi_week7_results.json
    if use_week7 and os.path.isfile(best_path):
        logger.info("=== Week7 test set evaluation (center patch) ===")
        sys.path.insert(0, os.path.join(_REPO_ROOT, "scripts"))
        from week7_data import get_week7_splits
        from week7_preprocess import metrics_in_brain
        _, _, test_pairs = get_week7_splits()
        model.load_state_dict(torch.load(best_path, map_location=device))
        model.eval()
        if hasattr(scheduler, "set_timesteps"):
            scheduler.set_timesteps(50, device=device)
        patch_size = 64
        mae_list, ssim_list, psnr_list = [], [], []
        with torch.no_grad():
            for pre_path, post_path in test_pairs:
                pre = load_volume_week7(pre_path, pad_shape=(128, 128, 128))
                post = load_volume_week7(post_path, pad_shape=(128, 128, 128))
                pre = (pre - pre.mean()) / (pre.std() + 1e-6)
                post = (post - post.mean()) / (post.std() + 1e-6)
                c = 32
                pre_p = pre[c:c+patch_size, c:c+patch_size, c:c+patch_size]
                post_p = post[c:c+patch_size, c:c+patch_size, c:c+patch_size]
                pre_t = torch.from_numpy(pre_p).float().unsqueeze(0).unsqueeze(0).to(device)
                x = torch.randn(1, 1, patch_size, patch_size, patch_size, device=device)
                for t in scheduler.timesteps:
                    t_b = t.unsqueeze(0).expand(1)
                    model_in = torch.cat([x, pre_t], dim=1)
                    pred_noise = model(model_in, t_b)
                    x, _ = scheduler.step(pred_noise, int(t.item()), x)
                pred_p = x[0, 0].cpu().numpy()
                post_p_np = post_p
                m = metrics_in_brain(pred_p, post_p_np, data_range=1.0)
                mae_list.append(m["mae_mean"])
                ssim_list.append(m["ssim_mean"])
                psnr_list.append(m["psnr_mean"])
        out = {
            "mae_mean": float(np.mean(mae_list)),
            "ssim_mean": float(np.mean(ssim_list)),
            "psnr_mean": float(np.mean(psnr_list)),
        }
        use_phase2 = os.environ.get("WEEK7_REGION_WEIGHT", "").lower() in ("1", "true", "yes")
        out_path = os.path.join(RUN_DIR, "maisi_week7_phase2_results.json" if use_phase2 else "maisi_week7_results.json")
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        logger.info(f"Week7 test metrics saved to {out_path}: MAE={out['mae_mean']:.4f} SSIM={out['ssim_mean']:.4f} PSNR={out['psnr_mean']:.2f}")


if __name__ == "__main__":
    main()
