#!/usr/bin/env python3
"""
Train FNO 3D with full spectral (Fourier) layers on Week7 data.
Uses same data/split/preprocessing as Week7/Week8: get_week7_splits(), Week7VolumePairsFNO
(91x109x91 brain-masked, pad 96x112x96), brain-mask + optional region-weighted loss.
Set SEED=42|123|456 for three-seed runs; run_week8_phase2_all_seeds.sh includes FNO_3D_Spectral.
Saves fno_3d_spectral_week7_best.pt and fno_3d_spectral_week7_results.json (or phase2).
Higher SSIM: set WEEK7_FNO_SSIM_WEIGHT=lambda (e.g. 0.5) to add lambda*(1-SSIM) to loss; 0=baseline. Compare test SSIM/MAE after one run.
"""
import os
import sys
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_REPO_ROOT, "pyproject.toml")):
    _parent = os.path.dirname(_REPO_ROOT)
    if _parent == _REPO_ROOT:
        raise RuntimeError("Could not locate repository root (pyproject.toml not found)")
    _REPO_ROOT = _parent

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fno_3d_cvr_spectral import SpectralFNO3D, Week7VolumePairsFNO, WEEK7_PAD_SHAPE, WEEK7_ORIGINAL_SHAPE
from fno_3d_finetune_slice_brain import composite_loss_slice_brain

sys.path.insert(0, os.path.join(_REPO_ROOT, "scripts"))
from week7_data import get_week7_splits
from week7_preprocess import get_region_weight_mask_for_shape

EPOCHS = int(os.environ.get("FNO_EPOCHS", "40"))
LR = 2e-4
# Optional: WEEK7_FNO_SSIM_WEIGHT=lambda_ssim adds lambda_ssim * (1 - SSIM) to loss; 0 = baseline (no SSIM term).
WEEK7_FNO_SSIM_WEIGHT = float(os.environ.get("WEEK7_FNO_SSIM_WEIGHT", "0"))


def main():
    seed = int(os.environ.get("SEED", "1337"))
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = os.path.dirname(os.path.abspath(__file__))

    train_pairs, val_pairs, test_pairs = get_week7_splits()
    print("Week7: %d train / %d val / %d test (FNO 3D spectral)" % (len(train_pairs), len(val_pairs), len(test_pairs)))
    region_weight_t = None
    if os.environ.get("WEEK7_REGION_WEIGHT", "").lower() in ("1", "true", "yes"):
        mask_np = get_region_weight_mask_for_shape(WEEK7_PAD_SHAPE, vascular_weight=1.5)
        region_weight_t = torch.from_numpy(mask_np).float().unsqueeze(0).unsqueeze(0)
    train_ds = Week7VolumePairsFNO(train_pairs)
    val_ds = Week7VolumePairsFNO(val_pairs)
    test_ds = Week7VolumePairsFNO(test_pairs)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

    H, W, D = WEEK7_PAD_SHAPE
    slice_weight_arr = np.ones(D, dtype=np.float32)
    in_ch, width, modes = 4, 64, 12
    model = SpectralFNO3D(in_ch=in_ch, out_ch=1, modes=modes, width=width).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=2e-6)

    criterion_ssim = None
    if WEEK7_FNO_SSIM_WEIGHT > 0:
        from monai.losses import SSIMLoss
        criterion_ssim = SSIMLoss(spatial_dims=3)
    print("Training FNO 3D spectral (Week7, brain-mask loss)%s..." % (" + SSIM weight=%.2f" % WEEK7_FNO_SSIM_WEIGHT if WEEK7_FNO_SSIM_WEIGHT > 0 else ""))
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for pre, post in train_loader:
            pre, post = pre.to(device), post.to(device)
            pred = model(pre)
            loss = composite_loss_slice_brain(
                pred, post, pre[:, :1], slice_weight_arr, use_brain_mask=True, loss_type="l1mse",
                region_weight_vol=region_weight_t,
            )
            if criterion_ssim is not None:
                loss = loss + WEEK7_FNO_SSIM_WEIGHT * criterion_ssim(pred, post)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        scheduler.step()

        model.eval()
        val_mae, val_ssim, val_psnr = [], [], []
        with torch.no_grad():
            for pre, post in val_loader:
                pre, post = pre.to(device), post.to(device)
                pred = model(pre)
                pred = torch.clamp(pred, 0, 1)
                p = pred[0, 0].cpu().numpy()
                t = post[0, 0].cpu().numpy()
                val_mae.append(np.abs(p - t).mean())
                val_ssim.append(ssim(t, p, data_range=1.0))
                val_psnr.append(psnr(t, p, data_range=1.0))
        print("Epoch %2d/%d Train: %.4f | Val MAE: %.4f SSIM: %.4f PSNR: %.2f dB" % (
            epoch, EPOCHS, train_loss, np.mean(val_mae), np.mean(val_ssim), np.mean(val_psnr)))

    out_path = os.path.join(base, "fno_3d_spectral_week7_best.pt")
    torch.save(model.state_dict(), out_path)
    print("Saved %s" % out_path)

    from week7_preprocess import metrics_in_brain
    model.eval()
    mae_list, ssim_list, psnr_list = [], [], []
    oh, ow, od = WEEK7_ORIGINAL_SHAPE
    with torch.no_grad():
        for pre, post in test_loader:
            pre, post = pre.to(device), post.to(device)
            pred = model(pre)
            pred = torch.clamp(pred, 0, 1)
            pred_np = pred[0, 0].cpu().numpy()[:oh, :ow, :od]
            post_np = post[0, 0].cpu().numpy()[:oh, :ow, :od]
            m = metrics_in_brain(pred_np, post_np, data_range=1.0)
            mae_list.append(m["mae_mean"])
            ssim_list.append(m["ssim_mean"])
            psnr_list.append(m["psnr_mean"])
    results = {
        "mae_mean": float(np.mean(mae_list)), "mae_std": float(np.std(mae_list)),
        "ssim_mean": float(np.mean(ssim_list)), "ssim_std": float(np.std(ssim_list)),
        "psnr_mean": float(np.mean(psnr_list)), "psnr_std": float(np.std(psnr_list)),
        "model": "SpectralFNO3D", "week7": True, "n_test": len(mae_list),
    }
    phase2 = os.environ.get("WEEK7_REGION_WEIGHT", "").lower() in ("1", "true", "yes")
    if phase2:
        results_path = os.path.join(base, "fno_3d_spectral_week7_phase2_results.json")
    else:
        results_path = out_path.replace(".pt", "_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print("Test MAE %.4f SSIM %.4f PSNR %.2f dB" % (results["mae_mean"], results["ssim_mean"], results["psnr_mean"]))
    print("Saved %s" % results_path)


if __name__ == "__main__":
    main()
