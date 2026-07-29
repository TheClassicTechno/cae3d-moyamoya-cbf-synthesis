#!/usr/bin/env python3
"""
Export per-subject metrics for week7_unet2d (test set) for Bland-Altman and Wilcoxon.
Writes week8_per_subject_metrics/week7_unet2d_<subject_id>.json with model, subject_id, mae, ssim, psnr, pred_mean, target_mean.
Uses same test set and metrics as Week 7 (metrics_in_brain_2d; pred_mean/target_mean = mean intensity in 2D brain mask).
"""
import os
import sys
import json

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_REPO_ROOT, "pyproject.toml")):
    _parent = os.path.dirname(_REPO_ROOT)
    if _parent == _REPO_ROOT:
        raise RuntimeError("Could not locate repository root (pyproject.toml not found)")
    _REPO_ROOT = _parent

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from week7_data import get_week7_splits, Week7SlicePairs2D, _subject_id_from_path
from week7_preprocess import metrics_in_brain_2d, get_brain_mask_2d_slice

DATA_DIR = _REPO_ROOT
OUT_DIR = os.path.join(DATA_DIR, "week8_per_subject_metrics")
CKPT_PATH = os.path.join(DATA_DIR, "scripts", "week7_results", "week7_unet2d_best.pt")
TARGET_2D = (96, 112)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _pad_to_target(pre_t, post_t, target_hw):
    import torch.nn.functional as F
    th, tw = target_hw
    _, _, h, w = pre_t.shape
    if h < th or w < tw:
        pd = (0, max(0, tw - w), 0, max(0, th - h))
        pre_t = F.pad(pre_t, pd, mode="constant", value=0)
        post_t = F.pad(post_t, pd, mode="constant", value=0)
    return pre_t[:, :, :th, :tw], post_t[:, :, :th, :tw]


def main():
    from monai.networks.nets import UNet

    def make_unet_2d():
        return UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            channels=(16, 32, 64, 128),
            strides=(2, 2, 2),
            num_res_units=2,
            act=("LeakyReLU", {"inplace": True}),
            norm="INSTANCE",
            dropout=0.0,
        )

    os.makedirs(OUT_DIR, exist_ok=True)
    _, _, test_pairs = get_week7_splits()
    test_ds = Week7SlicePairs2D(test_pairs, augment=False)

    if not os.path.isfile(CKPT_PATH):
        print("Skip week7_unet2d: checkpoint not found", CKPT_PATH)
        return

    model = make_unet_2d().to(DEVICE)
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt.get("model", ckpt))
    model.eval()

    mask_2d = get_brain_mask_2d_slice()
    if mask_2d.shape != (91, 109):
        from scipy.ndimage import zoom as _zoom
        factors = [91 / mask_2d.shape[0], 109 / mask_2d.shape[1]]
        mask_2d = _zoom(mask_2d.astype(np.float32), factors, order=0)
        mask_2d = (mask_2d > 0.5).astype(np.float32)

    n = 0
    with torch.no_grad():
        for i in range(len(test_pairs)):
            pre_path, post_path = test_pairs[i]
            sid = _subject_id_from_path(pre_path)
            pre_t, post_t = test_ds[i]
            pre_t = pre_t.unsqueeze(0).to(DEVICE)
            post_t = post_t.unsqueeze(0).to(DEVICE)
            pre_t, post_t = _pad_to_target(pre_t, post_t, TARGET_2D)
            pred_t = model(pre_t)
            # Crop back to 91x109 for metric (match middle slice)
            p = pred_t[0, 0].cpu().numpy()
            t = post_t[0, 0].cpu().numpy()
            if p.shape != (91, 109):
                p = p[:91, :109] if p.shape[0] >= 91 and p.shape[1] >= 109 else p
                t = t[:91, :109] if t.shape[0] >= 91 and t.shape[1] >= 109 else t
            if mask_2d.shape != p.shape:
                from scipy.ndimage import zoom as _z
                factors = [p.shape[0] / mask_2d.shape[0], p.shape[1] / mask_2d.shape[1]]
                m = _z(mask_2d.astype(np.float32), factors, order=0)
            else:
                m = mask_2d
            m = (m > 0.5).astype(np.float32)
            met = metrics_in_brain_2d(p, t, mask_2d=m, data_range=1.0)
            pred_mean = float((p * m).sum() / (m.sum() + 1e-8))
            target_mean = float((t * m).sum() / (m.sum() + 1e-8))
            out = {
                "model": "week7_unet2d",
                "subject_id": sid,
                "mae": float(met["mae_mean"]),
                "ssim": float(met["ssim_mean"]),
                "psnr": float(met["psnr_mean"]),
                "pred_mean": pred_mean,
                "target_mean": target_mean,
            }
            out_path = os.path.join(OUT_DIR, f"week7_unet2d_{sid}.json")
            with open(out_path, "w") as f:
                json.dump(out, f, indent=0)
            n += 1
    print("Wrote", n, "per-subject JSONs for week7_unet2d to", OUT_DIR)


if __name__ == "__main__":
    main()
