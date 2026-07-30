#!/usr/bin/env python3
"""
One-off verification: recompute per-subject Bland-Altman bias/LoA and R^2 for
CAE3D (week7_unet3d) directly from the checkpoint used for the promoted
MAE/SSIM/PSNR numbers (scripts/week7_results/week7_unet3d_best.pt), matching
unet3d_test_brain_metrics_post_promotion.json.

Motivation: the paper's Table 1 CAE3D row combines MAE/SSIM/PSNR from the
post-promotion checkpoint eval with Bias/LoA/R^2 that traced back to an older,
buggy per-subject export (PSNR -2.53 dB). This script recomputes both from the
same checkpoint and the same 32-subject held-out test set, so everything in
the row comes from one evaluation run.

Read-only with respect to the checkpoint: loads it, runs inference, writes a
new JSON under week9_stats/eval_runs/. Does not modify any existing file.
"""
import os
import sys
import json
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from week7_data import load_pre_post_pair  # noqa: E402
from week7_preprocess import (  # noqa: E402
    TARGET_SHAPE,
    metrics_in_brain,
    get_pre_post_pairs_with_subject_id,
)
from monai.networks.nets import UNet  # noqa: E402

ROOT = "/data1/julih"
CKPT_PATH = os.path.join(ROOT, "scripts", "week7_results", "week7_unet3d_best.pt")
TARGET_3D_PAD = (96, 112, 96)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
B_BOOTSTRAP = 2000
RNG_SEED = 12345


def make_unet_3d():
    return UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128),
        strides=(2, 2, 2),
        num_res_units=2,
        act=("LeakyReLU", {"inplace": True}),
        norm="INSTANCE",
        dropout=0.0,
    )


def pad_to(vol_t, target_shape):
    import torch.nn.functional as F
    _, _, h, w, d = vol_t.shape
    th, tw, td = target_shape
    if h < th or w < tw or d < td:
        pd = (0, max(0, td - d), 0, max(0, tw - w), 0, max(0, th - h))
        vol_t = F.pad(vol_t, pd, mode="constant", value=0)
    return vol_t[:, :, :th, :tw, :td]


def bootstrap_ci(values, n_boot=B_BOOTSTRAP, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    n = len(values)
    means = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[b] = values[idx].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot <= 0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def main():
    if not os.path.isfile(CKPT_PATH):
        print("Checkpoint not found:", CKPT_PATH)
        sys.exit(1)

    model = make_unet_3d().to(DEVICE)
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt)
    model.eval()

    test_triples = get_pre_post_pairs_with_subject_id("test")
    print(f"Loaded {len(test_triples)} test subjects from combined split.")

    rows = []
    with torch.no_grad():
        for sid, pre_path, post_path in test_triples:
            pre, post = load_pre_post_pair(pre_path, post_path, target_shape=TARGET_SHAPE)
            pre_t = torch.from_numpy(pre).unsqueeze(0).unsqueeze(0).float()
            post_t = torch.from_numpy(post).unsqueeze(0).unsqueeze(0).float()
            pre_t = pad_to(pre_t, TARGET_3D_PAD).to(DEVICE)
            post_t = pad_to(post_t, TARGET_3D_PAD).to(DEVICE)
            pred_t = model(pre_t)
            p = pred_t[0, 0].cpu().numpy()
            t = post_t[0, 0].cpu().numpy()
            m = metrics_in_brain(p, t, data_range=1.0)
            from week7_preprocess import get_brain_mask_for_shape
            mask = get_brain_mask_for_shape(p.shape) > 0.5
            pred_mean = float(p[mask].mean())
            target_mean = float(t[mask].mean())
            rows.append({
                "subject_id": sid,
                "mae": m["mae_mean"],
                "ssim": m["ssim_mean"],
                "psnr": m["psnr_mean"],
                "pred_mean": pred_mean,
                "target_mean": target_mean,
            })
            print(f"  {sid}: mae={m['mae_mean']:.4f} ssim={m['ssim_mean']:.4f} psnr={m['psnr_mean']:.2f} "
                  f"pred_mean={pred_mean:.4f} target_mean={target_mean:.4f}")

    n = len(rows)
    mae = np.array([r["mae"] for r in rows])
    ssim = np.array([r["ssim"] for r in rows])
    psnr = np.array([r["psnr"] for r in rows])
    pred_mean = np.array([r["pred_mean"] for r in rows])
    target_mean = np.array([r["target_mean"] for r in rows])
    diff = pred_mean - target_mean

    bias = float(diff.mean())
    std_diff = float(diff.std(ddof=1))
    loa_low = bias - 1.96 * std_diff
    loa_high = bias + 1.96 * std_diff
    r2 = r2_score(target_mean, pred_mean)

    bias_ci = bootstrap_ci(diff)
    mae_ci = bootstrap_ci(mae)
    ssim_ci = bootstrap_ci(ssim)
    psnr_ci = bootstrap_ci(psnr)

    summary = {
        "model": "week7_unet3d (CAE3D ours)",
        "checkpoint": CKPT_PATH,
        "n_subjects": n,
        "mae_mean": float(mae.mean()),
        "mae_ci95": mae_ci,
        "ssim_mean": float(ssim.mean()),
        "ssim_ci95": ssim_ci,
        "psnr_mean": float(psnr.mean()),
        "psnr_ci95": psnr_ci,
        "bias": bias,
        "bias_ci95": bias_ci,
        "std_diff": std_diff,
        "loa_low": loa_low,
        "loa_high": loa_high,
        "r2": r2,
        "per_subject": rows,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "bootstrap_B": B_BOOTSTRAP,
    }

    out_dir = os.path.join(ROOT, "week9_stats", "eval_runs")
    os.makedirs(out_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%SZ", time.gmtime())
    out_path = os.path.join(out_dir, f"cae3d_promoted_ckpt_bias_r2_recheck_{stamp}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Summary (promoted checkpoint, n=%d) ===" % n)
    print(f"MAE  = {mae.mean():.4f}  95% CI {mae_ci}")
    print(f"SSIM = {ssim.mean():.4f}  95% CI {ssim_ci}")
    print(f"PSNR = {psnr.mean():.2f}  95% CI {psnr_ci}")
    print(f"Bias = {bias:.4f}  95% CI {bias_ci}")
    print(f"LoA  = [{loa_low:.4f}, {loa_high:.4f}]")
    print(f"R2   = {r2:.4f}")
    print("\nWrote", out_path)


if __name__ == "__main__":
    main()
