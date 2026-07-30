#!/usr/bin/env python3
"""
Full 3-seed, 32-subject re-evaluation of CAE3D (week7_unet3d) from the original
per-seed checkpoints (week7_unet3d_seed{42,123,456}_best.pt), confirmed sane
(no PSNR bug) on a smoke test. Produces:
  - per-seed cohort means (for Table 2 three-seed mean+-std)
  - per-subject metrics averaged across the 3 seeds, then bootstrapped over
    subjects (for Table 1 cohort MAE/SSIM/PSNR CI, Bias/LoA, R^2)

Read-only w.r.t. checkpoints. Writes one consolidated JSON under
week9_stats/eval_runs/.
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
    get_brain_mask_for_shape,
)
from monai.networks.nets import UNet  # noqa: E402

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_REPO_ROOT, "pyproject.toml")):
    _parent = os.path.dirname(_REPO_ROOT)
    if _parent == _REPO_ROOT:
        break
    _REPO_ROOT = _parent

ROOT = _REPO_ROOT
CKPT_DIR = os.path.join(ROOT, "scripts", "week7_results")
SEEDS = [42, 123, 456]
TARGET_3D_PAD = (96, 112, 96)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
B_BOOTSTRAP = 2000
RNG_SEED = 12345


def make_unet_3d():
    return UNet(
        spatial_dims=3, in_channels=1, out_channels=1, channels=(16, 32, 64, 128),
        strides=(2, 2, 2), num_res_units=2, act=("LeakyReLU", {"inplace": True}),
        norm="INSTANCE", dropout=0.0,
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
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def run_one_checkpoint(ckpt_path, test_triples):
    model = make_unet_3d().to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt)
    model.eval()
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
            mask = get_brain_mask_for_shape(p.shape) > 0.5
            rows.append({
                "subject_id": sid,
                "mae": m["mae_mean"], "ssim": m["ssim_mean"], "psnr": m["psnr_mean"],
                "pred_mean": float(p[mask].mean()), "target_mean": float(t[mask].mean()),
            })
    return rows


def main():
    test_triples = get_pre_post_pairs_with_subject_id("test")
    n = len(test_triples)
    print(f"{n} test subjects.")

    per_seed = {}
    for seed in SEEDS:
        ckpt_path = os.path.join(CKPT_DIR, f"week7_unet3d_seed{seed}_best.pt")
        print(f"\n--- seed {seed}: {ckpt_path} ---")
        rows = run_one_checkpoint(ckpt_path, test_triples)
        mae = np.array([r["mae"] for r in rows])
        ssim = np.array([r["ssim"] for r in rows])
        psnr = np.array([r["psnr"] for r in rows])
        print(f"seed {seed}: MAE={mae.mean():.4f} SSIM={ssim.mean():.4f} PSNR={psnr.mean():.2f}")
        per_seed[seed] = rows

    # Table 2: three-seed cohort-mean mean+-std
    seed_cohort_mae = [np.mean([r["mae"] for r in per_seed[s]]) for s in SEEDS]
    seed_cohort_ssim = [np.mean([r["ssim"] for r in per_seed[s]]) for s in SEEDS]
    seed_cohort_psnr = [np.mean([r["psnr"] for r in per_seed[s]]) for s in SEEDS]

    table2 = {
        "mae_mean": float(np.mean(seed_cohort_mae)), "mae_std": float(np.std(seed_cohort_mae, ddof=1)),
        "ssim_mean": float(np.mean(seed_cohort_ssim)), "ssim_std": float(np.std(seed_cohort_ssim, ddof=1)),
        "psnr_mean": float(np.mean(seed_cohort_psnr)), "psnr_std": float(np.std(seed_cohort_psnr, ddof=1)),
        "per_seed_cohort_mae": seed_cohort_mae,
        "per_seed_cohort_ssim": seed_cohort_ssim,
        "per_seed_cohort_psnr": seed_cohort_psnr,
    }

    # Table 1: average each subject's metrics/means across the 3 seeds, then bootstrap over subjects
    by_subject = {}
    for s in SEEDS:
        for r in per_seed[s]:
            by_subject.setdefault(r["subject_id"], []).append(r)

    subj_mae, subj_ssim, subj_psnr, subj_pred_mean, subj_target_mean = [], [], [], [], []
    for sid, recs in by_subject.items():
        subj_mae.append(np.mean([r["mae"] for r in recs]))
        subj_ssim.append(np.mean([r["ssim"] for r in recs]))
        subj_psnr.append(np.mean([r["psnr"] for r in recs]))
        subj_pred_mean.append(np.mean([r["pred_mean"] for r in recs]))
        subj_target_mean.append(np.mean([r["target_mean"] for r in recs]))

    subj_mae = np.array(subj_mae); subj_ssim = np.array(subj_ssim); subj_psnr = np.array(subj_psnr)
    subj_pred_mean = np.array(subj_pred_mean); subj_target_mean = np.array(subj_target_mean)
    diff = subj_pred_mean - subj_target_mean

    bias = float(diff.mean())
    std_diff = float(diff.std(ddof=1))
    loa_low = bias - 1.96 * std_diff
    loa_high = bias + 1.96 * std_diff
    r2 = r2_score(subj_target_mean, subj_pred_mean)

    table1 = {
        "mae_mean": float(subj_mae.mean()), "mae_ci95": bootstrap_ci(subj_mae),
        "ssim_mean": float(subj_ssim.mean()), "ssim_ci95": bootstrap_ci(subj_ssim),
        "psnr_mean": float(subj_psnr.mean()), "psnr_ci95": bootstrap_ci(subj_psnr),
        "bias": bias, "bias_ci95": bootstrap_ci(diff),
        "std_diff": std_diff, "loa_low": loa_low, "loa_high": loa_high,
        "r2": r2,
    }

    out = {
        "model": "week7_unet3d (CAE3D ours), 3-seed re-eval from original seed checkpoints",
        "checkpoints": {s: os.path.join(CKPT_DIR, f"week7_unet3d_seed{s}_best.pt") for s in SEEDS},
        "n_subjects": n,
        "table2_three_seed": table2,
        "table1_pooled": table1,
        "per_seed_per_subject": {str(s): per_seed[s] for s in SEEDS},
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "bootstrap_B": B_BOOTSTRAP,
    }

    out_dir = os.path.join(ROOT, "week9_stats", "eval_runs")
    os.makedirs(out_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%SZ", time.gmtime())
    out_path = os.path.join(out_dir, f"cae3d_three_seed_full_recheck_{stamp}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print("\n=== Table 2 (three-seed mean+-std) ===")
    print(f"MAE  = {table2['mae_mean']:.4f} +- {table2['mae_std']:.4f}")
    print(f"SSIM = {table2['ssim_mean']:.4f} +- {table2['ssim_std']:.4f}")
    print(f"PSNR = {table2['psnr_mean']:.2f} +- {table2['psnr_std']:.2f}")
    print("\n=== Table 1 (pooled across seeds, bootstrap over 32 subjects) ===")
    print(f"MAE  = {table1['mae_mean']:.4f}  95% CI {table1['mae_ci95']}")
    print(f"SSIM = {table1['ssim_mean']:.4f}  95% CI {table1['ssim_ci95']}")
    print(f"PSNR = {table1['psnr_mean']:.2f}  95% CI {table1['psnr_ci95']}")
    print(f"Bias = {table1['bias']:.4f}  95% CI {table1['bias_ci95']}")
    print(f"LoA  = [{table1['loa_low']:.4f}, {table1['loa_high']:.4f}]")
    print(f"R2   = {table1['r2']:.4f}")
    print("\nWrote", out_path)


if __name__ == "__main__":
    main()
