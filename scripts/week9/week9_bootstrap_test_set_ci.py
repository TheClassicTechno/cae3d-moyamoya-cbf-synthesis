#!/usr/bin/env python3
"""
Test-set bootstrap 95% CI for mean MAE, SSIM, PSNR (percentile method).

For each model we have per-subject metrics (32 test subjects) in week8_per_subject_metrics.
We draw B bootstrap samples of size n (with replacement) from those n subjects,
compute the mean MAE (and SSIM, PSNR) in each bootstrap sample, then take the
2.5th and 97.5th percentiles of those B means as the 95% CI. This quantifies
uncertainty due to the finite test set; no normality assumption.

Usage (from repo root):
  python scripts/week9/week9_bootstrap_test_set_ci.py --per_subject_dir week8_per_subject_metrics --output_dir week9_stats
  python scripts/week9/week9_bootstrap_test_set_ci.py --per_subject_dir week8_per_subject_metrics --output_dir week9_stats --B 2000 --seed 42

Output:
  week9_stats/bootstrap_test_set_ci.csv   (model, n, mean_mae, ci_mae_lo, ci_mae_hi, mean_ssim, ...)
  week9_stats/bootstrap_test_set_ci.md    (human-readable table)
"""

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PER_SUBJECT = ROOT / "week8_per_subject_metrics"
DEFAULT_OUT = ROOT / "week9_stats"


def load_per_subject_by_model(per_subject_dir: Path) -> dict:
    """Return { model: [ { subject_id, mae, ssim, psnr }, ... ] }."""
    by_model = {}
    for p in sorted(per_subject_dir.glob("*.json")):
        try:
            with open(p) as f:
                d = json.load(f)
        except Exception:
            continue
        model = d.get("model")
        if not model:
            stem = p.stem
            # e.g. Cold_3D_2021_002 -> model Cold_3D
            model = stem.rsplit("_", 2)[0] if "_" in stem else stem
        if model not in by_model:
            by_model[model] = []
        mae = d.get("mae")
        ssim = d.get("ssim")
        psnr = d.get("psnr")
        if mae is None and ssim is None and psnr is None:
            continue
        by_model[model].append({
            "subject_id": d.get("subject_id", p.stem),
            "mae": float(mae) if mae is not None else np.nan,
            "ssim": float(ssim) if ssim is not None else np.nan,
            "psnr": float(psnr) if psnr is not None else np.nan,
        })
    return by_model


def bootstrap_percentile_ci(
    values: np.ndarray,
    B: int = 2000,
    seed: int = 42,
    ci_percent: float = 95.0,
) -> tuple[float, float, float]:
    """
    values: 1D array of length n (e.g. per-subject MAE for n=32).
    Returns (point_estimate_mean, ci_low, ci_high) using percentile bootstrap.
    """
    n = len(values)
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    if valid.sum() < 2:
        return float(np.nanmean(values)), np.nan, np.nan
    values = values[valid]
    n = len(values)
    rng = np.random.default_rng(seed)
    boot_means = []
    for _ in range(B):
        idx = rng.choice(n, size=n, replace=True)
        boot_means.append(np.mean(values[idx]))
    boot_means = np.array(boot_means)
    alpha = (100.0 - ci_percent) / 2.0
    ci_lo = np.percentile(boot_means, alpha)
    ci_hi = np.percentile(boot_means, 100.0 - alpha)
    return float(np.mean(values)), float(ci_lo), float(ci_hi)


def main():
    ap = argparse.ArgumentParser(description="Bootstrap 95%% CI on test set for mean MAE/SSIM/PSNR")
    ap.add_argument("--per_subject_dir", default=str(DEFAULT_PER_SUBJECT), help="Dir with model_subjectid.json")
    ap.add_argument("--output_dir", default=str(DEFAULT_OUT), help="Output dir for CSV and MD")
    ap.add_argument("--B", type=int, default=2000, help="Number of bootstrap samples")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for bootstrap")
    ap.add_argument("--ci_percent", type=float, default=95.0, help="CI level (default 95)")
    args = ap.parse_args()

    per_subject_dir = Path(args.per_subject_dir)
    if not per_subject_dir.is_dir():
        print("Per-subject dir not found:", per_subject_dir)
        return 1

    by_model = load_per_subject_by_model(per_subject_dir)
    if not by_model:
        print("No per-subject JSONs found in", per_subject_dir)
        return 1

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for model in sorted(by_model.keys()):
        rows_list = by_model[model]
        mae_arr = np.array([r["mae"] for r in rows_list], dtype=float)
        ssim_arr = np.array([r["ssim"] for r in rows_list], dtype=float)
        psnr_arr = np.array([r["psnr"] for r in rows_list], dtype=float)

        n = len(rows_list)
        mean_mae, ci_mae_lo, ci_mae_hi = bootstrap_percentile_ci(mae_arr, B=args.B, seed=args.seed, ci_percent=args.ci_percent)
        mean_ssim, ci_ssim_lo, ci_ssim_hi = bootstrap_percentile_ci(ssim_arr, B=args.B, seed=args.seed, ci_percent=args.ci_percent)
        mean_psnr, ci_psnr_lo, ci_psnr_hi = bootstrap_percentile_ci(psnr_arr, B=args.B, seed=args.seed, ci_percent=args.ci_percent)

        rows.append({
            "model": model,
            "n": n,
            "mean_mae": mean_mae, "ci_mae_lo": ci_mae_lo, "ci_mae_hi": ci_mae_hi,
            "mean_ssim": mean_ssim, "ci_ssim_lo": ci_ssim_lo, "ci_ssim_hi": ci_ssim_hi,
            "mean_psnr": mean_psnr, "ci_psnr_lo": ci_psnr_lo, "ci_psnr_hi": ci_psnr_hi,
        })

    # CSV
    csv_path = out_dir / "bootstrap_test_set_ci.csv"
    with open(csv_path, "w") as f:
        f.write("model,n,mean_mae,ci_mae_lo,ci_mae_hi,mean_ssim,ci_ssim_lo,ci_ssim_hi,mean_psnr,ci_psnr_lo,ci_psnr_hi\n")
        for r in rows:
            f.write(f"{r['model']},{r['n']},{r['mean_mae']:.6f},{r['ci_mae_lo']:.6f},{r['ci_mae_hi']:.6f},"
                    f"{r['mean_ssim']:.6f},{r['ci_ssim_lo']:.6f},{r['ci_ssim_hi']:.6f},"
                    f"{r['mean_psnr']:.4f},{r['ci_psnr_lo']:.4f},{r['ci_psnr_hi']:.4f}\n")
    print("Wrote", csv_path)

    # Markdown
    md_path = out_dir / "bootstrap_test_set_ci.md"
    with open(md_path, "w") as f:
        f.write("## Bootstrap 95% CI on test set (percentile method)\n\n")
        f.write("Per-subject MAE, SSIM, PSNR from week8_per_subject_metrics; B=2000 bootstrap samples of size n (test subjects) with replacement; 95% CI = 2.5th and 97.5th percentiles of the B means.\n\n")
        f.write("| Model | n | Mean MAE | 95% CI MAE | Mean SSIM | 95% CI SSIM | Mean PSNR | 95% CI PSNR |\n")
        f.write("|-------|---|----------|------------|-----------|-------------|-----------|-------------|\n")
        for r in rows:
            f.write(f"| {r['model']} | {r['n']} | {r['mean_mae']:.4f} | [{r['ci_mae_lo']:.4f}, {r['ci_mae_hi']:.4f}] | "
                    f"{r['mean_ssim']:.4f} | [{r['ci_ssim_lo']:.4f}, {r['ci_ssim_hi']:.4f}] | "
                    f"{r['mean_psnr']:.2f} | [{r['ci_psnr_lo']:.2f}, {r['ci_psnr_hi']:.2f}] |\n")
    print("Wrote", md_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
