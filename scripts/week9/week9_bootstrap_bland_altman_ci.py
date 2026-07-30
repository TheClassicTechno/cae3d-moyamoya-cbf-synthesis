#!/usr/bin/env python3
"""
Bootstrap 95% CI for Bland-Altman bias and limits of agreement (LoA).

Loads per-subject pred_mean/target_mean from week8_per_subject_metrics.
For each model: resample n subjects B=2000 times with replacement;
for each sample compute mean_diff (bias) and LoA (bias +/- 1.96*SD of diff).
Report point estimates and 95% percentile CI for bias, LoA low, LoA high.

Usage:
  python scripts/week9/week9_bootstrap_bland_altman_ci.py --per_subject_dir week8_per_subject_metrics --output_dir week9_stats
  python scripts/week9/week9_bootstrap_bland_altman_ci.py --B 2000 --seed 42

Output: week9_stats/bootstrap_bland_altman_ci.csv, .md
"""

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PER_SUBJECT = ROOT / "week8_per_subject_metrics"
DEFAULT_OUT = ROOT / "week9_stats"


def load_ba_by_model(per_subject_dir: Path) -> dict:
    """Return { model: { "diffs": [...], "means": [...] } } from pred_mean, target_mean."""
    by_model = {}
    for p in sorted(per_subject_dir.glob("*.json")):
        try:
            with open(p) as f:
                d = json.load(f)
        except Exception:
            continue
        pred_m = d.get("pred_mean")
        tgt_m = d.get("target_mean")
        if pred_m is None or tgt_m is None:
            continue
        model = d.get("model")
        if not model:
            stem = p.stem
            model = stem.rsplit("_", 2)[0] if "_" in stem else stem
        if model not in by_model:
            by_model[model] = {"diffs": [], "means": []}
        by_model[model]["diffs"].append(float(pred_m) - float(tgt_m))
        by_model[model]["means"].append((float(pred_m) + float(tgt_m)) / 2)
    return by_model


def bland_altman_from_arrays(diffs: np.ndarray, means: np.ndarray) -> tuple[float, float, float]:
    """Mean diff (bias), LoA low, LoA high."""
    mean_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs))
    if np.isnan(std_diff) or std_diff <= 0:
        std_diff = 0.0
    loa_low = mean_diff - 1.96 * std_diff
    loa_high = mean_diff + 1.96 * std_diff
    return mean_diff, loa_low, loa_high


def bootstrap_ba_ci(
    diffs: np.ndarray,
    means: np.ndarray,
    B: int = 2000,
    seed: int = 42,
    ci_percent: float = 95.0,
) -> dict:
    """Bootstrap percentile CI for bias, LoA low, LoA high."""
    n = len(diffs)
    diffs = np.asarray(diffs, dtype=float)
    means = np.asarray(means, dtype=float)
    if n < 2:
        return {
            "bias": np.nan, "bias_ci_lo": np.nan, "bias_ci_hi": np.nan,
            "loa_low": np.nan, "loa_low_ci_lo": np.nan, "loa_low_ci_hi": np.nan,
            "loa_high": np.nan, "loa_high_ci_lo": np.nan, "loa_high_ci_hi": np.nan,
        }
    rng = np.random.default_rng(seed)
    alpha = (100.0 - ci_percent) / 2.0
    bias_boot = []
    loa_low_boot = []
    loa_high_boot = []
    for _ in range(B):
        idx = rng.choice(n, size=n, replace=True)
        d, m = diffs[idx], means[idx]
        bias, loa_l, loa_h = bland_altman_from_arrays(d, m)
        bias_boot.append(bias)
        loa_low_boot.append(loa_l)
        loa_high_boot.append(loa_h)
    bias_boot = np.array(bias_boot)
    loa_low_boot = np.array(loa_low_boot)
    loa_high_boot = np.array(loa_high_boot)
    bias_pt, loa_l_pt, loa_h_pt = bland_altman_from_arrays(diffs, means)
    return {
        "bias": bias_pt,
        "bias_ci_lo": float(np.percentile(bias_boot, alpha)),
        "bias_ci_hi": float(np.percentile(bias_boot, 100.0 - alpha)),
        "loa_low": loa_l_pt,
        "loa_low_ci_lo": float(np.percentile(loa_low_boot, alpha)),
        "loa_low_ci_hi": float(np.percentile(loa_low_boot, 100.0 - alpha)),
        "loa_high": loa_h_pt,
        "loa_high_ci_lo": float(np.percentile(loa_high_boot, alpha)),
        "loa_high_ci_hi": float(np.percentile(loa_high_boot, 100.0 - alpha)),
    }


def main():
    ap = argparse.ArgumentParser(description="Bootstrap 95% CI for Bland-Altman bias and LoA")
    ap.add_argument("--per_subject_dir", default=str(DEFAULT_PER_SUBJECT), help="Dir with per-subject JSONs")
    ap.add_argument("--output_dir", default=str(DEFAULT_OUT), help="Output dir for CSV and MD")
    ap.add_argument("--B", type=int, default=2000, help="Bootstrap samples")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--ci_percent", type=float, default=95.0, help="CI level")
    args = ap.parse_args()

    per_subject_dir = Path(args.per_subject_dir)
    if not per_subject_dir.is_dir():
        print("Per-subject dir not found:", per_subject_dir)
        return 1

    by_model = load_ba_by_model(per_subject_dir)
    if not by_model:
        print("No pred_mean/target_mean data found in", per_subject_dir)
        return 1

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for model in sorted(by_model.keys()):
        data = by_model[model]
        diffs = np.array(data["diffs"])
        means = np.array(data["means"])
        res = bootstrap_ba_ci(diffs, means, B=args.B, seed=args.seed, ci_percent=args.ci_percent)
        rows.append({
            "model": model,
            "n": len(diffs),
            **res,
        })

    csv_path = out_dir / "bootstrap_bland_altman_ci.csv"
    with open(csv_path, "w") as f:
        f.write("model,n,bias,bias_ci_lo,bias_ci_hi,loa_low,loa_low_ci_lo,loa_low_ci_hi,loa_high,loa_high_ci_lo,loa_high_ci_hi\n")
        for r in rows:
            f.write(f"{r['model']},{r['n']},{r['bias']:.6f},{r['bias_ci_lo']:.6f},{r['bias_ci_hi']:.6f},"
                    f"{r['loa_low']:.6f},{r['loa_low_ci_lo']:.6f},{r['loa_low_ci_hi']:.6f},"
                    f"{r['loa_high']:.6f},{r['loa_high_ci_lo']:.6f},{r['loa_high_ci_hi']:.6f}\n")
    print("Wrote", csv_path)

    md_path = out_dir / "bootstrap_bland_altman_ci.md"
    with open(md_path, "w") as f:
        f.write("## Bootstrap 95% CI for Bland-Altman (bias and LoA)\n\n")
        f.write("Per-subject pred_mean, target_mean from week8_per_subject_metrics. ")
        f.write(f"B={args.B} bootstrap samples (n subjects with replacement); 95%% CI = percentile.\n\n")
        f.write("| Model | n | Bias | 95% CI bias | LoA low | 95% CI LoA low | LoA high | 95% CI LoA high |\n")
        f.write("|-------|---|------|-------------|---------|----------------|----------|------------------|\n")
        for r in rows:
            f.write(f"| {r['model']} | {r['n']} | {r['bias']:.4f} | [{r['bias_ci_lo']:.4f}, {r['bias_ci_hi']:.4f}] | "
                    f"{r['loa_low']:.4f} | [{r['loa_low_ci_lo']:.4f}, {r['loa_low_ci_hi']:.4f}] | "
                    f"{r['loa_high']:.4f} | [{r['loa_high_ci_lo']:.4f}, {r['loa_high_ci_hi']:.4f}] |\n")
    print("Wrote", md_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
