#!/usr/bin/env python3
"""
Week 8: Statistical significance and Bland-Altman for method comparison.

Input options:
  A) Per-run JSONs (e.g. from 3 seeds): aggregate to mean+/-std per model, then compare methods.
  B) Per-subject predictions: one file per (model, subject) with pred/target -> Bland-Altman per method.

This script supports:
  - Loading aggregated metrics (e.g. from aggregate_week8_seeds.py output or a CSV).
  - Pairwise statistical tests (e.g. Wilcoxon signed-rank or paired t-test) if per-subject metrics are available.
  - Bland-Altman: difference (pred - target) vs mean(pred, target) per subject; compute limits of agreement.

Usage:
  python scripts/week8_significance_and_bland_altman.py --aggregate_csv week8_aggregate.csv --output_dir week8_stats
  python scripts/week8_significance_and_bland_altman.py --per_subject_dir /path/to/per_subject_metrics --output_dir week8_stats

Output: Tables (CSV/MD) with method comparison and, if per-subject data provided, significance and Bland-Altman summary.
"""

import argparse
import json
import os
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from pipeline_model_registry import REPORTING_MODEL_ORDER


def load_aggregate_csv(path: str) -> list[dict]:
    """Load CSV with columns model, mae_mean, mae_std, ssim_mean, ssim_std, psnr_mean, psnr_std."""
    import csv
    rows = []
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows


def bland_altman_limits(diffs: list[float], mean_vals: list[float]) -> dict:
    """Compute mean difference, std of diff, limits of agreement (mean +/- 1.96*std)."""
    import numpy as np
    diffs = np.asarray(diffs, dtype=float)
    mean_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs))
    loa_low = mean_diff - 1.96 * std_diff
    loa_high = mean_diff + 1.96 * std_diff
    return {
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "loa_low": loa_low,
        "loa_high": loa_high,
    }


def pairwise_wilcoxon_or_ttest(metric_by_model: dict[str, list[float]], metric_name: str) -> list[tuple[str, str, float]]:
    """If we have per-subject metric per model: pairwise test. Returns [(model_a, model_b, p_value), ...]."""
    try:
        from scipy import stats
    except ImportError:
        return []
    models = list(metric_by_model.keys())
    results = []
    for i, ma in enumerate(models):
        for mb in models[i + 1 :]:
            a = metric_by_model[ma]
            b = metric_by_model[mb]
            if len(a) != len(b) or len(a) < 3:
                continue
            try:
                _, p = stats.wilcoxon(a, b)
                results.append((ma, mb, float(p)))
            except Exception:
                try:
                    _, p = stats.ttest_rel(a, b)
                    results.append((ma, mb, float(p)))
                except Exception:
                    pass
    return results


def main():
    ap = argparse.ArgumentParser(description="Week 8: significance tests and Bland-Altman")
    ap.add_argument("--aggregate_csv", default="", help="CSV from aggregate_week8_seeds.py")
    ap.add_argument("--per_subject_dir", default="", help="Dir with per-subject JSONs (e.g. model_subject_metrics.json)")
    ap.add_argument("--output_dir", default="week8_stats", help="Output dir for tables")
    ap.add_argument("--paper_order", action="store_true", help="Output Bland-Altman table in paper model order")
    args = ap.parse_args()
    PAPER_MODEL_ORDER = list(REPORTING_MODEL_ORDER)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.aggregate_csv and Path(args.aggregate_csv).is_file():
        rows = load_aggregate_csv(args.aggregate_csv)
        out_path = Path(args.output_dir) / "method_summary_table.md"
        with open(out_path, "w") as f:
            f.write("| Model | MAE (mean +/- std) | SSIM (mean +/- std) | PSNR (mean +/- std) |\n")
            f.write("|-------|------------------|-------------------|-------------------|\n")
            for r in rows:
                mae = f"{float(r.get('mae_mean', 0)):.4f} +/- {float(r.get('mae_std', 0)):.4f}"
                ssim = f"{float(r.get('ssim_mean', 0)):.4f} +/- {float(r.get('ssim_std', 0)):.4f}"
                psnr = f"{float(r.get('psnr_mean', 0)):.2f} +/- {float(r.get('psnr_std', 0)):.2f}"
                f.write(f"| {r.get('model', '')} | {mae} | {ssim} | {psnr} |\n")
        print(f"Wrote {out_path}")

    if args.per_subject_dir and Path(args.per_subject_dir).is_dir():
        # Expect JSONs with keys like subject_id, model, mae, ssim, psnr, pred_mean, target_mean
        metric_by_model = {}
        for p in Path(args.per_subject_dir).glob("*.json"):
            try:
                with open(p) as f:
                    d = json.load(f)
                model = d.get("model", p.stem)
                if model not in metric_by_model:
                    metric_by_model[model] = {"mae": [], "ssim": [], "psnr": [], "diffs": [], "means": []}
                if "mae" in d:
                    metric_by_model[model]["mae"].append(d["mae"])
                if "ssim" in d:
                    metric_by_model[model]["ssim"].append(d["ssim"])
                if "psnr" in d:
                    metric_by_model[model]["psnr"].append(d["psnr"])
                if "pred_mean" in d and "target_mean" in d:
                    pred_m = float(d["pred_mean"])
                    tgt_m = float(d["target_mean"])
                    metric_by_model[model]["diffs"].append(pred_m - tgt_m)
                    metric_by_model[model]["means"].append((pred_m + tgt_m) / 2)
            except Exception:
                continue

        # Bland-Altman per model (paper order when requested)
        ba_path = Path(args.output_dir) / "bland_altman_summary.md"
        models_iter = PAPER_MODEL_ORDER if args.paper_order else metric_by_model.keys()
        with open(ba_path, "w") as f:
            f.write("| Model | Mean diff | Std diff | LoA low | LoA high |\n")
            f.write("|-------|-----------|----------|---------|----------|\n")
            for model in models_iter:
                data = metric_by_model.get(model)
                if not data or not data["diffs"] or not data["means"]:
                    continue
                ba = bland_altman_limits(data["diffs"], data["means"])
                f.write(f"| {model} | {ba['mean_diff']:.4f} | {ba['std_diff']:.4f} | {ba['loa_low']:.4f} | {ba['loa_high']:.4f} |\n")
        print(f"Wrote {ba_path}")

        # Pairwise significance (e.g. MAE)
        for metric_key in ["mae", "ssim", "psnr"]:
            by_model = {m: data[metric_key] for m, data in metric_by_model.items() if data[metric_key]}
            if len(by_model) < 2:
                continue
            pairs = pairwise_wilcoxon_or_ttest(by_model, metric_key)
            if pairs:
                sig_path = Path(args.output_dir) / f"pairwise_{metric_key}.md"
                with open(sig_path, "w") as f:
                    f.write(f"| Model A | Model B | p-value ({metric_key}) |\n")
                    f.write("|---------|---------|--------------------------|\n")
                    for ma, mb, p in pairs:
                        f.write(f"| {ma} | {mb} | {p:.4f} |\n")
                print(f"Wrote {sig_path}")

    if not args.aggregate_csv and not args.per_subject_dir:
        print("Provide --aggregate_csv and/or --per_subject_dir. See script docstring for usage.")


if __name__ == "__main__":
    main()
