#!/usr/bin/env python3
"""
Week 9: R² and paired t-tests for best models (using per-subject metrics).

Reads week8_per_subject_metrics/*.json (model, subject_id, mae, ssim, psnr, pred_mean, target_mean).
For each model: computes R² = 1 - SS_res/SS_tot over subjects (pred_mean vs target_mean).
Pairwise paired t-tests (and Wilcoxon) on MAE, SSIM, PSNR across models that share the same subject set.

Primary model (CAE3D / week7_unet3d) and diffusion flagship (Residual_3D_tips) are defined in
pipeline_model_registry.py.

Usage (from repo root):
  python scripts/week9/week9_stats_best6_r2_and_ptests.py --per_subject_dir week8_per_subject_metrics --output_dir week9_stats
  python scripts/week9/week9_stats_best6_r2_and_ptests.py --per_subject_dir week8_per_subject_metrics --output_dir week9_stats --best6_only

Output: R² table, pairwise p-value tables (MAE, SSIM, PSNR), and a combined summary markdown.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("<repo-root>")
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from pipeline_model_registry import (
    CORE_COMPARISON_MODELS,
    MODEL_DISPLAY_NAME,
    sort_models_by_reporting_order,
)

DEFAULT_PER_SUBJECT = ROOT / "week8_per_subject_metrics"
DEFAULT_OUT = ROOT / "week9_stats"

# --best6_only filters to this set (historical flag name; may include >6 models).
CORE_MODEL_SET = set(CORE_COMPARISON_MODELS)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R² = 1 - SS_res / SS_tot. Returns 0 if SS_tot is 0."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if len(y_true) != len(y_pred) or len(y_true) < 2:
        return float("nan")
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot <= 0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def load_per_subject_by_model(per_subject_dir: Path) -> dict:
    """Return { model: [ { subject_id, mae, ssim, psnr, pred_mean, target_mean }, ... ] }."""
    by_model = {}
    for p in sorted(per_subject_dir.glob("*.json")):
        try:
            with open(p) as f:
                d = json.load(f)
        except Exception:
            continue
        model = d.get("model")
        if not model:
            # infer from filename: ModelName_subjectid.json
            stem = p.stem
            for m in list(by_model.keys()) + list(MODEL_DISPLAY_NAME.keys()):
                if stem.startswith(m + "_"):
                    model = m
                    break
            if not model:
                model = stem.rsplit("_", 2)[0] if "_" in stem else stem
        if model not in by_model:
            by_model[model] = []
        row = {
            "subject_id": d.get("subject_id", p.stem),
            "mae": d.get("mae"),
            "ssim": d.get("ssim"),
            "psnr": d.get("psnr"),
            "pred_mean": d.get("pred_mean"),
            "target_mean": d.get("target_mean"),
        }
        # skip if no metrics
        if row["mae"] is None and row["pred_mean"] is None:
            continue
        by_model[model].append(row)
    return by_model


def align_subjects(by_model: dict) -> tuple[list[str], dict[str, list]]:
    """Return (common_subject_ids, { model: [mae, ssim, psnr, pred_mean, target_mean] aligned }) for paired tests."""
    # subject_id -> { model -> row }
    by_subject = {}
    for model, rows in by_model.items():
        for r in rows:
            sid = r.get("subject_id", "")
            if sid not in by_subject:
                by_subject[sid] = {}
            by_subject[sid][model] = r
    models = list(by_model.keys())
    # common subjects: present in all models
    common = None
    for sid, per_model in by_subject.items():
        if len(per_model) != len(models):
            continue
        ok = all(m in per_model for m in models)
        if not ok:
            continue
        if common is None:
            common = []
        common.append(sid)
    if common is None:
        common = sorted(by_subject.keys())[:1]  # at least one
    common = sorted(common)
    # align
    aligned = {m: [] for m in models}
    for m in models:
        aligned[m] = {"mae": [], "ssim": [], "psnr": [], "pred_mean": [], "target_mean": []}
    for sid in common:
        per_model = by_subject.get(sid, {})
        for m in models:
            r = per_model.get(m)
            if not r:
                continue
            aligned[m]["mae"].append(r.get("mae") if r.get("mae") is not None else np.nan)
            aligned[m]["ssim"].append(r.get("ssim") if r.get("ssim") is not None else np.nan)
            aligned[m]["psnr"].append(r.get("psnr") if r.get("psnr") is not None else np.nan)
            aligned[m]["pred_mean"].append(r.get("pred_mean") if r.get("pred_mean") is not None else np.nan)
            aligned[m]["target_mean"].append(r.get("target_mean") if r.get("target_mean") is not None else np.nan)
    return common, aligned


def pairwise_ttest_and_wilcoxon(by_model: dict, metric_key: str) -> list[tuple[str, str, float, float]]:
    """Returns [(model_a, model_b, p_ttest, p_wilcoxon)] for pairs. Uses common subjects only."""
    try:
        from scipy import stats
    except ImportError:
        return []
    _, aligned = align_subjects(by_model)
    models = [m for m in aligned if aligned[m][metric_key]]
    results = []
    for i, ma in enumerate(models):
        for mb in models[i + 1 :]:
            a = np.array(aligned[ma][metric_key], dtype=float)
            b = np.array(aligned[mb][metric_key], dtype=float)
            # drop nan for this pair (same indices)
            mask = np.isfinite(a) & np.isfinite(b)
            a, b = a[mask], b[mask]
            n = len(a)
            if n < 3:
                continue
            p_ttest = float(stats.ttest_rel(a, b).pvalue) if n >= 2 else float("nan")
            try:
                p_wilcoxon = float(stats.wilcoxon(a, b).pvalue)
            except Exception:
                p_wilcoxon = float("nan")
            results.append((ma, mb, p_ttest, p_wilcoxon))
    return results


def main():
    ap = argparse.ArgumentParser(description="R² and paired t-tests for per-subject metrics")
    ap.add_argument("--per_subject_dir", default=str(DEFAULT_PER_SUBJECT), help="Dir with model_subjectid.json")
    ap.add_argument("--output_dir", default=str(DEFAULT_OUT), help="Output dir for tables")
    ap.add_argument("--best6_only", action="store_true", help="Restrict to best-6 model names and use display names")
    args = ap.parse_args()
    per_subject_dir = Path(args.per_subject_dir)
    out_dir = Path(args.output_dir)
    if not per_subject_dir.is_dir():
        print("Per-subject dir not found:", per_subject_dir)
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    by_model = load_per_subject_by_model(per_subject_dir)
    if args.best6_only:
        by_model = {m: rows for m, rows in by_model.items() if m in CORE_MODEL_SET}
        display = {m: MODEL_DISPLAY_NAME.get(m, m) for m in by_model}
    else:
        display = {m: m for m in by_model}

    # R² per model (using pred_mean vs target_mean); paper-friendly order when --best6_only
    model_order = (
        sort_models_by_reporting_order(list(by_model.keys()))
        if args.best6_only
        else sorted(by_model.keys())
    )
    r2_rows = []
    for model in model_order:
        rows = by_model.get(model)
        if not rows:
            continue
        pred = [r["pred_mean"] for r in rows if r.get("pred_mean") is not None]
        tgt = [r["target_mean"] for r in rows if r.get("target_mean") is not None]
        n = min(len(pred), len(tgt))
        if n < 2:
            r2_rows.append((display.get(model, model), len(rows), float("nan"), np.nan, np.nan, np.nan))
            continue
        pred = np.array(pred[:n], dtype=float)
        tgt = np.array(tgt[:n], dtype=float)
        r2 = r2_score(tgt, pred)
        mae_arr = np.array([r["mae"] for r in rows if r.get("mae") is not None][:n], dtype=float)
        ssim_arr = np.array([r["ssim"] for r in rows if r.get("ssim") is not None][:n], dtype=float)
        psnr_arr = np.array([r["psnr"] for r in rows if r.get("psnr") is not None][:n], dtype=float)
        mae_mean = float(np.mean(mae_arr)) if len(mae_arr) else np.nan
        ssim_mean = float(np.mean(ssim_arr)) if len(ssim_arr) else np.nan
        psnr_mean = float(np.mean(psnr_arr)) if len(psnr_arr) else np.nan
        r2_rows.append((display.get(model, model), n, r2, mae_mean, ssim_mean, psnr_mean))

    # Summary table (R², MAE, SSIM, PSNR)
    out_md = out_dir / "best6_r2_and_metrics.md"
    with open(out_md, "w") as f:
        f.write("## R² and summary metrics (per-subject)\n\n")
        f.write("| Model | n | R² | MAE (mean) | SSIM (mean) | PSNR (mean) |\n")
        f.write("|-------|---|-----|-------------|-------------|-------------|\n")
        for name, n, r2, mae, ssim, psnr in r2_rows:
            r2_s = f"{r2:.4f}" if not np.isnan(r2) else "—"
            mae_s = f"{mae:.4f}" if not np.isnan(mae) else "—"
            ssim_s = f"{ssim:.4f}" if not np.isnan(ssim) else "—"
            psnr_s = f"{psnr:.2f}" if not np.isnan(psnr) else "—"
            f.write(f"| {name} | {n} | {r2_s} | {mae_s} | {ssim_s} | {psnr_s} |\n")
    print("Wrote", out_md)

    # Pairwise p-values (paired t-test and Wilcoxon)
    for metric_key in ["mae", "ssim", "psnr"]:
        pairs = pairwise_ttest_and_wilcoxon(by_model, metric_key)
        if not pairs:
            continue
        path_ttest = out_dir / f"pairwise_{metric_key}_ttest.md"
        path_wilcoxon = out_dir / f"pairwise_{metric_key}_wilcoxon.md"
        with open(path_ttest, "w") as f:
            f.write(f"| Model A | Model B | p (paired t-test, {metric_key}) | Significant (p<0.05) |\n")
            f.write("|---------|---------|-----------------------------------|------------------------|\n")
            for ma, mb, p_t, _ in pairs:
                name_a = display.get(ma, ma)
                name_b = display.get(mb, mb)
                sig = "Yes" if p_t < 0.05 else "No"
                f.write(f"| {name_a} | {name_b} | {p_t:.4f} | {sig} |\n")
        with open(path_wilcoxon, "w") as f:
            f.write(f"| Model A | Model B | p (Wilcoxon, {metric_key}) | Significant (p<0.05) |\n")
            f.write("|---------|---------|----------------------------|------------------------|\n")
            for ma, mb, _, p_w in pairs:
                name_a = display.get(ma, ma)
                name_b = display.get(mb, mb)
                sig = "Yes" if p_w < 0.05 else "No"
                f.write(f"| {name_a} | {name_b} | {p_w:.4f} | {sig} |\n")
        print("Wrote", path_ttest, path_wilcoxon)

    # JSON summary for downstream
    summary = {
        "r2_per_model": [
            {"model": name, "n": n, "r2": r2, "mae_mean": mae, "ssim_mean": ssim, "psnr_mean": psnr}
            for name, n, r2, mae, ssim, psnr in r2_rows
        ],
        "pairwise_ttest_mae": [{"model_a": ma, "model_b": mb, "p": p_t} for ma, mb, p_t, _ in pairwise_ttest_and_wilcoxon(by_model, "mae")],
    }
    out_json = out_dir / "best6_r2_and_ptests.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print("Wrote", out_json)


if __name__ == "__main__":
    main()
