#!/usr/bin/env python3
"""
Aggregate Week 8 multi-seed results into mean +/- std for tables.

Expects: per-seed result JSONs with keys mae_mean, ssim_mean, psnr_mean.
  - Either one dir per model with files like results_seed42.json, results_seed123.json, results_seed456.json
  - Or a single dir with files like MODEL_NAME_seed42.json

Usage:
  python scripts/aggregate_week8_seeds.py --results_dir /path/to/results --output table_week8.md
  python scripts/aggregate_week8_seeds.py --results_dir <repo-root> --pattern "*_seed*.json" --output week8_aggregate.csv

Output: Table (Markdown or CSV) with columns Model, MAE (mean+/-std), SSIM (mean+/-std), PSNR (mean+/-std).
"""

import argparse
import json
import os
import re
from pathlib import Path

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_REPO_ROOT, "pyproject.toml")):
    _parent = os.path.dirname(_REPO_ROOT)
    if _parent == _REPO_ROOT:
        raise RuntimeError("Could not locate repository root (pyproject.toml not found)")
    _REPO_ROOT = _parent


def find_seed_jsons(results_dir: str, pattern: str = "*_seed*.json"):
    """Collect (model_name, seed, filepath). Model name = filename with _seedXX stripped."""
    results_dir = Path(results_dir)
    if not results_dir.is_dir():
        return []
    out = []
    for p in results_dir.rglob("*.json"):
        if not p.is_file():
            continue
        m = re.search(r"_seed(\d+)\.json$", p.name, re.IGNORECASE)
        if m:
            seed = int(m.group(1))
            base = re.sub(r"_seed\d+\.json$", "", p.name, flags=re.IGNORECASE)
            out.append((base, seed, str(p)))
    for p in sorted(results_dir.glob(pattern)):
        if p.is_file():
            m = re.search(r"_seed(\d+)\.json$", p.name, re.IGNORECASE)
            if m:
                seed = int(m.group(1))
                base = re.sub(r"_seed\d+\.json$", "", p.name, flags=re.IGNORECASE)
                out.append((base, seed, str(p)))
    seen = set()
    unique = []
    for t in out:
        if (t[0], t[1]) not in seen:
            seen.add((t[0], t[1]))
            unique.append(t)
    return unique


def load_metrics(path: str) -> dict | None:
    """Return dict with mae_mean, ssim_mean, psnr_mean or None."""
    try:
        with open(path) as f:
            d = json.load(f)
        # Support top-level or nested test_results (e.g. UNet_3D output)
        t = d.get("test_results") or d
        return {
            "mae_mean": float(t.get("mae_mean", t.get("MAE", float("nan")))),
            "ssim_mean": float(t.get("ssim_mean", t.get("SSIM", float("nan")))),
            "psnr_mean": float(t.get("psnr_mean", t.get("PSNR", float("nan")))),
        }
    except Exception:
        return None


def aggregate_by_model(items: list[tuple[str, int, str]]) -> dict[str, list[dict]]:
    """Group by model name; each value is list of metric dicts (one per seed)."""
    by_model = {}
    for model_name, seed, path in items:
        m = load_metrics(path)
        if m is None:
            continue
        m["_seed"] = seed
        m["_path"] = path
        by_model.setdefault(model_name, []).append(m)
    return by_model


def mean_std(vals: list[float]) -> tuple[float, float]:
    import numpy as np
    a = np.array([float(x) for x in vals if not (x != x or x == float("inf"))])
    if a.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(a)), float(np.std(a))


def ci_half(std: float, n: int, z: float = 1.96) -> float:
    """95%% CI half-width: z * std / sqrt(n). Use for supplement reporting."""
    if n < 2:
        return 0.0
    return z * std / (n ** 0.5)


def main():
    ap = argparse.ArgumentParser(description="Aggregate Week 8 seed results to mean +/- std")
    ap.add_argument("--results_dir", default=_REPO_ROOT, help="Root dir to search for *_seed*.json")
    ap.add_argument("--pattern", default="*_seed*.json", help="Glob for result files")
    ap.add_argument("--output", default="", help="Output path (e.g. week8_table.md or .csv)")
    ap.add_argument("--seeds", type=str, default="42,123,456", help="Comma-separated seeds to expect (for validation)")
    ap.add_argument("--ci", action="store_true", help="Add 95%% CI (mean +/- 1.96*std/sqrt(n)) to output for supplement")
    args = ap.parse_args()

    expected_seeds = [int(s.strip()) for s in args.seeds.split(",")]
    items = find_seed_jsons(args.results_dir, args.pattern)
    if not items:
        # Fallback: scan common dirs
        for sub in ["UNet_3D", "Diffusion_baseline_3D", "scripts", "Diffusion_3D_PatchVolume", "NeuralOperators"]:
            subdir = Path(args.results_dir) / sub
            if subdir.is_dir():
                items.extend(find_seed_jsons(str(subdir), args.pattern))

    by_model = aggregate_by_model(items)
    if not by_model:
        print("No *_seed*.json files found. Place result JSONs named like model_seed42.json in --results_dir.")
        return

    rows = []
    for model_name in sorted(by_model.keys()):
        metrics_list = by_model[model_name]
        mae_vals = [x["mae_mean"] for x in metrics_list]
        ssim_vals = [x["ssim_mean"] for x in metrics_list]
        psnr_vals = [x["psnr_mean"] for x in metrics_list]
        mae_m, mae_s = mean_std(mae_vals)
        ssim_m, ssim_s = mean_std(ssim_vals)
        psnr_m, psnr_s = mean_std(psnr_vals)
        n = len(metrics_list)
        row = {
            "model": model_name,
            "n_seeds": n,
            "mae_mean": mae_m, "mae_std": mae_s,
            "ssim_mean": ssim_m, "ssim_std": ssim_s,
            "psnr_mean": psnr_m, "psnr_std": psnr_s,
        }
        if args.ci and n >= 2:
            row["mae_ci95"] = ci_half(mae_s, n)
            row["ssim_ci95"] = ci_half(ssim_s, n)
            row["psnr_ci95"] = ci_half(psnr_s, n)
        rows.append(row)

    # Print to stdout
    print("Model\tMAE (mean+/-std)\tSSIM (mean+/-std)\tPSNR (mean+/-std)\tn_seeds")
    for r in rows:
        print(f"{r['model']}\t{r['mae_mean']:.4f} +/- {r['mae_std']:.4f}\t{r['ssim_mean']:.4f} +/- {r['ssim_std']:.4f}\t{r['psnr_mean']:.2f} +/- {r['psnr_std']:.2f}\t{r['n_seeds']}")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.suffix.lower() == ".csv":
            with open(out_path, "w") as f:
                f.write("model,n_seeds,mae_mean,mae_std,ssim_mean,ssim_std,psnr_mean,psnr_std\n")
                for r in rows:
                    f.write(f"{r['model']},{r['n_seeds']},{r['mae_mean']:.6f},{r['mae_std']:.6f},{r['ssim_mean']:.6f},{r['ssim_std']:.6f},{r['psnr_mean']:.4f},{r['psnr_std']:.4f}\n")
        else:
            with open(out_path, "w") as f:
                f.write("| Model | MAE (mean +/- std) | SSIM (mean +/- std) | PSNR (mean +/- std) | n_seeds |\n")
                f.write("|-------|------------------|-------------------|-------------------|--------|\n")
                for r in rows:
                    f.write(f"| {r['model']} | {r['mae_mean']:.4f} +/- {r['mae_std']:.4f} | {r['ssim_mean']:.4f} +/- {r['ssim_std']:.4f} | {r['psnr_mean']:.2f} +/- {r['psnr_std']:.2f} | {r['n_seeds']} |\n")
                if args.ci:
                    f.write("\n95% CI half-width (1.96*std/√n) for supplement:\n")
                    for r in rows:
                        if r.get("mae_ci95") is not None:
                            f.write(f"- {r['model']}: MAE ± {r['mae_ci95']:.4f}, SSIM ± {r['ssim_ci95']:.4f}, PSNR ± {r['psnr_ci95']:.2f}\n")
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
