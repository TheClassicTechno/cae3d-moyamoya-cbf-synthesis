#!/usr/bin/env python3
"""
Apply Holm-Bonferroni correction to pairwise comparison p-values.

Reads pairwise_mae.md, pairwise_ssim.md, pairwise_psnr.md from week8_stats (or given dir).
Parses (Model A, Model B, p-value), then for each metric applies Holm's method:
  - Sort p-values; adjusted p-value for the i-th smallest is min(1, max over j<=i of (k - j + 1) * p_j).
Reports adjusted p-values and "significant at alpha=0.05 after correction" for each pair.

Usage:
  python scripts/week9/week9_holm_bonferroni_pairwise.py --stats_dir week8_stats --output_dir week9_stats
  python scripts/week9/week9_holm_bonferroni_pairwise.py --alpha 0.05

Output: week9_stats/pairwise_holm_mae.md, pairwise_holm_ssim.md, pairwise_holm_psnr.md (and .csv).
"""

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATS_DIR = ROOT / "week8_stats"
DEFAULT_OUT = ROOT / "week9_stats"


def parse_pairwise_md(path: Path) -> list[tuple[str, str, float]]:
    """Return [(model_a, model_b, p_value), ...]. Skip header and non-table lines."""
    if not path.is_file():
        return []
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line.startswith("|") or "---" in line or "Model A" in line:
                continue
            parts = [p.strip() for p in line.split("|") if p.strip()]
            if len(parts) >= 3:
                try:
                    p_val = float(parts[2])
                    rows.append((parts[0], parts[1], p_val))
                except ValueError:
                    continue
    return rows


def holm_adjust(p_values: list[float]) -> list[float]:
    """
    Holm's method: for sorted p_(1) <= p_(2) <= ... <= p_(k),
    adjusted p for rank i is p_holm[i] = max_{j=1..i} min(1, (k - j + 1) * p_(j)).
    Returns adjusted p-values in the same order as the sorted input.
    """
    if not p_values:
        return []
    k = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * k
    running_max = 0.0
    for pos, (orig_idx, p) in enumerate(indexed):
        mult = k - pos
        candidate = min(1.0, p * mult)
        running_max = max(running_max, candidate)
        adjusted[orig_idx] = running_max
    return adjusted


def main():
    ap = argparse.ArgumentParser(description="Holm-Bonferroni correction for pairwise tests")
    ap.add_argument("--stats_dir", default=str(DEFAULT_STATS_DIR), help="Dir with pairwise_mae.md etc.")
    ap.add_argument("--output_dir", default=str(DEFAULT_OUT), help="Output dir for Holm tables")
    ap.add_argument("--alpha", type=float, default=0.05, help="Significance level after correction")
    args = ap.parse_args()

    stats_dir = Path(args.stats_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for metric in ["mae", "ssim", "psnr"]:
        path = stats_dir / f"pairwise_{metric}.md"
        pairs = parse_pairwise_md(path)
        if not pairs:
            print("No pairs for", metric, "in", path)
            continue

        model_a_list = [x[0] for x in pairs]
        model_b_list = [x[1] for x in pairs]
        p_list = [x[2] for x in pairs]
        adjusted = holm_adjust(p_list)
        alpha = args.alpha

        csv_path = out_dir / f"pairwise_holm_{metric}.csv"
        with open(csv_path, "w") as f:
            f.write("model_a,model_b,p_raw,p_holm,significant_after_holm\n")
            for i in range(len(pairs)):
                sig = "yes" if adjusted[i] <= alpha else "no"
                f.write(f"{model_a_list[i]},{model_b_list[i]},{p_list[i]:.6f},{adjusted[i]:.6f},{sig}\n")
        print("Wrote", csv_path)

        md_path = out_dir / f"pairwise_holm_{metric}.md"
        with open(md_path, "w") as f:
            f.write(f"# Pairwise {metric.upper()} with Holm-Bonferroni correction\n\n")
            f.write(f"Raw p-values from {path.name}; Holm-adjusted p-values; significant at α={alpha} after correction.\n\n")
            f.write("| Model A | Model B | p (raw) | p (Holm) | Significant (α=0.05) |\n")
            f.write("|---------|---------|---------|----------|------------------------|\n")
            for i in range(len(pairs)):
                sig = "yes" if adjusted[i] <= alpha else "no"
                f.write(f"| {model_a_list[i]} | {model_b_list[i]} | {p_list[i]:.4f} | {adjusted[i]:.4f} | {sig} |\n")
        print("Wrote", md_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
