#!/usr/bin/env python3
"""
Week 9: Subgroup comparison (sex, age, ethnicity, severity, surgery, side).

Requires a metadata file (CSV or subject_metadata in split) with columns like
subject_id, sex, age_group, ethnicity, severity, prior_surgery, side_moyamoya.
See SUBJECT_METADATA_SCHEMA.md.

If metadata is not provided or missing, exits with a clear message and does nothing.

Usage (when metadata exists):
  python scripts/week9/week9_subgroup_comparison.py --per_subject_dir week8_per_subject_metrics --metadata_csv subject_metadata.csv --output_dir week9_stats
"""

import argparse
import csv
import json
from pathlib import Path

ROOT = Path("<repo-root>")
DEFAULT_PER_SUBJECT = ROOT / "week8_per_subject_metrics"
DEFAULT_OUT = ROOT / "week9_stats"


def load_metadata_csv(path: Path) -> dict:
    """Return { subject_id: { col: value, ... } }."""
    out = {}
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            sid = row.get("subject_id", "").strip()
            if not sid:
                continue
            out[sid] = {k.strip(): v.strip() for k, v in row.items() if k}
    return out


def main():
    ap = argparse.ArgumentParser(description="Subgroup comparison (requires metadata)")
    ap.add_argument("--per_subject_dir", default=str(DEFAULT_PER_SUBJECT))
    ap.add_argument("--metadata_csv", default="", help="CSV with subject_id, sex, age_group, side_moyamoya, ...")
    ap.add_argument("--subgroup_key", default="sex", help="Column for stratification: sex, age_group, side_moyamoya, etc.")
    ap.add_argument("--output_dir", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    if not args.metadata_csv or not Path(args.metadata_csv).is_file():
        print("Subgroup comparison requires --metadata_csv pointing to a CSV with subject_id and subgroup columns (sex, age_group, etc.).")
        print("See scripts/week9/SUBJECT_METADATA_SCHEMA.md for the schema.")
        print("No data overwritten; exiting.")
        return

    meta = load_metadata_csv(Path(args.metadata_csv))
    if not meta:
        print("Metadata CSV is empty or has no subject_id column. Exiting.")
        return

    per_subject_dir = Path(args.per_subject_dir)
    out_dir = Path(args.output_dir)
    if not per_subject_dir.is_dir():
        print("Per-subject dir not found:", per_subject_dir)
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load per-subject metrics and merge with metadata
    rows = []
    for p in sorted(per_subject_dir.glob("*.json")):
        try:
            with open(p) as f:
                d = json.load(f)
        except Exception:
            continue
        sid = d.get("subject_id", p.stem)
        if sid not in meta:
            continue
        m = meta[sid]
        rows.append({
            "model": d.get("model", ""),
            "subject_id": sid,
            "mae": d.get("mae"),
            "ssim": d.get("ssim"),
            "psnr": d.get("psnr"),
            "sex": m.get("sex", ""),
            "age_group": m.get("age_group", ""),
            "ethnicity": m.get("ethnicity", ""),
            "severity": m.get("severity", ""),
            "prior_surgery": m.get("prior_surgery", ""),
            "side_moyamoya": m.get("side_moyamoya", ""),
        })

    if not rows:
        print("No per-subject rows could be matched to metadata. Check subject_id format.")
        return

    subgroup_key = args.subgroup_key
    try:
        from scipy import stats
    except ImportError:
        stats = None

    models = sorted(set(r["model"] for r in rows))
    groups = sorted(set(r[subgroup_key] for r in rows if r[subgroup_key]))
    out_path = Path(args.output_dir) / "subgroup_summary.md"
    with open(out_path, "w") as f:
        f.write("## Subgroup summary (mean MAE)\n\n")
        f.write("| Model | " + " | ".join(groups) + " |\n")
        f.write("|-------|" + "-----|" * len(groups) + "\n")
        for model in models:
            vals = []
            for g in groups:
                maes = [r["mae"] for r in rows if r["model"] == model and r[subgroup_key] == g and r["mae"] is not None]
                vals.append(f"{sum(maes)/len(maes):.4f}" if maes else "—")
            f.write("| " + model + " | " + " | ".join(vals) + " |\n")
        if stats and len(groups) == 2:
            f.write("\n### Pairwise test (group1 vs group2) per model\n")
            g1, g2 = groups[0], groups[1]
            for model in models:
                a = [r["mae"] for r in rows if r["model"] == model and r[subgroup_key] == g1 and r["mae"] is not None]
                b = [r["mae"] for r in rows if r["model"] == model and r[subgroup_key] == g2 and r["mae"] is not None]
                if len(a) >= 2 and len(b) >= 2:
                    _, p = stats.ttest_ind(a, b)
                    f.write(f"- {model}: p = {p:.4f}\n")
    print("Wrote", out_path)


if __name__ == "__main__":
    main()
