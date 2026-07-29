#!/usr/bin/env python3
"""
Week 9: Affected vs unaffected territory analysis (requires disease laterality metadata).

Reads delta_cbf_by_territory.csv and metadata CSV with subject_id, side_moyamoya (left | right | bilateral).
For each subject: labels left MCA/ACA/PCA as affected when side_moyamoya in (left, bilateral),
right as affected when side_moyamoya in (right, bilateral). Unaffected = contralateral.
Computes mean delta_cbf_gt and delta_cbf_pred per subject in affected vs unaffected territories,
then runs paired t-test and Wilcoxon (within-subject: affected vs unaffected).

Without --metadata_csv, exits with a clear message (no overwrite).

Usage (when metadata exists):
  python scripts/week9/week9_affected_vs_unaffected.py --delta_cbf_csv week9_stats/delta_cbf_by_territory.csv --metadata_csv subject_metadata.csv --output_dir week9_stats
"""

import argparse
import csv
from pathlib import Path

import numpy as np

ROOT = Path("<repo-root>")
DEFAULT_DELTA_CSV = ROOT / "week9_stats" / "delta_cbf_by_territory.csv"
DEFAULT_OUT = ROOT / "week9_stats"

# Territory name -> hemisphere (left / right). Vascular_territory = whole, skip or treat separately.
LEFT_TERRITORIES = {"MNI_left_ACA_2mm", "MNI_left_MCA_2mm", "MNI_left_PCA_2mm", "MNI_left_cerebellum_2mm", "MNI_left_pons_medulla_2mm"}
RIGHT_TERRITORIES = {"MNI_right_ACA_2mm", "MNI_right_MCA_2mm", "MNI_right_PCA_2mm", "MNI_right_cerebellum_2mm", "MNI_right_pons_medulla_2mm"}


def load_metadata(path: Path) -> dict:
    """Return { subject_id: { "side_moyamoya": "left"|"right"|"bilateral", ... } }."""
    out = {}
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            sid = (row.get("subject_id") or "").strip()
            if not sid:
                continue
            out[sid] = {k.strip(): v.strip() for k, v in row.items() if k}
    return out


def load_delta_cbf(path: Path) -> list:
    """Return list of dicts with model, subject_id, territory, delta_cbf_gt_pct, delta_cbf_pred_pct."""
    rows = []
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            row["delta_cbf_gt_pct"] = row.get("delta_cbf_gt_pct")
            row["delta_cbf_pred_pct"] = row.get("delta_cbf_pred_pct")
            rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser(description="Affected vs unaffected analysis (requires metadata)")
    ap.add_argument("--delta_cbf_csv", default=str(DEFAULT_DELTA_CSV), help="CSV from week9_delta_cbf_by_territory.py")
    ap.add_argument("--metadata_csv", default="", help="CSV with subject_id, side_moyamoya (left/right/bilateral)")
    ap.add_argument("--output_dir", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    if not args.metadata_csv or not Path(args.metadata_csv).is_file():
        print("Affected vs unaffected analysis requires --metadata_csv with subject_id and side_moyamoya (left/right/bilateral).")
        print("See scripts/week9/SUBJECT_METADATA_SCHEMA.md.")
        print("No data overwritten; exiting.")
        return

    delta_path = Path(args.delta_cbf_csv)
    if not delta_path.is_file():
        print("Delta CBF CSV not found:", delta_path)
        return
    meta = load_metadata(Path(args.metadata_csv))
    if not meta:
        print("Metadata CSV is empty or has no subject_id.")
        return

    rows = load_delta_cbf(delta_path)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build per (model, subject): list of (territory, delta_gt, delta_pred, is_affected)
    # is_affected: True if territory is on disease side
    by_model_subject = {}
    for r in rows:
        sid = (r.get("subject_id") or "").strip()
        side = (meta.get(sid) or {}).get("side_moyamoya", "").strip().lower()
        if not side or side not in ("left", "right", "bilateral"):
            continue
        ter = (r.get("territory") or "").strip()
        if ter not in LEFT_TERRITORIES and ter not in RIGHT_TERRITORIES:
            continue
        is_left = ter in LEFT_TERRITORIES
        if side == "left":
            affected = is_left
        elif side == "right":
            affected = not is_left
        else:
            affected = is_left  # bilateral: treat left as affected for this aggregation (or both)
        try:
            d_gt = float(r.get("delta_cbf_gt_pct") or np.nan)
            d_pred = float(r.get("delta_cbf_pred_pct") or np.nan)
        except (TypeError, ValueError):
            continue
        key = (r.get("model", ""), sid)
        if key not in by_model_subject:
            by_model_subject[key] = {"affected_gt": [], "affected_pred": [], "unaffected_gt": [], "unaffected_pred": []}
        if affected:
            by_model_subject[key]["affected_gt"].append(d_gt)
            by_model_subject[key]["affected_pred"].append(d_pred)
        else:
            by_model_subject[key]["unaffected_gt"].append(d_gt)
            by_model_subject[key]["unaffected_pred"].append(d_pred)

    # Per subject: mean affected, mean unaffected (for paired test)
    paired_data = {}
    for (model, sid), v in by_model_subject.items():
        if not v["affected_gt"] or not v["unaffected_gt"]:
            continue
        a_gt = np.nanmean(v["affected_gt"])
        a_pred = np.nanmean(v["affected_pred"]) if v["affected_pred"] else np.nan
        u_gt = np.nanmean(v["unaffected_gt"])
        u_pred = np.nanmean(v["unaffected_pred"]) if v["unaffected_pred"] else np.nan
        if model not in paired_data:
            paired_data[model] = {"affected_gt": [], "unaffected_gt": [], "affected_pred": [], "unaffected_pred": []}
        paired_data[model]["affected_gt"].append(a_gt)
        paired_data[model]["unaffected_gt"].append(u_gt)
        paired_data[model]["affected_pred"].append(a_pred)
        paired_data[model]["unaffected_pred"].append(u_pred)

    if not paired_data:
        print("No subject with both affected and unaffected territories after merging metadata.")
        return

    try:
        from scipy import stats
    except ImportError:
        stats = None

    out_md = out_dir / "affected_vs_unaffected_summary.md"
    lines = ["## Affected vs unaffected territories (delta CBF %)", ""]
    for model in sorted(paired_data.keys()):
        d = paired_data[model]
        a_gt = np.array(d["affected_gt"])
        u_gt = np.array(d["unaffected_gt"])
        a_pred = np.array(d["affected_pred"])
        u_pred = np.array(d["unaffected_pred"])
        n = len(a_gt)
        lines.append("### " + model)
        lines.append("- n subjects: %d" % n)
        lines.append("- Mean delta_cbf_gt: affected %.2f%%, unaffected %.2f%%" % (np.mean(a_gt), np.mean(u_gt)))
        lines.append("- Mean delta_cbf_pred: affected %.2f%%, unaffected %.2f%%" % (np.nanmean(a_pred), np.nanmean(u_pred)))
        if stats and n >= 2:
            t_gt, p_gt = stats.ttest_rel(a_gt, u_gt)
            try:
                w_gt, pw_gt = stats.wilcoxon(a_gt, u_gt)
            except Exception:
                pw_gt = float("nan")
            lines.append("- Paired t-test (affected vs unaffected, delta_cbf_gt): p = %.4f" % p_gt)
            lines.append("- Wilcoxon (affected vs unaffected, delta_cbf_gt): p = %.4f" % pw_gt)
        lines.append("")
    with open(out_md, "w") as f:
        f.write("\n".join(lines))
    print("Wrote", out_md)


if __name__ == "__main__":
    main()
