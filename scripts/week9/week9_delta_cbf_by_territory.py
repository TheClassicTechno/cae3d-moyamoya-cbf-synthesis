#!/usr/bin/env python3
"""
Week 9: Percentage CBF change (Delta CBF) by territory, aligned with Rydham et al.

Computes per territory: mean_pre (from pre-ACZ volume), mean_post_gt and mean_post_pred
(from regional JSON). Then:
  delta_cbf_gt_pct  = (mean_post_gt - mean_pre) / mean_pre * 100
  delta_cbf_pred_pct = (mean_post_pred - mean_pre) / mean_pre * 100

Uses same territory masks and TARGET_SHAPE as week7_regional_eval. Outputs CSV and
optional summary table. When metadata with side_moyamoya (affected/unaffected) is
provided, can stratify and run paired test (separate script or --metadata_csv).

Usage (from repo root):
  python scripts/week9/week9_delta_cbf_by_territory.py --regional_json week8_regional_unet3d.json --output_dir week9_stats
  python scripts/week9/week9_delta_cbf_by_territory.py --regional_json week8_regional_unet3d.json --regional_json week8_regional_resnet3d.json --output_dir week9_stats
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path("<repo-root>")
sys.path.insert(0, str(ROOT / "scripts"))
from week7_data import get_week7_splits, _subject_id_from_path
from week7_preprocess import TARGET_SHAPE, load_volume, load_territory_masks

DATA_DIR = ROOT
MASKS_DIR_DEFAULT = ROOT / "Masks"
MIN_MEAN_PRE_FOR_PCT = 1e-6  # avoid div by zero; below this delta_cbf is NaN


def build_sid_to_pre_path(test_pairs):
    """Return dict subject_id -> pre_path. Prefer exact _subject_id_from_path; fallback: sid in path."""
    sid_to_pre = {}
    for pre_path, _ in test_pairs:
        sid = _subject_id_from_path(pre_path)
        if sid and sid != "unknown":
            sid_to_pre[sid] = pre_path
    # Fallback: for paths where basename doesn't give clean id, try matching test_subjects from path
    for pre_path, _ in test_pairs:
        base = os.path.basename(pre_path)
        sid = _subject_id_from_path(pre_path)
        if sid in ("unknown", "") or not sid:
            for known_sid in list(sid_to_pre.keys()):
                if known_sid in pre_path:
                    break
            if "2020_" in pre_path or "moyamoya_stanford_2020" in pre_path:
                parts = pre_path.split("/")
                for p in parts:
                    if "2020_" in p:
                        cand = p.replace("moyamoya_stanford_", "").strip()
                        if cand not in sid_to_pre:
                            sid_to_pre[cand] = pre_path
                        break
    return sid_to_pre


def compute_mean_pre_per_territory(pre_vol, territory_masks):
    """pre_vol: (H,W,D) same shape as masks. Returns dict territory_name -> mean_pre."""
    if pre_vol.shape != TARGET_SHAPE:
        from scipy.ndimage import zoom
        factors = [TARGET_SHAPE[i] / pre_vol.shape[i] for i in range(3)]
        pre_vol = zoom(pre_vol.astype(np.float32), factors, order=1)
    out = {}
    for name, mask in territory_masks:
        if mask.shape != pre_vol.shape:
            from scipy.ndimage import zoom
            factors = [pre_vol.shape[i] / mask.shape[i] for i in range(3)]
            mask = zoom(mask, factors, order=1)
            mask = (mask > 0.5).astype(np.float32)
        n = float(mask.sum())
        if n < 10:
            out[name] = None
            continue
        out[name] = float((pre_vol * mask).sum() / n)
    return out


def main():
    ap = argparse.ArgumentParser(description="Delta CBF by territory from regional JSON + pre volumes")
    ap.add_argument("--regional_json", action="append", default=[], help="Path to regional JSON (can repeat)")
    ap.add_argument("--masks_dir", default=str(MASKS_DIR_DEFAULT), help="Territory masks dir")
    ap.add_argument("--output_dir", default=str(ROOT / "week9_stats"), help="Output dir for CSV and summary")
    args = ap.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.regional_json:
        # Default: all week8_regional_*.json in repo root (unet3d, resnet3d, FNO_3D, etc.)
        for p in sorted(ROOT.glob("week8_regional_*.json")):
            args.regional_json.append(str(p))
    if not args.regional_json:
        print("No regional JSONs found or provided. Run week9_regional_eval_all_models.py first, or pass --regional_json.")
        return

    _, _, test_pairs = get_week7_splits()
    sid_to_pre = build_sid_to_pre_path(test_pairs)
    masks_dir = args.masks_dir if os.path.isdir(args.masks_dir) else str(ROOT / "Masks")
    if not os.path.isdir(masks_dir):
        print("Masks dir not found:", masks_dir)
        return
    territory_masks = load_territory_masks(masks_dir, TARGET_SHAPE)
    if not territory_masks:
        print("No territory masks loaded from", masks_dir)
        return

    all_rows = []
    for jpath in args.regional_json:
        path = Path(jpath)
        if not path.is_file():
            continue
        model = path.stem.replace("week8_regional_", "").replace("regional_", "")
        with open(path) as f:
            data = json.load(f)
        for vol in data.get("per_volume", []):
            sid = vol.get("id", "")
            pre_path = sid_to_pre.get(sid)
            if not pre_path or not os.path.isfile(pre_path):
                continue
            try:
                pre_vol = load_volume(pre_path, target_shape=TARGET_SHAPE)
            except Exception as e:
                continue
            mean_pre_by_territory = compute_mean_pre_per_territory(pre_vol, territory_masks)
            per_region = vol.get("per_region", {})
            for ter_name, r in per_region.items():
                mean_gt = r.get("mean_perfusion_gt")
                mean_pred = r.get("mean_perfusion_pred")
                n_vox = r.get("n_voxels", 0)
                if n_vox < 10:
                    continue
                mean_pre = mean_pre_by_territory.get(ter_name)
                if mean_pre is None:
                    continue
                # Absolute change (normalized units): avoids denominator instability
                delta_gt_abs = float(mean_gt) - mean_pre if mean_gt is not None else float("nan")
                delta_pred_abs = float(mean_pred) - mean_pre if mean_pred is not None else float("nan")
                if mean_pre < MIN_MEAN_PRE_FOR_PCT:
                    delta_gt_pct = float("nan")
                    delta_pred_pct = float("nan")
                else:
                    delta_gt_pct = (float(mean_gt) - mean_pre) / mean_pre * 100.0 if mean_gt is not None else float("nan")
                    delta_pred_pct = (float(mean_pred) - mean_pre) / mean_pre * 100.0 if mean_pred is not None else float("nan")
                all_rows.append({
                    "model": model,
                    "subject_id": sid,
                    "territory": ter_name,
                    "mean_pre": round(mean_pre, 6),
                    "mean_post_gt": round(float(mean_gt), 6) if mean_gt is not None else "",
                    "mean_post_pred": round(float(mean_pred), 6) if mean_pred is not None else "",
                    "delta_cbf_gt_pct": round(delta_gt_pct, 4) if not np.isnan(delta_gt_pct) else "",
                    "delta_cbf_pred_pct": round(delta_pred_pct, 4) if not np.isnan(delta_pred_pct) else "",
                    "delta_cbf_gt_abs": round(delta_gt_abs, 6) if not np.isnan(delta_gt_abs) else "",
                    "delta_cbf_pred_abs": round(delta_pred_abs, 6) if not np.isnan(delta_pred_abs) else "",
                })

    csv_path = out_dir / "delta_cbf_by_territory.csv"
    if all_rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["model", "subject_id", "territory", "mean_pre", "mean_post_gt", "mean_post_pred", "delta_cbf_gt_pct", "delta_cbf_pred_pct", "delta_cbf_gt_abs", "delta_cbf_pred_abs"])
            w.writeheader()
            w.writerows(all_rows)
        print("Wrote", csv_path, "with", len(all_rows), "rows")
    else:
        print("No rows produced. Check regional JSON ids match test_pairs subject ids and pre paths exist.")

    summary_path = out_dir / "delta_cbf_summary.md"
    if all_rows:
        models = sorted(set(r["model"] for r in all_rows))
        territories = sorted(set(r["territory"] for r in all_rows))
        lines = ["## Delta CBF by territory (mean % change across subjects)", ""]
        for model in models:
            lines.append("### " + model)
            lines.append("")
            lines.append("| Territory | GT delta% | pred delta% | GT delta_abs | pred delta_abs | n |")
            lines.append("|-----------|-----------|-------------|--------------|----------------|---|")
            for ter in territories:
                vals_gt = [r["delta_cbf_gt_pct"] for r in all_rows if r["model"] == model and r["territory"] == ter and r["delta_cbf_gt_pct"] != ""]
                vals_pred = [r["delta_cbf_pred_pct"] for r in all_rows if r["model"] == model and r["territory"] == ter and r["delta_cbf_pred_pct"] != ""]
                vals_gt_abs = [r["delta_cbf_gt_abs"] for r in all_rows if r["model"] == model and r["territory"] == ter and r.get("delta_cbf_gt_abs","") != ""]
                vals_pred_abs = [r["delta_cbf_pred_abs"] for r in all_rows if r["model"] == model and r["territory"] == ter and r.get("delta_cbf_pred_abs","") != ""]
                n = len(vals_gt)
                if n:
                    m_gt = np.nanmean([float(x) for x in vals_gt]) if vals_gt else float("nan")
                    m_pred = np.nanmean([float(x) for x in vals_pred]) if vals_pred else float("nan")
                    m_gt_abs = np.nanmean([float(x) for x in vals_gt_abs]) if vals_gt_abs else float("nan")
                    m_pred_abs = np.nanmean([float(x) for x in vals_pred_abs]) if vals_pred_abs else float("nan")
                    lines.append("| %s | %.2f | %.2f | %.4f | %.4f | %d |" % (ter, m_gt, m_pred, m_gt_abs, m_pred_abs, n))
            lines.append("")
        with open(summary_path, "w") as f:
            f.write("\n".join(lines))
        print("Wrote", summary_path)


if __name__ == "__main__":
    main()
