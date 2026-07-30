#!/usr/bin/env python3
"""
Compute R² and Bland–Altman (per-subject mean in-brain intensity) for Med3DVLM and SAM-Med3D CBF
on the Week7 test set, using the same definitions as the main paper (32 subjects).

Uses best checkpoints for seed 42 (same convention as primary metrics row). Writes
third_party_foundation_3d/foundation_table1_r2_ba_seed42.json for pasting into Table 1.

Run from repo root:
  PYTHONPATH=/data1/julih/scripts:/data1/julih/third_party_foundation_3d/SAM-Med3D:/data1/julih/third_party_foundation_3d/Med3DVLM \\
    python3 third_party_foundation_3d/compute_foundation_r2_ba_table1.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = "/data1/julih"
REPO_SCRIPTS = os.path.join(ROOT, "scripts")
SAM_REPO = os.path.join(ROOT, "third_party_foundation_3d", "SAM-Med3D")
MED_REPO = os.path.join(ROOT, "third_party_foundation_3d", "Med3DVLM")
for p in (REPO_SCRIPTS, SAM_REPO, MED_REPO):
    if p not in sys.path:
        sys.path.insert(0, p)

from week7_data import get_week7_splits, Week7VolumePairs3D
from week7_preprocess import TARGET_SHAPE, get_brain_mask_for_shape

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
BATCH_SIZE = 2
OUT_JSON = os.path.join(ROOT, "third_party_foundation_3d", "foundation_table1_r2_ba_seed42.json")


def mean_intensity_in_brain(vol: np.ndarray) -> float:
    mask = get_brain_mask_for_shape(vol.shape, dtype=np.float32)
    m = (mask > 0.5).astype(bool)
    if not m.any():
        return float("nan")
    return float(vol.astype(np.float64)[m].mean())


def r2_and_bland_altman(pred_means: list[float], tgt_means: list[float]) -> dict:
    pr = np.asarray(pred_means, dtype=np.float64)
    tg = np.asarray(tgt_means, dtype=np.float64)
    ss_res = np.sum((tg - pr) ** 2)
    ss_tot = np.sum((tg - np.mean(tg)) ** 2)
    r2 = float(1.0 - ss_res / (ss_tot + 1e-12))
    diff = pr - tg
    bias = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0
    lo_lo = bias - 1.96 * sd
    lo_hi = bias + 1.96 * sd
    return {"r2": r2, "bias": bias, "loa_low": lo_lo, "loa_high": lo_hi, "n_subjects": len(pr)}


def run_sam() -> dict:
    import torch.nn.functional as F
    from run_sam_med3d_week7_cbf_regressor import SamMed3DCBFRegressor, load_sam_med3d

    ckpt = os.path.join(ROOT, "third_party_foundation_3d", "sam_med3d_week7_cbf", "sam_med3d_cbf_decoder_best_seed%d.pt" % SEED)
    if not os.path.isfile(ckpt):
        raise FileNotFoundError("Missing %s" % ckpt)
    sam = load_sam_med3d().to(DEVICE)
    model = SamMed3DCBFRegressor(sam, freeze_encoder=True).to(DEVICE)
    model.decoder.load_state_dict(torch.load(ckpt, map_location=DEVICE)["decoder"])
    model.eval()

    _, _, test_pairs = get_week7_splits()
    test_ds = Week7VolumePairs3D(test_pairs, augment=False)
    loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    pred_means, tgt_means = [], []
    with torch.no_grad():
        for pre, post in loader:
            pre = pre.to(DEVICE)
            post = post.to(DEVICE)
            pred = model.forward_pred(pre)
            for i in range(pred.shape[0]):
                p = pred[i, 0].cpu().numpy()
                t = post[i, 0].cpu().numpy()
                pred_means.append(mean_intensity_in_brain(p))
                tgt_means.append(mean_intensity_in_brain(t))
    ba = r2_and_bland_altman(pred_means, tgt_means)
    return {"model": "SAM-Med3D_CBF", "eval_seed": SEED, **ba}


def run_med3dvlm() -> dict:
    import torch.nn.functional as F
    from run_med3dvlm_week7_cvr import DCFormerCVR, ENC_SIZE, TARGET_SHAPE_3D, resize_vol

    ckpt = os.path.join(ROOT, "third_party_foundation_3d", "med3dvlm_week7_cvr", "med3dvlm_cvr_best_seed%d.pt" % SEED)
    if not os.path.isfile(ckpt):
        raise FileNotFoundError("Missing %s" % ckpt)
    model = DCFormerCVR(input_size=ENC_SIZE, freeze_encoder=True).to(DEVICE)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE)["model"])
    model.eval()

    _, _, test_pairs = get_week7_splits()
    test_ds = Week7VolumePairs3D(test_pairs, augment=False)
    loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    pred_means, tgt_means = [], []
    with torch.no_grad():
        for pre, post in loader:
            pre = pre.to(DEVICE)
            post = post.to(DEVICE)
            pre_128 = resize_vol(pre, ENC_SIZE)
            pred_128 = model(pre_128)
            pred = resize_vol(pred_128, TARGET_SHAPE_3D)
            for i in range(pred.shape[0]):
                p = pred[i, 0].cpu().numpy()
                t = post[i, 0].cpu().numpy()
                pred_means.append(mean_intensity_in_brain(p))
                tgt_means.append(mean_intensity_in_brain(t))
    ba = r2_and_bland_altman(pred_means, tgt_means)
    return {"model": "Med3DVLM", "eval_seed": SEED, **ba}


def main() -> None:
    out = {
        "note": "R² and Bland–Altman from per-subject mean in-brain intensity (same brain mask as metrics_in_brain). Seed 42 checkpoints.",
        "sam_med3d_cbf": run_sam(),
        "med3dvlm": run_med3dvlm(),
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print("Wrote", OUT_JSON)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
