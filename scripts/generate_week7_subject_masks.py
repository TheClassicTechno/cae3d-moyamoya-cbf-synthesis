#!/usr/bin/env python3
"""
Generate per-subject masks for Week 7 (Phase 3). Saves one mask per subject to a cache dir.
- If a mask already exists for a subject, skip.
- By default uses the MNI brain mask for all subjects (so the pipeline runs without external tools).
- You can replace cached masks with subject-specific segmentations (e.g. SAM-Med3D, nnUNet)
  by writing {subject_id}.nii.gz into OUT_DIR with the same dimensions (91, 109, 91).

Usage:
  cd <repo-root>/scripts && python3 generate_week7_subject_masks.py

Environment:
  WEEK7_SUBJECT_MASKS_DIR  Output directory (default: <repo-root>/week7_subject_masks)
"""
import os
import sys
import re
import nibabel as nib
import numpy as np

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_REPO_ROOT, "pyproject.toml")):
    _parent = os.path.dirname(_REPO_ROOT)
    if _parent == _REPO_ROOT:
        raise RuntimeError("Could not locate repository root (pyproject.toml not found)")
    _REPO_ROOT = _parent

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from week7_data import get_week7_splits
from week7_preprocess import get_brain_mask, TARGET_SHAPE

OUT_DIR_DEFAULT = os.path.join(_REPO_ROOT, "week7_subject_masks")


def subject_id_from_path(pre_path: str) -> str:
    """Derive a unique subject id from pre path (e.g. pre_2021_001.nii.gz -> 2021_001)."""
    base = os.path.basename(pre_path)
    # pre_2021_001.nii.gz or 2021_001_pre.nii.gz
    base = base.replace(".nii.gz", "").replace(".nii", "")
    if base.startswith("pre_"):
        return base.replace("pre_", "", 1)
    m = re.match(r"(.+)_pre$", base)
    if m:
        return m.group(1)
    return base.replace("pre", "").strip("_") or "unknown"


def main():
    out_dir = os.environ.get("WEEK7_SUBJECT_MASKS_DIR", OUT_DIR_DEFAULT)
    os.makedirs(out_dir, exist_ok=True)
    train_pairs, val_pairs, test_pairs = get_week7_splits()
    all_pairs = train_pairs + val_pairs + test_pairs
    seen = set()
    mni_mask = get_brain_mask()
    assert mni_mask.shape == TARGET_SHAPE, f"MNI mask shape {mni_mask.shape} != {TARGET_SHAPE}"
    n_created = 0
    n_skipped = 0
    for pre_path, _ in all_pairs:
        sid = subject_id_from_path(pre_path)
        if sid in seen:
            continue
        seen.add(sid)
        out_path = os.path.join(out_dir, f"{sid}.nii.gz")
        if os.path.isfile(out_path):
            n_skipped += 1
            continue
        # Save MNI mask as subject mask (same for all; replace with real segmentation if available)
        img = nib.Nifti1Image(mni_mask.astype(np.float32), np.eye(4))
        nib.save(img, out_path)
        n_created += 1
    print(f"Week7 subject masks: {out_dir}")
    print(f"  Created: {n_created}, Already present: {n_skipped}, Subjects: {len(seen)}")
    print("  To use subject-specific segmentations, write {subject_id}.nii.gz (shape 91,109,91) into the same dir.")


if __name__ == "__main__":
    main()
