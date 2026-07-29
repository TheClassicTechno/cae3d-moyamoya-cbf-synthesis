#!/usr/bin/env python3
"""
Week 7 data loader for Diffusion_3D_Latent.
Use get_week7_splits() and load_volume_week7() when training/eval with Week 7 pipeline
(91×109×91 brain mask, combined 2020-2023 split, pad 96×112×96 for model).
"""
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_REPO_ROOT, "pyproject.toml")):
    _parent = os.path.dirname(_REPO_ROOT)
    if _parent == _REPO_ROOT:
        raise RuntimeError("Could not locate repository root (pyproject.toml not found)")
    _REPO_ROOT = _parent

sys.path.insert(0, os.path.join(_REPO_ROOT, 'scripts'))

def get_week7_splits():
    """Return (train_pairs, val_pairs, test_pairs) from combined 2020-2023. Each pair is (pre_path, post_path)."""
    from week7_data import get_week7_splits as _get
    return _get()


def load_volume_week7(nii_path: str, pad_shape=(96, 112, 96)):
    """Load with Week7 preprocessing (91×109×91, brain mask, minmax) then pad to pad_shape. Returns float32."""
    from week7_preprocess import load_volume, TARGET_SHAPE
    import numpy as np
    vol = load_volume(nii_path, target_shape=TARGET_SHAPE, apply_mask=True, minmax=True)
    if vol.shape != pad_shape:
        out = np.zeros(pad_shape, dtype=vol.dtype)
        sh = [min(vol.shape[i], pad_shape[i]) for i in range(3)]
        out[:sh[0], :sh[1], :sh[2]] = vol[:sh[0], :sh[1], :sh[2]]
        return out.astype(np.float32)
    return vol.astype(np.float32)
