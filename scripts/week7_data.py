#!/usr/bin/env python3
"""
Week 7 unified dataset for 2D and 3D models.
- Uses combined 2020-2023 split only.
- Same preprocessing: brain mask, 91x109x91, pad, min-max (via week7_preprocess).
- Same augmentations at train time: flip_lr, flip_ud, intensity_scale (optional flip_fb for 3D).
- 2D: returns middle axial slice (91, 109) from the preprocessed volume.
- 3D: returns full volume (1, 91, 109, 91).
Metrics should be computed in same dimensionality (e.g. 2D middle slice vs 2D middle slice; 3D full vs 3D full).
Phase 3: Week7VolumePairs3DWithMasks returns (pre, post, mask) for mask-weighted loss.
"""
import os
import random
from typing import List, Tuple, Optional

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_REPO_ROOT, "pyproject.toml")):
    _parent = os.path.dirname(_REPO_ROOT)
    if _parent == _REPO_ROOT:
        raise RuntimeError("Could not locate repository root (pyproject.toml not found)")
    _REPO_ROOT = _parent

import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.ndimage import zoom
import nibabel as nib

from week7_preprocess import (
    load_volume,
    load_volume_cropped,
    load_pre_post_pair,
    augment_volume,
    get_pre_post_pairs,
    get_week7_splits,
    get_brain_mask_for_shape,
    get_brain_bounding_box,
    get_brain_mask,
    TARGET_SHAPE,
    USE_AFFINE,
    _subject_id_from_path,
)


# Same augmentation range for 2D and 3D (from week7tasks: left/right, top/bottom flips; future: intensity)
def _random_augment() -> Tuple[bool, bool, bool, Optional[float]]:
    flip_lr = random.random() < 0.5
    flip_ud = random.random() < 0.5
    flip_fb = random.random() < 0.5
    # intensity scale in [0.9, 1.1]
    intensity_scale = 0.9 + 0.2 * random.random()
    return flip_lr, flip_ud, flip_fb, intensity_scale


def _random_augment_geometric_only() -> Tuple[bool, bool, bool, float]:
    """Random flips only; intensity_scale=1.0 (no intensity change). For Tied-Augment Phase 2: second view geometric-only."""
    flip_lr = random.random() < 0.5
    flip_ud = random.random() < 0.5
    flip_fb = random.random() < 0.5
    return flip_lr, flip_ud, flip_fb, 1.0


class Week7VolumePairs3D(Dataset):
    """3D dataset: returns (pre_vol, post_vol) as (1, H, W, D) tensors, same preprocessing and aug."""

    def __init__(
        self,
        pairs: List[Tuple[str, str]],
        augment: bool = False,
        target_shape: Tuple[int, int, int] = TARGET_SHAPE,
        use_affine: Optional[bool] = None,
    ):
        self.pairs = pairs
        self.augment = augment
        self.target_shape = target_shape
        self.use_affine = use_affine if use_affine is not None else USE_AFFINE

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pre_path, post_path = self.pairs[idx]
        pre, post = load_pre_post_pair(
            pre_path, post_path, use_affine=self.use_affine, target_shape=self.target_shape
        )
        if self.augment:
            fl, fu, ff, scale = _random_augment()
            pre = augment_volume(pre, flip_lr=fl, flip_ud=fu, flip_fb=ff, intensity_scale=scale)
            post = augment_volume(post, flip_lr=fl, flip_ud=fu, flip_fb=ff, intensity_scale=scale)
        pre_t = torch.from_numpy(pre).unsqueeze(0).float()   # (1, H, W, D)
        post_t = torch.from_numpy(post).unsqueeze(0).float()
        return pre_t, post_t


class Week7VolumePairs3DCropped(Dataset):
    """3D dataset with brain-only crop: load 91x109x91, apply mask, crop to bbox, pad to crop_pad_shape.
    Same split and augmentations as Week7VolumePairs3D. For controlled crop experiment vs full-volume."""

    def __init__(
        self,
        pairs: List[Tuple[str, str]],
        augment: bool = False,
        target_shape: Tuple[int, int, int] = TARGET_SHAPE,
        crop_pad_shape: Tuple[int, int, int] = (72, 88, 72),
        use_affine: Optional[bool] = None,
    ):
        self.pairs = list(pairs)
        self.augment = augment
        self.target_shape = target_shape
        self.crop_pad_shape = crop_pad_shape
        self.use_affine = use_affine if use_affine is not None else USE_AFFINE

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pre_path, post_path = self.pairs[idx]
        pre, post = load_pre_post_pair(
            pre_path, post_path, use_affine=self.use_affine, target_shape=self.target_shape
        )
        # Crop to brain bbox and pad (same as load_volume_cropped logic)
        from week7_preprocess import get_brain_bounding_box, get_brain_mask
        mask = get_brain_mask()
        sl_d, sl_h, sl_w = get_brain_bounding_box(mask)
        pre = pre[sl_d, sl_h, sl_w].copy()
        post = post[sl_d, sl_h, sl_w].copy()
        pd, ph, pw = self.crop_pad_shape
        out_pre = np.zeros(self.crop_pad_shape, dtype=np.float32)
        out_post = np.zeros(self.crop_pad_shape, dtype=np.float32)
        cd, ch, cw = pre.shape[0], pre.shape[1], pre.shape[2]
        out_pre[: min(cd, pd), : min(ch, ph), : min(cw, pw)] = pre[: min(cd, pd), : min(ch, ph), : min(cw, pw)]
        out_post[: min(cd, pd), : min(ch, ph), : min(cw, pw)] = post[: min(cd, pd), : min(ch, ph), : min(cw, pw)]
        pre, post = out_pre, out_post
        if self.augment:
            fl, fu, ff, scale = _random_augment()
            pre = augment_volume(pre, flip_lr=fl, flip_ud=fu, flip_fb=ff, intensity_scale=scale)
            post = augment_volume(post, flip_lr=fl, flip_ud=fu, flip_fb=ff, intensity_scale=scale)
        pre_t = torch.from_numpy(pre).unsqueeze(0).float()
        post_t = torch.from_numpy(post).unsqueeze(0).float()
        return pre_t, post_t


class Week7SlicePairs2D(Dataset):
    """2D dataset: same preprocessing as 3D, then take middle axial slice. Same augment (flip_lr, flip_ud, scale)."""

    def __init__(
        self,
        pairs: List[Tuple[str, str]],
        augment: bool = False,
        target_shape: Tuple[int, int, int] = TARGET_SHAPE,
        use_affine: Optional[bool] = None,
    ):
        self.pairs = pairs
        self.augment = augment
        self.target_shape = target_shape
        self.use_affine = use_affine if use_affine is not None else USE_AFFINE

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pre_path, post_path = self.pairs[idx]
        pre, post = load_pre_post_pair(
            pre_path, post_path, use_affine=self.use_affine, target_shape=self.target_shape
        )
        # Middle axial slice: index D//2
        mid = self.target_shape[2] // 2
        pre_2d = pre[:, :, mid]   # (H, W)
        post_2d = post[:, :, mid]
        if self.augment:
            fl, fu, _, scale = _random_augment()
            # (H,W) -> (H,W,1) so augment_volume works; flip_fb on dim 2 is no-op for 1 slice
            pre_2d = augment_volume(pre_2d[:, :, np.newaxis], flip_lr=fl, flip_ud=fu, flip_fb=False, intensity_scale=scale).squeeze(-1)
            post_2d = augment_volume(post_2d[:, :, np.newaxis], flip_lr=fl, flip_ud=fu, flip_fb=False, intensity_scale=scale).squeeze(-1)
        pre_t = torch.from_numpy(pre_2d).unsqueeze(0).float()   # (1, H, W)
        post_t = torch.from_numpy(post_2d).unsqueeze(0).float()
        return pre_t, post_t


# Phase 3: subject-specific masks dir (run generate_week7_subject_masks.py first)
SUBJECT_MASKS_DIR_DEFAULT = os.path.join(_REPO_ROOT, "week7_subject_masks")


class Week7VolumePairs3DWithMasks(Dataset):
    """3D dataset returning (pre_vol, post_vol, mask_vol). Mask is per-subject from cache or MNI fallback."""

    def __init__(
        self,
        pairs: List[Tuple[str, str]],
        augment: bool = False,
        target_shape: Tuple[int, int, int] = TARGET_SHAPE,
        masks_dir: Optional[str] = None,
        pad_shape: Optional[Tuple[int, int, int]] = None,
        use_affine: Optional[bool] = None,
    ):
        self.pairs = pairs
        self.augment = augment
        self.target_shape = target_shape
        self.masks_dir = masks_dir or os.environ.get("WEEK7_SUBJECT_MASKS_DIR", SUBJECT_MASKS_DIR_DEFAULT)
        self.pad_shape = pad_shape  # if set, mask is resized to pad_shape for loss (e.g. 96,112,96)
        self.use_affine = use_affine if use_affine is not None else USE_AFFINE

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pre_path, post_path = self.pairs[idx]
        pre, post = load_pre_post_pair(
            pre_path, post_path, use_affine=self.use_affine, target_shape=self.target_shape
        )
        sid = _subject_id_from_path(pre_path)
        mask_path = os.path.join(self.masks_dir, f"{sid}.nii.gz")
        if os.path.isfile(mask_path):
            img = nib.load(mask_path)
            mask = np.asarray(img.dataobj).squeeze().astype(np.float32)
            if mask.shape != self.target_shape:
                factors = [self.target_shape[i] / mask.shape[i] for i in range(3)]
                mask = zoom(mask, factors, order=0)
            mask = (mask > 0.5).astype(np.float32)
        else:
            mask = get_brain_mask_for_shape(self.target_shape)
        if self.augment:
            fl, fu, ff, scale = _random_augment()
            pre = augment_volume(pre, flip_lr=fl, flip_ud=fu, flip_fb=ff, intensity_scale=scale)
            post = augment_volume(post, flip_lr=fl, flip_ud=fu, flip_fb=ff, intensity_scale=scale)
            mask = augment_volume(mask, flip_lr=fl, flip_ud=fu, flip_fb=ff, intensity_scale=1.0)
            mask = (mask > 0.5).astype(np.float32)
        out_shape = self.pad_shape if self.pad_shape else self.target_shape
        if mask.shape != out_shape:
            factors = [out_shape[i] / mask.shape[i] for i in range(3)]
            mask = zoom(mask, factors, order=0)
            mask = (mask > 0.5).astype(np.float32)
        pre_t = torch.from_numpy(pre).unsqueeze(0).float()
        post_t = torch.from_numpy(post).unsqueeze(0).float()
        mask_t = torch.from_numpy(mask).unsqueeze(0).float()
        return pre_t, post_t, mask_t


class Week7VolumePairs3DTiedAugment(Dataset):
    """
    Tied-Augment dataset: returns two independently augmented views per sample.
    For each sample: ((pre1, post1), (pre2, post2)) where aug1 and aug2 are different.
    Same (fl, fu, ff, scale) applied to pre and post within each view (paired).
    Used for L = L_recon + tw * MSE(features(pre1), features(pre2)).
    Phase 2 (WEEK89_IMPLEMENTATION_PLAN): geometric_only_second_view=True uses only flips for view2 (no intensity scaling).
    """

    def __init__(
        self,
        pairs: List[Tuple[str, str]],
        augment: bool = True,
        target_shape: Tuple[int, int, int] = TARGET_SHAPE,
        geometric_only_second_view: bool = False,
        use_affine: Optional[bool] = None,
    ):
        self.pairs = pairs
        self.augment = augment
        self.target_shape = target_shape
        self.geometric_only_second_view = geometric_only_second_view
        self.use_affine = use_affine if use_affine is not None else USE_AFFINE

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pre_path, post_path = self.pairs[idx]
        pre, post = load_pre_post_pair(
            pre_path, post_path, use_affine=self.use_affine, target_shape=self.target_shape
        )
        if self.augment:
            fl1, fu1, ff1, scale1 = _random_augment()
            if self.geometric_only_second_view:
                fl2, fu2, ff2, scale2 = _random_augment_geometric_only()
            else:
                fl2, fu2, ff2, scale2 = _random_augment()
            pre1 = augment_volume(pre, flip_lr=fl1, flip_ud=fu1, flip_fb=ff1, intensity_scale=scale1)
            post1 = augment_volume(post, flip_lr=fl1, flip_ud=fu1, flip_fb=ff1, intensity_scale=scale1)
            pre2 = augment_volume(pre, flip_lr=fl2, flip_ud=fu2, flip_fb=ff2, intensity_scale=scale2)
            post2 = augment_volume(post, flip_lr=fl2, flip_ud=fu2, flip_fb=ff2, intensity_scale=scale2)
        else:
            pre1 = pre.copy()
            post1 = post.copy()
            pre2 = pre.copy()
            post2 = post.copy()
        pre1_t = torch.from_numpy(pre1).unsqueeze(0).float()
        post1_t = torch.from_numpy(post1).unsqueeze(0).float()
        pre2_t = torch.from_numpy(pre2).unsqueeze(0).float()
        post2_t = torch.from_numpy(post2).unsqueeze(0).float()
        return (pre1_t, post1_t), (pre2_t, post2_t)
