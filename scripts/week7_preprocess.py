#!/usr/bin/env python3
"""
Week 7 unified preprocessing for fair 2D vs 3D comparison.
- Optional affine registration to MNI152 2mm (12-DOF, FSL FLIRT): set WEEK7_AFFINE=1.
  When enabled, pre-ACZ is registered to the reference; the same transform is applied to
  post-ACZ so the pair remains aligned. Registered volumes are cached under AFFINE_CACHE_DIR.
- Apply MNI brain mask to all volumes (outside mask = 0).
- Same dimensions: 91 x 109 x 91 (match MNI152_T1_2mm_brain_mask_dil).
- Pad with 0s when needed; resize to target; min-max norm.
- Same augmentations for 2D and 3D: flip LR, flip UD, intensity scale (optional, at train time).
Use with combined 2020-2023 split so train/val/test are identical across all models.
"""
import os
import re
import json
import hashlib
import subprocess
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
from typing import Tuple, Optional, List

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_REPO_ROOT, "pyproject.toml")):
    _parent = os.path.dirname(_REPO_ROOT)
    if _parent == _REPO_ROOT:
        raise RuntimeError("Could not locate repository root (pyproject.toml not found)")
    _REPO_ROOT = _parent

# Same for everyone (from week7tasks.txt)
TARGET_SHAPE = (91, 109, 91)  # MNI 2mm brain mask dimensions
BRAIN_MASK_PATH = os.path.join(_REPO_ROOT, "MNI152_T1_2mm_brain_mask_dil.nii.gz")
COMBINED_SPLIT_PATH = os.path.join(_REPO_ROOT, "combined_subject_split.json")

# Affine registration (Path A): reference = MNI 2mm T1 brain or mask; cache under this dir
_MNI_DIR = os.path.dirname(BRAIN_MASK_PATH)
MNI_REFERENCE_PATH = os.environ.get(
    "WEEK7_MNI_REFERENCE",
    os.path.join(_MNI_DIR, "MNI152_T1_2mm_brain.nii.gz"),
)
if not os.path.isfile(MNI_REFERENCE_PATH):
    MNI_REFERENCE_PATH = BRAIN_MASK_PATH  # fallback: use mask as reference (same grid)
AFFINE_CACHE_DIR = os.environ.get("WEEK7_AFFINE_CACHE", os.path.join(_MNI_DIR, "week7_affine_cache"))
USE_AFFINE = os.environ.get("WEEK7_AFFINE", "0").lower() in ("1", "true", "yes")

_mask_cache = None
_flirt_warned = False


def _flirt_available() -> bool:
    """True if FSL flirt and applyxfm are on PATH."""
    import shutil
    return bool(shutil.which("flirt") and shutil.which("applyxfm"))


def _affine_cache_key(pre_path: str, post_path: str) -> str:
    """Stable key for caching registered pair (pre and post paths + ref mtime)."""
    ref_mtime = str(os.path.getmtime(MNI_REFERENCE_PATH)) if os.path.isfile(MNI_REFERENCE_PATH) else "0"
    raw = f"{os.path.abspath(pre_path)}|{os.path.abspath(post_path)}|{ref_mtime}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def _register_pair_to_mni_dipy(
    pre_path: str,
    post_path: str,
    ref_path: str,
    pre_out: str,
    post_out: str,
) -> None:
    """
    12-DOF affine registration using DIPY (fallback when FSL not available).
    Registers pre to reference, applies same transform to post; writes pre_out, post_out.
    """
    try:
        from dipy.align import affine_registration
    except ImportError:
        raise RuntimeError("WEEK7_AFFINE=1 and FSL not available. Install dipy: pip install dipy")
    ref_img = nib.load(ref_path)
    ref_data = np.asarray(ref_img.dataobj).squeeze().astype(np.float64)
    ref_affine = ref_img.affine.copy()
    pre_img = nib.load(pre_path)
    pre_data = np.asarray(pre_img.dataobj).squeeze().astype(np.float64)
    if pre_data.ndim == 4:
        pre_data = pre_data[..., 0]
    post_img = nib.load(post_path)
    post_data = np.asarray(post_img.dataobj).squeeze().astype(np.float64)
    if post_data.ndim == 4:
        post_data = post_data[..., 0]
    pipeline = ["center_of_mass", "translation", "rigid", "affine"]
    pre_reg_data, reg_affine = affine_registration(
        pre_data, ref_data,
        moving_affine=pre_img.affine,
        static_affine=ref_affine,
        pipeline=pipeline,
        nbins=32,
        level_iters=[100, 50, 20],
        sigmas=[3.0, 1.0, 0.0],
    )
    from scipy.ndimage import affine_transform
    T = np.linalg.inv(post_img.affine) @ np.linalg.inv(reg_affine) @ ref_affine
    post_reg_data = affine_transform(
        post_data.astype(np.float64), T[:3, :3], offset=T[:3, 3],
        output_shape=ref_data.shape, order=1, cval=0.0,
    ).astype(np.float32)
    nib.save(nib.Nifti1Image(np.asarray(pre_reg_data, dtype=np.float32), ref_affine), pre_out)
    nib.save(nib.Nifti1Image(post_reg_data, ref_affine), post_out)


def register_pair_to_mni(
    pre_path: str,
    post_path: str,
    ref_path: Optional[str] = None,
    cache_dir: Optional[str] = None,
    dof: int = 12,
) -> Tuple[str, str]:
    """
    Register pre-ACZ to MNI (12-DOF affine), apply same transform to post-ACZ.
    Returns (path_to_registered_pre, path_to_registered_post). Uses cache when present.
    Uses FSL FLIRT when available; otherwise DIPY (pip install dipy).
    """
    ref_path = ref_path or MNI_REFERENCE_PATH
    cache_dir = cache_dir or AFFINE_CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)
    key = _affine_cache_key(pre_path, post_path)
    pre_out = os.path.join(cache_dir, f"pre_{key}.nii.gz")
    post_out = os.path.join(cache_dir, f"post_{key}.nii.gz")
    mat_path = os.path.join(cache_dir, f"{key}.mat")
    if os.path.isfile(pre_out) and os.path.isfile(post_out):
        return pre_out, post_out
    if _flirt_available():
        # FSL FLIRT path
        subprocess.run(
            ["flirt", "-in", pre_path, "-ref", ref_path, "-out", pre_out, "-omat", mat_path, "-dof", str(dof), "-interp", "trilinear"],
            check=True, capture_output=True, timeout=300,
        )
        subprocess.run(
            ["applyxfm", "-in", post_path, "-ref", ref_path, "-out", post_out, "-init", mat_path, "-interp", "trilinear"],
            check=True, capture_output=True, timeout=120,
        )
    else:
        # DIPY fallback (12-DOF affine)
        _register_pair_to_mni_dipy(pre_path, post_path, ref_path, pre_out, post_out)
    return pre_out, post_out


def load_pre_post_pair(
    pre_path: str,
    post_path: str,
    use_affine: Optional[bool] = None,
    target_shape: Tuple[int, int, int] = TARGET_SHAPE,
    apply_mask: bool = True,
    minmax: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load (pre_vol, post_vol) as float32 arrays of shape target_shape.
    When use_affine is True (or WEEK7_AFFINE=1), volumes are affinely registered to MNI
    first (same transform applied to both); then mask and min-max are applied.
    When use_affine is False, loads with resampling only (no registration).
    """
    global _flirt_warned
    do_affine = use_affine if use_affine is not None else USE_AFFINE
    if do_affine:
        try:
            pre_reg, post_reg = register_pair_to_mni(pre_path, post_path)
            pre_vol = load_volume(pre_reg, target_shape=target_shape, apply_mask=apply_mask, minmax=minmax)
            post_vol = load_volume(post_reg, target_shape=target_shape, apply_mask=apply_mask, minmax=minmax)
            return pre_vol, post_vol
        except Exception as e:
            if not _flirt_warned:
                _flirt_warned = True
                import warnings
                warnings.warn(f"WEEK7_AFFINE=1 but registration failed ({e}); loading without affine.", UserWarning)
    pre_vol = load_volume(pre_path, target_shape=target_shape, apply_mask=apply_mask, minmax=minmax)
    post_vol = load_volume(post_path, target_shape=target_shape, apply_mask=apply_mask, minmax=minmax)
    return pre_vol, post_vol


def get_brain_mask() -> np.ndarray:
    """Load MNI brain mask (91, 109, 91), 1 = brain, 0 = non-brain. Cached."""
    global _mask_cache
    if _mask_cache is None:
        m = nib.load(BRAIN_MASK_PATH)
        _mask_cache = np.asarray(m.dataobj).squeeze().astype(np.float32)
        if _mask_cache.max() > 1:
            _mask_cache = (_mask_cache > 0).astype(np.float32)
    return _mask_cache


def get_brain_bounding_box(mask: Optional[np.ndarray] = None) -> Tuple[slice, slice, slice]:
    """Axis-aligned bounding box of brain (where mask > 0). Returns (sl_d, sl_h, sl_w) to crop volume to brain only."""
    if mask is None:
        mask = get_brain_mask()
    mask = (mask > 0.5) if mask.dtype != bool else mask
    where = np.argwhere(mask)
    if where.size == 0:
        return slice(0, mask.shape[0]), slice(0, mask.shape[1]), slice(0, mask.shape[2])
    d_min, d_max = where[:, 0].min(), where[:, 0].max() + 1
    h_min, h_max = where[:, 1].min(), where[:, 1].max() + 1
    w_min, w_max = where[:, 2].min(), where[:, 2].max() + 1
    return slice(d_min, d_max), slice(h_min, h_max), slice(w_min, w_max)


def get_brain_crop_shape(mask: Optional[np.ndarray] = None) -> Tuple[int, int, int]:
    """Shape of the brain-only crop (same for all subjects when using fixed MNI mask). Returns (D, H, W)."""
    sl_d, sl_h, sl_w = get_brain_bounding_box(mask)
    return (sl_d.stop - sl_d.start, sl_h.stop - sl_h.start, sl_w.stop - sl_w.start)


def load_volume_cropped(
    nii_path: str,
    target_shape: Tuple[int, int, int] = TARGET_SHAPE,
    apply_mask: bool = True,
    pad_to_shape: Optional[Tuple[int, int, int]] = None,
    minmax: bool = True,
) -> np.ndarray:
    """
    Load NIfTI, resize to target_shape, apply brain mask, crop to brain bbox, optionally pad to pad_to_shape.
    Returns float32 array of shape pad_to_shape if given, else get_brain_crop_shape().
    Use for brain-only crop experiment: smaller volume, same preprocessing otherwise.
    """
    vol = load_volume(nii_path, target_shape=target_shape, apply_mask=apply_mask, pad_zeros=True, minmax=minmax)
    # Bbox from MNI mask (same shape as TARGET_SHAPE)
    sl_d, sl_h, sl_w = get_brain_bounding_box(get_brain_mask())
    cropped = vol[sl_d, sl_h, sl_w].copy()
    if pad_to_shape is not None:
        out = np.zeros(pad_to_shape, dtype=np.float32)
        cd, ch, cw = cropped.shape
        pd, ph, pw = pad_to_shape
        out[:min(cd, pd), :min(ch, ph), :min(cw, pw)] = cropped[:min(cd, pd), :min(ch, ph), :min(cw, pw)]
        return out
    return cropped


def load_volume(
    nii_path: str,
    target_shape: Tuple[int, int, int] = TARGET_SHAPE,
    apply_mask: bool = True,
    pad_zeros: bool = True,
    minmax: bool = True,
) -> np.ndarray:
    """
    Load NIfTI, optionally resize to target_shape, apply brain mask, pad, min-max norm.
    Returns float32 array of shape target_shape.
    """
    img = nib.load(nii_path)
    data = np.asarray(img.get_fdata()).astype(np.float32).squeeze()
    if data.ndim == 4:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {data.shape} from {nii_path}")

    # Resize to target (trilinear)
    if data.shape != target_shape:
        factors = [target_shape[i] / data.shape[i] for i in range(3)]
        data = zoom(data, factors, order=1)

    # Pad with 0s if smaller (shouldn't happen after zoom, but if target was larger)
    if pad_zeros and (data.shape[0] < target_shape[0] or data.shape[1] < target_shape[1] or data.shape[2] < target_shape[2]):
        out = np.zeros(target_shape, dtype=np.float32)
        s0 = min(data.shape[0], target_shape[0])
        s1 = min(data.shape[1], target_shape[1])
        s2 = min(data.shape[2], target_shape[2])
        out[:s0, :s1, :s2] = data[:s0, :s1, :s2]
        data = out
    elif data.shape != target_shape:
        # Crop to target if larger
        data = data[: target_shape[0], : target_shape[1], : target_shape[2]]

    # Apply brain mask: outside mask = 0
    if apply_mask:
        mask = get_brain_mask()
        if mask.shape != target_shape:
            factors = [target_shape[i] / mask.shape[i] for i in range(3)]
            mask = zoom(mask, factors, order=0)
        mask = (mask > 0.5).astype(np.float32)
        data = data * mask

    if minmax:
        mn, mx = data.min(), data.max()
        if (mx - mn) > 1e-8:
            data = (data - mn) / (mx - mn)
        else:
            data = np.zeros_like(data)

    return data.astype(np.float32)


def augment_volume(
    vol: np.ndarray,
    flip_lr: bool = False,
    flip_ud: bool = False,
    flip_fb: bool = False,
    intensity_scale: Optional[float] = None,
) -> np.ndarray:
    """In-place style augmentations; returns augmented copy. Same for 2D and 3D."""
    out = vol.copy()
    if flip_lr:
        out = np.flip(out, axis=1).copy()
    if flip_ud:
        out = np.flip(out, axis=0).copy()
    if flip_fb:
        out = np.flip(out, axis=2).copy()
    if intensity_scale is not None and abs(intensity_scale - 1.0) > 1e-6:
        out = (out * intensity_scale).clip(0.0, 1.0)
    return out


def is_env_flag(var: str) -> bool:
    """Return True if env var is set to a truthy string (1/true/yes, case-insensitive)."""
    return os.environ.get(var, "").lower() in ("1", "true", "yes")


def is_week7_kfold() -> bool:
    """Return True when a per-fold split path is active (K-fold mode)."""
    return bool(os.environ.get("WEEK7_SPLIT_PATH", "").strip())


def get_kfold_seed(base: int = None) -> int:
    """Return effective training seed: base + fold_index when in K-fold mode, else base.

    Reads SEED env (default 42) for base; reads WEEK11_KFOLD_FOLD for fold offset.
    Centralizes the conditional arithmetic scattered across model training scripts.
    """
    base = base if base is not None else int(os.environ.get("SEED", 42))
    if not is_week7_kfold():
        return base
    fold_env = os.environ.get("WEEK11_KFOLD_FOLD", "").strip()
    fold_idx = int(fold_env) if fold_env.isdigit() else 0
    return base + fold_idx


def get_combined_split_path() -> str:
    """Path to combined split JSON. Use WEEK7_SPLIT_PATH env for K-fold per-fold splits."""
    return os.environ.get("WEEK7_SPLIT_PATH", "").strip() or COMBINED_SPLIT_PATH


def week7_kfold_results_tag() -> str:
    """Return e.g. ``_fold2`` for unique K-fold checkpoints and result JSONs.

    Prefer ``WEEK11_KFOLD_FOLD`` (set by ``run_week11_kfold*.sh``). Otherwise parse
    ``split_foldN.json`` from ``WEEK7_SPLIT_PATH``. Empty if not in K-fold.
    """
    fold = os.environ.get("WEEK11_KFOLD_FOLD", "").strip()
    if fold.isdigit():
        return "_fold%s" % fold
    path = os.environ.get("WEEK7_SPLIT_PATH", "")
    m = re.search(r"split_fold(\d+)\.json", path, re.I)
    if m:
        return "_fold%s" % m.group(1)
    return ""


def week7_kfold_suffix_paths(ckpt_name: str, results_name: str):
    """Append K-fold tag before ``.pt`` / ``.json`` so folds do not clobber each other."""
    tag = week7_kfold_results_tag()
    if not tag:
        return ckpt_name, results_name
    if ckpt_name.endswith(".pt"):
        ckpt_name = ckpt_name[:-3] + tag + ".pt"
    if results_name.endswith(".json"):
        results_name = results_name[:-5] + tag + ".json"
    return ckpt_name, results_name


def week7_kfold_suffix_checkpoint(ckpt_name: str) -> str:
    """Append K-fold tag to a ``.pt`` path (EMA/VAE/aux checkpoints)."""
    tag = week7_kfold_results_tag()
    if not tag or not ckpt_name.endswith(".pt"):
        return ckpt_name
    return ckpt_name[:-3] + tag + ".pt"


def load_combined_split() -> dict:
    """Load combined 2020-2024 subject-level split (from combined_subject_split.json or WEEK7_SPLIT_PATH)."""
    path = get_combined_split_path()
    if not path or not os.path.isfile(path):
        path = COMBINED_SPLIT_PATH
    with open(path) as f:
        return json.load(f)


def get_pre_post_pairs(split_key: str = "train") -> List[Tuple[str, str]]:
    """Return list of (pre_path, post_path) for split_key in combined split."""
    data = load_combined_split()
    pairs = []
    for item in data.get(split_key, []):
        pre = item.get("pre_path")
        post = item.get("post_path")
        if pre and post and os.path.isfile(pre) and os.path.isfile(post):
            pairs.append((pre, post))
    return pairs


def _subject_id_from_path(pre_path: str) -> str:
    """Derive subject id from pre path (e.g. pre_2021_001.nii.gz -> 2021_001).

    Stanford 2020 cohort uses the same CBF filename under different subject folders
    (e.g. .../moyamoya_stanford_2020_007/...); basename-only ids collide. Also used
    as fallback when split JSON has no subject_id.
    """
    m = re.search(r"moyamoya_stanford_(\d{4}_\d+)", pre_path)
    if m:
        return m.group(1)
    base = os.path.basename(pre_path).replace(".nii.gz", "").replace(".nii", "")
    if base.startswith("pre_"):
        return base.replace("pre_", "", 1)
    m2 = re.match(r"(.+)_pre$", base)
    if m2:
        return m2.group(1)
    return base.replace("pre", "").strip("_") or "unknown"


def get_week7_splits():
    """Return train, val, test pairs (pre_path, post_path) from combined 2020–2023 split."""
    return (
        get_pre_post_pairs("train"),
        get_pre_post_pairs("val"),
        get_pre_post_pairs("test"),
    )


def get_pre_post_pairs_with_subject_id(split_key: str = "train") -> List[Tuple[str, str, str]]:
    """Return list of (subject_id, pre_path, post_path) for split_key. Uses canonical subject_id from split when present (ensures n=32 unique IDs for test)."""
    data = load_combined_split()
    out = []
    for item in data.get(split_key, []):
        pre = item.get("pre_path")
        post = item.get("post_path")
        if not pre or not post or not os.path.isfile(pre) or not os.path.isfile(post):
            continue
        sid = item.get("subject_id") or _subject_id_from_path(pre)
        out.append((sid, pre, post))
    return out


def collect_pre_post_quads_by_splits(split_keys: List[str]) -> List[Tuple[str, str, str, str]]:
    """Concatenate several splits: each row is (subject_id, pre_path, post_path, split_key).

    Typical use: ``split_keys=["train","val","test"]`` → all subjects in combined_subject_split
    with valid paths (e.g. 252 rows when 193+27+32).
    """
    out: List[Tuple[str, str, str, str]] = []
    for sk in split_keys:
        for sid, pre, post in get_pre_post_pairs_with_subject_id(sk):
            out.append((sid, pre, post, sk))
    return out


def get_subject_center_map(split_path: Optional[str] = None) -> dict:
    """
    Return subject_id -> center_id for multi-center reporting.
    Optional: if split contains subject_metadata with center_id per subject, use that.
    Else use subject ID prefix (e.g. 2020_051 -> "2020") as proxy. Single-center if no metadata.
    Does not change get_week7_splits or reproducibility; for reporting only.
    """
    path = split_path or COMBINED_SPLIT_PATH
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        data = json.load(f)
    meta = data.get("subject_metadata") or {}
    # All subject IDs from train/val/test subject lists if present
    subject_ids = []
    for key in ("train_subjects", "val_subjects", "test_subjects"):
        subject_ids.extend(data.get(key, []))
    if not subject_ids:
        # Fallback: derive from train/val/test path lists
        seen = set()
        for key in ("train", "val", "test"):
            for item in data.get(key, []):
                pre = item.get("pre_path", "")
                base = os.path.basename(pre).replace(".nii.gz", "").replace(".nii", "").strip()
                if base.startswith("pre_"):
                    sid = base[4:]
                else:
                    sid = base
                if sid and sid not in seen:
                    seen.add(sid)
                    subject_ids.append(sid)
    out = {}
    for sid in subject_ids:
        if isinstance(meta.get(sid), dict) and "center_id" in meta[sid]:
            out[sid] = str(meta[sid]["center_id"])
        elif "_" in sid:
            out[sid] = sid.split("_")[0]
        else:
            out[sid] = sid or "unknown"
    return out


def metrics_in_brain(
    pred: np.ndarray,
    target: np.ndarray,
    mask: Optional[np.ndarray] = None,
    data_range: float = 1.0,
) -> dict:
    """
    MAE, SSIM, PSNR computed only inside the brain (best for reporting CVR quality).
    pred, target: 3D arrays same shape as mask (e.g. 91,109,91).
    mask: same shape, 1 = brain, 0 = outside. If None, uses get_brain_mask().
    Returns dict with mae_mean, ssim_mean, psnr_mean (brain-only).
    """
    from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr

    if mask is None:
        mask = get_brain_mask()
    if mask.shape != pred.shape:
        from scipy.ndimage import zoom as _zoom
        factors = [pred.shape[i] / mask.shape[i] for i in range(3)]
        mask = _zoom(mask.astype(np.float32), factors, order=0) > 0.5
    mask_bool = (mask > 0.5).astype(bool)
    n = mask_bool.sum()
    if n == 0:
        return {"mae_mean": float("nan"), "ssim_mean": float("nan"), "psnr_mean": float("nan")}
    mae = np.abs(pred.astype(np.float64) - target.astype(np.float64))[mask_bool].mean()
    sl_d, sl_h, sl_w = get_brain_bounding_box(mask)
    p_crop = pred[sl_d, sl_h, sl_w]
    t_crop = target[sl_d, sl_h, sl_w]
    ssim_val = ssim(t_crop, p_crop, data_range=data_range)
    psnr_val = psnr(t_crop, p_crop, data_range=data_range)
    return {"mae_mean": float(mae), "ssim_mean": float(ssim_val), "psnr_mean": float(psnr_val)}


def get_brain_mask_2d_slice(mask_3d: Optional[np.ndarray] = None) -> np.ndarray:
    """Middle axial slice of brain mask (D//2), shape (H, W). For 2D models."""
    if mask_3d is None:
        mask_3d = get_brain_mask()
    d = mask_3d.shape[0] // 2
    return (mask_3d[d] > 0.5).astype(np.float32)


def metrics_in_brain_2d(
    pred: np.ndarray,
    target: np.ndarray,
    mask_2d: Optional[np.ndarray] = None,
    data_range: float = 1.0,
) -> dict:
    """
    MAE, SSIM, PSNR for 2D (single slice) computed only inside brain.
    pred, target: (H, W). mask_2d: (H, W), 1 = brain. If None, uses middle slice of get_brain_mask().
    """
    from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr

    if mask_2d is None:
        mask_2d = get_brain_mask_2d_slice()
    if mask_2d.shape != pred.shape:
        from scipy.ndimage import zoom as _zoom
        factors = [pred.shape[i] / mask_2d.shape[i] for i in range(2)]
        mask_2d = (_zoom(mask_2d.astype(np.float32), factors, order=0) > 0.5).astype(np.float32)
    mask_bool = (mask_2d > 0.5).astype(bool)
    n = mask_bool.sum()
    if n == 0:
        return {"mae_mean": float("nan"), "ssim_mean": float("nan"), "psnr_mean": float("nan")}
    mae = np.abs(pred.astype(np.float64) - target.astype(np.float64))[mask_bool].mean()
    # SSIM/PSNR on full slice (skimage doesn't support mask); for consistency crop to bbox
    where = np.argwhere(mask_bool)
    if where.size == 0:
        return {"mae_mean": float(mae), "ssim_mean": float("nan"), "psnr_mean": float("nan")}
    rmin, rmax = where[:, 0].min(), where[:, 0].max() + 1
    cmin, cmax = where[:, 1].min(), where[:, 1].max() + 1
    p_crop = pred[rmin:rmax, cmin:cmax]
    t_crop = target[rmin:rmax, cmin:cmax]
    ssim_val = ssim(t_crop, p_crop, data_range=data_range)
    psnr_val = psnr(t_crop, p_crop, data_range=data_range)
    return {"mae_mean": float(mae), "ssim_mean": float(ssim_val), "psnr_mean": float(psnr_val)}


def masked_loss_3d(pred: np.ndarray, target: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """L1 loss only over voxels where mask > 0. pred, target, mask: (D,H,W) or (1,D,H,W)."""
    if pred.ndim == 4:
        pred, target = pred.squeeze(0), target.squeeze(0)
    if mask is None:
        mask = get_brain_mask()
    if mask.shape != pred.shape:
        from scipy.ndimage import zoom as _zoom
        factors = [pred.shape[i] / mask.shape[i] for i in range(3)]
        mask = _zoom(mask.astype(np.float32), factors, order=0) > 0.5
    m = (mask > 0.5).astype(bool)
    if m.sum() == 0:
        return float(np.abs(pred - target).mean())
    return float(np.abs(pred.astype(np.float64) - target.astype(np.float64))[m].mean())


def masked_loss_2d(pred: np.ndarray, target: np.ndarray, mask_2d: Optional[np.ndarray] = None) -> float:
    """L1 loss only over pixels where mask > 0. pred, target: (H,W)."""
    if mask_2d is None:
        mask_2d = get_brain_mask_2d_slice()
    if mask_2d.shape != pred.shape:
        from scipy.ndimage import zoom as _zoom
        factors = [pred.shape[i] / mask_2d.shape[i] for i in range(2)]
        mask_2d = _zoom(mask_2d.astype(np.float32), factors, order=0) > 0.5
    m = (mask_2d > 0.5).astype(bool)
    if m.sum() == 0:
        return float(np.abs(pred - target).mean())
    return float(np.abs(pred.astype(np.float64) - target.astype(np.float64))[m].mean())


def get_brain_mask_for_shape(shape: Tuple[int, ...], dtype=np.float32) -> np.ndarray:
    """Brain mask resized to shape (e.g. (96,112,96) or (96,112)). Returns 0/1 float array."""
    mask = get_brain_mask()
    if len(shape) == 3:
        if mask.shape != shape:
            factors = [shape[i] / mask.shape[i] for i in range(3)]
            mask = zoom(mask.astype(np.float32), factors, order=0)
    else:
        mask = get_brain_mask_2d_slice(mask)
        if mask.shape != shape:
            factors = [shape[i] / mask.shape[i] for i in range(2)]
            mask = zoom(mask.astype(np.float32), factors, order=0)
    return (mask > 0.5).astype(dtype)


# ---------------------------------------------------------------------------
# Phase 2: Vascular / MNI territory region-weighted loss
# ---------------------------------------------------------------------------
MASKS_DIR_DEFAULT = os.path.join(_REPO_ROOT, "Masks")
MASKS_DIR_EXTRA = os.environ.get("MOYAMOYA_MASKS_DIR", "")


def _get_masks_dir() -> Optional[str]:
    """Return first existing of MASKS_DIR_DEFAULT, MASKS_DIR_EXTRA (env override), or None."""
    if os.path.isdir(MASKS_DIR_DEFAULT):
        return MASKS_DIR_DEFAULT
    if MASKS_DIR_EXTRA and os.path.isdir(MASKS_DIR_EXTRA):
        return MASKS_DIR_EXTRA
    return None


def load_territory_masks(
    masks_dir: str,
    target_shape: Tuple[int, int, int],
) -> List[Tuple[str, np.ndarray]]:
    """
    Load MNI territory masks (*2mm*.nii.gz or MNI_*.nii.gz) and resize to target_shape.
    Returns [(name, mask_3d), ...] with mask_3d float 0/1.
    """
    import glob
    out = []
    pattern = os.path.join(masks_dir, "*.nii.gz")
    files = sorted(glob.glob(pattern))
    files = [f for f in files if "2mm" in f]
    if not files:
        files = sorted(glob.glob(os.path.join(masks_dir, "MNI_*.nii.gz")))
    for path in files:
        try:
            img = nib.load(path)
            data = np.asarray(img.dataobj).squeeze().astype(np.float32)
            if data.ndim == 4:
                data = data[..., 0]
            mask = (data > 0).astype(np.float32)
            if mask.shape != target_shape:
                factors = [target_shape[i] / mask.shape[i] for i in range(3)]
                mask = zoom(mask, factors, order=0)
                mask = (mask > 0.5).astype(np.float32)
            name = os.path.basename(path).replace(".nii.gz", "")
            out.append((name, mask))
        except Exception:
            continue
    return out


def get_region_weight_mask_for_shape(
    shape: Tuple[int, ...],
    masks_dir: Optional[str] = None,
    vascular_weight: float = 1.5,
    dtype=np.float32,
) -> np.ndarray:
    """
    Weight map for region-weighted loss: brain = 1.0, vascular territories = vascular_weight.
    If masks_dir is None or no territory masks found, returns brain mask (1.0 in brain, 0 outside).
    shape: (D,H,W) for 3D or (H,W) for 2D.
    """
    base = get_brain_mask_for_shape(shape, dtype=np.float32)  # 0/1
    mdir = masks_dir or _get_masks_dir()
    if not mdir or not os.path.isdir(mdir):
        return base.astype(dtype)
    if len(shape) == 2:
        target_3d = TARGET_SHAPE
        territory_list = load_territory_masks(mdir, target_3d)
        if not territory_list:
            return base.astype(dtype)
        mid = target_3d[0] // 2
        weight = base.copy()
        for _name, m3 in territory_list:
            m2 = (m3[mid] > 0.5).astype(np.float32)
            if m2.shape != shape:
                factors = [shape[i] / m2.shape[i] for i in range(2)]
                m2 = zoom(m2, factors, order=0)
            weight[m2 > 0.5] = vascular_weight
        return weight.astype(dtype)
    else:
        territory_list = load_territory_masks(mdir, shape)
        if not territory_list:
            return base.astype(dtype)
        weight = base.astype(np.float32)
        for _name, m in territory_list:
            weight[m > 0.5] = vascular_weight
        return weight.astype(dtype)


# Low-baseline region handling (future work): down-weight voxels with low pre (or post) to avoid over-penalizing.
# Threshold: treat voxels with pre < threshold as low-baseline. Alternative: 10th percentile per volume (not implemented).
LOW_BASELINE_THRESHOLD_DEFAULT = 0.1
LOW_BASELINE_WEIGHT_DEFAULT = 0.5  # weight in [0, 1] for low-baseline voxels; 1.0 = no down-weighting.


def get_low_baseline_weight_map_np(
    pre: np.ndarray,
    brain_mask: Optional[np.ndarray] = None,
    threshold: float = LOW_BASELINE_THRESHOLD_DEFAULT,
    low_weight: float = LOW_BASELINE_WEIGHT_DEFAULT,
) -> np.ndarray:
    """
    Weight map for loss: 0 outside brain; low_weight where pre < threshold (low baseline); 1 elsewhere in brain.
    pre: (H,W,D) or (1,H,W,D). brain_mask: same shape, 1=brain 0=out. If None, uses get_brain_mask_for_shape(pre.shape).
    Edge: whole volume above threshold -> no down-weighting (all 1 in brain).
    """
    if pre.ndim == 4:
        pre = pre.squeeze(0)
    shape = pre.shape
    if brain_mask is None:
        brain_mask = get_brain_mask_for_shape(shape, dtype=np.float32)
    if brain_mask.shape != shape:
        factors = [shape[i] / brain_mask.shape[i] for i in range(3)]
        brain_mask = zoom(brain_mask.astype(np.float32), factors, order=0)
        brain_mask = (brain_mask > 0.5).astype(np.float32)
    weight = np.where(pre < threshold, low_weight, 1.0).astype(np.float32)
    weight = weight * brain_mask
    return weight
