"""Metric functions: sanity-check MAE/SSIM/PSNR on synthetic data (no GPU or patient data needed)."""
import numpy as np

from regional_eval_3d import safe_ssim, safe_psnr


def _synthetic_volume_pair(seed=42, shape=(32, 32, 32), noise_std=0.05):
    rng = np.random.default_rng(seed)
    gt = rng.random(shape).astype("float32")
    pred = gt + rng.normal(0, noise_std, shape).astype("float32")
    mask = np.ones(shape, dtype="float32")
    return gt, pred, mask


def test_ssim_near_one_for_small_noise():
    gt, pred, mask = _synthetic_volume_pair()
    s = safe_ssim(gt, pred, mask)
    assert 0.9 < s <= 1.0001


def test_psnr_positive_and_plausible():
    gt, pred, mask = _synthetic_volume_pair()
    p = safe_psnr(gt, pred, mask)
    assert 15 < p < 60


def test_ssim_psnr_nan_when_mask_too_small():
    gt, pred, mask = _synthetic_volume_pair()
    mask = np.zeros_like(mask)  # < 10 voxels -> functions should return NaN, not crash
    assert np.isnan(safe_ssim(gt, pred, mask))
    assert np.isnan(safe_psnr(gt, pred, mask))
