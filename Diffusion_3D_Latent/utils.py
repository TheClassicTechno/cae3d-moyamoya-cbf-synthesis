"""
Shared utilities for Diffusion_3D_Latent and consumers (PatchVolume, Hybrid, evaluate_cold_diffusion).
"""
import os
import numpy as np
import torch
import nibabel as nib
from scipy.ndimage import zoom


class EMA:
    """Exponential moving average of model parameters. decay in (0,1); shadow = decay*shadow + (1-decay)*param."""
    def __init__(self, model: torch.nn.Module, decay: float = 0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}

    def update(self):
        for n, p in self.model.named_parameters():
            if not p.requires_grad or n not in self.shadow:
                continue
            self.shadow[n].mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def apply_shadow(self):
        """Replace model params with shadow (EMA) for eval/save."""
        self.backup = {n: p.data.clone() for n, p in self.model.named_parameters() if n in self.shadow}
        for n, p in self.model.named_parameters():
            if n in self.shadow:
                p.data.copy_(self.shadow[n])

    def restore(self):
        """Restore model params from backup after apply_shadow."""
        for n, p in self.model.named_parameters():
            if n in self.backup:
                p.data.copy_(self.backup[n])


def strict_normalize_volume(vol: np.ndarray) -> np.ndarray:
    """Normalize volume to roughly [0,1] via (x - min) / (max - min + eps)."""
    v = vol.astype(np.float32)
    mn, mx = v.min(), v.max()
    if mx - mn < 1e-8:
        return np.zeros_like(v)
    return ((v - mn) / (mx - mn + 1e-8)).astype(np.float32)


def bland_altman_analysis(pred_flat: np.ndarray, gt_flat: np.ndarray) -> dict:
    """Bland-Altman: bias, SD of diff, limits of agreement, optional CIs. Flat 1D arrays."""
    pred_flat = np.asarray(pred_flat).ravel()
    gt_flat = np.asarray(gt_flat).ravel()
    n = min(len(pred_flat), len(gt_flat))
    if n == 0:
        return {
            'mean_bias': np.nan, 'std_diff': np.nan, 'upper_loa': np.nan, 'lower_loa': np.nan,
            'loa_upper_ci': np.nan, 'loa_lower_ci': np.nan, 'bias_ci_upper': np.nan, 'bias_ci_lower': np.nan, 'n_samples': 0
        }
    pred_flat = pred_flat[:n]
    gt_flat = gt_flat[:n]
    diff = pred_flat - gt_flat
    mean_bias = float(np.mean(diff))
    std_diff = float(np.std(diff))
    upper_loa = mean_bias + 1.96 * std_diff
    lower_loa = mean_bias - 1.96 * std_diff
    # Approximate 95% CIs (large n)
    se_bias = std_diff / np.sqrt(n)
    bias_ci_lower = mean_bias - 1.96 * se_bias
    bias_ci_upper = mean_bias + 1.96 * se_bias
    se_loa = std_diff * np.sqrt(3 / n)  # approx for LOA
    loa_upper_ci = upper_loa + 1.96 * se_loa
    loa_lower_ci = lower_loa - 1.96 * se_loa
    return {
        'mean_bias': mean_bias, 'std_diff': std_diff, 'upper_loa': upper_loa, 'lower_loa': lower_loa,
        'loa_upper_ci': loa_upper_ci, 'loa_lower_ci': loa_lower_ci,
        'bias_ci_upper': bias_ci_upper, 'bias_ci_lower': bias_ci_lower, 'n_samples': int(n)
    }


def load_full_volume(nii_path: str, target_size=(128, 128, 64)) -> np.ndarray:
    """Load NIfTI and resize to target_size."""
    img = nib.load(nii_path)
    data = img.get_fdata(dtype=np.float32)
    if data.ndim == 4:
        data = data[..., 0]
    zoom_factors = [target_size[i] / data.shape[i] for i in range(3)]
    resized = zoom(data, zoom_factors, order=1)
    return resized.astype(np.float32)


def pre_to_post_path(pre_path: str) -> str:
    """Convert pre-scan path to post-scan path."""
    basename = os.path.basename(pre_path).replace('pre_', 'post_')
    dirname = os.path.dirname(pre_path).replace('/pre', '/post')
    if dirname == '':
        dirname = 'post'
    return os.path.join(dirname, basename)
