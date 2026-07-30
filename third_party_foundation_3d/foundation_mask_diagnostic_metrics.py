#!/usr/bin/env python3
"""
Shared metrics for legacy SAM / STU-Net mask diagnostics (not CBF synthesis).

Uses the same brain-mask resampling as Week7 CBF code (get_brain_mask_for_shape) so the
reference aligns with each prediction grid. Reports Dice + IoU as primary overlap metrics;
MAE / SSIM / PSNR on 0/1 maps are kept for backward compatibility but are not comparable to
continuous CBF metrics (see JSON field pixel_image_metrics_note).
"""
from __future__ import annotations

import os
import sys
from typing import Tuple

import numpy as np
from scipy.ndimage import zoom
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPTS = os.path.join(_REPO, "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from week7_preprocess import get_brain_mask_for_shape  # noqa: E402


def dice_coefficient(pred: np.ndarray, gt: np.ndarray, thresh: float = 0.5) -> float:
    a = (pred > thresh).flatten()
    b = (gt > thresh).flatten()
    if a.sum() == 0 and b.sum() == 0:
        return 1.0
    inter = (a & b).sum()
    return float(2 * inter / (a.sum() + b.sum() + 1e-8))


def iou_binary(pred: np.ndarray, gt: np.ndarray, thresh: float = 0.5) -> float:
    a = (pred > thresh).flatten()
    b = (gt > thresh).flatten()
    inter = (a & b).sum()
    union = (a | b).sum()
    if union == 0:
        return 1.0
    return float(inter / (union + 1e-8))


def align_pred_to_shape(pred: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
    if pred.shape == target_shape:
        return pred
    factors = [target_shape[k] / pred.shape[k] for k in range(3)]
    return zoom(pred, factors, order=0)


def mask_diagnostic_metrics(pred: np.ndarray) -> dict:
    """
    pred: float array (D,H,W), segmentation probability or binarized mask.
    Returns dict with overlap metrics + optional pixel metrics on binary maps.
    """
    pred = np.asarray(pred).squeeze().astype(np.float32)
    if pred.ndim != 3:
        raise ValueError("Expected 3D pred, got shape %s" % (pred.shape,))
    gt = get_brain_mask_for_shape(pred.shape, dtype=np.float32)
    pred_c = np.clip(pred, 0, 1).astype(np.float32)
    gt_b = (gt > 0.5).astype(np.float32)

    mae = float(np.abs(pred_c - gt_b).mean())
    try:
        ssim_v = float(ssim(gt_b, pred_c, data_range=1.0))
    except Exception:
        ssim_v = 0.0
    try:
        psnr_v = float(psnr(gt_b, pred_c, data_range=1.0))
    except Exception:
        psnr_v = 0.0
    dice_v = dice_coefficient(pred_c, gt_b)
    iou_v = iou_binary(pred_c, gt_b)

    return {
        "mae": mae,
        "ssim": ssim_v,
        "psnr": psnr_v,
        "dice": dice_v,
        "iou": iou_v,
    }


PIXEL_METRICS_NOTE = (
    "MAE/SSIM/PSNR here are computed on binary (or soft) masks vs the Week7 brain mask; "
    "they are not comparable to continuous CBF metrics in Table 1 / Med3DVLM / SAM-CBF rows. "
    "Use Dice and IoU for this diagnostic."
)
