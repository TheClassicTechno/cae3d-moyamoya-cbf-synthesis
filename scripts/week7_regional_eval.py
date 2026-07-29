#!/usr/bin/env python3
"""
Week 7 regional evaluation: metrics inside vascular/territory masks and per-region.
Supports (1) predictions from NIfTI dir, or (2) inference from script-based 3D models (UNet 3D, ResNet 3D).

Usage:
  # From a directory of prediction NIfTIs (post_{id}_pred.nii.gz):
  python week7_regional_eval.py --pred-dir /path/to/pred_niftis --masks-dir <repo-root>/Masks --out regional_week7.json

  # From a trained script model (runs test set inference, then regional metrics):
  python week7_regional_eval.py --model unet3d --out regional_week7_unet3d.json
  python week7_regional_eval.py --model resnet3d --out regional_week7_resnet3d.json
  # For variants (low-baseline, Phase 2/3), pass --checkpoint explicitly:
  python week7_regional_eval.py --model unet3d --checkpoint week7_results/week7_unet3d_best_lowbaseline.pt --out regional_week7_unet3d_lowbaseline.json
"""
import os
import sys
import json
import argparse
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_REPO_ROOT, "pyproject.toml")):
    _parent = os.path.dirname(_REPO_ROOT)
    if _parent == _REPO_ROOT:
        raise RuntimeError("Could not locate repository root (pyproject.toml not found)")
    _REPO_ROOT = _parent

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from week7_data import get_week7_splits, _subject_id_from_path, Week7VolumePairs3D
from week7_preprocess import (
    TARGET_SHAPE,
    load_volume,
    load_territory_masks,
)

DATA_DIR = _REPO_ROOT
MASKS_DIR_DEFAULT = os.path.join(_REPO_ROOT, "Masks")
MASKS_DIR_EXTRA = os.environ.get("MOYAMOYA_MASKS_DIR", "")


def load_nifti(path):
    img = nib.load(path)
    return np.asarray(img.dataobj).astype(np.float32).squeeze()


def _crop_to_target(vol: np.ndarray) -> np.ndarray:
    """Crop volume to TARGET_SHAPE (91,109,91) if it was padded (e.g. 96,112,96)."""
    if vol.shape == TARGET_SHAPE:
        return vol
    h, w, d = TARGET_SHAPE
    if vol.ndim == 3:
        return vol[:h, :w, :d].copy()
    return vol[:h, :w, :d]


def safe_ssim(gt, pred, mask, data_range=1.0):
    if mask.sum() < 10:
        return float("nan")
    gt_m = gt * mask
    pred_m = pred * mask
    return ssim(gt_m, pred_m, data_range=data_range)


def safe_psnr(gt, pred, mask, data_range=1.0):
    if mask.sum() < 10:
        return float("nan")
    gt_m = gt[mask > 0.5]
    pred_m = pred[mask > 0.5]
    mse = np.mean((gt_m - pred_m) ** 2)
    if mse <= 0:
        return 60.0
    return 10 * np.log10((data_range ** 2) / mse)


def compute_regional_metrics(pred_vol, gt_vol, territory_masks, data_range=1.0):
    """pred_vol, gt_vol: (H,W,D). territory_masks: [(name, mask_3d), ...]. Returns dict per_volume, per_region."""
    pred_vol = _crop_to_target(pred_vol)
    gt_vol = _crop_to_target(gt_vol)
    if pred_vol.shape != gt_vol.shape:
        return None
    target_shape = pred_vol.shape

    rec = {
        "full_mae": float(np.abs(pred_vol - gt_vol).mean()),
        "full_ssim": float(ssim(gt_vol, pred_vol, data_range=data_range)),
        "full_psnr": float(psnr(gt_vol, pred_vol, data_range=data_range)),
    }
    per_region = {}
    for name, mask in territory_masks:
        if mask.shape != target_shape:
            factors = [target_shape[i] / mask.shape[i] for i in range(3)]
            mask = zoom(mask, factors, order=1)
            mask = (mask > 0.5).astype(np.float32)
        n = float(mask.sum())
        if n < 10:
            per_region[name] = {"mae": None, "ssim": None, "psnr": None, "mean_perfusion_pred": None, "mean_perfusion_gt": None}
            continue
        mae = float((np.abs(pred_vol - gt_vol) * mask).sum() / n)
        per_region[name] = {
            "mae": mae,
            "ssim": float(safe_ssim(gt_vol, pred_vol, mask, data_range)),
            "psnr": float(safe_psnr(gt_vol, pred_vol, mask, data_range)),
            "mean_perfusion_pred": float((pred_vol * mask).sum() / n),
            "mean_perfusion_gt": float((gt_vol * mask).sum() / n),
            "n_voxels": int(n),
        }
    rec["per_region"] = per_region
    return rec


def run_from_pred_dir(pred_dir, post_dir, masks_dir, out_path):
    """Load test pairs, for each load pred NIfTI and GT, compute regional metrics."""
    _, _, test_pairs = get_week7_splits()
    masks_dir = masks_dir or (MASKS_DIR_DEFAULT if os.path.isdir(MASKS_DIR_DEFAULT) else MASKS_DIR_EXTRA)
    if not os.path.isdir(masks_dir):
        print("Masks dir not found:", masks_dir)
        return
    territory_masks = load_territory_masks(masks_dir, TARGET_SHAPE)
    mask_names = [n for n, _ in territory_masks]
    if not mask_names:
        print("No territory masks found in", masks_dir)
        return

    results = {"per_volume": [], "per_region": {}, "mask_names": mask_names, "source": "pred_dir", "pred_dir": pred_dir}

    for pre_path, post_path in test_pairs:
        sid = _subject_id_from_path(pre_path)
        pred_path = os.path.join(pred_dir, f"post_{sid}_pred.nii.gz")
        if not os.path.isfile(pred_path):
            print("Skip (no pred):", pred_path)
            continue
        pred_vol = load_nifti(pred_path)
        if pred_vol.ndim == 4:
            pred_vol = pred_vol[..., 0]
        gt_vol = load_volume(post_path, target_shape=TARGET_SHAPE)
        rec = compute_regional_metrics(pred_vol, gt_vol, territory_masks)
        if rec is None:
            continue
        rec["id"] = sid
        results["per_volume"].append(rec)

    _aggregate_per_region(results)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print("Wrote", out_path)
    _print_summary(results)
    return results


def _aggregate_per_region(results):
    mask_names = results["mask_names"]
    for mname in mask_names:
        maes, ssims, psnrs = [], [], []
        for rec in results["per_volume"]:
            r = rec.get("per_region", {}).get(mname, {})
            if r.get("mae") is not None:
                maes.append(r["mae"])
            if r.get("ssim") is not None and not (isinstance(r["ssim"], float) and np.isnan(r["ssim"])):
                ssims.append(r["ssim"])
            if r.get("psnr") is not None and not (isinstance(r["psnr"], float) and np.isnan(r["psnr"])):
                psnrs.append(r["psnr"])
        results["per_region"][mname] = {
            "mae_mean": float(np.mean(maes)) if maes else None,
            "mae_std": float(np.std(maes)) if maes else None,
            "ssim_mean": float(np.mean(ssims)) if ssims else None,
            "psnr_mean": float(np.mean(psnrs)) if psnrs else None,
            "n_volumes": len(maes),
        }


def _print_summary(results):
    for mname in results["mask_names"]:
        r = results["per_region"][mname]
        if r["n_volumes"]:
            print("  %s: MAE %.4f  SSIM %.4f  PSNR %.2f dB (n=%d)" % (
                mname, r["mae_mean"] or 0, r["ssim_mean"] or 0, r["psnr_mean"] or 0, r["n_volumes"]))


def run_from_model(model_name, checkpoint_path, masks_dir, out_path):
    """Load model, run test set inference, compute regional metrics (no NIfTI export)."""
    import torch
    from torch.utils.data import DataLoader

    _, _, test_pairs = get_week7_splits()
    masks_dir = masks_dir or (MASKS_DIR_DEFAULT if os.path.isdir(MASKS_DIR_DEFAULT) else MASKS_DIR_EXTRA)
    if not os.path.isdir(masks_dir):
        print("Masks dir not found:", masks_dir)
        return
    territory_masks = load_territory_masks(masks_dir, TARGET_SHAPE)
    mask_names = [n for n, _ in territory_masks]
    if not mask_names:
        print("No territory masks found in", masks_dir)
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    TARGET_3D_PAD = (96, 112, 96)

    if model_name == "unet3d":
        from monai.networks.nets import UNet
        model = UNet(
            spatial_dims=3, in_channels=1, out_channels=1,
            channels=(16, 32, 64, 128), strides=(2, 2, 2), num_res_units=2,
            act=("LeakyReLU", {"inplace": True}), norm="INSTANCE", dropout=0.0,
        )
    elif model_name == "resnet3d":
        from week7_train_resnet3d import ResNet3DCVR
        model = ResNet3DCVR(pretrained=False)
    else:
        print("Unknown --model. Use unet3d or resnet3d.")
        return

    ckpt = torch.load(checkpoint_path, map_location=device)
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)
    model = model.to(device).eval()

    test_ds = Week7VolumePairs3D(test_pairs, augment=False)
    test_loader = DataLoader(test_ds, batch_size=1, num_workers=0)

    def _pad_3d(pre_t, post_t, target_shape):
        import torch.nn.functional as F
        _, _, h, w, d = pre_t.shape
        th, tw, td = target_shape
        if h < th or w < tw or d < td:
            pd = (0, max(0, td - d), 0, max(0, tw - w), 0, max(0, th - h))
            pre_t = F.pad(pre_t, pd, mode='constant', value=0)
            post_t = F.pad(post_t, pd, mode='constant', value=0)
        return pre_t[:, :, :th, :tw, :td], post_t[:, :, :th, :tw, :td]

    results = {"per_volume": [], "per_region": {}, "mask_names": mask_names, "source": "model", "model": model_name, "checkpoint": checkpoint_path}
    with torch.no_grad():
        for idx, (pre_path, post_path) in enumerate(test_pairs):
            pre, post = test_ds[idx]
            pre_t = pre.unsqueeze(0).to(device)
            post_t = post.unsqueeze(0).to(device)
            pre_t, post_t = _pad_3d(pre_t, post_t, TARGET_3D_PAD)
            pred_t = model(pre_t)
            pred_vol = pred_t[0, 0].cpu().numpy()
            gt_vol = post_t[0, 0].cpu().numpy()
            pred_vol = _crop_to_target(pred_vol)
            gt_vol = _crop_to_target(gt_vol)
            sid = _subject_id_from_path(pre_path)
            rec = compute_regional_metrics(pred_vol, gt_vol, territory_masks)
            if rec is None:
                continue
            rec["id"] = sid
            results["per_volume"].append(rec)

    _aggregate_per_region(results)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print("Wrote", out_path)
    _print_summary(results)
    return results


def main():
    ap = argparse.ArgumentParser(description="Week 7 regional eval: metrics inside vascular/territory and per-region")
    ap.add_argument("--pred-dir", default="", help="Directory with post_{id}_pred.nii.gz (optional if --model given)")
    ap.add_argument("--post-dir", default=os.path.join(DATA_DIR, "post"), help="GT post NIfTIs if not in pred-dir")
    ap.add_argument("--model", default="", choices=("", "unet3d", "resnet3d"), help="Run inference with this model")
    ap.add_argument("--checkpoint", default="", help="Path to model checkpoint (optional: defaults to week7_unet3d_best.pt or week7_resnet3d_best.pt; use for variants e.g. week7_unet3d_best_lowbaseline.pt)")
    ap.add_argument("--masks-dir", default="", help="MNI territory masks (default: <repo-root>/Masks)")
    ap.add_argument("--out", default="regional_week7.json")
    args = ap.parse_args()

    if args.model:
        if not args.checkpoint:
            scripts_dir = os.path.dirname(os.path.abspath(__file__))
            default_ckpts = {
                "unet3d": os.path.join(scripts_dir, "week7_results", "week7_unet3d_best.pt"),
                "resnet3d": os.path.join(scripts_dir, "week7_results", "week7_resnet3d_best.pt"),
            }
            args.checkpoint = default_ckpts.get(args.model, "")
        if not args.checkpoint or not os.path.isfile(args.checkpoint):
            print("With --model you must provide --checkpoint to an existing .pt file (or use default Phase 1 checkpoint)")
            sys.exit(1)
        run_from_model(args.model, args.checkpoint, args.masks_dir or None, args.out)
    elif args.pred_dir and os.path.isdir(args.pred_dir):
        run_from_pred_dir(args.pred_dir, args.post_dir, args.masks_dir or None, args.out)
    else:
        print("Provide either --pred-dir (with post_{id}_pred.nii.gz) or --model + --checkpoint")
        sys.exit(1)


if __name__ == "__main__":
    main()
