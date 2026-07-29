#!/usr/bin/env python3
"""
Ensemble: FNO 3D + 3D DDPM (simple). Same test split (seed 42), average predictions.
FNO runs at (128,128,64); diffusion at (64,64,32) then upsampled to (128,128,64).
Usage:
  python3 scripts/ensemble_fno_diffusion_3d.py --w-fno 0.5 --w-diff 0.5 --max-n 10
"""
import os
import sys
import glob
import json
import random
import argparse
import numpy as np
import nibabel as nib
import torch
from scipy.ndimage import zoom
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_REPO_ROOT, "pyproject.toml")):
    _parent = os.path.dirname(_REPO_ROOT)
    if _parent == _REPO_ROOT:
        raise RuntimeError("Could not locate repository root (pyproject.toml not found)")
    _REPO_ROOT = _parent

DATA_DIR = _REPO_ROOT
SEED = 42
FNO_SIZE = (128, 128, 64)
DIFF_SIZE = (64, 64, 32)


def load_volume(path, target_size):
    import nibabel as nib
    from scipy.ndimage import zoom as z
    nii = nib.load(path)
    vol = np.asarray(nii.dataobj).astype(np.float32).squeeze()
    if vol.ndim == 4:
        vol = vol[..., 0]
    factors = [target_size[i] / vol.shape[i] for i in range(3)]
    vol = z(vol, factors, order=1)
    vmin, vmax = vol.min(), vol.max()
    vol = (vol - vmin) / (vmax - vmin + 1e-8)
    return vol.astype(np.float32)


def pre_to_post(pre_path):
    base = os.path.basename(pre_path).replace("pre_", "post_")
    d = os.path.dirname(pre_path).replace("/pre", "/post")
    return os.path.join(d or "post", base)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fno-ckpt", default=os.path.join(_REPO_ROOT, "NeuralOperators/fno_3d_finetuned_slice_brain_80ep.pt"))
    ap.add_argument("--diff-ckpt", default=os.path.join(_REPO_ROOT, "Diffusion_baseline_3D/ddpm_3d_best.pt"))
    ap.add_argument("--w-fno", type=float, default=0.5)
    ap.add_argument("--w-diff", type=float, default=0.5)
    ap.add_argument("--out-dir", default=os.path.join(_REPO_ROOT, "ensemble_fno_diffusion_3d"))
    ap.add_argument("--max-n", type=int, default=10)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    random.seed(SEED)
    all_pre = sorted(glob.glob(os.path.join(DATA_DIR, "pre", "pre_*.nii.gz")))
    all_pre = [p for p in all_pre if os.path.exists(pre_to_post(p))]
    n = len(all_pre)
    n_train, n_val = int(0.75 * n), int(0.125 * n)
    test_pre = all_pre[n_train + n_val :][: args.max_n]

    os.makedirs(args.out_dir, exist_ok=True)

    # Load FNO
    sys.path.insert(0, os.path.join(_REPO_ROOT, "NeuralOperators"))
    from fno_3d_cvr import SimpleFNO3D, add_position_channels
    fno_ckpt = torch.load(args.fno_ckpt, map_location=device)
    width = fno_ckpt["fc0.weight"].shape[0]
    in_ch = fno_ckpt["fc0.weight"].shape[1]
    modes = 12 if width == 64 else 8
    fno = SimpleFNO3D(in_ch=in_ch, out_ch=1, modes=modes, width=width).to(device)
    fno.load_state_dict(fno_ckpt)
    fno.eval()

    # Load diffusion (simple model)
    sys.path.insert(0, os.path.join(_REPO_ROOT, "Diffusion_baseline_3D"))
    from diffusion_model_3d import (
        SimpleCondDiffusion3D, make_beta_schedule, p_sample_loop_ddim,
    )
    diff_ckpt = torch.load(args.diff_ckpt, map_location=device)
    betas = make_beta_schedule(1000)
    alphas = 1.0 - betas
    alphas_bar_sqrt = torch.from_numpy(np.sqrt(np.cumprod(alphas))).float().to(device)
    alphas_t = torch.from_numpy(alphas).float().to(device)
    diff = SimpleCondDiffusion3D(ch=16).to(device)
    diff.load_state_dict(diff_ckpt)
    diff.eval()

    results = []
    aff = np.eye(4)
    for pre_path in test_pre:
        name = os.path.basename(pre_path).replace(".nii.gz", "").replace("pre_", "")
        post_path = pre_to_post(pre_path)
        gt = load_volume(post_path, FNO_SIZE)

        # FNO at 128,128,64
        pre_fno = load_volume(pre_path, FNO_SIZE)
        pre_4ch = add_position_channels(pre_fno)
        x = torch.from_numpy(pre_4ch).unsqueeze(0).float().to(device)
        with torch.no_grad():
            pred_fno = fno(x)[0, 0].cpu().numpy()

        # Diffusion at 64,64,32 then upsample
        pre_diff = load_volume(pre_path, DIFF_SIZE)
        xd = torch.from_numpy(pre_diff).unsqueeze(0).unsqueeze(0).float().to(device)
        torch.manual_seed(SEED)
        with torch.no_grad():
            pred_diff_small = p_sample_loop_ddim(diff, xd, 1000, alphas_t, alphas_bar_sqrt, device)[0, 0].cpu().numpy()
        factors = [FNO_SIZE[i] / DIFF_SIZE[i] for i in range(3)]
        pred_diff = zoom(pred_diff_small, factors, order=1).astype(np.float32)

        # Ensemble
        pred_ens = args.w_fno * pred_fno + args.w_diff * pred_diff
        pred_ens = np.clip(pred_ens, 0, 1).astype(np.float32)

        mae = float(np.abs(pred_ens - gt).mean())
        ssim_val = float(ssim(gt, pred_ens, data_range=1.0))
        psnr_val = float(psnr(gt, pred_ens, data_range=1.0))
        results.append({"id": name, "mae": mae, "ssim": ssim_val, "psnr": psnr_val})

        nib.save(nib.Nifti1Image(pred_ens, aff), os.path.join(args.out_dir, f"post_{name}_ensemble.nii.gz"))
        print(name, "MAE %.4f SSIM %.4f PSNR %.2f dB" % (mae, ssim_val, psnr_val))

    summary = {"w_fno": args.w_fno, "w_diff": args.w_diff, "n": len(results)}
    if results:
        summary["mae_mean"] = float(np.mean([r["mae"] for r in results]))
        summary["ssim_mean"] = float(np.mean([r["ssim"] for r in results]))
        summary["psnr_mean"] = float(np.mean([r["psnr"] for r in results]))
    with open(os.path.join(args.out_dir, "ensemble_results.json"), "w") as f:
        json.dump({"summary": summary, "per_volume": results}, f, indent=2)
    print("Ensemble PSNR mean %.2f dB" % (summary.get("psnr_mean") or 0))
    print("Saved", args.out_dir)


if __name__ == "__main__":
    main()
