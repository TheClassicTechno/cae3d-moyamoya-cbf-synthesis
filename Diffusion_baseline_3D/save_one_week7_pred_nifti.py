#!/usr/bin/env python3
"""Save one DDPM Week7 test prediction and GT as NIfTI for inspection (constant/wrong scale check)."""
import os
import sys
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import DataLoader

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_REPO_ROOT, "pyproject.toml")):
    _parent = os.path.dirname(_REPO_ROOT)
    if _parent == _REPO_ROOT:
        raise RuntimeError("Could not locate repository root (pyproject.toml not found)")
    _REPO_ROOT = _parent

# Week7 setup
os.environ['WEEK7'] = '1'
sys.path.insert(0, os.path.join(_REPO_ROOT, 'scripts'))
from week7_data import get_week7_splits, Week7VolumePairs3D
from week7_preprocess import TARGET_SHAPE

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diffusion_model_3d import (
    SimpleCondDiffusion3D, make_beta_schedule, p_sample_loop,
    PAD_3D_WEEK7, WEEK7_ORIGINAL_SHAPE, _pad_3d,
)

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_timesteps = 1000
    ckpt_path = os.path.join(os.path.dirname(__file__), 'ddpm_3d_week7_best.pt')
    out_dir = os.path.dirname(__file__)

    if not os.path.isfile(ckpt_path):
        print("Missing:", ckpt_path)
        return

    print("Loading Week7 test data (one batch)...")
    _, _, test_pairs = get_week7_splits()
    test_dataset = Week7VolumePairs3D(test_pairs, augment=False, target_shape=TARGET_SHAPE)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    pre_vol, post_vol = next(iter(test_loader))
    pre_vol = pre_vol.to(device)
    post_vol = post_vol.to(device)
    pre_vol, post_vol = _pad_3d(pre_vol, post_vol, PAD_3D_WEEK7)

    print("Building schedule and model...")
    betas = make_beta_schedule(n_timesteps)
    alphas = 1.0 - betas
    alphas_bar = np.cumprod(alphas)
    alphas_bar_sqrt = np.sqrt(alphas_bar)
    one_minus_alphas_bar_sqrt = np.sqrt(1.0 - alphas_bar)
    betas_t = torch.from_numpy(betas).float().to(device)
    alphas_t = torch.from_numpy(alphas).float().to(device)
    alphas_bar_sqrt_t = torch.from_numpy(alphas_bar_sqrt).float().to(device)
    one_minus_t = torch.from_numpy(one_minus_alphas_bar_sqrt).float().to(device)

    model = SimpleCondDiffusion3D(ch=16).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    print("Running DDPM sampling (1000 steps)...")
    with torch.no_grad():
        pred_vol = p_sample_loop(model, pre_vol, n_timesteps, betas_t, alphas_t,
                                  alphas_bar_sqrt_t, one_minus_t, device)

    oh, ow, od = WEEK7_ORIGINAL_SHAPE
    pred_np = pred_vol[0, 0].cpu().numpy()[:oh, :ow, :od]
    gt_np = post_vol[0, 0].cpu().numpy()[:oh, :ow, :od]

    pred_nii = nib.Nifti1Image(pred_np.astype(np.float32), np.eye(4))
    gt_nii = nib.Nifti1Image(gt_np.astype(np.float32), np.eye(4))
    pred_path = os.path.join(out_dir, 'week7_one_pred.nii.gz')
    gt_path = os.path.join(out_dir, 'week7_one_gt.nii.gz')
    nib.save(pred_nii, pred_path)
    nib.save(gt_nii, gt_path)

    mae = np.abs(pred_np - gt_np).mean()
    print(f"  Pred range [{pred_np.min():.4f}, {pred_np.max():.4f}], GT [{gt_np.min():.4f}, {gt_np.max():.4f}]")
    print(f"  MAE (this sample): {mae:.4f}")
    print(f"Saved: {pred_path}, {gt_path}")

if __name__ == '__main__':
    main()
