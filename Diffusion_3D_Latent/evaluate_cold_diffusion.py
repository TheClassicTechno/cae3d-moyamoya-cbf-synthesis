#!/usr/bin/env python3
"""
Evaluation-only script for Cold Diffusion Latent
Re-evaluates the trained model with Bland-Altman analysis
"""

import os
import glob
import inspect
import json
import random
import time
import sys

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_REPO_ROOT, "pyproject.toml")):
    _parent = os.path.dirname(_REPO_ROOT)
    if _parent == _REPO_ROOT:
        raise RuntimeError("Could not locate repository root (pyproject.toml not found)")
    _REPO_ROOT = _parent

import numpy as np
import torch
import torch.nn as nn

from monai.networks.nets import UNet

from vae_3d import VAE3D
from utils import EMA, strict_normalize_volume, load_full_volume, pre_to_post_path
from cold_diffusion_latent import (
    make_cosine_schedule, cold_sample_ddim_latent, evaluate_model
)

# Add paths
sys.path.append(os.path.join(_REPO_ROOT, 'Diffusion_3D_Latent'))

def main():
    use_week7 = '--week7' in sys.argv or os.environ.get('WEEK7') == '1'
    print("="*70)
    print("RE-EVALUATING COLD DIFFUSION LATENT WITH BLAND-ALTMAN ANALYSIS" + (" [Week7]" if use_week7 else ""))
    print("="*70)

    if use_week7:
        CONFIG = {
            'target_size': (96, 112, 96),
            'latent_size': (24, 28, 24),
            'latent_channels': 4,
            'n_timesteps_train': 200,
            'n_steps_ddim': 25,
            'vae_path': 'vae_3d_week7_best.pt',
            'model_path': 'cold_diffusion_latent_week7_ema_best.pt',
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'seed': 42,
        }
        sys.path.insert(0, os.path.join(_REPO_ROOT, 'scripts'))
        from week7_data import get_week7_splits
        _, _, test_pairs = get_week7_splits()
        test_items = test_pairs
        sys.path.insert(0, os.path.join(_REPO_ROOT, 'Diffusion_3D_Latent'))
        from week7_loader import load_volume_week7
        load_fn = lambda p: load_volume_week7(p, pad_shape=(96, 112, 96))
        results_fname = 'cold_diffusion_latent_week7_results.json'
    else:
        CONFIG = {
            'target_size': (128, 128, 64),
            'latent_size': (32, 32, 16),
            'latent_channels': 4,
            'n_timesteps_train': 200,
            'n_steps_ddim': 25,
            'vae_path': 'vae_3d_best.pt',
            'model_path': 'cold_diffusion_latent_ema_best.pt',  # Use EMA model
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'seed': 42,
        }
        data_dir = _REPO_ROOT
        all_pre = sorted(glob.glob(f"{data_dir}/pre/pre_*.nii.gz"))
        all_pre_paired = [p for p in all_pre if os.path.exists(pre_to_post_path(p))]
        random.shuffle(all_pre_paired)
        n_train = int(0.75 * len(all_pre_paired))
        n_val = int(0.12 * len(all_pre_paired))
        test_paths = all_pre_paired[n_train+n_val:]
        test_items = [(p, pre_to_post_path(p)) for p in test_paths if os.path.exists(pre_to_post_path(p))]
        load_fn = None
        results_fname = 'cold_diffusion_latent_results.json'
    
    random.seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])
    torch.manual_seed(CONFIG['seed'])
    
    # Load VAE
    print(f"\n Loading VAE from {CONFIG['vae_path']}...")
    vae_model = VAE3D(
        in_channels=1,
        latent_channels=CONFIG['latent_channels'],
        channels=(32, 64, 128, 256),
        num_res_blocks=2,
        downsample_factor=4,
    ).to(CONFIG['device'])
    vae_model.load_state_dict(torch.load(CONFIG['vae_path'], map_location=CONFIG['device']))
    vae_model.eval()
    print("   VAE loaded")
    
    # Cosine alpha schedule
    alpha_np = make_cosine_schedule(CONFIG['n_timesteps_train'])
    alpha_schedule = torch.from_numpy(alpha_np).float().to(CONFIG['device'])
    
    # Load diffusion model
    print(f"\n Loading Cold Diffusion model from {CONFIG['model_path']}...")
    model = UNet(
        spatial_dims=3,
        in_channels=CONFIG['latent_channels'] * 2 + 1,  # noisy + pre + time
        out_channels=CONFIG['latent_channels'],
        channels=(32, 64, 128, 256),
        strides=(2, 2, 2),
        num_res_units=2,
    ).to(CONFIG['device'])
    
    if os.path.exists(CONFIG['model_path']):
        model.load_state_dict(torch.load(CONFIG['model_path'], map_location=CONFIG['device']))
        print("   Model loaded")
    else:
        print(f"   Model not found: {CONFIG['model_path']}")
        return
    
    print(f"  Test set: {len(test_items)} volumes")
    
    # Evaluate (evaluate_model may accept load_fn= for Week7)
    print(f"\n Evaluating on test set...")
    sig = inspect.signature(evaluate_model)
    kwargs = {}
    if load_fn is not None and 'load_fn' in sig.parameters:
        kwargs['load_fn'] = load_fn
    results = evaluate_model(model, vae_model, test_items, 
                            CONFIG['n_timesteps_train'], CONFIG['n_steps_ddim'],
                            CONFIG['target_size'], CONFIG['device'], **kwargs)
    
    print(f"\nCold Diffusion Latent Results:")
    print(f"  MAE:  {results['mae_mean']:.4f} ± {results['mae_std']:.4f}")
    print(f"  SSIM: {results['ssim_mean']:.4f} ± {results['ssim_std']:.4f}")
    print(f"  PSNR: {results['psnr_mean']:.2f} ± {results['psnr_std']:.2f} dB")
    
    # Print Bland-Altman analysis
    ba = results.get('bland_altman', {})
    if ba.get('n_samples', 0) > 0:
        print(f"\nBland-Altman Analysis (Clinical Agreement):")
        print(f"  Mean Bias: {ba['mean_bias']:.6f} (95% CI: [{ba['bias_ci_lower']:.6f}, {ba['bias_ci_upper']:.6f}])")
        print(f"  Upper LOA: {ba['upper_loa']:.6f} (95% CI: [{ba['loa_upper_ci']:.6f}, ...])")
        print(f"  Lower LOA: {ba['lower_loa']:.6f} (95% CI: [..., {ba['loa_lower_ci']:.6f}])")
        print(f"  SD of Differences: {ba['std_diff']:.6f}")
        print(f"  N Samples: {ba['n_samples']:,}")
    
    # Save results
    with open(results_fname, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n Evaluation complete!")
    print(f"  Results saved to: {results_fname}")
    print("="*70)

if __name__ == '__main__':
    main()
