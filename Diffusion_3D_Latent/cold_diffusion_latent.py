"""
Cold Diffusion in latent space for 3D CVR prediction.
Exports: make_cosine_schedule, cold_sample_ddim_latent, evaluate_model.
"""
import os
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_REPO_ROOT, "pyproject.toml")):
    _parent = os.path.dirname(_REPO_ROOT)
    if _parent == _REPO_ROOT:
        raise RuntimeError("Could not locate repository root (pyproject.toml not found)")
    _REPO_ROOT = _parent


def make_cosine_schedule(n_timesteps=200):
    """Cosine degradation schedule. alpha[t] from 0 (t=0, clean post) to 1 (t=T, pre)."""
    steps = np.arange(n_timesteps + 1, dtype=np.float64) / n_timesteps
    alpha_bar = np.cos((steps + 0.008) / 1.008 * np.pi / 2) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    alpha = 1 - alpha_bar
    return alpha.astype(np.float32)


def cold_sample_ddim_latent_simple(model, vae_model, pre_vol, n_timesteps_train, n_steps_ddim, alpha_schedule, target_size, device):
    """Simpler: model(x_t, pre_latent, t) expects (noisy_latent, pre_latent, t). Input concat is (x_t, pre_latent, time_emb)."""
    from utils import strict_normalize_volume
    if pre_vol.ndim == 3:
        pre_vol = strict_normalize_volume(pre_vol)
        pre_t = torch.from_numpy(pre_vol).unsqueeze(0).unsqueeze(0).float().to(device)
    else:
        pre_t = torch.from_numpy(np.asarray(pre_vol)).float().to(device)
        if pre_t.dim() == 4:
            pre_t = pre_t.unsqueeze(0)
    with torch.no_grad():
        pre_latent = vae_model.encode_to_latent(pre_t)
    x_t = pre_latent.clone()
    step_indices = np.linspace(n_timesteps_train - 1, 0, n_steps_ddim, dtype=int).tolist()
    alpha_schedule_t = torch.from_numpy(alpha_schedule).float().to(device)
    with torch.no_grad():
        for t_val in step_indices:
            t_batch = torch.full((x_t.shape[0],), t_val, dtype=torch.long, device=device)
            # Model expects (noisy, pre, time): in_channels = latent*2+1 -> concat [x_t, pre_latent, time_emb]
            time_emb = (t_batch.float() / max(n_timesteps_train - 1, 1)).view(-1, 1, 1, 1, 1).expand(-1, 1, *x_t.shape[2:])
            model_in = torch.cat([x_t, pre_latent, time_emb], dim=1)
            # MONAI UNet takes only (x); time is already in model_in
            x0_pred = model(model_in)
            x0_pred = torch.clamp(x0_pred, -10.0, 10.0)
            if t_val > 0:
                alpha_prev = alpha_schedule_t[t_val - 1]
                alpha_prev = alpha_prev.view(1, 1, 1, 1, 1).expand_as(x0_pred)
                x_t = (1 - alpha_prev) * x0_pred + alpha_prev * pre_latent
            else:
                x_t = x0_pred
    with torch.no_grad():
        pred_vol = vae_model.decode_from_latent(x_t)
    out = pred_vol[0, 0].cpu().numpy()
    if target_size and out.shape != target_size:
        from scipy.ndimage import zoom
        zoom_factors = [target_size[i] / out.shape[i] for i in range(3)]
        out = zoom(out, zoom_factors, order=1)
    return out


def evaluate_model(model, vae_model, test_items, n_timesteps_train, n_steps_ddim, target_size, device, load_fn=None):
    """Evaluate cold diffusion latent on test (pre, post) path pairs. If load_fn given, use it to load volumes."""
    from utils import strict_normalize_volume, load_full_volume, bland_altman_analysis
    model.eval()
    vae_model.eval()
    mae_list, ssim_list, psnr_list = [], [], []
    all_predicted = []
    all_ground_truth = []
    alpha_np = make_cosine_schedule(n_timesteps_train)
    with torch.no_grad():
        for pre_p, post_p in test_items:
            try:
                if load_fn is not None:
                    pre_vol = strict_normalize_volume(load_fn(pre_p))
                    post_vol = strict_normalize_volume(load_fn(post_p))
                else:
                    pre_vol = strict_normalize_volume(load_full_volume(pre_p, target_size))
                    post_vol = strict_normalize_volume(load_full_volume(post_p, target_size))
                pred_vol = cold_sample_ddim_latent_simple(
                    model, vae_model, pre_vol, n_timesteps_train, n_steps_ddim,
                    alpha_np, target_size, device
                )
                if np.isnan(pred_vol).any() or np.isinf(pred_vol).any():
                    continue
                sh = min(pred_vol.shape[0], post_vol.shape[0]), min(pred_vol.shape[1], post_vol.shape[1]), min(pred_vol.shape[2], post_vol.shape[2])
                pred_slice = pred_vol[:sh[0], :sh[1], :sh[2]]
                post_slice = post_vol[:sh[0], :sh[1], :sh[2]]
                all_predicted.append(pred_slice.flatten())
                all_ground_truth.append(post_slice.flatten())
                if load_fn is not None:
                    import sys
                    if os.path.join(_REPO_ROOT, "scripts") not in sys.path:
                        sys.path.insert(0, os.path.join(_REPO_ROOT, "scripts"))
                    from week7_preprocess import metrics_in_brain
                    m = metrics_in_brain(pred_slice, post_slice, data_range=1.0)
                    mae_list.append(m["mae_mean"])
                    ssim_list.append(m["ssim_mean"])
                    psnr_list.append(m["psnr_mean"])
                else:
                    mae_list.append(np.abs(pred_slice - post_slice).mean())
                    ssim_list.append(ssim(post_slice, pred_slice, data_range=1.0))
                    psnr_list.append(psnr(post_slice, pred_slice, data_range=1.0))
            except Exception as e:
                print(f"Error evaluating {pre_p}: {e}, skipping")
                continue
    if len(all_predicted) > 0:
        all_pred_flat = np.concatenate(all_predicted)
        all_gt_flat = np.concatenate(all_ground_truth)
        ba_results = bland_altman_analysis(all_pred_flat, all_gt_flat)
    else:
        ba_results = {'mean_bias': np.nan, 'std_diff': np.nan, 'upper_loa': np.nan, 'lower_loa': np.nan,
                      'loa_upper_ci': np.nan, 'loa_lower_ci': np.nan, 'bias_ci_upper': np.nan, 'bias_ci_lower': np.nan, 'n_samples': 0}
    return {
        'mae_mean': float(np.mean(mae_list)) if mae_list else np.nan,
        'mae_std': float(np.std(mae_list)) if mae_list else np.nan,
        'ssim_mean': float(np.mean(ssim_list)) if ssim_list else np.nan,
        'ssim_std': float(np.std(ssim_list)) if ssim_list else np.nan,
        'psnr_mean': float(np.mean(psnr_list)) if psnr_list else np.nan,
        'psnr_std': float(np.std(psnr_list)) if psnr_list else np.nan,
        'bland_altman': ba_results,
    }


# API alias for scripts that import this name
cold_sample_ddim_latent = cold_sample_ddim_latent_simple
