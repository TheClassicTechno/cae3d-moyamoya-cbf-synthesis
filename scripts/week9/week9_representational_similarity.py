#!/usr/bin/env python3
"""
Representational similarity matrix (RSM) for UNet 3D: encoder bottleneck features.
For each test subject we extract the bottleneck activation (smallest spatial size),
flatten to a vector, then compute pairwise similarity (Pearson or cosine).
Output: week9_stats/rsm_unet3d.csv, rsm_unet3d.png, subject_ids.txt.

Usage (from repo root):
  python scripts/week9/week9_representational_similarity.py --checkpoint scripts/week7_results/week7_unet3d_best.pt --output_dir week9_stats
  python scripts/week9/week9_representational_similarity.py --metric cosine
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from pipeline_model_registry import PRIMARY_RECON_CHECKPOINT

TARGET_3D_PAD = (96, 112, 96)


def _pad_3d(pre_t, post_t, target_shape):
    import torch.nn.functional as F
    _, _, h, w, d = pre_t.shape
    th, tw, td = target_shape
    if h < th or w < tw or d < td:
        pd = (0, max(0, td - d), 0, max(0, tw - w), 0, max(0, th - h))
        pre_t = F.pad(pre_t, pd, mode="constant", value=0)
        post_t = F.pad(post_t, pd, mode="constant", value=0)
    return pre_t[:, :, :th, :tw, :td], post_t[:, :, :th, :tw, :td]


def capture_bottleneck_features(model, pre_t, device):
    """Run forward and return flattened bottleneck (smallest spatial) activation."""
    smallest = [None]
    smallest_size = [float("inf")]

    def hook_save(module, input, output):
        if torch.is_tensor(output) and output.dim() == 5:
            sp = output.shape[2] * output.shape[3] * output.shape[4]
            if sp < smallest_size[0]:
                smallest_size[0] = sp
                smallest[0] = output.detach().cpu().float().numpy().copy()

    handles = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:
            try:
                h = module.register_forward_hook(hook_save)
                handles.append(h)
            except Exception:
                pass
    model.eval()
    with torch.no_grad():
        pre_t = pre_t.to(device)
        model(pre_t)
    for h in handles:
        h.remove()
    if smallest[0] is None:
        return np.array([])
    return smallest[0].reshape(smallest[0].shape[0], -1).squeeze()


def similarity_matrix(features_list, metric="pearson"):
    """features_list: list of 1D arrays. Return NxN similarity matrix."""
    X = np.array(features_list, dtype=np.float64)
    if metric == "cosine":
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1
        X = X / norms
        R = np.dot(X, X.T)
    else:
        X = X - X.mean(axis=1, keepdims=True)
        std = X.std(axis=1, keepdims=True)
        std[std == 0] = 1
        X = X / std
        R = np.dot(X, X.T) / X.shape[1]
    return np.clip(R, -1, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(PRIMARY_RECON_CHECKPOINT))
    ap.add_argument("--output_dir", default=str(ROOT / "week9_stats"))
    ap.add_argument("--metric", default="pearson", choices=("pearson", "cosine"))
    ap.add_argument("--prefix", default="rsm_unet3d")
    args = ap.parse_args()
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        print("Checkpoint not found:", ckpt_path)
        return 1
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from monai.networks.nets import UNet
    model = UNet(
        spatial_dims=3, in_channels=1, out_channels=1,
        channels=(16, 32, 64, 128), strides=(2, 2, 2), num_res_units=2,
        act=("LeakyReLU", {"inplace": True}), norm="INSTANCE", dropout=0.0,
    )
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)
    model = model.to(device).eval()
    from week7_data import get_week7_splits, _subject_id_from_path, Week7VolumePairs3D
    _, _, test_pairs = get_week7_splits()
    test_ds = Week7VolumePairs3D(test_pairs, augment=False)
    subject_ids = []
    features_list = []
    for idx in range(len(test_pairs)):
        pre_t, post_t = test_ds[idx]
        sid = _subject_id_from_path(test_pairs[idx][0])
        pre_t, post_t = _pad_3d(pre_t.unsqueeze(0), post_t.unsqueeze(0), TARGET_3D_PAD)
        feats = capture_bottleneck_features(model, pre_t, device)
        if feats.size > 0:
            features_list.append(feats)
            subject_ids.append(sid)
    if not features_list:
        print("No bottleneck features captured.")
        return 1
    R = similarity_matrix(features_list, args.metric)
    np.savetxt(out_dir / ("%s.csv" % args.prefix), R, delimiter=",")
    with open(out_dir / ("%s_subject_ids.txt" % args.prefix), "w") as f:
        f.write("\n".join(subject_ids))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(8, 7))
        im = ax.imshow(R, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_title("Representational similarity (%s)" % args.metric)
        plt.colorbar(im, ax=ax)
        fig.savefig(out_dir / ("%s.png" % args.prefix), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("Wrote", out_dir / ("%s.png" % args.prefix))
    except Exception as e:
        print("Plot failed:", e)
    print("Wrote", out_dir / ("%s.csv" % args.prefix), "and subject_ids.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
