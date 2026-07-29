#!/usr/bin/env python3
"""
Export four PNGs for the paper figure: pre-ACZ, post-ACZ (target), predicted, brain mask (middle axial slice).
Usage:
  cd <repo-root> && python scripts/export_qualitative_figures.py [--checkpoint PATH] [--out figures]
Without --checkpoint, predicted slice is set to post (placeholder); with checkpoint, run one forward pass.
"""
import os
import sys
import argparse
import numpy as np

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_REPO_ROOT, "pyproject.toml")):
    _parent = os.path.dirname(_REPO_ROOT)
    if _parent == _REPO_ROOT:
        raise RuntimeError("Could not locate repository root (pyproject.toml not found)")
    _REPO_ROOT = _parent

sys.path.insert(0, os.path.join(_REPO_ROOT, "scripts"))
from week7_preprocess import load_volume, TARGET_SHAPE, get_brain_mask, get_pre_post_pairs

OUT_DIR = os.path.join(_REPO_ROOT, "figures")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_DIR, help="Output directory for PNGs")
    ap.add_argument("--checkpoint", default="", help="Optional: path to model .pt for predicted slice")
    ap.add_argument("--index", type=int, default=0, help="Test set subject index (0..31)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    _, _, test_pairs = get_week7_splits()
    if args.index >= len(test_pairs):
        args.index = 0
    pre_path, post_path = test_pairs[args.index]
    pre_vol = load_volume(pre_path, target_shape=TARGET_SHAPE, apply_mask=False, minmax=True)
    post_vol = load_volume(post_path, target_shape=TARGET_SHAPE, apply_mask=False, minmax=True)
    D, H, W = pre_vol.shape
    z_mid = D // 2
    pre_slice = pre_vol[z_mid]
    post_slice = post_vol[z_mid]

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required for PNG export")
        return

    def save_slice(arr, path, title=""):
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.imshow(arr.T, origin="lower", cmap="gray", vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"Saved {path}")

    save_slice(pre_slice, os.path.join(args.out, "qualitative_pre_ACZ.png"), "Pre-ACZ")
    save_slice(post_slice, os.path.join(args.out, "qualitative_post_ACZ.png"), "Post-ACZ (target)")

    if args.checkpoint and os.path.isfile(args.checkpoint):
        # Optional: load model and run inference (UNet 3D / script style)
        sys.path.insert(0, os.path.join(_REPO_ROOT, "scripts"))
        import torch
        from torch.utils.data import DataLoader
        test_ds = Week7VolumePairs3D([(pre_path, post_path)], augment=False, target_shape=TARGET_SHAPE)
        loader = DataLoader(test_ds, batch_size=1, num_workers=0)
        pre_t, _ = next(iter(loader))
        # Try loading as MONAI UNet 3D
        try:
            from monai.networks.nets import UNet
            model = UNet(spatial_dims=3, in_channels=1, out_channels=1, channels=(16, 32, 64, 128), strides=(2, 2, 2)).eval()
            model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
            with torch.no_grad():
                pred = model(pre_t)
            pred_slice = pred[0, 0, z_mid].numpy()
            pred_slice = np.clip(pred_slice, 0, 1)
            save_slice(pred_slice, os.path.join(args.out, "qualitative_predicted.png"), "Predicted")
        except Exception as e:
            print("Model load failed, using post as placeholder for predicted:", e)
            save_slice(post_slice, os.path.join(args.out, "qualitative_predicted.png"), "Predicted (placeholder)")
    else:
        save_slice(post_slice, os.path.join(args.out, "qualitative_predicted.png"), "Predicted (placeholder)")

    mask_3d = get_brain_mask(BRAIN_MASK_PATH, TARGET_SHAPE)
    mask_slice = mask_3d[z_mid]
    save_slice(mask_slice, os.path.join(args.out, "qualitative_mask.png"), "Brain mask")

    print("Done. Place the four PNGs in figures/ for the paper.")


if __name__ == "__main__":
    main()
