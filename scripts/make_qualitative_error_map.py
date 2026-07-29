#!/usr/bin/env python3
"""
Create qualitative_error.png = |predicted - target| from existing qualitative_predicted.png
and qualitative_post_ACZ.png. Run from repo root: python scripts/make_qualitative_error_map.py
"""
import os
import argparse
import numpy as np

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_REPO_ROOT, "pyproject.toml")):
    _parent = os.path.dirname(_REPO_ROOT)
    if _parent == _REPO_ROOT:
        raise RuntimeError("Could not locate repository root (pyproject.toml not found)")
    _REPO_ROOT = _parent

FIG_DIR = os.path.join(_REPO_ROOT, "figures")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=FIG_DIR, help="Output directory")
    ap.add_argument("--pred", default=None, help="Path to predicted PNG (default: out/qualitative_predicted.png)")
    ap.add_argument("--target", default=None, help="Path to target PNG (default: out/qualitative_post_ACZ.png)")
    args = ap.parse_args()
    pred_path = args.pred or os.path.join(args.out, "qualitative_predicted.png")
    target_path = args.target or os.path.join(args.out, "qualitative_post_ACZ.png")
    out_path = os.path.join(args.out, "qualitative_error.png")

    if not os.path.isfile(pred_path):
        print("Not found:", pred_path)
        return
    if not os.path.isfile(target_path):
        print("Not found:", target_path)
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from PIL import Image
    except ImportError as e:
        print("Need matplotlib and Pillow:", e)
        return

    pred_img = np.array(Image.open(pred_path))
    target_img = np.array(Image.open(target_path))

    # Handle RGB/RGBA: convert to grayscale (0-1)
    if pred_img.ndim == 3:
        pred_gray = np.mean(pred_img[..., :3], axis=-1) / 255.0
    else:
        pred_gray = pred_img.astype(np.float64) / 255.0
    if target_img.ndim == 3:
        target_gray = np.mean(target_img[..., :3], axis=-1) / 255.0
    else:
        target_gray = target_img.astype(np.float64) / 255.0

    # Match sizes (in case of small differences from savefig)
    if pred_gray.shape != target_gray.shape:
        from scipy.ndimage import zoom
        h, w = target_gray.shape
        ph, pw = pred_gray.shape
        pred_gray = zoom(pred_gray, (h / ph, w / pw), order=1)

    error = np.abs(pred_gray.astype(np.float64) - target_gray.astype(np.float64))
    error = np.clip(error, 0, 1)

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    im = ax.imshow(error.T, origin="lower", cmap="hot", vmin=0, vmax=0.5)
    ax.set_title("|Pred $-$ Target|")
    ax.axis("off")
    plt.colorbar(im, ax=ax, shrink=0.7, label="Absolute error")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print("Saved", out_path)


if __name__ == "__main__":
    main()
