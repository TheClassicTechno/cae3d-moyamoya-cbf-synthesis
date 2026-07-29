#!/usr/bin/env python3
"""
Verify 3D volume integrity: ensure resize never grows dimensions, and export
sample resized/pre/post volumes as NIfTI for visual inspection.
Usage: python verify_volume_integrity.py [--data-dir <repo-root>] [--out-dir <repo-root>/visual_checks/volumes] [--max-volumes 5]
"""
import os
import sys
import argparse
import glob
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_REPO_ROOT, "pyproject.toml")):
    _parent = os.path.dirname(_REPO_ROOT)
    if _parent == _REPO_ROOT:
        raise RuntimeError("Could not locate repository root (pyproject.toml not found)")
    _REPO_ROOT = _parent

TARGET_SIZE = (128, 128, 64)
DATA_DIR = _REPO_ROOT


def load_volume_resize(nii_path: str, target_size: tuple):
    """
    Load NIfTI and resize to target_size with scipy.ndimage.zoom.
    Returns (data_orig_shape, data_resized, zoom_factors_used).
    """
    img = nib.load(nii_path)
    data = np.asarray(img.dataobj).astype(np.float32).squeeze()
    if data.ndim == 4:
        data = data[..., 0]
    assert data.ndim == 3, f"Expected 3D, got {data.ndim}D"
    orig_shape = data.shape
    zoom_factors = [target_size[i] / data.shape[i] for i in range(3)]
    resized = zoom(data, zoom_factors, order=1).astype(np.float32)
    return orig_shape, resized, zoom_factors


def main():
    ap = argparse.ArgumentParser(description="Verify 3D volume integrity and export sample NIfTIs")
    ap.add_argument("--data-dir", default=DATA_DIR, help="Root dir containing pre/ and post/")
    ap.add_argument("--out-dir", default=None, help="Output dir for NIfTI samples (default: data_dir/visual_checks/volumes)")
    ap.add_argument("--max-volumes", type=int, default=5, help="Max number of pre/post pairs to check and export")
    args = ap.parse_args()

    data_dir = args.data_dir
    out_dir = args.out_dir or os.path.join(data_dir, "visual_checks", "volumes")
    os.makedirs(out_dir, exist_ok=True)

    pre_dir = os.path.join(data_dir, "pre")
    post_dir = os.path.join(data_dir, "post")
    if not os.path.isdir(pre_dir):
        print("Pre dir not found:", pre_dir)
        return

    all_pre = sorted(glob.glob(os.path.join(pre_dir, "pre_*.nii.gz")))
    if not all_pre:
        print("No pre_*.nii.gz found in", pre_dir)
        return

    def post_path(pre_path):
        base = os.path.basename(pre_path).replace("pre_", "post_")
        return os.path.join(post_dir, base)

    pairs = [(p, post_path(p)) for p in all_pre if os.path.exists(post_path(p))][: args.max_volumes]
    print("=" * 60)
    print("3D Volume integrity check (no dimension growth)")
    print("=" * 60)
    print("Target size:", TARGET_SIZE)
    print("Checking and exporting", len(pairs), "pairs to", out_dir)
    print()

    for pre_path, post_path_str in pairs:
        name = os.path.basename(pre_path).replace(".nii.gz", "")
        try:
            orig_pre, resized_pre, zf_pre = load_volume_resize(pre_path, TARGET_SIZE)
            orig_post, resized_post, zf_post = load_volume_resize(post_path_str, TARGET_SIZE)
        except Exception as e:
            print("  ERROR", name, e)
            continue

        # Check: no dimension should grow (target should be <= original in each dim for safety)
        for i in range(3):
            if resized_pre.shape[i] > orig_pre[i]:
                print("  WARN", name, "pre: dim", i, "grew", orig_pre[i], "->", resized_pre.shape[i])
            if resized_post.shape[i] > orig_post[i]:
                print("  WARN", name, "post: dim", i, "grew", orig_post[i], "->", resized_post.shape[i])
        if resized_pre.shape != TARGET_SIZE:
            print("  WARN", name, "pre resized shape", resized_pre.shape, "!= target", TARGET_SIZE)

        # Export as NIfTI for visual inspection
        aff = np.eye(4)
        nib.save(nib.Nifti1Image(resized_pre, aff), os.path.join(out_dir, f"{name}_pre_resized.nii.gz"))
        nib.save(nib.Nifti1Image(resized_post, aff), os.path.join(out_dir, f"{name.replace('pre_', 'post_')}_post_resized.nii.gz"))
        print("  OK", name, "orig_pre", orig_pre, "resized", resized_pre.shape, "zoom_pre", [round(z, 3) for z in zf_pre])

    print()
    print("Done. Inspect NIfTIs in", out_dir)


if __name__ == "__main__":
    main()
