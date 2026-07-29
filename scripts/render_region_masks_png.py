#!/usr/bin/env python3
"""
Render vascular/territory masks from <repo-root>/Masks as PNG slices.
Output: Masks/region_mask_pngs/ with one PNG per mask (axial mid-slices) and a composite.
"""
import os
import sys
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_REPO_ROOT, "pyproject.toml")):
    _parent = os.path.dirname(_REPO_ROOT)
    if _parent == _REPO_ROOT:
        raise RuntimeError("Could not locate repository root (pyproject.toml not found)")
    _REPO_ROOT = _parent

# Allow import of week7_preprocess when run from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TARGET_SHAPE = (91, 109, 91)
BRAIN_MASK_PATH = os.path.join(_REPO_ROOT, "MNI152_T1_2mm_brain_mask_dil.nii.gz")
MASKS_DIR = os.path.join(_REPO_ROOT, "Masks")
OUT_DIR = os.path.join(_REPO_ROOT, "Masks/region_mask_pngs")


def load_vol(path, target_shape):
    img = nib.load(path)
    data = np.asarray(img.get_fdata()).squeeze().astype(np.float32)
    if data.ndim == 4:
        data = data[..., 0]
    if data.shape != target_shape:
        factors = [target_shape[i] / data.shape[i] for i in range(3)]
        data = zoom(data, factors, order=0)
    return (data > 0.5).astype(np.float32)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(OUT_DIR, exist_ok=True)

    # Brain mask
    brain = load_vol(BRAIN_MASK_PATH, TARGET_SHAPE)
    if brain.max() > 1:
        brain = (brain > 0.5).astype(np.float32)

    # Territory NIfTIs (*2mm*.nii.gz)
    files = sorted(f for f in os.listdir(MASKS_DIR) if f.endswith(".nii.gz") and "2mm" in f)
    # Exclude combined so we show individual territories clearly
    files = [f for f in files if "vascular_territory" not in f]

    D, H, W = TARGET_SHAPE
    # Axial slices (z = depth): show a few representative slices
    slices_axial = [D // 4, D // 2, 3 * D // 4]

    for fname in files:
        path = os.path.join(MASKS_DIR, fname)
        name = fname.replace(".nii.gz", "")
        mask = load_vol(path, TARGET_SHAPE)
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for i, z in enumerate(slices_axial):
            sl = mask[z]
            axes[i].imshow(sl.T, origin="lower", cmap="hot", vmin=0, vmax=1)
            axes[i].set_title(f"{name} (z={z})")
            axes[i].axis("off")
        plt.suptitle(f"Vascular/territory mask: {name}", fontsize=12)
        plt.tight_layout()
        out_path = os.path.join(OUT_DIR, f"{name}.png")
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"Saved {out_path}")

    # Composite: one row per territory (ACA, MCA, PCA, cerebellum, pons_medulla), L and R columns, mid slice
    z_mid = D // 2
    groups = {}
    for fname in files:
        name = fname.replace(".nii.gz", "")
        base = name.replace("MNI_left_", "").replace("MNI_right_", "")
        if base not in groups:
            groups[base] = []
        groups[base].append((name, load_vol(os.path.join(MASKS_DIR, fname), TARGET_SHAPE)))

    nrows = len(groups)
    fig, axes = plt.subplots(nrows, 2, figsize=(8, 4 * nrows))
    for row, (base, items) in enumerate(sorted(groups.items())):
        for col, (name, mask) in enumerate(sorted(items, key=lambda x: x[0])):
            if col >= 2:
                break
            ax = axes[row, col]
            ax.imshow(mask[z_mid].T, origin="lower", cmap="hot", vmin=0, vmax=1)
            ax.set_title(f"{name} (z={z_mid})")
            ax.axis("off")
        for c in range(len(items), 2):
            axes[row, c].axis("off")
    plt.suptitle("Vascular/territory masks (axial mid-slice) - WEEK7_REGION_WEIGHT=1", fontsize=11)
    plt.tight_layout()
    composite_path = os.path.join(OUT_DIR, "composite_territories_mid_slice.png")
    plt.savefig(composite_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved {composite_path}")

    # Brain mask only (reference)
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(brain[z_mid].T, origin="lower", cmap="gray", vmin=0, vmax=1)
    ax.set_title("Brain mask (MNI 2mm, mid axial slice)")
    ax.axis("off")
    plt.tight_layout()
    brain_path = os.path.join(OUT_DIR, "brain_mask_mid_slice.png")
    plt.savefig(brain_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved {brain_path}")

    # Combined region weight map (1.0 brain, 1.5 vascular) at mid slice
    from week7_preprocess import get_region_weight_mask_for_shape

    weight = get_region_weight_mask_for_shape(TARGET_SHAPE, masks_dir=MASKS_DIR, vascular_weight=1.5)
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    im = ax.imshow(weight[z_mid].T, origin="lower", cmap="viridis", vmin=0, vmax=1.5)
    plt.colorbar(im, ax=ax, label="Loss weight")
    ax.set_title("Region weight map (0=outside, 1=brain, 1.5=vascular)")
    ax.axis("off")
    plt.tight_layout()
    weight_path = os.path.join(OUT_DIR, "region_weight_map_mid_slice.png")
    plt.savefig(weight_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved {weight_path}")

    print(f"All PNGs in {OUT_DIR}")


if __name__ == "__main__":
    main()
