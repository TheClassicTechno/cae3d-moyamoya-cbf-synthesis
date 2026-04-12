#!/usr/bin/env python3
# Creates 2x3 visualisations for diffusion model outputs

import os
import glob
import csv
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as sk_ssim
from skimage.metrics import peak_signal_noise_ratio as sk_psnr

PRED_DIR = os.path.join("runs_diffusion_post_from_pre", "pred_samples")
OUT_DIR  = os.path.join("runs_diffusion_post_from_pre", "analysis")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------
def ssim_np(a, b):
    dr = b.max() - b.min()
    if dr == 0:
        dr = 1.0
    return float(sk_ssim(a, b, data_range=dr))

def psnr_np(a, b):
    dr = b.max() - b.min()
    if dr == 0:
        dr = 1.0
    return float(sk_psnr(b, a, data_range=dr))

def mae_np(a, b):
    return float(np.mean(np.abs(a - b)))

# ---------------------------------------------------------
# File discovery
# ---------------------------------------------------------
def find_indices(pred_dir):
    outs = sorted(glob.glob(os.path.join(pred_dir, "out_*.npy")))
    idxs = [int(os.path.splitext(os.path.basename(p))[0].split("_")[1]) for p in outs]
    return sorted(idxs)

# ---------------------------------------------------------
# Panel creation (2 rows, 3 columns)
# ---------------------------------------------------------
def panel_2x3(inp, pred, tgt, diff1, diff2, diff3, mae1, mae2, mae3, fname):
    import matplotlib.gridspec as gridspec

    # Rotate all images 90° counter-clockwise for radiology standard
    inp = np.rot90(inp, k=1)
    pred = np.rot90(pred, k=1)
    tgt = np.rot90(tgt, k=1)
    diff1 = np.rot90(diff1, k=1)
    diff2 = np.rot90(diff2, k=1)
    diff3 = np.rot90(diff3, k=1)

    # Shared vmax for all three heatmaps so colors are comparable.
    # Use the non-zero 99th percentile to avoid outlier-driven colormap saturation
    combined = np.concatenate([diff1.ravel(), diff2.ravel(), diff3.ravel()])
    nonzero = combined[combined > 0]
    if nonzero.size == 0:
        # Fallback to max-based scaling (and avoid zero)
        vmax = max(diff1.max(), diff2.max(), diff3.max())
        if vmax == 0:
            vmax = 1.0
    else:
        # Use a robust percentile (ignore zeros) to set the color scale
        vmax = float(np.nanpercentile(nonzero, 99))

    # Larger figure, reduced spacing
    fig = plt.figure(figsize=(12, 7))
    gs = gridspec.GridSpec(2, 4, width_ratios=[1, 1, 1, 0.05], 
                           wspace=0.08, hspace=0.15,
                           left=0.02, right=0.92, top=0.92, bottom=0.05)

    # --- Top row (grayscale) ---
    # Use consistent intensity scaling (vmin=0, vmax=1) so predictions don't look faded
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(inp, cmap="gray", vmin=0, vmax=1)
    ax1.set_title("Input (Pre)", fontsize=14, fontweight='bold', pad=8)
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(pred, cmap="gray", vmin=0, vmax=1)
    ax2.set_title("Prediction", fontsize=14, fontweight='bold', pad=8)
    ax2.axis("off")

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(tgt, cmap="gray", vmin=0, vmax=1)
    ax3.set_title("Target (Post)", fontsize=14, fontweight='bold', pad=8)
    ax3.axis("off")

    # --- Bottom row (heatmaps) ---
    ax4 = fig.add_subplot(gs[1, 0])
    im = ax4.imshow(diff1, cmap="magma", vmin=0, vmax=vmax)
    ax4.set_title(f"|Pre–Post|\nMAE={mae1:.4f}", fontsize=12, pad=6)
    ax4.axis("off")

    ax5 = fig.add_subplot(gs[1, 1])
    ax5.imshow(diff2, cmap="magma", vmin=0, vmax=vmax)
    ax5.set_title(f"|Pre–Pred|\nMAE={mae2:.4f}", fontsize=12, pad=6)
    ax5.axis("off")

    ax6 = fig.add_subplot(gs[1, 2])
    ax6.imshow(diff3, cmap="magma", vmin=0, vmax=vmax)
    ax6.set_title(f"|Pred–Post|\nMAE={mae3:.4f}", fontsize=12, pad=6)
    ax6.axis("off")

    # Shared colorbar on the right
    cax = fig.add_subplot(gs[:, 3])
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label("Absolute Difference", fontsize=11)
    cbar.ax.tick_params(labelsize=10)

    plt.savefig(fname, dpi=150, bbox_inches='tight', pad_inches=0.05)
    plt.close()


# ---------------------------------------------------------
# Main script
# ---------------------------------------------------------
def main():
    idxs = find_indices(PRED_DIR)
    if not idxs:
        raise RuntimeError(f"No prediction files found in {PRED_DIR}")

    csv_rows = []
    ssims, psnrs, maes = [], [], []

    for i in idxs:
        paths = {
            "inp":  os.path.join(PRED_DIR, f"in_{i:04d}.npy"),
            "pred": os.path.join(PRED_DIR, f"out_{i:04d}.npy"),
            "tgt":  os.path.join(PRED_DIR, f"tgt_{i:04d}.npy"),
        }

        if not all(os.path.exists(p) for p in paths.values()):
            print(f"[WARN] Missing files for {i:04d}, skipping.")
            continue

        inp  = np.load(paths["inp"])
        pred = np.load(paths["pred"])
        tgt  = np.load(paths["tgt"])

        # Metrics for pred vs tgt
        ssim = ssim_np(pred, tgt)
        psnr = psnr_np(pred, tgt)
        mae  = mae_np(pred, tgt)

        ssims.append(ssim)
        psnrs.append(psnr)
        maes.append(mae)

        csv_rows.append({
            "index": i,
            "SSIM": ssim,
            "PSNR_dB": psnr,
            "MAE": mae,
        })

        # Difference images
        diff_pre_post  = np.abs(inp - tgt)
        diff_pre_pred  = np.abs(inp - pred)
        diff_pred_post = np.abs(pred - tgt)

        # MAEs used in titles
        mae1 = mae_np(inp, tgt)
        mae2 = mae_np(inp, pred)
        mae3 = mae_np(pred, tgt)

        out_path = os.path.join(OUT_DIR, f"sample_{i:04d}.png")

        panel_2x3(
            inp, pred, tgt,
            diff_pre_post, diff_pre_pred, diff_pred_post,
            mae1, mae2, mae3,
            out_path
        )

    # Save metrics CSV
    csv_path = os.path.join(OUT_DIR, "test_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "SSIM", "PSNR_dB", "MAE"])
        writer.writeheader()
        writer.writerows(csv_rows)

    # Print summary
    print("\n==== Diffusion Test Set Summary ====")
    print(f"Samples: {len(ssims)}")
    print(f"Mean SSIM : {np.mean(ssims):.4f}")
    print(f"Mean PSNR : {np.mean(psnrs):.2f} dB")
    print(f"Mean MAE  : {np.mean(maes):.4f}")
    print("\nSaved panels to:", OUT_DIR)
    print("Saved CSV:", csv_path)


if __name__ == "__main__":
    main()
