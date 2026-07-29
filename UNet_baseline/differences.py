#!/usr/bin/env python3
import os
import glob
import csv
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as sk_ssim
from skimage.metrics import peak_signal_noise_ratio as sk_psnr

# ============================================================
# Automatically detect pred_samples directory in current folder
# ============================================================
def find_pred_dir():
    here = os.getcwd()
    candidates = [
        os.path.join(here, "pred_samples"),
        os.path.join(here, "runs_axial_post_from_pre", "pred_samples"),
        os.path.join(here, "preds"),   # fallback names
        os.path.join(here, "outputs"),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    raise RuntimeError("❌ Could not find a pred_samples/ folder in this directory.")

PRED_DIR = find_pred_dir()

# Output directory (safe, isolated)
OUT_DIR = os.path.join(os.getcwd(), "differences")
os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# Metrics
# ============================================================
def mae_np(a, b):
    return float(np.mean(np.abs(a - b)))

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


# ============================================================
# Panel Plot (2×3 with colorbar)
# ============================================================
def make_panel(pre, pred, post, idx, out_path):
    import matplotlib.gridspec as gridspec

    # Rotate all images 90° counter-clockwise for radiology standard
    pre = np.rot90(pre, k=1)
    pred = np.rot90(pred, k=1)
    post = np.rot90(post, k=1)

    diff_pp   = np.abs(pre - post)
    diff_ppr  = np.abs(pre - pred)
    diff_prpo = np.abs(pred - post)

    # Shared vmax for all three heatmaps so colors are comparable.
    # Use the non-zero 99th percentile to avoid outlier-driven colormap saturation
    combined = np.concatenate([diff_pp.ravel(), diff_ppr.ravel(), diff_prpo.ravel()])
    nonzero = combined[combined > 0]
    if nonzero.size == 0:
        # Fallback to max-based scaling (and avoid zero)
        vmax = max(diff_pp.max(), diff_ppr.max(), diff_prpo.max())
        if vmax == 0:
            vmax = 1.0
    else:
        # Use a robust percentile (ignore zeros) to set the color scale
        vmax = float(np.nanpercentile(nonzero, 99))

    mae_pp   = mae_np(pre, post)
    mae_ppr  = mae_np(pre, pred)
    mae_prpo = mae_np(pred, post)

    # Larger figure, reduced spacing
    fig = plt.figure(figsize=(12, 7))
    gs = gridspec.GridSpec(2, 4, width_ratios=[1, 1, 1, 0.05], 
                           wspace=0.08, hspace=0.15,
                           left=0.02, right=0.92, top=0.92, bottom=0.05)

    # -------- TOP: Pre / Pred / Post --------
    # Use consistent intensity scaling (vmin=0, vmax=1) so predictions don't look faded
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(pre, cmap="gray", vmin=0, vmax=1)
    ax1.set_title("Input (Pre)", fontsize=14, fontweight='bold', pad=8)
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(pred, cmap="gray", vmin=0, vmax=1)
    ax2.set_title("Prediction", fontsize=14, fontweight='bold', pad=8)
    ax2.axis("off")

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(post, cmap="gray", vmin=0, vmax=1)
    ax3.set_title("Target (Post)", fontsize=14, fontweight='bold', pad=8)
    ax3.axis("off")

    # -------- SECOND ROW: Difference heatmaps --------
    ax4 = fig.add_subplot(gs[1, 0])
    im = ax4.imshow(diff_pp, cmap="magma", vmin=0, vmax=vmax)
    ax4.set_title(f"|Pre–Post|\nMAE={mae_pp:.4f}", fontsize=12, pad=6)
    ax4.axis("off")

    ax5 = fig.add_subplot(gs[1, 1])
    ax5.imshow(diff_ppr, cmap="magma", vmin=0, vmax=vmax)
    ax5.set_title(f"|Pre–Pred|\nMAE={mae_ppr:.4f}", fontsize=12, pad=6)
    ax5.axis("off")

    ax6 = fig.add_subplot(gs[1, 2])
    ax6.imshow(diff_prpo, cmap="magma", vmin=0, vmax=vmax)
    ax6.set_title(f"|Pred–Post|\nMAE={mae_prpo:.4f}", fontsize=12, pad=6)
    ax6.axis("off")

    # -------- COLORBAR (shared) --------
    cax = fig.add_subplot(gs[:, 3])
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label("Absolute Difference", fontsize=11)
    cbar.ax.tick_params(labelsize=10)

    plt.savefig(out_path, dpi=150, bbox_inches='tight', pad_inches=0.05)
    plt.close()



# ============================================================
# Main Loop
# ============================================================
def main():
    outs = sorted(glob.glob(os.path.join(PRED_DIR, "out_*.npy")))
    if not outs:
        raise RuntimeError(f"❌ No prediction files found in: {PRED_DIR}")

    csv_path = os.path.join(OUT_DIR, "difference_metrics.csv")
    csv_rows = []

    for out_file in outs:
        idx = int(os.path.splitext(os.path.basename(out_file))[0].split("_")[1])

        pre  = np.load(os.path.join(PRED_DIR, f"in_{idx:04d}.npy"))
        pred = np.load(out_file)
        post = np.load(os.path.join(PRED_DIR, f"tgt_{idx:04d}.npy"))

        out_path = os.path.join(OUT_DIR, f"panel_{idx:04d}.png")
        make_panel(pre, pred, post, idx, out_path)

        csv_rows.append({
            "index": idx,
            "MAE_pre_post": mae_np(pre, post),
            "MAE_pre_pred": mae_np(pre, pred),
            "MAE_pred_post": mae_np(pred, post),
            "SSIM_pred_post": ssim_np(pred, post),
            "PSNR_pred_post": psnr_np(pred, post),
        })

    # write CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"✅ Saved 2×3 panels → {OUT_DIR}")
    print(f"✅ Saved metrics CSV → {csv_path}")


if __name__ == "__main__":
    main()
