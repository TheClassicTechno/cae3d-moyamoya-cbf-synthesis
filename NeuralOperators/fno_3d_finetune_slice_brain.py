#!/usr/bin/env python3
"""
FNO 3D fine-tune with slice-weighted + brain-mask loss.
Loads fno_3d_cvr_optimized.pt (when width=64), trains with optional edge slice weighting
and brain-only loss. Saves fno_3d_finetuned_slice_brain_<suffix>.pt and results JSON.
"""
import os
import sys
import glob
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_REPO_ROOT, "pyproject.toml")):
    _parent = os.path.dirname(_REPO_ROOT)
    if _parent == _REPO_ROOT:
        raise RuntimeError("Could not locate repository root (pyproject.toml not found)")
    _REPO_ROOT = _parent

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fno_3d_cvr import (
    load_volume, pre_to_post, VolumePairs, VolumePairsFromPairs, SimpleFNO3D,
    TARGET_SIZE, DATA_DIR, add_position_channels,
    Week7VolumePairsFNO, WEEK7_PAD_SHAPE, WEEK7_ORIGINAL_SHAPE,
)

SEED = int(os.environ.get("SEED", 1337))


def slice_weights(edge_weight=1.5, num_slices=64):
    """Weight per slice index: higher at edges."""
    w = np.ones(num_slices, dtype=np.float32)
    q = num_slices // 4
    w[:q] = edge_weight
    w[-q:] = edge_weight
    return w


def composite_loss_slice_brain(pred, target, pre_vol, slice_weight_arr, use_brain_mask, loss_type="l1mse", region_weight_vol=None):
    """pred/target (B,1,H,W,D), pre_vol (B,1,H,W,D), slice_weight_arr (D,). Optional region_weight_vol (1,1,H,W,D) for Phase 2."""
    B, _, H, W, D = pred.shape
    device = pred.device
    sw = torch.from_numpy(slice_weight_arr).to(device).view(1, 1, 1, 1, D)
    if use_brain_mask:
        mask = (pre_vol > 0.05).float()
    else:
        mask = torch.ones_like(pred)
    if region_weight_vol is not None:
        mask = mask * region_weight_vol.to(device)
    err = pred - target
    if loss_type == "l1mse":
        l1 = torch.abs(err) * mask * sw
        mse = (err ** 2) * mask * sw
        return l1.mean() + mse.mean()
    else:
        l1 = torch.abs(err) * mask * sw
        mse = (err ** 2) * mask * sw
        return 0.2 * l1.mean() + 0.8 * mse.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice-weight", action="store_true", help="Use slice weighting (edge slices higher)")
    ap.add_argument("--brain-mask", action="store_true", help="Restrict loss to brain voxels (pre > 0.05)")
    ap.add_argument("--edge-weight", type=float, default=1.5)
    ap.add_argument("--epochs", type=int, default=int(os.environ.get("WEEK9_QUICK_EPOCHS", 40)))
    ap.add_argument("--out-suffix", type=str, default="")
    ap.add_argument("--loss", type=str, default="l1mse", choices=["l1mse", "psnr"])
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--ckpt", type=str, default="")
    ap.add_argument("--use-2020", action="store_true", help="Use 2020 Single-delay split JSON")
    ap.add_argument("--split-2020-json", type=str, default=os.path.join(_REPO_ROOT, "2020_single_delay_split.json"))
    ap.add_argument("--combined-split-json", type=str, default="", help="Use combined (existing+2020) subject-level split; overrides --use-2020")
    ap.add_argument("--week7", action="store_true", help="Week7: 91x109x91 + brain mask, pad 96x112x96, same split as other Week7 models")
    args = ap.parse_args()

    use_week7 = args.week7 or os.environ.get("WEEK7", "").lower() in ("1", "true", "yes")
    if use_week7:
        H, W, D = WEEK7_PAD_SHAPE
        MID_Z = D // 2
    else:
        H, W, D = TARGET_SIZE
        MID_Z = D // 2

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = os.path.dirname(os.path.abspath(__file__))
    default_ckpt = os.path.join(base, "fno_3d_cvr_optimized.pt")
    ckpt_path = args.ckpt or default_ckpt
    load_ckpt = os.path.exists(ckpt_path) and args.width == 64

    in_ch = 4
    modes = 12 if args.width == 64 else 8
    model = SimpleFNO3D(in_ch=in_ch, out_ch=1, modes=modes, width=args.width)
    if load_ckpt:
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state)
        print("  Loaded: %s  (modes=%s, width=%s)" % (ckpt_path, modes, args.width))
    else:
        print("  Training from scratch (width=%s)" % args.width)

    model = model.to(device)
    slice_weight_arr = slice_weights(edge_weight=args.edge_weight, num_slices=D) if args.slice_weight else np.ones(D, dtype=np.float32)

    region_weight_t = None
    if use_week7:
        sys.path.insert(0, os.path.join(_REPO_ROOT, "scripts"))
        from week7_data import get_week7_splits
        from week7_preprocess import get_region_weight_mask_for_shape
        train_pairs, val_pairs, test_pairs = get_week7_splits()
        print("  Week7: 91x109x91 + brain mask, pad 96x112x96: %d train / %d val / %d test" % (len(train_pairs), len(val_pairs), len(test_pairs)))
        if os.environ.get("WEEK7_REGION_WEIGHT", "").lower() in ("1", "true", "yes"):
            mask_np = get_region_weight_mask_for_shape(WEEK7_PAD_SHAPE, vascular_weight=1.5)
            region_weight_t = torch.from_numpy(mask_np).float().unsqueeze(0).unsqueeze(0)  # (1,1,H,W,D)
        train_ds = Week7VolumePairsFNO(train_pairs)
        val_ds = Week7VolumePairsFNO(val_pairs)
        test_ds = Week7VolumePairsFNO(test_pairs)
    else:
        split_json = None
        if args.combined_split_json and os.path.isfile(args.combined_split_json):
            split_json = args.combined_split_json
        elif args.use_2020 and os.path.isfile(args.split_2020_json):
            split_json = args.split_2020_json
        if split_json:
            with open(split_json) as f:
                data = json.load(f)
            train_pairs = [(x["pre_path"], x["post_path"]) for x in data["train"]]
            val_pairs = [(x["pre_path"], x["post_path"]) for x in data["val"]]
            test_pairs = [(x["pre_path"], x["post_path"]) for x in data["test"]]
            print("  From %s: %d train / %d val / %d test" % (split_json, len(train_pairs), len(val_pairs), len(test_pairs)))
            train_ds = VolumePairsFromPairs(train_pairs)
            val_ds = VolumePairsFromPairs(val_pairs)
            test_ds = VolumePairsFromPairs(test_pairs)
        else:
            all_pre = sorted(glob.glob(os.path.join(DATA_DIR, "pre", "pre_*.nii.gz")))
            trainval, test_paths = train_test_split(all_pre, test_size=0.25, random_state=SEED, shuffle=True)
            train_paths, val_paths = train_test_split(trainval, test_size=0.15, random_state=SEED, shuffle=True)
            train_ds = VolumePairs(train_paths)
            val_ds = VolumePairs(val_paths)
            test_ds = VolumePairs(test_paths)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=2e-6)

    print("============================================================")
    print("FNO 3D Fine-tune (Slice-Weighted + Brain-Mask)")
    print("============================================================")
    print("  Slice-weighted loss: %s  (edge slices %.1fx)" % (args.slice_weight, args.edge_weight))
    print("  Brain-mask loss:    %s" % args.brain_mask)
    print("  Loss: %s  Epochs: %s, LR: 0.0002 -> 2e-06" % (args.loss, args.epochs))
    print()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for pre, post in train_loader:
            pre, post = pre.to(device), post.to(device)
            pred = model(pre)
            loss = composite_loss_slice_brain(
                pred, post, pre[:, :1], slice_weight_arr, args.brain_mask, args.loss,
                region_weight_vol=region_weight_t,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        scheduler.step()

        model.eval()
        val_mae, val_ssim, val_psnr = [], [], []
        with torch.no_grad():
            for pre, post in val_loader:
                pre, post = pre.to(device), post.to(device)
                pred = model(pre)
                pred = torch.clamp(pred, 0, 1)
                p = pred[0, 0].cpu().numpy()
                t = post[0, 0].cpu().numpy()
                val_mae.append(np.abs(p - t).mean())
                val_ssim.append(ssim(t, p, data_range=1.0))
                val_psnr.append(psnr(t, p, data_range=1.0))
        print("Epoch %2d/%d Train: %.4f | Val MAE: %.4f SSIM: %.4f PSNR: %.2f dB" % (
            epoch, args.epochs, train_loss, np.mean(val_mae), np.mean(val_ssim), np.mean(val_psnr)))

    if use_week7:
        out_name = "fno_3d_week7_best.pt"
        out_path = os.path.join(base, out_name)
    else:
        suffix = args.out_suffix or ("combined" if (args.combined_split_json and os.path.isfile(args.combined_split_json)) else ("2020" if args.use_2020 else ""))
        out_name = "fno_3d_finetuned_slice_brain" + ("_" + suffix if suffix else "") + ".pt"
        out_path = os.path.join(base, out_name)
    torch.save(model.state_dict(), out_path)
    print("\nSaved to %s" % out_path)

    # Test with simple ensemble (identity + 3 flips, average)
    model.eval()
    mae_list, ssim_list, psnr_list = [], [], []
    mae_slice, ssim_slice, psnr_slice = [], [], []
    transforms = [
        (lambda x: x, lambda x: x),
        (lambda x: torch.flip(x, (-1,)), lambda x: torch.flip(x, (-1,))),
        (lambda x: torch.flip(x, (-2,)), lambda x: torch.flip(x, (-2,))),
        (lambda x: torch.flip(torch.flip(x, (-1,)), (-2,)), lambda x: torch.flip(torch.flip(x, (-1,)), (-2,))),
    ]
    with torch.no_grad():
        for pre, post in test_loader:
            pre, post = pre.to(device), post.to(device)
            preds = []
            for t_in, t_out in transforms:
                preds.append(t_out(model(t_in(pre))))
            pred = torch.stack(preds).mean(0)
            pred = torch.clamp(pred, 0, 1)
            pred_np = pred[0, 0].cpu().numpy()
            post_np = post[0, 0].cpu().numpy()
            if use_week7:
                oh, ow, od = WEEK7_ORIGINAL_SHAPE
                pred_np = pred_np[:oh, :ow, :od]
                post_np = post_np[:oh, :ow, :od]
                from week7_preprocess import metrics_in_brain
                m = metrics_in_brain(pred_np, post_np, data_range=1.0)
                mae_list.append(m["mae_mean"])
                ssim_list.append(m["ssim_mean"])
                psnr_list.append(m["psnr_mean"])
            else:
                mae_list.append(np.abs(pred_np - post_np).mean())
                ssim_list.append(ssim(post_np, pred_np, data_range=1.0))
                psnr_list.append(psnr(post_np, pred_np, data_range=1.0))
            mid_z = (WEEK7_ORIGINAL_SHAPE[2] // 2) if use_week7 else MID_Z
            mid_p = pred_np[:, :, mid_z]
            mid_t = post_np[:, :, mid_z]
            mae_slice.append(np.abs(mid_p - mid_t).mean())
            ssim_slice.append(ssim(mid_t, mid_p, data_range=1.0))
            psnr_slice.append(psnr(mid_t, mid_p, data_range=1.0))

    results = {
        "mae_mean": float(np.mean(mae_list)),
        "mae_std": float(np.std(mae_list)),
        "ssim_mean": float(np.mean(ssim_list)),
        "ssim_std": float(np.std(ssim_list)),
        "psnr_mean": float(np.mean(psnr_list)),
        "psnr_std": float(np.std(psnr_list)),
        "mae_middle_slice": float(np.mean(mae_slice)),
        "ssim_middle_slice": float(np.mean(ssim_slice)),
        "psnr_middle_slice": float(np.mean(psnr_slice)),
        "slice_weighted": args.slice_weight,
        "brain_mask": args.brain_mask,
        "edge_weight": args.edge_weight,
        "loss_type": args.loss,
        "width": args.width,
        "epochs": args.epochs,
        "test_time_ensemble": True,
        "n_test": len(mae_list),
    }
    phase2 = use_week7 and os.environ.get("WEEK7_REGION_WEIGHT", "").lower() in ("1", "true", "yes")
    if phase2:
        results_path = os.path.join(base, "fno_3d_week7_phase2_results.json")
    else:
        results_path = out_path.replace(".pt", "_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f)

    print("============================================================")
    print("FNO 3D Fine-tuned (Slice+Brain) Test Results")
    print("============================================================")
    print("  Full volume: MAE %.4f SSIM %.4f PSNR %.2f dB" % (results["mae_mean"], results["ssim_mean"], results["psnr_mean"]))
    print("  Middle slice: MAE %.4f SSIM %.4f PSNR %.2f dB" % (results["mae_middle_slice"], results["ssim_middle_slice"], results["psnr_middle_slice"]))
    pct = (results["psnr_mean"] / 21.49 - 1) * 100
    print("  vs 2D paper (PSNR 21.49 dB): full-volume PSNR %+.1f%%" % pct)
    print("============================================================")
    print("Saved to %s" % results_path)
    print("\nNext: run brain-mask eval and extended eval on this checkpoint:")
    print("  python evaluate_fno_3d_brainmask.py --ckpt %s --ensemble" % out_name)
    print("  python evaluate_fno_3d_extended.py --ckpt %s --ensemble" % out_name)


if __name__ == "__main__":
    main()
