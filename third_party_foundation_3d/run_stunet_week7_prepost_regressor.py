#!/usr/bin/env python3
"""
Week7 pre→post **voxel regression** with the **STU-Net architecture** (nnUNet 1.7 STUNet base).

This is the fair comparator to `run_sam_med3d_week7_cbf_regressor.py` / `run_med3dvlm_week7_cvr.py`:
same `get_week7_splits()`, masked L1, `metrics_in_brain` on test.

**Not** `run_stunet_week7_test_set.py` (segmentation vs brain mask, no post target).

Architecture: `STUNet` (1 input, 1 output channel) with sigmoid; volumes resized to 128³ (divisible
by STU-Net pooling), prediction resized back to Week7 `TARGET_SHAPE`.

**Pretrained weights (optional):** place a Task101 **base_ep4k** `.model` checkpoint (or any compatible
STU-Net `state_dict`) and set:

  export STUNET_PRETRAINED_WEIGHTS=/path/to/base_ep4k.model

Only non-`seg_outputs` layers are loaded (105-class heads are skipped). If unset or missing,
training starts from scratch (still the same STU-Net architecture).

Run from repo root:

  PYTHONNOUSERSITE=1 PYTHONPATH=/data1/julih/scripts \\
    python third_party_foundation_3d/run_stunet_week7_prepost_regressor.py --seeds 42,123,456

See `FOUNDATION_PRE_TO_POST_FINETUNING.md`.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = "/data1/julih"
sys.path.insert(0, os.path.join(ROOT, "scripts"))
NNUNET_ROOT = os.path.join(ROOT, "third_party_foundation_3d", "STU-Net-unimedical", "nnUNet-1.7.1")
sys.path.insert(0, NNUNET_ROOT)

from week7_data import get_week7_splits, Week7VolumePairs3D
from week7_preprocess import TARGET_SHAPE, get_brain_mask_for_shape, get_region_weight_mask_for_shape, metrics_in_brain
from nnunet.network_architecture.STUNet import STUNet

OUT_DIR = os.path.join(ROOT, "third_party_foundation_3d", "stunet_week7_prepost_regressor")
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_SHAPE_3D = TARGET_SHAPE
BATCH_SIZE = int(os.environ.get("STUNET_BATCH_SIZE", "1"))
EPOCHS = int(os.environ.get("STUNET_EPOCHS", "30"))
LR = float(os.environ.get("STUNET_LR", "1e-4"))
INTERNAL = (128, 128, 128)
POOL_OP_KERNEL_SIZES = [[2, 2, 2]] * 5
CONV_KERNEL_SIZES = [[3, 3, 3]] * 6


def load_stunet_pretrained_skip_seg(net: nn.Module, fname: str) -> bool:
    """Load encoder/decoder blocks; skip seg head (shape mismatch for 1-class vs 105-class)."""
    if not os.path.isfile(fname):
        return False
    saved = torch.load(fname, map_location="cpu")
    if isinstance(saved, dict):
        if "state_dict" in saved:
            pre = saved["state_dict"]
        elif "network_weights" in saved:
            pre = saved["network_weights"]
        else:
            pre = saved
    else:
        return False
    msd = net.state_dict()
    loaded = {}
    for k, v in pre.items():
        kk = k[7:] if k.startswith("module.") else k
        if "seg_outputs" in kk:
            continue
        if kk in msd and msd[kk].shape == v.shape:
            loaded[kk] = v
    if not loaded:
        print("STU-Net: no compatible keys from", fname)
        return False
    msd.update(loaded)
    net.load_state_dict(msd)
    print("Loaded STU-Net pretrained tensors (%d non-seg keys) from %s" % (len(loaded), fname))
    return True


class STUNetPrePostRegressor(nn.Module):
    """STU-Net with single-channel sigmoid output for [0,1] CBF-style volumes."""

    def __init__(self, pretrained_path: str | None):
        super().__init__()
        self.net = STUNet(
            1,
            1,
            depth=[1, 1, 1, 1, 1, 1],
            dims=[32, 64, 128, 256, 512, 512],
            pool_op_kernel_sizes=POOL_OP_KERNEL_SIZES,
            conv_kernel_sizes=CONV_KERNEL_SIZES,
        )
        self.net.do_ds = False
        self.net.inference_apply_nonlin = lambda x: x
        if pretrained_path:
            load_stunet_pretrained_skip_seg(self.net, pretrained_path)

    def forward(self, pre: torch.Tensor) -> torch.Tensor:
        """pre (B,1,D,H,W) in [0,1] -> pred (B,1,D,H,W) in [0,1] on TARGET_SHAPE."""
        x = F.interpolate(pre, size=INTERNAL, mode="trilinear", align_corners=False)
        out = self.net(x)
        if isinstance(out, (list, tuple)):
            out = out[-1]
        out = torch.sigmoid(out)
        return F.interpolate(out, size=TARGET_SHAPE_3D, mode="trilinear", align_corners=False)


def train_epoch(model: STUNetPrePostRegressor, loader, optimizer, mask_t):
    model.train()
    total, n = 0.0, 0
    for pre, post in loader:
        pre = pre.to(DEVICE)
        post = post.to(DEVICE)
        optimizer.zero_grad()
        pred = model(pre)
        l1 = (torch.abs(pred - post) * mask_t).sum() / (mask_t.sum() + 1e-8)
        l1.backward()
        optimizer.step()
        total += l1.item()
        n += 1
    return total / max(n, 1)


def evaluate(model: STUNetPrePostRegressor, loader):
    model.eval()
    mae_list, ssim_list, psnr_list = [], [], []
    with torch.no_grad():
        for pre, post in loader:
            pre = pre.to(DEVICE)
            post = post.to(DEVICE)
            pred = model(pre)
            for i in range(pred.shape[0]):
                p = pred[i, 0].cpu().numpy()
                t = post[i, 0].cpu().numpy()
                m = metrics_in_brain(p, t, data_range=1.0)
                mae_list.append(m["mae_mean"])
                ssim_list.append(m["ssim_mean"])
                psnr_list.append(m["psnr_mean"])
    return {
        "mae_mean": float(np.mean(mae_list)),
        "ssim_mean": float(np.mean(ssim_list)),
        "psnr_mean": float(np.mean(psnr_list)),
    }


def train_eval_one_seed(seed: int, pretrained_path: str | None, tag: str) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print("STU-Net pre→post regressor, seed=%s, tag=%s" % (seed, tag))
    model = STUNetPrePostRegressor(pretrained_path).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    train_pairs, val_pairs, test_pairs = get_week7_splits()
    train_ds = Week7VolumePairs3D(train_pairs, augment=True)
    val_ds = Week7VolumePairs3D(val_pairs, augment=False)
    test_ds = Week7VolumePairs3D(test_pairs, augment=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    use_region_weight = os.environ.get("WEEK7_REGION_WEIGHT", "").lower() in ("1", "true", "yes")
    mask_np = get_region_weight_mask_for_shape(TARGET_SHAPE_3D, vascular_weight=1.5) if use_region_weight else get_brain_mask_for_shape(TARGET_SHAPE_3D)
    mask_t = torch.from_numpy(mask_np).float().to(DEVICE).unsqueeze(0).unsqueeze(0)

    ckpt_path = os.path.join(OUT_DIR, "stunet_week7_prepost_best_seed%d_%s.pt" % (seed, tag))
    best_val_psnr = -1.0
    for ep in range(EPOCHS):
        loss = train_epoch(model, train_loader, optimizer, mask_t)
        metrics = evaluate(model, val_loader)
        if metrics["psnr_mean"] > best_val_psnr:
            best_val_psnr = metrics["psnr_mean"]
            torch.save({"model": model.state_dict(), "epoch": ep, "seed": seed, "tag": tag}, ckpt_path)
        if (ep + 1) % 5 == 0:
            print("Epoch %d loss=%.4f val MAE=%.4f SSIM=%.4f PSNR=%.2f" % (ep + 1, loss, metrics["mae_mean"], metrics["ssim_mean"], metrics["psnr_mean"]))

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    test_metrics = evaluate(model, test_loader)
    test_metrics["seed"] = seed
    test_metrics["task"] = "pre_to_post_stunet_arch_in_brain"
    test_metrics["pretrained_loaded"] = bool(pretrained_path and os.path.isfile(pretrained_path))
    test_metrics["tag"] = tag
    print("Test (seed %d):" % seed, test_metrics)
    return test_metrics


def default_pretrained_path() -> str | None:
    p = os.environ.get("STUNET_PRETRAINED_WEIGHTS", "").strip()
    if p:
        return p
    cand = os.path.join(
        ROOT,
        "third_party_foundation_3d",
        "stunet_results",
        "nnUNet",
        "3d_fullres",
        "Task101_TotalSegmentator",
        "STUNetTrainer_base__nnUNetPlansv2.1",
        "fold_0",
        "base_ep4k.model",
    )
    return cand if os.path.isfile(cand) else None


def main():
    ap = argparse.ArgumentParser(description="STU-Net architecture pre→post for Week7.")
    ap.add_argument("--seeds", type=str, default="42", help="Comma-separated seeds, e.g. 42,123,456")
    args = ap.parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    pretrained_path = default_pretrained_path()
    tag = "from_pretrained" if pretrained_path and os.path.isfile(pretrained_path) else "from_scratch"

    all_metrics = []
    for sd in seeds:
        all_metrics.append(train_eval_one_seed(sd, pretrained_path, tag))
        if len(seeds) > 1:
            pj = os.path.join(OUT_DIR, "stunet_week7_prepost_results_%s_seed%d.json" % (tag, sd))
            with open(pj, "w") as f:
                json.dump(all_metrics[-1], f, indent=2)
            print("Saved", pj)

    stem = "stunet_week7_prepost_results_%s" % tag
    if len(seeds) == 1:
        out_json = os.path.join(OUT_DIR, stem + ".json")
        with open(out_json, "w") as f:
            json.dump(all_metrics[0], f, indent=2)
        print("Saved", out_json)
    else:
        agg = {
            "seeds": seeds,
            "n_seeds": len(seeds),
            "tag": tag,
            "pretrained_path": pretrained_path,
            "lr": LR,
            "mae_mean": float(np.mean([m["mae_mean"] for m in all_metrics])),
            "mae_std": float(np.std([m["mae_mean"] for m in all_metrics])),
            "ssim_mean": float(np.mean([m["ssim_mean"] for m in all_metrics])),
            "ssim_std": float(np.std([m["ssim_mean"] for m in all_metrics])),
            "psnr_mean": float(np.mean([m["psnr_mean"] for m in all_metrics])),
            "psnr_std": float(np.std([m["psnr_mean"] for m in all_metrics])),
            "per_seed": all_metrics,
        }
        multi = os.path.join(OUT_DIR, stem + "_multiseed.json")
        with open(multi, "w") as f:
            json.dump(agg, f, indent=2)
        print("Saved aggregate", multi)


if __name__ == "__main__":
    main()
