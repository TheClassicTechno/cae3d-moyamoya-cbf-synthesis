#!/usr/bin/env python3
"""
Week 9: Tied-Augment for ResNet 3D CVR (Phase 1).
Implements L = L_recon + tw * MSE(f1, f2) with two augmented views per sample.
See TIED_AUGMENT_ANALYSIS.txt and week9/NOTES.txt.
"""
import os
import sys
import json
import random

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_REPO_ROOT, "pyproject.toml")):
    _parent = os.path.dirname(_REPO_ROOT)
    if _parent == _REPO_ROOT:
        raise RuntimeError("Could not locate repository root (pyproject.toml not found)")
    _REPO_ROOT = _parent

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from week7_data import get_week7_splits, Week7VolumePairs3DTiedAugment, Week7VolumePairs3D
from week7_preprocess import TARGET_SHAPE, metrics_in_brain

from monai.networks.nets import ResNetFeatures

DATA_DIR = _REPO_ROOT
OUT_DIR = os.path.join(DATA_DIR, "scripts", "week9_results")
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = int(os.environ.get("SEED", 42))
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

TARGET_3D_PAD = (96, 112, 96)
BATCH_SIZE = int(os.environ.get("WEEK9_BATCH_SIZE", 2))
EPOCHS = int(os.environ.get("WEEK9_EPOCHS", "50"))
LR = 1e-3
TIED_WEIGHT = float(os.environ.get("TIED_WEIGHT", "10"))
# Week89: scheduled tw (use early, decay to 0). If set > 0, tw(epoch) = TIED_WEIGHT_MAX * max(0, 1 - epoch/TIED_DECAY_BY_EPOCH)
TIED_WEIGHT_MAX = float(os.environ.get("TIED_WEIGHT_MAX", os.environ.get("TIED_WEIGHT", "10")))
TIED_DECAY_BY_EPOCH = int(os.environ.get("TIED_DECAY_BY_EPOCH", "0"))  # 0 = constant tw
USE_POOLED_TIE = os.environ.get("USE_POOLED_TIE", "0").lower() in ("1", "true", "yes")
# Phase 2 (WEEK89_IMPLEMENTATION_PLAN): second view geometric-only (flips, no intensity scaling)
GEOMETRIC_ONLY_SECOND_VIEW = os.environ.get("WEEK9_GEOMETRIC_ONLY_SECOND_VIEW", "0").lower() in ("1", "true", "yes")
ENC_CH = 512


def tied_weight_schedule(epoch, tw_max, decay_by_epoch):
    """tw = tw_max * max(0, 1 - epoch / decay_by_epoch). At epoch 0: tw_max; at epoch >= decay_by_epoch: 0."""
    if decay_by_epoch <= 0:
        return tw_max
    return tw_max * max(0.0, 1.0 - epoch / decay_by_epoch)


def _pad_3d(pre_t, post_t, target_shape):
    _, _, h, w, d = pre_t.shape
    th, tw, td = target_shape
    if h < th or w < tw or d < td:
        pd = (0, max(0, td - d), 0, max(0, tw - w), 0, max(0, th - h))
        pre_t = F.pad(pre_t, pd, mode="constant", value=0)
        post_t = F.pad(post_t, pd, mode="constant", value=0)
    return pre_t[:, :, :th, :tw, :td], post_t[:, :, :th, :tw, :td]


def _collate_tied_augment(batch):
    """Collate ((pre1,post1),(pre2,post2)) per sample into 4 tensors."""
    pre1_list, post1_list, pre2_list, post2_list = [], [], [], []
    for (p1, t1), (p2, t2) in batch:
        pre1_list.append(p1)
        post1_list.append(t1)
        pre2_list.append(p2)
        post2_list.append(t2)
    return (
        torch.stack(pre1_list),
        torch.stack(post1_list),
        torch.stack(pre2_list),
        torch.stack(post2_list),
    )


class ResNet3DCVRTied(nn.Module):
    """ResNet18 3D encoder + decoder. Supports return_features for Tied-Augment."""

    def __init__(self, pretrained=False):
        super().__init__()
        self.encoder = ResNetFeatures(
            model_name="resnet18",
            pretrained=pretrained,
            spatial_dims=3,
            in_channels=1,
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(ENC_CH, 256, 4, stride=2, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(32, 1, 4, stride=2, padding=1),
        )

    def forward(self, x, return_features=False):
        feats = self.encoder(x)
        bottleneck = feats[-1]
        x = self.decoder(bottleneck)
        _, _, h, w, d = x.shape
        if h != TARGET_3D_PAD[0] or w != TARGET_3D_PAD[1] or d != TARGET_3D_PAD[2]:
            x = F.interpolate(x, size=TARGET_3D_PAD, mode="trilinear", align_corners=False)
        if return_features:
            # Pooled = global summary (B, C) for optional tie on global rep only (week89)
            pooled = F.adaptive_avg_pool3d(bottleneck, 1).flatten(1)
            return x, bottleneck, pooled
        return x


def train_epoch(model, loader, criterion_l1, criterion_ssim, optimizer, tied_weight, use_pooled_tie=False):
    model.train()
    total, n = 0.0, 0
    for pre1, post1, pre2, post2 in loader:
        pre1, post1 = _pad_3d(pre1, post1, TARGET_3D_PAD)
        pre2, post2 = _pad_3d(pre2, post2, TARGET_3D_PAD)
        pre1 = pre1.to(DEVICE)
        post1 = post1.to(DEVICE)
        pre2 = pre2.to(DEVICE)
        post2 = post2.to(DEVICE)

        optimizer.zero_grad()

        # Separate forward passes (paper: avoid concat before BN)
        out1, bot1, pool1 = model(pre1, return_features=True)
        out2, bot2, pool2 = model(pre2, return_features=True)

        L_recon = criterion_l1(out1, post1) + criterion_ssim(out1, post1)
        L_recon += criterion_l1(out2, post2) + criterion_ssim(out2, post2)
        L_recon = L_recon / 2  # average over two views

        # Week89: tie pooled (global) features to preserve spatial detail in bottleneck
        f1, f2 = (pool1, pool2) if use_pooled_tie else (bot1, bot2)
        L_feat = F.mse_loss(f1, f2)

        loss = L_recon + tied_weight * L_feat
        loss.backward()
        optimizer.step()
        total += loss.item()
        n += 1
    return total / max(n, 1)


def evaluate(model, loader):
    model.eval()
    mae_list, ssim_list, psnr_list = [], [], []
    with torch.no_grad():
        for batch in loader:
            pre, post = batch[0], batch[1]
            pre, post = _pad_3d(pre, post, TARGET_3D_PAD)
            pre, post = pre.to(DEVICE), post.to(DEVICE)
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


def main():
    from monai.losses import SSIMLoss

    use_schedule = TIED_DECAY_BY_EPOCH > 0
    tw_max = TIED_WEIGHT_MAX
    use_pooled = USE_POOLED_TIE
    print(
        f"Week9 Tied-Augment ResNet 3D: epochs={EPOCHS}, batch_size={BATCH_SIZE}"
        f"{', tw_schedule: max=%g decay_by_ep=%d' % (tw_max, TIED_DECAY_BY_EPOCH) if use_schedule else ', tw=%g (constant)' % TIED_WEIGHT}"
        f", use_pooled_tie={use_pooled}, geometric_only_second_view={GEOMETRIC_ONLY_SECOND_VIEW}"
    )

    train_pairs, val_pairs, test_pairs = get_week7_splits()

    train_ds = Week7VolumePairs3DTiedAugment(
        train_pairs, augment=True, geometric_only_second_view=GEOMETRIC_ONLY_SECOND_VIEW
    )
    val_ds = Week7VolumePairs3D(val_pairs, augment=False)
    test_ds = Week7VolumePairs3D(test_pairs, augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=_collate_tied_augment,
    )
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, num_workers=0)

    model = ResNet3DCVRTied(pretrained=False).to(DEVICE)
    criterion_l1 = nn.L1Loss()
    criterion_ssim = SSIMLoss(spatial_dims=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_psnr = -1.0
    for ep in range(EPOCHS):
        tw = tied_weight_schedule(ep, tw_max, TIED_DECAY_BY_EPOCH) if use_schedule else TIED_WEIGHT
        loss = train_epoch(
            model, train_loader, criterion_l1, criterion_ssim, optimizer,
            tied_weight=tw, use_pooled_tie=use_pooled,
        )
        metrics = evaluate(model, val_loader)
        if metrics["psnr_mean"] > best_val_psnr:
            best_val_psnr = metrics["psnr_mean"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": ep,
                    "tied_weight": tw,
                    "tied_weight_max": tw_max,
                    "tied_decay_by_epoch": TIED_DECAY_BY_EPOCH,
                    "use_pooled_tie": use_pooled,
                },
                os.path.join(OUT_DIR, "week9_resnet3d_tied_best.pt"),
            )
        if (ep + 1) % 10 == 0:
            print(
                f"Epoch {ep+1} tw={tw:.2f} loss={loss:.4f} val MAE={metrics['mae_mean']:.4f} "
                f"SSIM={metrics['ssim_mean']:.4f} PSNR={metrics['psnr_mean']:.2f}"
            )

    ckpt = torch.load(os.path.join(OUT_DIR, "week9_resnet3d_tied_best.pt"), map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    test_metrics = evaluate(model, test_loader)
    print("Test:", test_metrics)

    suffix = "sched" if use_schedule else f"tw{int(TIED_WEIGHT)}"
    if use_schedule and TIED_DECAY_BY_EPOCH != 10:
        suffix += "_decay%d" % TIED_DECAY_BY_EPOCH
    if use_schedule and int(tw_max) != 10:
        suffix += "_tw%d" % int(tw_max)
    if use_pooled:
        suffix += "_pooled"
    if GEOMETRIC_ONLY_SECOND_VIEW:
        suffix += "_geo2"
    out_json = os.path.join(OUT_DIR, f"week9_resnet3d_tied_{suffix}_results.json")
    with open(out_json, "w") as f:
        json.dump(
            {
                "tied_weight_max": tw_max,
                "tied_decay_by_epoch": TIED_DECAY_BY_EPOCH,
                "use_pooled_tie": use_pooled,
                "geometric_only_second_view": GEOMETRIC_ONLY_SECOND_VIEW,
                "epochs": EPOCHS,
                "seed": SEED,
                **test_metrics,
            },
            f,
            indent=2,
        )
    print("Saved", out_json)


if __name__ == "__main__":
    main()
