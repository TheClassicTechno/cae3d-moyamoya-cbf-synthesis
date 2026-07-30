#!/usr/bin/env python3
"""
Fair baseline: finetune (decoder) on OUR pre→post task — not run_sam_med3d_week7_test_set.py.
See third_party_foundation_3d/FOUNDATION_PRE_TO_POST_FINETUNING.md.

Train a CBF-matched regressor: frozen SAM-Med3D image encoder + trainable 3D decoder
pre -> post on Week7 combined split. Same data path, in-brain metrics, and masked L1
objective style as run_med3dvlm_week7_cvr.py.

Requires: medim, SAM-Med3D under third_party_foundation_3d/SAM-Med3D, checkpoint
sam_med3d_turbo.pth (see run_sam_med3d_week7_test_set.py).

Run from repo root:
  PYTHONPATH=/data1/julih/scripts:/data1/julih/third_party_foundation_3d/SAM-Med3D \\
    python3 third_party_foundation_3d/run_sam_med3d_week7_cbf_regressor.py --seeds 42,123,456

Stronger adaptation (train SAM image encoder + decoder), same env pattern as Med3DVLM:
  SAM_MED3D_FREEZE_ENCODER=0 python3 .../run_sam_med3d_week7_cbf_regressor.py --seeds 42,123,456
Checkpoints/JSONs use enc_frozen vs enc_train suffixes.
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
REPO_SCRIPTS = os.path.join(ROOT, "scripts")
SAM_REPO = os.path.join(ROOT, "third_party_foundation_3d", "SAM-Med3D")
for p in (REPO_SCRIPTS, SAM_REPO):
    if p not in sys.path:
        sys.path.insert(0, p)

from week7_data import get_week7_splits, Week7VolumePairs3D
from week7_preprocess import TARGET_SHAPE, get_brain_mask_for_shape, get_region_weight_mask_for_shape, metrics_in_brain

OUT_DIR = os.path.join(ROOT, "third_party_foundation_3d", "sam_med3d_week7_cbf")
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_SHAPE_3D = TARGET_SHAPE
BATCH_SIZE = 2
EPOCHS = 30
LR = float(os.environ.get("SAM_MED3D_LR", "1e-4"))
UP_SPATIAL = 128


def sam_med3d_freeze_encoder_from_env() -> bool:
    """Default True (decoder-only). Set SAM_MED3D_FREEZE_ENCODER=0|false|unfreeze to train encoder + decoder."""
    v = os.environ.get("SAM_MED3D_FREEZE_ENCODER", "1").strip().lower()
    return v not in ("0", "false", "no", "unfreeze")


def load_sam_med3d():
    import medim

    ckpt_path = "https://huggingface.co/blueyo0/SAM-Med3D/blob/main/sam_med3d_turbo.pth"
    for loc in [
        os.path.join(ROOT, "third_party_foundation_3d", "SAM-Med3D", "ckpt", "sam_med3d_turbo.pth"),
        os.path.join(ROOT, "third_party_foundation_3d", "ckpt", "sam_med3d_turbo.pth"),
    ]:
        if os.path.isfile(loc):
            ckpt_path = loc
            break
    model = medim.create_model("SAM-Med3D", pretrained=True, checkpoint_path=ckpt_path)
    return model


def _sam_norm_1ch(sam: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Normalize single-channel input; avoid (B,1,...) - (1,3,...) -> 3-channel broadcast."""
    pm = sam.pixel_mean.to(x.device).flatten()[0].view(1, 1, 1, 1, 1)
    ps = sam.pixel_std.to(x.device).flatten()[0].view(1, 1, 1, 1, 1)
    return (x - pm) / (ps + 1e-8)


@torch.no_grad()
def probe_encoder_out(sam: nn.Module, device: str) -> tuple[int, int, int, int]:
    isz = int(sam.image_encoder.img_size)
    dummy = torch.zeros(1, 1, isz, isz, isz, device=device)
    x = _sam_norm_1ch(sam, dummy * 255.0)
    emb = sam.image_encoder(x)
    if emb.dim() != 5:
        raise RuntimeError("Unexpected embedding shape: %s" % (emb.shape,))
    _b, c, d, h, w = emb.shape
    if d != h or h != w:
        raise RuntimeError("Non-cubic SAM embedding grid not handled: %s" % (emb.shape,))
    return c, d, h, w


def make_decoder(in_ch: int, grid0: int) -> nn.Module:
    layers: list[nn.Module] = []
    c = in_ch
    d = grid0
    while d < UP_SPATIAL:
        out_c = max(c // 2, 16)
        layers += [
            nn.ConvTranspose3d(c, out_c, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(out_c),
            nn.GELU(),
        ]
        c = out_c
        d *= 2
    layers.append(nn.Conv3d(c, 1, kernel_size=1))
    return nn.Sequential(*layers)


def preprocess_pre_for_sam(sam: nn.Module, pre: torch.Tensor) -> torch.Tensor:
    """pre: (B,1,D,H,W) in [0,1] -> SAM encoder input cube (B,1,S,S,S)."""
    x = _sam_norm_1ch(sam, pre * 255.0)
    isz = int(sam.image_encoder.img_size)
    return F.interpolate(x, size=(isz, isz, isz), mode="trilinear", align_corners=False)


class SamMed3DCBFRegressor(nn.Module):
    def __init__(self, sam: nn.Module, freeze_encoder: bool = True):
        super().__init__()
        self.sam = sam
        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            for p in self.sam.image_encoder.parameters():
                p.requires_grad = False
        c0, g, _, _ = probe_encoder_out(sam, DEVICE)
        self.decoder = make_decoder(c0, g).to(DEVICE)

    def forward_pred(self, pre: torch.Tensor, for_training: bool = True) -> torch.Tensor:
        """for_training=False: inference-style (no grad)."""
        x_in = preprocess_pre_for_sam(self.sam, pre)
        if not for_training:
            self.sam.image_encoder.eval()
            self.decoder.eval()
            with torch.no_grad():
                emb = self.sam.image_encoder(x_in)
                y = self.decoder(emb)
        elif self.freeze_encoder:
            self.sam.image_encoder.eval()
            with torch.no_grad():
                emb = self.sam.image_encoder(x_in)
            y = self.decoder(emb)
        else:
            emb = self.sam.image_encoder(x_in)
            y = self.decoder(emb)
        pred = F.interpolate(y, size=TARGET_SHAPE_3D, mode="trilinear", align_corners=False)
        return pred.clamp(0.0, 1.0)


def resize_vol(vol, size):
    if vol.dim() == 3:
        return F.interpolate(vol.unsqueeze(0).unsqueeze(0), size=size, mode="trilinear", align_corners=False).squeeze(0).squeeze(0)
    return F.interpolate(vol, size=size, mode="trilinear", align_corners=False)


def train_epoch(model: SamMed3DCBFRegressor, loader, optimizer, mask_t):
    model.decoder.train()
    if model.freeze_encoder:
        model.sam.image_encoder.eval()
    else:
        model.sam.image_encoder.train()
    total, n = 0.0, 0
    for pre, post in loader:
        pre = pre.to(DEVICE)
        post = post.to(DEVICE)
        optimizer.zero_grad()
        pred = model.forward_pred(pre, for_training=True)
        l1 = (torch.abs(pred - post) * mask_t).sum() / (mask_t.sum() + 1e-8)
        l1.backward()
        optimizer.step()
        total += l1.item()
        n += 1
    return total / max(n, 1)


def evaluate(model: SamMed3DCBFRegressor, loader):
    model.decoder.eval()
    model.sam.image_encoder.eval()
    mae_list, ssim_list, psnr_list = [], [], []
    with torch.no_grad():
        for pre, post in loader:
            pre = pre.to(DEVICE)
            post = post.to(DEVICE)
            pred = model.forward_pred(pre, for_training=False)
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


def train_eval_one_seed(seed: int, freeze_encoder: bool, enc_tag: str) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print("SAM-Med3D CBF (freeze_encoder=%s, %s), Week7 pre->post, seed=%s" % (freeze_encoder, enc_tag, seed))
    sam = load_sam_med3d().to(DEVICE)
    model = SamMed3DCBFRegressor(sam, freeze_encoder=freeze_encoder).to(DEVICE)
    params = list(model.decoder.parameters())
    if not freeze_encoder:
        params += list(model.sam.image_encoder.parameters())
    optimizer = torch.optim.Adam(params, lr=LR)

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

    ckpt_path = os.path.join(OUT_DIR, "sam_med3d_cbf_best_seed%d_%s.pt" % (seed, enc_tag))
    best_val_psnr = -1.0
    for ep in range(EPOCHS):
        loss = train_epoch(model, train_loader, optimizer, mask_t)
        metrics = evaluate(model, val_loader)
        if metrics["psnr_mean"] > best_val_psnr:
            best_val_psnr = metrics["psnr_mean"]
            blob = {"decoder": model.decoder.state_dict(), "epoch": ep, "seed": seed, "freeze_encoder": freeze_encoder}
            if not freeze_encoder:
                blob["image_encoder"] = model.sam.image_encoder.state_dict()
            torch.save(blob, ckpt_path)
        if (ep + 1) % 5 == 0:
            print("Epoch %d loss=%.4f val MAE=%.4f SSIM=%.4f PSNR=%.2f" % (ep + 1, loss, metrics["mae_mean"], metrics["ssim_mean"], metrics["psnr_mean"]))

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.decoder.load_state_dict(ckpt["decoder"])
    if not freeze_encoder and ckpt.get("image_encoder") is not None:
        model.sam.image_encoder.load_state_dict(ckpt["image_encoder"])
    test_metrics = evaluate(model, test_loader)
    test_metrics["seed"] = seed
    test_metrics["task"] = "pre_to_post_cbf_in_brain"
    test_metrics["freeze_encoder"] = freeze_encoder
    test_metrics["encoder_tag"] = enc_tag
    test_metrics["encoder"] = "SAM-Med3D_image_encoder_frozen" if freeze_encoder else "SAM-Med3D_image_encoder_trainable"
    print("Test (seed %d):" % seed, test_metrics)
    return test_metrics


def main():
    ap = argparse.ArgumentParser(description="SAM-Med3D encoder + CBF decoder for Week7.")
    ap.add_argument("--seeds", type=str, default="42", help="Comma-separated seeds, e.g. 42,123,456")
    args = ap.parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    freeze_encoder = sam_med3d_freeze_encoder_from_env()
    enc_tag = "enc_frozen" if freeze_encoder else "enc_train"
    out_single = os.path.join(OUT_DIR, "sam_med3d_week7_cbf_results_%s.json" % enc_tag)
    all_metrics = []
    for sd in seeds:
        all_metrics.append(train_eval_one_seed(sd, freeze_encoder, enc_tag))
        if len(seeds) > 1:
            pj = os.path.join(OUT_DIR, "sam_med3d_week7_cbf_results_%s_seed%d.json" % (enc_tag, sd))
            with open(pj, "w") as f:
                json.dump(all_metrics[-1], f, indent=2)
            print("Saved", pj)

    if len(seeds) == 1:
        with open(out_single, "w") as f:
            json.dump(all_metrics[0], f, indent=2)
        print("Saved", out_single)
    else:
        agg = {
            "seeds": seeds,
            "n_seeds": len(seeds),
            "freeze_encoder": freeze_encoder,
            "encoder_tag": enc_tag,
            "lr": LR,
            "mae_mean": float(np.mean([m["mae_mean"] for m in all_metrics])),
            "mae_std": float(np.std([m["mae_mean"] for m in all_metrics])),
            "ssim_mean": float(np.mean([m["ssim_mean"] for m in all_metrics])),
            "ssim_std": float(np.std([m["ssim_mean"] for m in all_metrics])),
            "psnr_mean": float(np.mean([m["psnr_mean"] for m in all_metrics])),
            "psnr_std": float(np.std([m["psnr_mean"] for m in all_metrics])),
            "per_seed": all_metrics,
        }
        multi = os.path.join(OUT_DIR, "sam_med3d_week7_cbf_results_%s_multiseed.json" % enc_tag)
        with open(multi, "w") as f:
            json.dump(agg, f, indent=2)
        print("Saved aggregate", multi)


if __name__ == "__main__":
    main()
