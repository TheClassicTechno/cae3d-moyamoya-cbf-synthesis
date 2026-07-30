#!/usr/bin/env python3
"""
Med3DVLMv2: DCFormer encoder with FiLM text conditioning for CVR synthesis.

Architecture improvements over Med3DVLMv1 (run_med3dvlm_week7_cvr.py):
  1. Text encoder: learned bag-of-words token embedding (BoWTextEncoder) instead
     of no text conditioning at all. Med3DVLMv1 had clinical text infrastructure
     designed into the codebase but commented "not wired in by default." V2 wires
     it in.
  2. FiLM at DCFormer bottleneck: the (B, 768, 2, 2, 2) feature map output from
     the DCFormer encoder is modulated by γ, β produced from the text embedding.
  3. Three-seed training for reproducibility.

Architecture:
  pre-ACZ volume (B,1,128³) → DCFormer encoder → (B, 768, 2, 2, 2)
                                                        ↑
  clinical text prompt → BoWTextEncoder → (B, 768) → FiLM(γ,β)
                                                        ↓
                          FiLM-modulated bottleneck → CVRDecoder3D → (B,1,128³)
                          → interpolate to (91,109,91) → predicted post-ACZ

Note: the DCFormer bottleneck has 768 channels (decomp_small output).
      text_embed_dim is set to 768 to produce per-channel FiLM parameters
      without a separate projection layer.

Run:
  PYTHONNOUSERSITE=1 \
  PYTHONPATH=/data1/julih/scripts:/data1/julih/scripts/vlm \
             :/data1/julih/third_party_foundation_3d/Med3DVLM \
  /data1/julih/miniconda3/envs/julih_monai/bin/python \
  /data1/julih/third_party_foundation_3d/run_med3dvlm_v2_cvr.py --seeds 42

Run all three seeds:
  ... run_med3dvlm_v2_cvr.py --seeds 42,123,456

Outputs:
  third_party_foundation_3d/med3dvlm_v2_cvr/
    med3dvlm_v2_best_seed{N}.pt
    med3dvlm_v2_results_seed{N}.json
    med3dvlm_v2_multiseed_summary.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as ssim_fn, peak_signal_noise_ratio as psnr_fn

ROOT = "/data1/julih"
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "vlm"))
sys.path.insert(0, os.path.join(ROOT, "third_party_foundation_3d", "Med3DVLM"))

from week7_preprocess import TARGET_SHAPE, metrics_in_brain, get_brain_mask_for_shape
from week7_data import get_week7_splits, Week7VolumePairs3D
from text_encoder_v2 import ClinicalTextEncoder, PROMPTS_PATH
from src.model.encoder.dcformer import decomp_small

OUT_DIR = os.path.join(ROOT, "third_party_foundation_3d", "med3dvlm_v2_cvr")
os.makedirs(OUT_DIR, exist_ok=True)

METADATA_PATH   = os.path.join(ROOT, "VLMCSVFILES", "clinical_metadata.json")
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
ENC_SIZE        = (128, 128, 128)
EPOCHS          = int(os.environ.get("MED3DVLM_V2_EPOCHS", "30"))
LR              = float(os.environ.get("MED3DVLM_V2_LR", "1e-4"))
BATCH_SIZE      = 2
DCF_BOTTLENECK_CH = 768   # decomp_small last feature channels


# ---------------------------------------------------------------------------
# FiLM layer (self-contained here so no cross-import with film_multilevel)
# ---------------------------------------------------------------------------

class FiLMLayer(nn.Module):
    def __init__(self, embed_dim: int, n_channels: int) -> None:
        super().__init__()
        self.gamma = nn.Linear(embed_dim, n_channels)
        self.beta  = nn.Linear(embed_dim, n_channels)
        nn.init.zeros_(self.gamma.weight);  nn.init.ones_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight);   nn.init.zeros_(self.beta.bias)

    def forward(self, embed: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        g = self.gamma(embed).view(embed.size(0), -1, 1, 1, 1)
        b = self.beta(embed).view(embed.size(0),  -1, 1, 1, 1)
        return g * feat + b


# ---------------------------------------------------------------------------
# CVRDecoder3D (same as Med3DVLMv1)
# ---------------------------------------------------------------------------

class CVRDecoder3D(nn.Module):
    def __init__(self, in_ch: int = DCF_BOTTLENECK_CH) -> None:
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose3d(in_ch, 384, 4, stride=2, padding=1), nn.BatchNorm3d(384), nn.GELU(),
            nn.ConvTranspose3d(384,   192, 4, stride=2, padding=1), nn.BatchNorm3d(192), nn.GELU(),
            nn.ConvTranspose3d(192,    96, 4, stride=2, padding=1), nn.BatchNorm3d(96),  nn.GELU(),
            nn.ConvTranspose3d(96,     48, 4, stride=2, padding=1), nn.BatchNorm3d(48),  nn.GELU(),
            nn.ConvTranspose3d(48,     24, 4, stride=2, padding=1), nn.BatchNorm3d(24),  nn.GELU(),
            nn.ConvTranspose3d(24,      1, 4, stride=2, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)


# ---------------------------------------------------------------------------
# DCFormerCVRv2: DCFormer encoder + FiLM on bottleneck + CVRDecoder3D
# ---------------------------------------------------------------------------

class DCFormerCVRv2(nn.Module):
    """
    Med3DVLMv2: DCFormer backbone + FiLM text conditioning at bottleneck.

    When text_embed is None, FiLM is bypassed (identity) and the model
    is equivalent to DCFormerCVRv1 (pure vision baseline).
    """

    def __init__(
        self,
        input_size:     tuple = ENC_SIZE,
        freeze_encoder: bool = True,
        text_embed_dim: int  = DCF_BOTTLENECK_CH,
    ) -> None:
        super().__init__()
        self.encoder = decomp_small(input_size=input_size)
        self.enc_size = input_size
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        enc_ch       = self.encoder.channels[-1]
        self.decoder = CVRDecoder3D(in_ch=enc_ch)
        self.film    = FiLMLayer(text_embed_dim, enc_ch)

    def forward(
        self,
        x:          torch.Tensor,
        text_embed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x          : (B, 1, 128, 128, 128) in (B,C,D,H,W)
            text_embed : (B, text_embed_dim) or None

        Returns:
            (B, 1, 128, 128, 128) predicted post-ACZ volume
        """
        feats = self.encoder(x)
        last  = feats[-1]                              # (B, N, C)
        B, N, C = last.shape
        s = 2
        bottleneck = last.permute(0, 2, 1).view(B, C, s, s, s)   # (B, C, 2, 2, 2)

        if text_embed is not None:
            bottleneck = self.film(text_embed, bottleneck)

        return self.decoder(bottleneck)


# ---------------------------------------------------------------------------
# Dataset wrapper with text prompts
# ---------------------------------------------------------------------------

def _load_prompts(path: str = PROMPTS_PATH) -> dict[str, str]:
    out = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            out[d["subject_id"]] = d["clinical_context_prompt"]
    return out


class VolumePairsWithText(torch.utils.data.Dataset):
    def __init__(self, pairs, prompts, augment=False):
        from week7_preprocess import load_pre_post_pair, augment_volume, _subject_id_from_path
        self._load = load_pre_post_pair
        self._aug  = augment_volume
        self._sid  = _subject_id_from_path
        self.pairs   = pairs
        self.prompts = prompts
        self.augment = augment

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        pre_p, post_p = self.pairs[idx]
        pre, post = self._load(pre_p, post_p, target_shape=TARGET_SHAPE)
        if self.augment:
            import random as _r
            fl = _r.random() < 0.5; fu = _r.random() < 0.5; ff = _r.random() < 0.5
            sc = 0.9 + 0.2 * _r.random()
            pre  = self._aug(pre,  flip_lr=fl, flip_ud=fu, flip_fb=ff, intensity_scale=sc)
            post = self._aug(post, flip_lr=fl, flip_ud=fu, flip_fb=ff, intensity_scale=sc)
        pre_t  = torch.from_numpy(pre).unsqueeze(0).float()
        post_t = torch.from_numpy(post).unsqueeze(0).float()
        sid    = self._sid(pre_p)
        return pre_t, post_t, self.prompts.get(sid, "")


def _collate(batch):
    pres    = torch.stack([b[0] for b in batch])
    posts   = torch.stack([b[1] for b in batch])
    prompts = [b[2] for b in batch]
    return pres, posts, prompts


def resize_vol(vol, size):
    if vol.dim() == 3:
        return F.interpolate(vol.unsqueeze(0).unsqueeze(0), size=size,
                             mode="trilinear", align_corners=False).squeeze(0).squeeze(0)
    return F.interpolate(vol, size=size, mode="trilinear", align_corners=False)


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------

def train_epoch(model, text_enc, loader, optimizer, mask_t, use_condition):
    model.train(); text_enc.train()
    total, n = 0.0, 0
    criterion = nn.L1Loss()

    for pre, post, prompts in loader:
        pre  = pre.to(DEVICE);  post = post.to(DEVICE)
        pre_128  = resize_vol(pre,  ENC_SIZE)

        if use_condition:
            embed = text_enc(prompts, device=DEVICE)
        else:
            embed = None

        optimizer.zero_grad()
        pred_128 = model(pre_128, embed)
        pred     = resize_vol(pred_128, TARGET_SHAPE)

        if mask_t is not None:
            loss = (torch.abs(pred - post) * mask_t).sum() / (mask_t.sum() + 1e-8)
        else:
            loss = criterion(pred, post)

        loss.backward()
        optimizer.step()
        total += loss.item(); n += 1

    return total / max(n, 1)


@torch.no_grad()
def evaluate(model, text_enc, loader, use_condition):
    model.eval(); text_enc.eval()
    mae_l, ssim_l, psnr_l = [], [], []

    for pre, post, prompts in loader:
        pre  = pre.to(DEVICE);  post = post.to(DEVICE)
        pre_128  = resize_vol(pre,  ENC_SIZE)

        if use_condition:
            embed = text_enc(prompts, device=DEVICE)
        else:
            embed = None

        pred_128 = model(pre_128, embed)
        pred     = resize_vol(pred_128, TARGET_SHAPE)

        for i in range(pred.shape[0]):
            m = metrics_in_brain(pred[i, 0].cpu().numpy(),
                                 post[i, 0].cpu().numpy(), data_range=1.0)
            mae_l.append(m["mae_mean"])
            ssim_l.append(m["ssim_mean"])
            psnr_l.append(m["psnr_mean"])

    return {"mae_mean": float(np.mean(mae_l)),  "mae_std":  float(np.std(mae_l)),
            "ssim_mean": float(np.mean(ssim_l)), "ssim_std": float(np.std(ssim_l)),
            "psnr_mean": float(np.mean(psnr_l)), "psnr_std": float(np.std(psnr_l))}


# ---------------------------------------------------------------------------
# Single-seed run
# ---------------------------------------------------------------------------

def train_one_seed(seed: int, use_condition: bool, freeze_encoder: bool = True) -> dict:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    enc_tag = "enc_frozen" if freeze_encoder else "enc_train"
    cond_tag = "conditioned" if use_condition else "nocond"
    tag = f"{enc_tag}_{cond_tag}"
    print(f"\nMed3DVLMv2 seed={seed} condition={use_condition} freeze_enc={freeze_encoder}")

    prompts = _load_prompts()
    train_pairs, val_pairs, test_pairs = get_week7_splits()

    train_ds = VolumePairsWithText(train_pairs, prompts, augment=True)
    val_ds   = VolumePairsWithText(val_pairs,   prompts, augment=False)
    test_ds  = VolumePairsWithText(test_pairs,  prompts, augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, collate_fn=_collate)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, collate_fn=_collate)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, collate_fn=_collate)

    model    = DCFormerCVRv2(input_size=ENC_SIZE, freeze_encoder=freeze_encoder,
                              text_embed_dim=DCF_BOTTLENECK_CH).to(DEVICE)
    text_enc = ClinicalTextEncoder.from_prompts_file(out_dim=DCF_BOTTLENECK_CH).to(DEVICE)

    mask_np = get_brain_mask_for_shape(TARGET_SHAPE)
    mask_t  = torch.from_numpy(mask_np).float().to(DEVICE).unsqueeze(0).unsqueeze(0)

    trainable = [p for p in model.parameters() if p.requires_grad]
    trainable += list(text_enc.parameters())
    optimizer = torch.optim.Adam(trainable, lr=LR)

    ckpt_path = os.path.join(OUT_DIR, f"med3dvlm_v2_best_seed{seed}_{tag}.pt")
    best_psnr = -float("inf")
    history   = []

    for ep in range(1, EPOCHS + 1):
        t0   = time.time()
        loss = train_epoch(model, text_enc, train_loader, optimizer, mask_t, use_condition)
        val  = evaluate(model, text_enc, val_loader, use_condition)

        if val["psnr_mean"] > best_psnr:
            best_psnr = val["psnr_mean"]
            torch.save({"model": model.state_dict(), "text_enc": text_enc.state_dict(),
                        "epoch": ep, "seed": seed, "tag": tag}, ckpt_path)

        history.append({"epoch": ep, "train_loss": loss,
                        "val_ssim": val["ssim_mean"], "val_psnr": val["psnr_mean"]})
        if ep % 5 == 0 or ep == 1:
            print(f"  Ep {ep:3d} loss={loss:.4f}  val SSIM={val['ssim_mean']:.3f}"
                  f"  PSNR={val['psnr_mean']:.2f}  ({time.time()-t0:.1f}s)")

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    text_enc.load_state_dict(ckpt["text_enc"])
    test_metrics = evaluate(model, text_enc, test_loader, use_condition)
    test_metrics.update({"seed": seed, "tag": tag, "condition": use_condition,
                          "freeze_encoder": freeze_encoder, "epochs": EPOCHS,
                          "lr": LR, "history": history})

    out_json = os.path.join(OUT_DIR, f"med3dvlm_v2_results_seed{seed}_{tag}.json")
    with open(out_json, "w") as f:
        json.dump(test_metrics, f, indent=2)

    print(f"  Test seed={seed}: SSIM={test_metrics['ssim_mean']:.3f}"
          f"±{test_metrics['ssim_std']:.3f}  "
          f"MAE={test_metrics['mae_mean']:.3f}  "
          f"PSNR={test_metrics['psnr_mean']:.2f}")
    return test_metrics


def multiseed_summary(all_results):
    ssims = [r["ssim_mean"] for r in all_results]
    maes  = [r["mae_mean"]  for r in all_results]
    psnrs = [r["psnr_mean"] for r in all_results]
    return {"model": "Med3DVLMv2", "n_seeds": len(all_results),
            "seeds": [r["seed"] for r in all_results],
            "ssim_mean_across_seeds": float(np.mean(ssims)),
            "ssim_std_across_seeds":  float(np.std(ssims)),
            "mae_mean_across_seeds":  float(np.mean(maes)),
            "mae_std_across_seeds":   float(np.std(maes)),
            "psnr_mean_across_seeds": float(np.mean(psnrs)),
            "psnr_std_across_seeds":  float(np.std(psnrs)),
            "per_seed": all_results}


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def _run_tests() -> None:
    print("=== Med3DVLMv2 Unit Tests ===\n")
    passed = failed = 0

    def ok(name, cond, detail=""):
        nonlocal passed, failed
        tag = "PASS" if cond else "FAIL"
        print(f"  [{tag}] {name}" + (f"  ({detail})" if detail else ""))
        if cond: passed += 1
        else:    failed += 1

    # FiLMLayer
    film = FiLMLayer(embed_dim=768, n_channels=768)
    feat = torch.ones(2, 768, 2, 2, 2)
    emb  = torch.zeros(2, 768)   # zeros → identity (γ=1, β=0 at init)
    out  = film(emb, feat)
    ok("FiLM identity with zero embed", torch.allclose(out, feat, atol=1e-5))

    # CVRDecoder3D output shape
    dec = CVRDecoder3D(in_ch=768)
    x   = torch.randn(1, 768, 2, 2, 2)
    out_dec = dec(x)
    ok("CVRDecoder3D output shape", out_dec.shape == (1, 1, 128, 128, 128), str(out_dec.shape))

    # DCFormerCVRv2 with text
    model    = DCFormerCVRv2(freeze_encoder=True, text_embed_dim=768).to(DEVICE)
    text_enc = ClinicalTextEncoder.from_prompts_file(out_dim=768).to(DEVICE)
    pre_128  = torch.randn(1, 1, 128, 128, 128).to(DEVICE)
    embed    = text_enc(["Age: 45, Sex: M, Laterality: Bilateral"], device=DEVICE)
    ok("text_enc embed shape", embed.shape == (1, 768), str(embed.shape))
    pred     = model(pre_128, embed)
    ok("DCFormerCVRv2 output shape", pred.shape == (1, 1, 128, 128, 128), str(pred.shape))

    # Without text (ablation)
    pred_no_text = model(pre_128, None)
    ok("DCFormerCVRv2 no-text output shape", pred_no_text.shape == (1, 1, 128, 128, 128))
    ok("text conditioning changes output",
       not torch.allclose(pred, pred_no_text))

    # _collate
    batch = [(torch.zeros(1,91,109,91), torch.zeros(1,91,109,91), "a"),
             (torch.zeros(1,91,109,91), torch.zeros(1,91,109,91), "b")]
    p, q, texts = _collate(batch)
    ok("_collate shape", p.shape == (2,1,91,109,91))
    ok("_collate texts", texts == ["a", "b"])

    # multiseed_summary
    ms = multiseed_summary([{"ssim_mean":0.5,"mae_mean":0.1,"psnr_mean":20.0,"seed":42},
                             {"ssim_mean":0.6,"mae_mean":0.1,"psnr_mean":21.0,"seed":123}])
    ok("multiseed ssim mean", abs(ms["ssim_mean_across_seeds"] - 0.55) < 1e-6)

    print(f"\nTests: {passed} passed, {failed} failed out of {passed+failed}")
    if failed:
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="42")
    p.add_argument("--no-condition", action="store_true")
    p.add_argument("--unfreeze-encoder", action="store_true")
    p.add_argument("--test", action="store_true")
    args = p.parse_args()

    if args.test:
        _run_tests()
        return

    seeds         = [int(s) for s in args.seeds.split(",")]
    use_condition = not args.no_condition
    freeze        = not args.unfreeze_encoder

    all_results = []
    for seed in seeds:
        r = train_one_seed(seed, use_condition, freeze_encoder=freeze)
        all_results.append(r)

    if len(all_results) > 1:
        summ = multiseed_summary(all_results)
        out  = os.path.join(OUT_DIR, "med3dvlm_v2_multiseed_summary.json")
        with open(out, "w") as f:
            json.dump(summ, f, indent=2)
        print(f"\n{summ['model']} multi-seed SSIM = "
              f"{summ['ssim_mean_across_seeds']:.3f} ± {summ['ssim_std_across_seeds']:.3f}")
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
