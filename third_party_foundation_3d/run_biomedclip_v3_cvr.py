#!/usr/bin/env python3
"""
BiomedCLIPv3: Cross-attention + FiLM conditioned UNet3D with learned text encoding.

Improvement over BiomedCLIPv2 (multi-level FiLM only):
  FiLM applies identical modulation to every spatial position.  Cross-attention
  at the bottleneck lets each spatial token selectively attend to the most
  relevant clinical text tokens (e.g., a lateralised voxel can up-weight
  "laterality right").  Decoder levels retain FiLM (cross-attention is too
  memory-intensive at higher spatial resolutions: 24×28×24 = 16k tokens).

Architecture (CrossAttnFiLMUNet3D):
  - Bottleneck (128ch, ~12×14×12)  : SpatialTextCrossAttention
  - Decoder L1 (32ch)              : FiLMLayer
  - Decoder L2 (16ch)              : FiLMLayer

Text encoder (ClinicalTextEncoder / BoWTextEncoder):
  - forward()         → (B, 128) pooled embedding   (for FiLM)
  - forward_tokens()  → (B, L, 128) per-token embeds (K/V for cross-attention)

Run (single seed):
  PYTHONNOUSERSITE=1 \
  PYTHONPATH=/data1/julih/scripts:/data1/julih/scripts/vlm:/data1/julih/UNet_3D \
  /data1/julih/miniconda3/envs/julih_monai/bin/python \
  /data1/julih/third_party_foundation_3d/run_biomedclip_v3_cvr.py --seeds 42

Run (three seeds):
  ... run_biomedclip_v3_cvr.py --seeds 42,123,456

Outputs:
  results/vlm_v3/biomedclip_v3_best_seed{N}.pt
  results/vlm_v3/biomedclip_v3_results_seed{N}.json
  results/vlm_v3/biomedclip_v3_multiseed_summary.json
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
sys.path.insert(0, os.path.join(ROOT, "UNet_3D"))

from week7_preprocess import TARGET_SHAPE, metrics_in_brain, get_brain_mask_for_shape
from week7_data import get_week7_splits
from text_encoder_v2 import ClinicalTextEncoder, PROMPTS_PATH, build_vocab_from_files
from cross_attention_fusion import CrossAttnFiLMUNet3D

try:
    from monai.losses import SSIMLoss
    _SSIM_LOSS = SSIMLoss(spatial_dims=3, data_range=1.0)
except Exception:
    _SSIM_LOSS = None

OUT_DIR = os.path.join(ROOT, "results", "vlm_v3")
os.makedirs(OUT_DIR, exist_ok=True)

METADATA_PATH      = os.path.join(ROOT, "VLMCSVFILES", "clinical_metadata_v2.json")
HC_PROMPTS_PATH    = os.path.join(ROOT, "VLMCSVFILES", "hc_structured_prompts.jsonl")
MM_PROMPTS_V2_PATH = os.path.join(ROOT, "VLMCSVFILES", "moyamoya_structured_prompts_v2.jsonl")
PAD_SHAPE          = (96, 112, 96)
DEVICE             = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS             = int(os.environ.get("BIOMEDCLIP_V3_EPOCHS", "50"))
LR                 = float(os.environ.get("BIOMEDCLIP_V3_LR", "1e-3"))
BATCH_SIZE         = 2
TEXT_EMBED_DIM     = 128


# ---------------------------------------------------------------------------
# Dataset — identical to v2 (same text prompt loading)
# ---------------------------------------------------------------------------

class VolumePairsWithText(torch.utils.data.Dataset):
    def __init__(self, pairs, metadata, prompts, augment=False,
                 target_shape=TARGET_SHAPE):
        from week7_data import Week7VolumePairs3D
        from week7_preprocess import load_pre_post_pair, augment_volume, _subject_id_from_path
        self._load_pair    = load_pre_post_pair
        self._augment_vol  = augment_volume
        self._sid_from     = _subject_id_from_path
        self.pairs         = pairs
        self.metadata      = metadata
        self.prompts       = prompts
        self.augment       = augment
        self.target_shape  = target_shape

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pre_path, post_path = self.pairs[idx]
        pre, post = self._load_pair(pre_path, post_path, target_shape=self.target_shape)
        if self.augment:
            import random as _r
            fl = _r.random() < 0.5; fu = _r.random() < 0.5; ff = _r.random() < 0.5
            sc = 0.9 + 0.2 * _r.random()
            pre  = self._augment_vol(pre,  flip_lr=fl, flip_ud=fu, flip_fb=ff, intensity_scale=sc)
            post = self._augment_vol(post, flip_lr=fl, flip_ud=fu, flip_fb=ff, intensity_scale=sc)
        pre_t  = torch.from_numpy(pre).unsqueeze(0).float()
        post_t = torch.from_numpy(post).unsqueeze(0).float()
        sid    = self._sid_from(pre_path)
        prompt = self.prompts.get(sid, "")
        return pre_t, post_t, prompt


def _load_prompts_from_jsonl(path):
    prompts = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            prompts[d["subject_id"]] = d["clinical_context_prompt"]
    return prompts


def _load_prompts():
    prompts = {}
    mm_path = MM_PROMPTS_V2_PATH if os.path.isfile(MM_PROMPTS_V2_PATH) else PROMPTS_PATH
    prompts.update(_load_prompts_from_jsonl(mm_path))
    if os.path.isfile(HC_PROMPTS_PATH):
        prompts.update(_load_prompts_from_jsonl(HC_PROMPTS_PATH))
    return prompts


# ---------------------------------------------------------------------------
# Pad / crop
# ---------------------------------------------------------------------------

def _pad(t, shape=PAD_SHAPE):
    _, _, d, h, w = t.shape
    pd, ph, pw = shape
    return F.pad(t, [0, pw - w, 0, ph - h, 0, pd - d])


def _crop(t, shape=TARGET_SHAPE):
    return t[:, :, :shape[0], :shape[1], :shape[2]]


def _collate(batch):
    pres    = torch.stack([b[0] for b in batch])
    posts   = torch.stack([b[1] for b in batch])
    prompts = [b[2] for b in batch]
    return pres, posts, prompts


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------

def train_epoch(model, text_enc, loader, optimizer, mask_t):
    model.train(); text_enc.train()
    total, n = 0.0, 0
    criterion_l1   = nn.L1Loss()
    criterion_ssim = _SSIM_LOSS

    for pre, post, prompts in loader:
        pre  = _pad(pre.to(DEVICE))
        post = _pad(post.to(DEVICE))

        pooled, tok_embeds, pad_mask = text_enc.forward_tokens(prompts, device=DEVICE)

        optimizer.zero_grad()
        pred = model(pre, text_tokens=tok_embeds, key_padding_mask=pad_mask,
                     text_embed=pooled)

        pred_c = _crop(pred); post_c = _crop(post)

        if mask_t is not None:
            B = pred_c.shape[0]
            m = mask_t.expand(B, -1, -1, -1, -1)
            loss_l1 = (torch.abs(pred_c - post_c) * m).sum() / (m.sum() + 1e-8)
        else:
            loss_l1 = criterion_l1(pred_c, post_c)

        if criterion_ssim is not None:
            loss = loss_l1 + criterion_ssim(pred_c, post_c)
        else:
            loss = loss_l1

        loss.backward()
        optimizer.step()
        total += loss.item(); n += 1

    return total / max(n, 1)


@torch.no_grad()
def evaluate(model, text_enc, loader):
    model.eval(); text_enc.eval()
    mae_list, ssim_list, psnr_list = [], [], []

    for pre, post, prompts in loader:
        pre      = _pad(pre.to(DEVICE))
        post_dev = post.to(DEVICE)

        pooled, tok_embeds, pad_mask = text_enc.forward_tokens(prompts, device=DEVICE)

        pred   = model(pre, text_tokens=tok_embeds, key_padding_mask=pad_mask,
                       text_embed=pooled)
        pred_c = _crop(pred)
        post_c = _crop(post_dev)

        for i in range(pred_c.shape[0]):
            p = pred_c[i, 0].cpu().numpy()
            t = post_c[i, 0].cpu().numpy()
            m = metrics_in_brain(p, t, data_range=1.0)
            mae_list.append(m["mae_mean"])
            ssim_list.append(m["ssim_mean"])
            psnr_list.append(m["psnr_mean"])

    return {
        "mae_mean":  float(np.mean(mae_list)),  "mae_std":  float(np.std(mae_list)),
        "ssim_mean": float(np.mean(ssim_list)), "ssim_std": float(np.std(ssim_list)),
        "psnr_mean": float(np.mean(psnr_list)), "psnr_std": float(np.std(psnr_list)),
    }


# ---------------------------------------------------------------------------
# Single-seed run
# ---------------------------------------------------------------------------

def train_one_seed(seed: int) -> dict:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    print(f"\nBiomedCLIPv3 seed={seed}  (cross-attention bottleneck + FiLM decoders)")

    prompts  = _load_prompts()
    metadata = json.load(open(METADATA_PATH))
    train_pairs, val_pairs, test_pairs = get_week7_splits()

    train_ds = VolumePairsWithText(train_pairs, metadata, prompts, augment=True)
    val_ds   = VolumePairsWithText(val_pairs,   metadata, prompts, augment=False)
    test_ds  = VolumePairsWithText(test_pairs,  metadata, prompts, augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, collate_fn=_collate)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, collate_fn=_collate)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, collate_fn=_collate)

    # Build combined vocab (MM V2 + HC)
    mm_path = MM_PROMPTS_V2_PATH if os.path.isfile(MM_PROMPTS_V2_PATH) else PROMPTS_PATH
    vocab_paths = [mm_path]
    if os.path.isfile(HC_PROMPTS_PATH):
        vocab_paths.append(HC_PROMPTS_PATH)

    text_enc = ClinicalTextEncoder.from_prompts_files(
        vocab_paths, out_dim=TEXT_EMBED_DIM
    ).to(DEVICE)

    model = CrossAttnFiLMUNet3D(
        text_embed_dim = TEXT_EMBED_DIM,
        text_token_dim = TEXT_EMBED_DIM,
        n_attn_heads   = 4,
    ).to(DEVICE)

    mask_np = get_brain_mask_for_shape(TARGET_SHAPE)
    mask_t  = torch.from_numpy(mask_np).float().to(DEVICE).unsqueeze(0).unsqueeze(0)

    params    = list(model.parameters()) + list(text_enc.parameters())
    optimizer = torch.optim.Adam(params, lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    ckpt_path = os.path.join(OUT_DIR, f"biomedclip_v3_best_seed{seed}.pt")
    best_psnr = -float("inf")
    history   = []

    for ep in range(1, EPOCHS + 1):
        t0   = time.time()
        loss = train_epoch(model, text_enc, train_loader, optimizer, mask_t)
        val  = evaluate(model, text_enc, val_loader)
        scheduler.step()

        if val["psnr_mean"] > best_psnr:
            best_psnr = val["psnr_mean"]
            torch.save({"model": model.state_dict(),
                        "text_enc": text_enc.state_dict(),
                        "epoch": ep, "seed": seed}, ckpt_path)

        history.append({"epoch": ep, "train_loss": loss,
                        "val_ssim": val["ssim_mean"], "val_psnr": val["psnr_mean"]})
        if ep % 5 == 0 or ep == 1:
            print(f"  Ep {ep:3d} loss={loss:.4f}  val SSIM={val['ssim_mean']:.3f}"
                  f"  PSNR={val['psnr_mean']:.2f}  ({time.time()-t0:.1f}s)")

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    text_enc.load_state_dict(ckpt["text_enc"])
    test_metrics = evaluate(model, text_enc, test_loader)
    test_metrics.update({"seed": seed, "model": "BiomedCLIPv3",
                          "epochs": EPOCHS, "lr": LR, "history": history})

    out_json = os.path.join(OUT_DIR, f"biomedclip_v3_results_seed{seed}.json")
    with open(out_json, "w") as f:
        json.dump(test_metrics, f, indent=2)

    print(f"  Test seed={seed}: SSIM={test_metrics['ssim_mean']:.3f}"
          f"±{test_metrics['ssim_std']:.3f}  "
          f"MAE={test_metrics['mae_mean']:.3f}  "
          f"PSNR={test_metrics['psnr_mean']:.2f}")
    return test_metrics


# ---------------------------------------------------------------------------
# Multi-seed summary
# ---------------------------------------------------------------------------

def multiseed_summary(all_results):
    ssims = [r["ssim_mean"] for r in all_results]
    maes  = [r["mae_mean"]  for r in all_results]
    psnrs = [r["psnr_mean"] for r in all_results]
    return {
        "model":                    "BiomedCLIPv3",
        "n_seeds":                  len(all_results),
        "seeds":                    [r["seed"] for r in all_results],
        "ssim_mean_across_seeds":   float(np.mean(ssims)),
        "ssim_std_across_seeds":    float(np.std(ssims)),
        "mae_mean_across_seeds":    float(np.mean(maes)),
        "mae_std_across_seeds":     float(np.std(maes)),
        "psnr_mean_across_seeds":   float(np.mean(psnrs)),
        "psnr_std_across_seeds":    float(np.std(psnrs)),
        "per_seed":                 all_results,
    }


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def _run_tests() -> None:
    print("=== BiomedCLIPv3 Unit Tests ===\n")
    passed = failed = 0

    def ok(name, cond, detail=""):
        nonlocal passed, failed
        tag = "PASS" if cond else "FAIL"
        print(f"  [{tag}] {name}" + (f"  ({detail})" if detail else ""))
        if cond: passed += 1
        else:    failed += 1

    # Prompt loading
    prompts = _load_prompts()
    ok("prompts loaded", len(prompts) > 0, f"n={len(prompts)}")
    ok("prompt is string", isinstance(list(prompts.values())[0], str))

    # VolumePairsWithText
    import tempfile, nibabel as nib
    dummy = np.random.rand(91, 109, 91).astype(np.float32)
    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tf:
        nib.save(nib.Nifti1Image(dummy, np.eye(4)), tf.name)
        tmp_p = tf.name
    ds = VolumePairsWithText([(tmp_p, tmp_p)], {}, {})
    pre, post, prompt = ds[0]
    ok("dataset pre shape",    pre.shape == (1,) + TARGET_SHAPE, str(pre.shape))
    ok("dataset prompt str",   isinstance(prompt, str))
    os.unlink(tmp_p)

    # _collate
    batch = [(torch.zeros(1, 91, 109, 91), torch.zeros(1, 91, 109, 91), "text A"),
             (torch.zeros(1, 91, 109, 91), torch.zeros(1, 91, 109, 91), "text B")]
    pres, posts, texts = _collate(batch)
    ok("collate pre shape", pres.shape == (2, 1, 91, 109, 91))
    ok("collate texts",     texts == ["text A", "text B"])

    # Full forward pass
    model    = CrossAttnFiLMUNet3D(text_embed_dim=128, text_token_dim=128).to(DEVICE)
    text_enc = ClinicalTextEncoder.from_prompts_file(out_dim=128).to(DEVICE)
    pre_t    = _pad(torch.zeros(1, 1, 91, 109, 91)).to(DEVICE)

    sample_texts = ["Cohort year 2020; sex female; laterality bilateral."]
    pooled, tok_embeds, pad_mask = text_enc.forward_tokens(sample_texts, device=DEVICE)
    ok("forward_tokens pooled shape", pooled.shape == (1, 128),    str(pooled.shape))
    ok("forward_tokens tok shape",    tok_embeds.shape[2] == 128,  str(tok_embeds.shape))
    ok("forward_tokens pad_mask bool", pad_mask.dtype == torch.bool)

    pred = model(pre_t, text_tokens=tok_embeds, key_padding_mask=pad_mask,
                 text_embed=pooled)
    ok("pipeline output shape", pred.shape == (1, 1) + PAD_SHAPE, str(pred.shape))
    pred_c = _crop(pred)
    ok("cropped shape", pred_c.shape == (1, 1) + TARGET_SHAPE)

    # multiseed_summary
    fake = [{"ssim_mean": 0.78, "mae_mean": 0.09, "psnr_mean": 22.0, "seed": 42},
            {"ssim_mean": 0.79, "mae_mean": 0.08, "psnr_mean": 22.5, "seed": 123}]
    summ = multiseed_summary(fake)
    ok("multiseed ssim mean", abs(summ["ssim_mean_across_seeds"] - 0.785) < 1e-6)
    ok("multiseed n_seeds=2",  summ["n_seeds"] == 2)
    ok("model tag correct",    summ["model"] == "BiomedCLIPv3")

    model.remove_hooks()
    print(f"\nTests: {passed} passed, {failed} failed out of {passed+failed}")
    if failed:
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="42", help="comma-separated seed list")
    p.add_argument("--test",  action="store_true")
    args = p.parse_args()

    if args.test:
        _run_tests()
        return

    seeds = [int(s) for s in args.seeds.split(",")]
    all_results = [train_one_seed(seed) for seed in seeds]

    if len(all_results) > 1:
        summ = multiseed_summary(all_results)
        out  = os.path.join(OUT_DIR, "biomedclip_v3_multiseed_summary.json")
        with open(out, "w") as f:
            json.dump(summ, f, indent=2)
        print(f"\nMulti-seed summary ({len(seeds)} seeds):")
        print(f"  SSIM = {summ['ssim_mean_across_seeds']:.3f} ± {summ['ssim_std_across_seeds']:.3f}")
        print(f"  MAE  = {summ['mae_mean_across_seeds']:.3f} ± {summ['mae_std_across_seeds']:.3f}")
        print(f"  PSNR = {summ['psnr_mean_across_seeds']:.2f} ± {summ['psnr_std_across_seeds']:.2f}")
        print(f"  Saved: {out}")


if __name__ == "__main__":
    main()
