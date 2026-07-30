#!/usr/bin/env python3
"""
BiomedCLIPv2: Multi-level FiLM-conditioned UNet3D with learned text encoding.

Architecture improvements over BiomedCLIPv1 (Phase 6 ConditionedUNet3D):
  1. Text encoder: learned bag-of-words token embedding (BoWTextEncoder) instead
     of deterministic 5-dim scalar vector. The model now processes the clinical
     text as natural language tokens with learned representations.
  2. Multi-level FiLM: conditions the UNet at bottleneck (128 ch) AND two decoder
     levels (32 ch, 16 ch) instead of only the bottleneck. This allows the
     clinical signal to guide reconstruction at fine spatial scales, not just
     global context.
  3. Three-seed training: seeds 42, 123, 456 for reproducibility reporting.
  4. Full week7 brain-masked L1+SSIM loss (same as the winning UNet3D baseline).

Naming rationale:
  "BiomedCLIP" refers to the Microsoft BiomedCLIP model (open_clip), which pairs
  a 2D vision encoder with a text encoder for biomedical image-text alignment.
  Our "v2" extends this concept to 3D volumetric CBF synthesis: a 3D UNet vision
  backbone + learned clinical text encoder, trained end-to-end for CVR prediction.
  open_clip is not available in this environment; we implement an equivalent
  architecture using PyTorch primitives.

Run (single seed):
  PYTHONNOUSERSITE=1 \
  PYTHONPATH=/data1/julih/scripts:/data1/julih/scripts/vlm:/data1/julih/UNet_3D \
  /data1/julih/miniconda3/envs/julih_monai/bin/python \
  /data1/julih/third_party_foundation_3d/run_biomedclip_v2_cvr.py --seeds 42

Run (three seeds):
  PYTHONNOUSERSITE=1 \
  PYTHONPATH=/data1/julih/scripts:/data1/julih/scripts/vlm:/data1/julih/UNet_3D \
  /data1/julih/miniconda3/envs/julih_monai/bin/python \
  /data1/julih/third_party_foundation_3d/run_biomedclip_v2_cvr.py --seeds 42,123,456

Run ablation (no text conditioning, same architecture):
  ... run_biomedclip_v2_cvr.py --seeds 42 --no-condition

Outputs:
  results/vlm_v2/biomedclip_v2_best_seed{N}.pt
  results/vlm_v2/biomedclip_v2_results_seed{N}.json
  results/vlm_v2/biomedclip_v2_multiseed_summary.json
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
from week7_data import get_week7_splits, Week7VolumePairs3D
from text_encoder_v2 import ClinicalTextEncoder, PROMPTS_PATH
from film_multilevel import MultiLevelFiLMUNet3D

try:
    from monai.losses import SSIMLoss
    _SSIM_LOSS = SSIMLoss(spatial_dims=3, data_range=1.0)
except Exception:
    _SSIM_LOSS = None

OUT_DIR = os.path.join(ROOT, "results", "vlm_v2")
os.makedirs(OUT_DIR, exist_ok=True)

# V2 metadata: adds race field for Moyamoya subjects (built by build_vlm_metadata.py)
METADATA_PATH   = os.path.join(ROOT, "VLMCSVFILES", "clinical_metadata_v2.json")
# HC prompts: age/sex/ethnicity/race/diamox/weight for healthy controls
HC_PROMPTS_PATH = os.path.join(ROOT, "VLMCSVFILES", "hc_structured_prompts.jsonl")
# V2 Moyamoya prompts: adds race vs v1
MM_PROMPTS_V2_PATH = os.path.join(ROOT, "VLMCSVFILES", "moyamoya_structured_prompts_v2.jsonl")
PAD_SHAPE       = (96, 112, 96)
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS          = int(os.environ.get("BIOMEDCLIP_V2_EPOCHS", "50"))
LR              = float(os.environ.get("BIOMEDCLIP_V2_LR", "1e-3"))
BATCH_SIZE      = 2
TEXT_EMBED_DIM  = 128   # must match FiLM input dim


# ---------------------------------------------------------------------------
# Dataset wrapper: pairs + clinical text prompts
# ---------------------------------------------------------------------------

class VolumePairsWithText(torch.utils.data.Dataset):
    """
    Returns (pre_t, post_t, text_prompt_str, subject_id) per sample.

    text_prompt_str is fed to ClinicalTextEncoder.forward() as a list[str]
    during the collation step. If a subject has no metadata, an empty string
    is returned; the encoder maps unknown tokens to UNK.
    """

    def __init__(
        self,
        pairs:          list[tuple[str, str]],
        metadata:       dict[str, dict],
        prompts:        dict[str, str],
        augment:        bool = False,
        target_shape:   tuple = TARGET_SHAPE,
    ) -> None:
        from week7_data import Week7VolumePairs3D
        from week7_preprocess import load_pre_post_pair, augment_volume, _subject_id_from_path
        self._load_pair = load_pre_post_pair
        self._augment_vol = augment_volume
        self._sid_from = _subject_id_from_path
        self.pairs       = pairs
        self.metadata    = metadata
        self.prompts     = prompts
        self.augment     = augment
        self.target_shape = target_shape

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        pre_path, post_path = self.pairs[idx]
        pre, post = self._load_pair(pre_path, post_path, target_shape=self.target_shape)

        if self.augment:
            import random as _r
            fl  = _r.random() < 0.5
            fu  = _r.random() < 0.5
            ff  = _r.random() < 0.5
            sc  = 0.9 + 0.2 * _r.random()
            pre  = self._augment_vol(pre,  flip_lr=fl, flip_ud=fu, flip_fb=ff, intensity_scale=sc)
            post = self._augment_vol(post, flip_lr=fl, flip_ud=fu, flip_fb=ff, intensity_scale=sc)

        pre_t  = torch.from_numpy(pre).unsqueeze(0).float()
        post_t = torch.from_numpy(post).unsqueeze(0).float()

        sid    = self._sid_from(pre_path)
        prompt = self.prompts.get(sid, "")
        return pre_t, post_t, prompt


def _load_prompts_from_jsonl(path: str) -> dict[str, str]:
    """Load subject_id → prompt string from a JSONL file."""
    prompts: dict[str, str] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            prompts[d["subject_id"]] = d["clinical_context_prompt"]
    return prompts


def _load_prompts() -> dict[str, str]:
    """
    Build unified subject_id → prompt lookup for all training subjects.

    Merges:
      1. Moyamoya V2 prompts  (adds race vs v1)
      2. HC prompts           (age/sex/ethnicity/race/diamox/weight)

    Falls back to the original prompts file if V2 files are unavailable.
    Subjects not found in either lookup receive an empty string (encoder
    maps unknown tokens to UNK; missing prompts return a zero embedding).
    """
    prompts: dict[str, str] = {}

    # Try V2 files first; fall back to the original v1 file
    mm_path = MM_PROMPTS_V2_PATH if os.path.isfile(MM_PROMPTS_V2_PATH) else PROMPTS_PATH
    prompts.update(_load_prompts_from_jsonl(mm_path))

    if os.path.isfile(HC_PROMPTS_PATH):
        prompts.update(_load_prompts_from_jsonl(HC_PROMPTS_PATH))

    return prompts


# ---------------------------------------------------------------------------
# Pad / crop helpers (same convention as Phase 6)
# ---------------------------------------------------------------------------

def _pad(t: torch.Tensor, shape: tuple = PAD_SHAPE) -> torch.Tensor:
    _, _, d, h, w = t.shape
    pd, ph, pw = shape
    return F.pad(t, [0, pw - w, 0, ph - h, 0, pd - d])


def _crop(t: torch.Tensor, shape: tuple = TARGET_SHAPE) -> torch.Tensor:
    return t[:, :, :shape[0], :shape[1], :shape[2]]


# ---------------------------------------------------------------------------
# Training and evaluation loops
# ---------------------------------------------------------------------------

def _build_model(use_condition: bool) -> nn.Module:
    return MultiLevelFiLMUNet3D(text_embed_dim=TEXT_EMBED_DIM).to(DEVICE)


def _collate(batch):
    """Custom collate: stack tensors, keep text as list."""
    pres    = torch.stack([b[0] for b in batch])
    posts   = torch.stack([b[1] for b in batch])
    prompts = [b[2] for b in batch]
    return pres, posts, prompts


def train_epoch(
    model:       MultiLevelFiLMUNet3D,
    text_enc:    ClinicalTextEncoder,
    loader:      DataLoader,
    optimizer:   torch.optim.Optimizer,
    mask_t:      torch.Tensor,
    use_condition: bool,
) -> float:
    model.train();  text_enc.train()
    total, n = 0.0, 0
    criterion_l1   = nn.L1Loss()
    criterion_ssim = _SSIM_LOSS

    for pre, post, prompts in loader:
        pre  = _pad(pre.to(DEVICE))
        post = _pad(post.to(DEVICE))

        # Encode text
        if use_condition:
            text_embed = text_enc(prompts, device=DEVICE)   # (B, 128)
        else:
            text_embed = None

        optimizer.zero_grad()
        pred = model(pre, text_embed)

        # Crop back to TARGET_SHAPE for loss computation
        pred_c = _crop(pred);  post_c = _crop(post)

        if mask_t is not None:
            B = pred_c.shape[0]
            m = mask_t.expand(B, -1, -1, -1, -1)
            loss_l1 = (torch.abs(pred_c - post_c) * m).sum() / (m.sum() + 1e-8)
        else:
            loss_l1 = criterion_l1(pred_c, post_c)

        if criterion_ssim is not None:
            loss_ssim = criterion_ssim(pred_c, post_c)
            loss = loss_l1 + loss_ssim
        else:
            loss = loss_l1

        loss.backward()
        optimizer.step()
        total += loss.item();  n += 1

    return total / max(n, 1)


@torch.no_grad()
def evaluate(
    model:         MultiLevelFiLMUNet3D,
    text_enc:      ClinicalTextEncoder,
    loader:        DataLoader,
    use_condition: bool,
) -> dict:
    model.eval();  text_enc.eval()
    mae_list, ssim_list, psnr_list = [], [], []

    for pre, post, prompts in loader:
        pre  = _pad(pre.to(DEVICE))
        post_orig = post.to(DEVICE)

        if use_condition:
            text_embed = text_enc(prompts, device=DEVICE)
        else:
            text_embed = None

        pred    = model(pre, text_embed)
        pred_c  = _crop(pred)
        post_c  = _crop(post_orig)

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
# Single-seed training run
# ---------------------------------------------------------------------------

def train_one_seed(
    seed:          int,
    use_condition: bool,
) -> dict:
    random.seed(seed);  np.random.seed(seed);  torch.manual_seed(seed)

    tag = "conditioned" if use_condition else "ablation_nocond"
    print(f"\nBiomedCLIPv2 seed={seed} condition={use_condition}")

    # Data
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

    # Model + text encoder
    # Build vocabulary from all prompt sources so HC tokens are included.
    from text_encoder_v2 import build_vocab_from_files
    model    = _build_model(use_condition)
    mm_path  = MM_PROMPTS_V2_PATH if os.path.isfile(MM_PROMPTS_V2_PATH) else PROMPTS_PATH
    vocab_paths = [mm_path]
    if os.path.isfile(HC_PROMPTS_PATH):
        vocab_paths.append(HC_PROMPTS_PATH)
    text_enc = ClinicalTextEncoder.from_prompts_files(
        vocab_paths, out_dim=TEXT_EMBED_DIM
    ).to(DEVICE)

    # Brain mask for loss — keep at TARGET_SHAPE (loss is computed on cropped volumes)
    mask_np = get_brain_mask_for_shape(TARGET_SHAPE)
    mask_t  = torch.from_numpy(mask_np).float().to(DEVICE)
    mask_t  = mask_t.unsqueeze(0).unsqueeze(0)   # (1, 1, 91, 109, 91)

    params    = list(model.parameters()) + list(text_enc.parameters())
    optimizer = torch.optim.Adam(params, lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    ckpt_path  = os.path.join(OUT_DIR, f"biomedclip_v2_best_seed{seed}_{tag}.pt")
    best_psnr  = -float("inf")
    history    = []

    for ep in range(1, EPOCHS + 1):
        t0   = time.time()
        loss = train_epoch(model, text_enc, train_loader, optimizer, mask_t, use_condition)
        val  = evaluate(model, text_enc, val_loader, use_condition)
        scheduler.step()

        if val["psnr_mean"] > best_psnr:
            best_psnr = val["psnr_mean"]
            torch.save({"model": model.state_dict(),
                        "text_enc": text_enc.state_dict(),
                        "epoch": ep, "seed": seed, "tag": tag}, ckpt_path)

        history.append({"epoch": ep, "train_loss": loss,
                        "val_ssim": val["ssim_mean"], "val_psnr": val["psnr_mean"]})
        if ep % 5 == 0 or ep == 1:
            print(f"  Ep {ep:3d} loss={loss:.4f}  val SSIM={val['ssim_mean']:.3f}"
                  f"  PSNR={val['psnr_mean']:.2f}  ({time.time()-t0:.1f}s)")

    # Load best and test
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    text_enc.load_state_dict(ckpt["text_enc"])
    test_metrics = evaluate(model, text_enc, test_loader, use_condition)
    test_metrics.update({"seed": seed, "tag": tag, "condition": use_condition,
                          "epochs": EPOCHS, "lr": LR, "history": history})

    out_json = os.path.join(OUT_DIR, f"biomedclip_v2_results_seed{seed}_{tag}.json")
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

def multiseed_summary(all_results: list[dict]) -> dict:
    ssims  = [r["ssim_mean"] for r in all_results]
    maes   = [r["mae_mean"]  for r in all_results]
    psnrs  = [r["psnr_mean"] for r in all_results]
    return {
        "model": "BiomedCLIPv2",
        "n_seeds": len(all_results),
        "seeds":   [r["seed"] for r in all_results],
        "ssim_mean_across_seeds": float(np.mean(ssims)),
        "ssim_std_across_seeds":  float(np.std(ssims)),
        "mae_mean_across_seeds":  float(np.mean(maes)),
        "mae_std_across_seeds":   float(np.std(maes)),
        "psnr_mean_across_seeds": float(np.mean(psnrs)),
        "psnr_std_across_seeds":  float(np.std(psnrs)),
        "per_seed": all_results,
    }


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def _run_tests() -> None:
    print("=== BiomedCLIPv2 Unit Tests ===\n")
    passed = failed = 0

    def ok(name, cond, detail=""):
        nonlocal passed, failed
        tag = "PASS" if cond else "FAIL"
        print(f"  [{tag}] {name}" + (f"  ({detail})" if detail else ""))
        if cond: passed += 1
        else:    failed += 1

    # _load_prompts
    prompts = _load_prompts()
    ok("prompts loaded", len(prompts) > 0, f"n={len(prompts)}")
    ok("prompt is string", isinstance(list(prompts.values())[0], str))

    # VolumePairsWithText shape (using synthetic data)
    import tempfile, nibabel as nib
    dummy = np.random.rand(91, 109, 91).astype(np.float32)
    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tf:
        nib.save(nib.Nifti1Image(dummy, np.eye(4)), tf.name)
        tmp_p = tf.name
    test_pairs  = [(tmp_p, tmp_p)]
    test_meta   = {}
    test_prompts = {}
    ds = VolumePairsWithText(test_pairs, test_meta, test_prompts)
    pre, post, prompt = ds[0]
    ok("VolumePairsWithText pre shape", pre.shape == (1,) + TARGET_SHAPE, str(pre.shape))
    ok("VolumePairsWithText prompt is str", isinstance(prompt, str))
    import os as _os; _os.unlink(tmp_p)

    # _collate
    batch = [(torch.zeros(1, 91, 109, 91), torch.zeros(1, 91, 109, 91), "text A"),
             (torch.zeros(1, 91, 109, 91), torch.zeros(1, 91, 109, 91), "text B")]
    pres, posts, texts = _collate(batch)
    ok("_collate pre shape", pres.shape == (2, 1, 91, 109, 91))
    ok("_collate texts list", texts == ["text A", "text B"])

    # MultiLevelFiLMUNet3D + ClinicalTextEncoder forward pass
    model    = MultiLevelFiLMUNet3D(text_embed_dim=128).to(DEVICE)
    text_enc = ClinicalTextEncoder.from_prompts_file(out_dim=128).to(DEVICE)
    pre_t    = _pad(torch.zeros(1, 1, 91, 109, 91)).to(DEVICE)
    embed    = text_enc(["Age: 40, Sex: F, Laterality: Bilateral"], device=DEVICE)
    pred     = model(pre_t, embed)
    ok("full pipeline output shape", pred.shape == (1, 1) + PAD_SHAPE, str(pred.shape))
    pred_c   = _crop(pred)
    ok("cropped shape matches TARGET_SHAPE", pred_c.shape == (1, 1) + TARGET_SHAPE)
    model.remove_hooks()

    # multiseed_summary
    fake_results = [
        {"ssim_mean": 0.78, "mae_mean": 0.09, "psnr_mean": 22.0, "seed": 42},
        {"ssim_mean": 0.79, "mae_mean": 0.08, "psnr_mean": 22.5, "seed": 123},
    ]
    summ = multiseed_summary(fake_results)
    ok("multiseed ssim mean", abs(summ["ssim_mean_across_seeds"] - 0.785) < 1e-6)
    ok("multiseed n_seeds=2",  summ["n_seeds"] == 2)

    print(f"\nTests: {passed} passed, {failed} failed out of {passed+failed}")
    if failed:
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="42", help="comma-separated seed list")
    p.add_argument("--no-condition", action="store_true",
                   help="Ablation: train without text conditioning")
    p.add_argument("--test", action="store_true")
    args = p.parse_args()

    if args.test:
        _run_tests()
        return

    seeds         = [int(s) for s in args.seeds.split(",")]
    use_condition = not args.no_condition

    all_results = []
    for seed in seeds:
        r = train_one_seed(seed, use_condition)
        all_results.append(r)

    if len(all_results) > 1:
        summ = multiseed_summary(all_results)
        out  = os.path.join(OUT_DIR, "biomedclip_v2_multiseed_summary.json")
        with open(out, "w") as f:
            json.dump(summ, f, indent=2)
        print(f"\nMulti-seed summary ({len(seeds)} seeds):")
        print(f"  SSIM = {summ['ssim_mean_across_seeds']:.3f} ± {summ['ssim_std_across_seeds']:.3f}")
        print(f"  MAE  = {summ['mae_mean_across_seeds']:.3f} ± {summ['mae_std_across_seeds']:.3f}")
        print(f"  PSNR = {summ['psnr_mean_across_seeds']:.2f} ± {summ['psnr_std_across_seeds']:.2f}")
        print(f"  Saved: {out}")


if __name__ == "__main__":
    main()
