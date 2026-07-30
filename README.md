# Medical_Diffusion_Research

### Synthesizing post-acetazolamide cerebral blood flow maps from baseline ASL MRI in Moyamoya disease

Part of CNS Lab. Code accompanying: *"Synthesizing Post-Acetazolamide Cerebral Blood Flow Maps from Baseline MRI
in Moyamoya Using 3D Generative AI"* (Machine Learning for Healthcare (MLHC) 2026; see [Citation](#citation)).

## Overview

This repository trains models to predict a post-acetazolamide (ACZ) cerebral blood flow (CBF) map from a
baseline (pre-ACZ) CBF map, so that, in principle, a single acquisition could support a CVR-style (cerebrovascular
reactivity) assessment without an acetazolamide challenge. The codebase implements a unified pipeline — same
subject-level data split, same MNI-space preprocessing, same brain-masked metrics — for a range of 2D and 3D
model families (a 3D conditional autoencoder, DDPM/Cold/Residual diffusion variants, a Fourier Neural Operator,
a hybrid UNet-diffusion model, and adapters for two pretrained 3D foundation models), so that reported
differences reflect the model rather than the evaluation protocol.

## Clinical and scientific motivation

Acetazolamide (Diamox) challenge is used to probe cerebrovascular reactivity in Moyamoya disease, but the
challenge itself carries procedural burden and risk. This project investigates whether the post-ACZ CBF map can
be synthesized from the pre-ACZ (baseline) scan alone, which — if validated — could reduce reliance on the
pharmacological challenge for this specific assessment. This is a research investigation into feasibility, not a
validated clinical tool (see [Responsible use and disclaimer](#responsible-use-and-medical-disclaimer)).

## Main contributions

- A single, unified 2D/3D pipeline (same split, same MNI152 preprocessing, same in-brain metrics) used to
  compare a 3D conditional autoencoder (CAE3D), 2D/3D DDPM, Cold Diffusion, Residual Diffusion, a 3D Fourier
  Neural Operator, and a hybrid UNet-diffusion model.
- Adapters wiring two pretrained 3D foundation models (Med3DVLM, SAM-Med3D) into the same evaluation protocol
  as fixed-baseline comparisons (see [Limitations](#limitations-and-intended-use) for what is and is not
  comparable about these).
- Per-region (vascular territory) error analysis, Bland–Altman agreement analysis, and pairwise significance
  testing across models, in addition to the global MAE/SSIM/PSNR metrics.

## Repository structure

```
.
├── README.md                     This file
├── LICENSE                       BSD-3-Clause
├── CITATION.cff                  Citation metadata
├── CONTRIBUTING.md               Development setup and contribution guidelines
├── pyproject.toml / requirements.txt   Dependencies (pinned to the environment used for reported results)
├── data/README.md                Data availability and access — no data is stored in this repository
├── results/README.md             Where local runs write outputs — no results are stored in this repository
├── src/moyamoya_cvr/             Installable package (in progress: currently a package placeholder;
│                                  code below is being migrated here — see CONTRIBUTING.md)
├── UNet_3D/                      CAE3D / 3D conditional autoencoder (proposed model)
├── Diffusion_baseline/           2D DDPM baseline
├── Diffusion_baseline_3D/        3D DDPM baseline, plus a "with_tips" variant and NIfTI export/inference utilities
├── Diffusion_ColdDiffusion/      2D Cold Diffusion baseline
├── Diffusion_ColdDiffusion_3D/   3D Cold Diffusion baseline, plus improved/"with_tips" variants
├── Diffusion_ResidualDiffusion/  2D Residual Diffusion baseline
├── Diffusion_ResidualDiffusion_3D/  3D Residual Diffusion, plus a "with_tips" variant (best single-run model, see Key results)
├── Diffusion_3D_PatchVolume/     Patch-based 3D diffusion (Patch_3D) and its VAE component
├── Diffusion_3D_Latent/          Exploratory latent-space diffusion variants (cold diffusion latent, I2SB, DDIM sampler);
│                                  not part of the headline results table
├── Diffusion_MAISI/              Conditional training on top of MONAI's MAISI diffusion U-Net; exploratory
├── Diffusion_Option1/            Exploratory 2D conditional DDPM trained from scratch (slice-level)
├── Diffusion_Option2/            Exploratory 2D conditional DDPM fine-tuned from a pretrained MONAI DiffusionModelUNet
├── Hybrid_UNet_Diffusion/        Hybrid_3D: UNet + diffusion hybrid model
├── NeuralOperators/              FNO_3D: 3D Fourier Neural Operator, plus a spectral variant and slice-finetuning script
├── UNet_baseline/                Diff/visualization script for a 2D UNet baseline (no model definition currently tracked here)
├── third_party_foundation_3d/    Adapters and comparison scripts for Med3DVLM, SAM-Med3D, and STU-Net (nnU-Net-based)
└── scripts/                      Shared data pipeline (splits, preprocessing, dataset classes), training entry
                                   points (`week7_train_*.py`), and evaluation/statistics/aggregation scripts
                                   (`week8_*`, `week9/`)
```

`mlhc_code_release/` (present in the working tree but not tracked in git) is a superseded snapshot and is not
part of this repository; do not treat it as authoritative.

## Installation

```bash
git clone <this-repository-url>
cd Medical_Diffusion_Research
pip install -r requirements.txt
# or, for an editable install of the moyamoya_cvr package scaffold:
pip install -e .
```

Dependency versions in `requirements.txt` / `pyproject.toml` are pinned to the environment actually used to
produce the reported results (Python 3.12; PyTorch 2.10, MONAI 1.5.2, DIPY 1.11.0 — see `requirements.txt` for
the full pinned list). Two packages used by some evaluation/statistics scripts (`statsmodels`, `natsort`) are
listed unpinned because a pinned version could not be confirmed against the training environment; verify the
installed version before relying on exact numerical reproduction of anything that depends on them.

The Med3DVLM / SAM-Med3D adapters under `third_party_foundation_3d/` require those projects' own
environments/dependencies (e.g. SAM-Med3D's script is documented to be run "from the SAM-Med3D repo root with
their env"); they are not covered by `requirements.txt`.

## Data availability

**No patient data is included in this repository.** The dataset is a clinical ASL-CBF cohort (pre- and
post-acetazolamide CBF maps) collected under an IRB-approved protocol and cannot be redistributed publicly. See
[`data/README.md`](data/README.md) for how authorized researchers can request access and the expected on-disk
layout that the code assumes.

## Preprocessing

There is no standalone "run once, save processed volumes to disk" preprocessing script in this repository.
Preprocessing (skull-stripping/brain-mask application, resampling to MNI152 91×109×91, padding to 96×112×96,
per-volume min-max normalization to [0,1]) is applied on-the-fly at data-loading time by
`scripts/week7_preprocess.py` and `scripts/week7_data.py`, which are imported by the training scripts below.

Given access to the restricted raw data (see Data availability), subject-level train/val/test splits are built
with:

```bash
python scripts/combined_subject_split.py --out combined_subject_split.json
# or, for the 2020 single-delay cohort specifically:
python scripts/data_2020_single_delay.py --out 2020_single_delay_split.json
```

Both scripts default several paths (raw data root, xlsx clinical-score file, output path) to locations
under the detected repository root; pass `--data-root` / `--xlsx` / `--out` explicitly if your data lives
elsewhere.

## Training

Training scripts are invoked directly (no subcommand framework) and are primarily configured through
environment variables rather than CLI flags. Verified examples from the repository's own run scripts:

```bash
# CAE3D / UNet_3D, week7 pipeline, seed 42 (default)
cd UNet_3D && WEEK7=1 SEED=42 python model_3d.py

# 3D Residual Diffusion ("with tips" variant)
cd Diffusion_ResidualDiffusion_3D && WEEK7=1 SEED=42 python model_3d_with_tips.py

# 3D Fourier Neural Operator
cd NeuralOperators && WEEK7=1 SEED=42 python fno_3d_finetune_slice_brain.py --week7
```

Common environment variables (not all scripts support all of these — check each script's module docstring):
`SEED` (default 42), `WEEK7_EPOCHS` (default 50 where used), `WEEK7_EVAL_ONLY=1` (skip training, evaluate an
existing checkpoint), `WEEK7_REGION_WEIGHT=1` (vascular-region-weighted loss variant, where implemented).

The training scripts above derive their data/output root from the detected repository location rather
than a hard-coded path, so these commands reproduce as written regardless of where the repository is
checked out.

## Evaluation

```bash
# Per-region (vascular territory) evaluation for a model's predictions
python scripts/regional_eval_3d.py --pred-dir <dir with post_*_pred.nii.gz> --out regional_results.json

# Aggregate multi-seed results with bootstrap CIs
python scripts/aggregate_week8_seeds.py --results_dir week8_results --output week8_summary.md --ci

# Significance testing and Bland-Altman analysis
python scripts/week8_significance_and_bland_altman.py --aggregate_csv <csv> --output_dir week8_stats
```

## Reproducing main tables and figures

```bash
# Combined 2D+3D results table (writes WEEK7_TABLE_RESULTS.md)
python scripts/build_final_week7_table.py

# Qualitative comparison figures (pre / post / predicted / brain mask, one test subject)
python scripts/export_qualitative_figures.py --checkpoint <path/to/checkpoint.pt> --index 0 --out figures/

# Vascular territory mask overlays
python scripts/render_region_masks_png.py
```

As with training, several of these scripts hard-code the original development machine's absolute path; see
[Limitations](#limitations-and-intended-use).

## Expected outputs

Training scripts write a checkpoint (`*.pt`, e.g. `unet_3d_week7_best.pt`) and a metrics JSON (e.g.
`unet_3d_results_week7.json`, `fno_3d_week7_best_results.json`) containing test-set MAE/SSIM/PSNR (mean ± std
across seeds where applicable). Aggregation scripts under `scripts/` and `scripts/week9/` combine these into
Markdown/CSV summary tables and CSVs (e.g. per-territory ΔCBF tables, Bland–Altman tables). None of these
outputs are stored in this repository — see [`results/README.md`](results/README.md).

## Model checkpoints

**No trained model checkpoints are included in this repository.** See [`results/README.md`](results/README.md).
Checkpoint release, if any, is expected to accompany paper acceptance; no distribution link is available yet
(TODO: add link once published).

## Hardware and computational requirements

Formal hardware requirements are not documented in this repository. Scripts detect and use a CUDA GPU when
available (`torch.cuda.is_available()`), falling back to CPU otherwise; CPU execution on full 3D volumes is
untested and likely impractical. The only in-code hardware note found is a comment in
`Diffusion_baseline_3D/diffusion_model_3d_with_tips.py` reducing patch size and batch size "to fit [a] 47GB
GPU" for that specific script — this is not a repository-wide requirement, and no training-time estimates are
documented anywhere in the tracked code.

## Key results

From the MLHC 2026 manuscript (see the paper for full methodology): 252 subject-level pairs identified
(2020–2023); one excluded after failed affine registration, giving a final cohort of 251 subjects (188 train /
31 val / 32 held-out test), MNI152 91×109×91, brain-masked MAE/SSIM/PSNR.

Held-out test set, primary comparison across all eleven models (mean ± half-width of bootstrap 95% CI; one
representative trained instance per in-house model, mean ± std over seeds for the two foundation adapters):

| Model | MAE ↓ | SSIM ↑ | PSNR (dB) ↑ | R² |
|-------|-------|--------|--------------|-----|
| **CAE3D (ours)** | **0.066 ± 0.001** | **0.80 ± 0.01** | **24.0 ± 0.1** | 0.35 |
| FNO_3D | 0.072 ± 0.016 | 0.78 ± 0.06 | 24.0 ± 1.3 | 0.47 |
| ResNet_3D | 0.072 ± 0.008 | 0.80 ± 0.04 | 23.2 ± 0.8 | 0.32 |
| Patch_3D | 0.078 ± 0.016 | 0.71 ± 0.06 | 22.9 ± 1.4 | 0.28 |
| CAE_2D | 0.081 ± 0.015 | 0.68 ± 0.09 | 20.8 ± 1.2 | 0.58 |
| Hybrid_3D | 0.120 ± 0.013 | 0.59 ± 0.04 | 18.8 ± 0.7 | −0.03 |
| Cold_3D | 0.179 ± 0.011 | 0.37 ± 0.02 | 14.6 ± 0.3 | 0.31 |
| Residual_3D | 0.250 ± 0.028 | 0.31 ± 0.02 | 13.3 ± 0.95 | −7.44 |
| DDPM_3D | 0.749 ± 0.027 | 0.04 ± 0.00 | 1.12 ± 0.11 | −68.64 |
| SAM-Med3D † | 0.083 ± 0.001 | 0.701 ± 0.004 | 22.20 ± 0.03 | 0.45 |
| Med3DVLM † | 0.098 ± 0.003 | 0.418 ± 0.035 | 15.98 ± 1.86 | 0.46 |

† Foundation adapters (Med3DVLM, SAM-Med3D) use a different evaluation protocol and are not directly comparable
to the in-house rows above.

Seed-to-seed stability (mean ± std over three independent training seeds), fixed held-out test set:

| Model | MAE ↓ | SSIM ↑ | PSNR (dB) ↑ |
|-------|-------|--------|--------------|
| **CAE3D (ours)** | **0.0663 ± 0.0008** | **0.7986 ± 0.0011** | **24.00 ± 0.08** |
| Residual_3D_tips | 0.0675 ± 0.0000 | 0.7885 ± 0.0000 | 23.98 ± 0.00 |
| CAE3D-ES | 0.0722 ± 0.0013 | 0.7933 ± 0.0009 | 23.31 ± 0.15 |
| Patch_3D | 0.0721 ± 0.0000 | 0.7235 ± 0.0000 | 23.32 ± 0.00 |
| FNO_3D | 0.0725 ± 0.0001 | 0.7744 ± 0.0003 | 23.87 ± 0.02 |
| ResNet_3D | 0.0767 ± 0.0008 | 0.7233 ± 0.0056 | 22.32 ± 0.08 |
| Hybrid_3D | 0.1038 ± 0.0035 | 0.6255 ± 0.0160 | 19.54 ± 0.26 |
| Cold_3D | 0.2476 ± 0.0173 | 0.2707 ± 0.0390 | 12.27 ± 0.33 |

CAE3D achieves the lowest held-out MAE among all eleven models, including the two foundation adapters, and its
MAE advantage over the other trained-from-scratch baselines is statistically significant (paired Wilcoxon,
Holm-adjusted). Full-volume 3D (CAE3D) outperforms the 2D middle-slice baseline (CAE_2D) on all three metrics,
though the two are evaluated on different spatial domains and this comparison is supportive rather than a
controlled dimensionality ablation. Per-region ΔCBF by vascular territory, Bland–Altman limits of agreement, and
pairwise significance across models are also reported; see the manuscript for the full tables.

## Limitations and intended use

- **Portability.** Most training, preprocessing, and figure/table scripts detect the repository root
  dynamically rather than hard-coding a path, and are checked in `tests/test_repo_root_detection.py`; this
  was previously a broader issue and has largely been fixed. Some scripts still accept a data/output root
  via a CLI override or environment variable rather than deriving it automatically — pass `--data-dir` /
  `--output-dir` (or the applicable env var) where offered.
- **Single-institution cohort.** Models are trained and evaluated on one clinical Moyamoya cohort; performance
  on other scanners, protocols, or populations is not established.
- **Foundation-model baselines have different comparability caveats**, per their own script documentation:
  the STU-Net script runs off-the-shelf segmentation (Dice/IoU vs. a brain mask) rather than pre→post CBF
  regression, and is explicitly documented in-repo as not comparable to the CAE3D/diffusion reconstruction
  metrics. The SAM-Med3D adapter requires a separate environment (the SAM-Med3D project's own).
- **Several directories are exploratory and not part of the headline results**: `Diffusion_3D_Latent/`,
  `Diffusion_Option1/`, `Diffusion_Option2/`, and `Diffusion_MAISI/`.
- **`src/moyamoya_cvr/` is a partially-completed package migration**, not yet the primary way to run this code.

## Responsible use and medical disclaimer

This is research software for a feasibility investigation. It is **not a validated clinical or diagnostic
tool**, has not undergone prospective clinical validation, and is not cleared or approved by any regulatory
body. Outputs must not be used for clinical decision-making, diagnosis, or patient management. Any clinical
translation would require independent validation on appropriate cohorts and regulatory review.

## Citation

See [`CITATION.cff`](CITATION.cff). Preferred citation:

> Huang, J. et al. "Synthesize Post-Acetazolamide Cerebral Blood Flow Maps from Baseline ASL MRI in Moyamoya
> Using a 3D Conditional Autoencoder." Machine Learning for Healthcare (MLHC), 2026.

(TODO: full author list, venue proceedings link/DOI to be added once assigned — see `CITATION.cff`.)

## License

Released under the BSD 3-Clause License — see [`LICENSE`](LICENSE).

## Contact

Please open a GitHub issue on this repository for bugs, reproducibility questions, or data-access inquiries
that don't involve sharing identifying information. (TODO: add a maintainer contact email if desired.)

## Acknowledgments

Built with PyTorch, MONAI, and DIPY. The `third_party_foundation_3d/` adapters build on and compare against
Med3DVLM, SAM-Med3D, and STU-Net (nnU-Net-based). Part of CNS Lab.
