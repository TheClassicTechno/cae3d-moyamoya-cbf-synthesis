# Medical_Diffusion_Research

### Synthesizing post-acetazolamide cerebral blood flow maps from baseline ASL MRI in Moyamoya disease

Part of CNS Lab. Code accompanying: *"Synthesize Post-Acetazolamide Cerebral Blood Flow Maps from Baseline ASL MRI
in Moyamoya Using a 3D Conditional Autoencoder"* (Machine Learning for Healthcare (MLHC) 2026; see [Citation](#citation)).

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

Both scripts currently default several paths (raw data root, xlsx clinical-score file, output path) to an
absolute path specific to the original development machine; pass `--data-root` / `--xlsx` / `--out` explicitly
if running elsewhere (see [Limitations](#limitations-and-intended-use)).

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

**Important:** several training scripts (including the two above) currently hard-code a data/output root path
specific to the original development machine rather than deriving it from the repository location. As shipped,
these commands only reproduce as written if the repository is checked out at that same path, or if that
constant is edited first — see [Limitations](#limitations-and-intended-use).

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

From the project's own reporting (verified against this repository's README history; see the manuscript for
full methodology): 252 scans (2020–2023), subject-level split 189 train / 31 val / 32 test, MNI 91×109×91,
brain-masked MAE/SSIM/PSNR, three seeds (42, 123, 456) with early stopping for the "reproducible" table below.

2D baseline (middle slice): MAE 0.0497, SSIM 0.7886, PSNR 21.49 dB.

Reproducible (three-seed) best:

| Model | MAE | SSIM | PSNR (dB) |
|-------|-----|------|-----------|
| CAE_3D_s (script 3D) | 0.0689 ± 0.0008 | 0.7971 ± 0.0017 | 23.73 ± 0.09 |
| CAE_3D (external) | 0.0742 ± 0.0039 | 0.7909 ± 0.0039 | 23.10 ± 0.43 |

Extended (single-run) best — full-volume 3D beats the 2D baseline on all three metrics:

| Model | MAE | SSIM | PSNR (dB) |
|-------|-----|------|-----------|
| Residual Diffusion 3D (tips) | 0.0228 | 0.8528 | 26.13 |
| CAE_3D | 0.0253 | 0.8513 | 25.27 |
| FNO 3D | 0.0301 | 0.696 | 25.58 |

Per-region MAE in vascular territories, Bland–Altman limits of agreement, and pairwise significance across
models are also reported; see the manuscript for the full tables.

## Limitations and intended use

- **Not yet portable.** Multiple training, preprocessing, and figure/table scripts hard-code an absolute
  filesystem path from the original development machine (see Training/Evaluation sections above). Running them
  outside that exact path currently requires editing the relevant path constant or passing the applicable CLI
  override where one exists.
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

> Huang, J., Gonzalez, C., Goyal, R., Zou, A., Alexander, S., Moseley, M., Zhao, M.Y., and Steinberg, G.K.
> "Synthesize Post-Acetazolamide Cerebral Blood Flow Maps from Baseline ASL MRI in Moyamoya Using a 3D
> Conditional Autoencoder." Machine Learning for Healthcare (MLHC), 2026.

(TODO: venue proceedings link/DOI to be added once assigned — see `CITATION.cff`.)

## License

Released under the BSD 3-Clause License — see [`LICENSE`](LICENSE).

## Contact

Please open a GitHub issue on this repository for bugs, reproducibility questions, or data-access inquiries
that don't involve sharing identifying information. (TODO: add a maintainer contact email if desired.)

## Acknowledgments

Built with PyTorch, MONAI, and DIPY. The `third_party_foundation_3d/` adapters build on and compare against
Med3DVLM, SAM-Med3D, and STU-Net (nnU-Net-based). Part of CNS Lab.
