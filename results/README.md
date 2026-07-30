# Results

This directory is where scripts under `scripts/` and
`src/moyamoya_cvr/evaluation/` write their outputs (metrics JSON/CSV,
figures, checkpoints) when run locally. Its contents are not tracked in
git (see `.gitignore`) because they are fully regenerable from the code
plus the private dataset described in `data/README.md`.

## Regenerating published results

Given access to the dataset (see `data/README.md`) and the pinned
environment in `requirements.txt`, use the actual per-model entry points
documented in the top-level `README.md`, e.g.:

```bash
# CAE3D / UNet_3D, week7 pipeline, seed 42
cd UNet_3D && WEEK7=1 SEED=42 python model_3d.py

# Per-region (vascular territory) evaluation for a model's predictions
python scripts/regional_eval_3d.py --pred-dir <dir with post_*_pred.nii.gz> --out regional_results.json

# Aggregate multi-seed results with bootstrap CIs
python scripts/aggregate_week8_seeds.py --results_dir week8_results --output week8_summary.md --ci
```

There is no single `scripts/preprocess.py` / `train.py` / `evaluate.py` entry point; preprocessing is
applied on-the-fly by `scripts/week7_preprocess.py` / `scripts/week7_data.py`, and each model family has
its own training script (see the top-level README's "Training" and "Reproducing main tables and
figures" sections for the full, verified command list).

## Trained checkpoints

Trained model weights are not included in this repository due to size.
They will be released via a separate archive (Zenodo/HuggingFace, link
TBD) upon paper acceptance — see `CITATION.cff` and the top-level
`README.md` "Notes" section.
