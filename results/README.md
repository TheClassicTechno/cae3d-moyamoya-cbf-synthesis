# Results

This directory is where scripts under `scripts/` and
`src/moyamoya_cvr/evaluation/` write their outputs (metrics JSON/CSV,
figures, checkpoints) when run locally. Its contents are not tracked in
git (see `.gitignore`) because they are fully regenerable from the code
plus the private dataset described in `data/README.md`.

## Regenerating published results

Given access to the dataset (see `data/README.md`) and the pinned
environment in `requirements.txt`:

```bash
# 1. Build the subject-level train/val/test split (no standalone
#    preprocessing script — brain masking, MNI resampling, and
#    normalization happen on-the-fly at data-loading time)
python scripts/combined_subject_split.py --out combined_subject_split.json

# 2. Train a model (env-var configured, invoked directly, e.g. CAE3D)
cd UNet_3D && WEEK7=1 SEED=42 python model_3d.py

# 3. Evaluate and aggregate (from the repo root)
python scripts/regional_eval_3d.py --pred-dir <dir with post_*_pred.nii.gz> --out regional_results.json
python scripts/aggregate_week8_seeds.py --results_dir week8_results --output week8_summary.md --ci
python scripts/week8_significance_and_bland_altman.py --aggregate_csv <csv> --output_dir week8_stats
```

See the top-level `README.md`'s Training/Evaluation/"Reproducing main
tables and figures" sections for the full per-model command list and the
paper's reported metrics table.

## Trained checkpoints

Trained model weights are not included in this repository due to size.
They will be released via a separate archive (Zenodo/HuggingFace, link
TBD) upon paper acceptance — see `CITATION.cff` and the top-level
`README.md` "Notes" section.
