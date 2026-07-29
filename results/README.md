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
python scripts/preprocess.py --data-dir /path/to/raw --output-dir /path/to/processed
python scripts/train.py --model cae3d --seed 42
python scripts/evaluate.py --model cae3d --checkpoint results/checkpoints/cae3d_best.pt
```

See the top-level `README.md` for the full model list and the paper's
reported metrics table.

## Trained checkpoints

Trained model weights are not included in this repository due to size.
They will be released via a separate archive (Zenodo/HuggingFace, link
TBD) upon paper acceptance — see `CITATION.cff` and the top-level
`README.md` "Notes" section.
