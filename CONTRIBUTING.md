# Contributing

This repository accompanies a research paper. Contributions are welcome for
bug fixes, reproducibility issues, and documentation improvements.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Before opening a pull request

- Run the test suite: `pytest tests/`. It currently covers config validation
  (`pyproject.toml`, `CITATION.cff`, `LICENSE` presence), metrics correctness
  (SSIM/PSNR sanity and NaN-on-small-mask behavior), missing-data error
  handling (subject-split and region-mask failure modes), a CAE3D
  forward-pass smoke test, the package import, and repo-root detection. Add
  new tests alongside new code rather than treating this as a gap to fill
  later.
- Do not commit data, model checkpoints, or anything under `data/` or
  `results/` other than their `README.md` placeholders — see `.gitignore`.
- Do not introduce hard-coded absolute paths; use the `--data-dir` /
  `--output-dir` style arguments (or environment variables) that the
  existing scripts use.
- Keep new modules under `src/moyamoya_cvr/` and new executable entry
  points under `scripts/`, following the existing action-oriented naming
  (`train.py`, `evaluate.py`, `preprocess.py`, `generate_figures.py`).

## Reporting issues

Please open a GitHub issue with enough detail to reproduce (command run,
environment, and error output). Do not include patient data or any
identifying information in issue reports or attached files.
