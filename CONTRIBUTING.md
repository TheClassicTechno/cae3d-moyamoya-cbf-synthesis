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

- Run the test suite: `pytest tests/` (or `pytest -q` from the repo root; a `pytest.ini` is provided).
  It requires no private data and no GPU.
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
