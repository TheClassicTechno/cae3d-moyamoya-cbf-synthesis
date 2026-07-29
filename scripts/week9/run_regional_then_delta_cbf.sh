#!/usr/bin/env bash
# Run missing regional evals (Residual_3D, DDPM_3D, Hybrid_3D, Patch_3D) then update ΔCBF.
# Use from repo root: bash scripts/week9/run_regional_then_delta_cbf.sh

set -e
_find_repo_root() {
  local d
  d="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  while [[ ! -f "$d/pyproject.toml" ]]; do
    local parent
    parent="$(dirname "$d")"
    if [[ "$parent" == "$d" ]]; then
      echo "Could not locate repository root (pyproject.toml not found)" >&2
      exit 1
    fi
    d="$parent"
  done
  echo "$d"
}
cd "$(_find_repo_root)"

echo "=== Step 1: Regional eval for Residual_3D ==="
python3 scripts/week9/week9_regional_eval_all_models.py --only Residual_3D

echo "=== Step 2: Regional eval for DDPM_3D ==="
python3 scripts/week9/week9_regional_eval_all_models.py --only DDPM_3D

echo "=== Step 3: Regional eval for Hybrid_3D ==="
python3 scripts/week9/week9_regional_eval_all_models.py --only Hybrid_3D

echo "=== Step 4: Regional eval for Patch_3D ==="
python3 scripts/week9/week9_regional_eval_all_models.py --only Patch_3D

echo "=== Step 5: Update ΔCBF CSV and summary (all week8_regional_*.json) ==="
python3 scripts/week9/week9_delta_cbf_by_territory.py --output_dir week9_stats

echo "=== Done. Check week9_stats/delta_cbf_by_territory.csv and delta_cbf_summary.md ==="
