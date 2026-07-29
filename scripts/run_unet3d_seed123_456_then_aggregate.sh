#!/bin/bash
# Run AFTER UNet_3D seed 123 training has finished.
# 1. Copy current unet_3d_results_week7.json to week8_results/UNet_3D_seed123.json
# 2. Run UNet_3D with SEED=456
# 3. Copy result to week8_results/UNet_3D_seed456.json
# 4. Run aggregate and stats
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
ROOT="$(_find_repo_root)"
PYTHON="${ROOT}/miniconda3/envs/julih_monai/bin/python3"
cd "$ROOT"

echo "Step 1: Copy seed 123 result to week8_results..."
cp UNet_3D/unet_3d_results_week7.json week8_results/UNet_3D_seed123.json
echo "  -> week8_results/UNet_3D_seed123.json"

echo "Step 2: Run UNet_3D seed 456..."
cd UNet_3D
export WEEK7=1
unset WEEK7_REGION_WEIGHT
export SEED=456
"$PYTHON" model_3d.py
cd "$ROOT"

echo "Step 3: Copy seed 456 result..."
cp UNet_3D/unet_3d_results_week7.json week8_results/UNet_3D_seed456.json
echo "  -> week8_results/UNet_3D_seed456.json"

echo "Step 4: Aggregate and stats..."
./scripts/week8_aggregate_and_stats.sh

echo "Done. UNet_3D has real 3-seed results; week8_table.md and week8_stats updated."
