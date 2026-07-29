#!/bin/bash
# Run 3 seeds (42, 123, 456) for the 5 "best" models that need mean±std (excluding UNet 3D scripts which has K-fold).
# Results go to week8_results/<Model>_seed42.json etc. Then run aggregate_week8_seeds.py.
#
# Usage:
#   ./scripts/run_week9_best5_three_seeds.sh              # full epochs (long)
#   WEEK9_QUICK_EPOCHS=2 ./scripts/run_week9_best5_three_seeds.sh   # 2 epochs per run (pipeline test)

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
cd "$ROOT"
mkdir -p week8_results

PYTHON="python3"
[[ -x "${ROOT}/miniconda3/envs/julih_monai/bin/python3" ]] && PYTHON="${ROOT}/miniconda3/envs/julih_monai/bin/python3"
export PYTHON
export WEEK7=1
unset WEEK7_REGION_WEIGHT
unset WEEK7_SUBJECT_MASKS

MODELS=(UNet_3D Residual_3D_tips Patch_3D Hybrid_3D FNO_3D)
for m in "${MODELS[@]}"; do
  echo "========== $m (3 seeds) =========="
  ./scripts/run_week8_all_seeds.sh "$m" 3
done

echo ""
echo "Done. Aggregate with:"
echo "  python3 scripts/aggregate_week8_seeds.py --results_dir week8_results --output scripts/week9/week8_best5_mean_std.md --ci"
