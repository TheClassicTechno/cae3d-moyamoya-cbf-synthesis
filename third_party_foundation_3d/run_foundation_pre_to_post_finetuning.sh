#!/usr/bin/env bash
# Fair pre-to-post CBF finetuning: SAM-Med3D CBF regressor, then Med3DVLM CVR.
# Not run_all_foundation_3d.sh (mask/segmentation demos).
#
# Usage:
#   cd /data1/julih && bash third_party_foundation_3d/run_foundation_pre_to_post_finetuning.sh
#   SEEDS=42 bash third_party_foundation_3d/run_foundation_pre_to_post_finetuning.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
SEEDS="${SEEDS:-42,123,456}"
# Avoid broken torch in ~/.local shadowing conda (PEP 668 / mixed installs).
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
PYTHON_BIN="${PYTHON_BIN:-/data1/julih/miniconda3/envs/julih_monai/bin/python}"
export PYTHONPATH="${ROOT}/scripts:${ROOT}/third_party_foundation_3d/Med3DVLM:${ROOT}/third_party_foundation_3d/SAM-Med3D${PYTHONPATH:+:$PYTHONPATH}"

echo "Using PYTHON_BIN=${PYTHON_BIN} PYTHONNOUSERSITE=${PYTHONNOUSERSITE}"

echo "=== 1/2 SAM-Med3D CBF regressor (seeds: ${SEEDS}) ==="
# Decoder-only by default; set SAM_MED3D_FREEZE_ENCODER=0 to train SAM image encoder + decoder.
export SAM_MED3D_FREEZE_ENCODER="${SAM_MED3D_FREEZE_ENCODER:-1}"
"$PYTHON_BIN" "${ROOT}/third_party_foundation_3d/run_sam_med3d_week7_cbf_regressor.py" --seeds "$SEEDS"

echo "=== 2/2 Med3DVLM CVR (seeds: ${SEEDS}) ==="
# Decoder-only by default; set MED3DVLM_FREEZE_ENCODER=0 for full DCFormer + decoder training.
export MED3DVLM_FREEZE_ENCODER="${MED3DVLM_FREEZE_ENCODER:-1}"
"$PYTHON_BIN" "${ROOT}/third_party_foundation_3d/run_med3dvlm_week7_cvr.py" --seeds "$SEEDS"

echo "Done. Outputs: ${ROOT}/third_party_foundation_3d/sam_med3d_week7_cbf/ and med3dvlm_week7_cvr/"
