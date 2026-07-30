#!/usr/bin/env bash
# Queue: SAM-Med3D CBF with trainable encoder (enc_train), then STU-Net pre→post regression.
# Seeds: 42, 123, 456 for each. Same PYTHONNOUSERSITE / conda python as foundation wrapper.
#
# Usage:
#   cd /data1/julih && nohup bash third_party_foundation_3d/run_sam_enc_train_then_stunet_prepost.sh \
#     > week9_stats/foundation_sam_enc_train_stunet_prepost_nohup.out 2>&1 &
#
# Logs (tee from this script):
#   week9_stats/foundation_sam_enc_train_YYYYMMDD.log
#   week9_stats/foundation_stunet_prepost_YYYYMMDD.log
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
DAY="$(date +%Y%m%d)"
LOG_SAM="${ROOT}/week9_stats/foundation_sam_enc_train_${DAY}.log"
LOG_STU="${ROOT}/week9_stats/foundation_stunet_prepost_${DAY}.log"
SEEDS="${SEEDS:-42,123,456}"

export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
PYTHON_BIN="${PYTHON_BIN:-/data1/julih/miniconda3/envs/julih_monai/bin/python}"

echo "=== $(date -Is) START sam_enc_train_then_stunet ==="
echo "ROOT=${ROOT} PYTHON_BIN=${PYTHON_BIN} SEEDS=${SEEDS}"

echo ""
echo "=== 1/2 SAM-Med3D CBF enc_train (SAM_MED3D_FREEZE_ENCODER=0) ===" | tee -a "$LOG_SAM"
export SAM_MED3D_FREEZE_ENCODER=0
export PYTHONPATH="${ROOT}/scripts:${ROOT}/third_party_foundation_3d/SAM-Med3D${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON_BIN" "${ROOT}/third_party_foundation_3d/run_sam_med3d_week7_cbf_regressor.py" --seeds "$SEEDS" 2>&1 | tee -a "$LOG_SAM"

echo ""
echo "=== 2/2 STU-Net pre→post (run_stunet_week7_prepost_regressor.py) ===" | tee -a "$LOG_STU"
unset SAM_MED3D_FREEZE_ENCODER
export PYTHONPATH="${ROOT}/scripts:${ROOT}/third_party_foundation_3d/STU-Net-unimedical/nnUNet-1.7.1${PYTHONPATH:+:$PYTHONPATH}"
# Optional: export STUNET_PRETRAINED_WEIGHTS=/path/to/base_ep4k.model
"$PYTHON_BIN" "${ROOT}/third_party_foundation_3d/run_stunet_week7_prepost_regressor.py" --seeds "$SEEDS" 2>&1 | tee -a "$LOG_STU"

echo "=== $(date -Is) DONE ==="
echo "SAM log: $LOG_SAM"
echo "STU-Net log: $LOG_STU"
echo "SAM aggregate: ${ROOT}/third_party_foundation_3d/sam_med3d_week7_cbf/sam_med3d_week7_cbf_results_enc_train_multiseed.json"
echo "STU-Net aggregate: ${ROOT}/third_party_foundation_3d/stunet_week7_prepost_regressor/stunet_week7_prepost_results_*_multiseed.json"
