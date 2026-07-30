#!/bin/bash
# Run STU-Net inference on one Week7 pre volume.
# Prerequisites:
#   1. Install nnU-Net with STU-Net: cd third_party_foundation_3d/STU-Net-unimedical/nnUNet-1.7.1 && pip install -e .
#   2. Set env: export nnUNet_raw_data_base=... nnUNet_preprocessed=... RESULTS_FOLDER=...
#   3. Download a STU-Net checkpoint (e.g. base_ep4k) from openmedlab/STU-Net README and place in:
#      RESULTS_FOLDER/nnUNet/3d_fullres/Task101_TotalSegmentator/STUNetTrainer_base__nnUNetPlansv2.1/fold_0/
#      as base_ep4k.model and base_ep4k.model.pkl (and plans.pkl from STU-Net plan_files).
#   4. Copy plan_files/*.pkl into that trainer folder if needed.
#
# This script runs nnUNet_predict on the first Week7 pre volume (copied to a temp input dir).

set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
WEEK7_SPLIT="${WEEK7_SPLIT:-/data1/julih/combined_subject_split.json}"
PRE_PATH=$(python3 -c "
import json, sys
with open('$WEEK7_SPLIT') as f: d = json.load(f)
for p in d.get('train',[]):
    pre = p.get('pre_path')
    if pre:
        print(pre)
        break
" 2>/dev/null)
if [ -z "$PRE_PATH" ] || [ ! -f "$PRE_PATH" ]; then
  echo "No Week7 pre path found. Set WEEK7_SPLIT or check $WEEK7_SPLIT"
  exit 1
fi

INPUT_DIR="$ROOT/stunet_week7_input"
OUTPUT_DIR="$ROOT/stunet_week7_output"
rm -rf "$INPUT_DIR" "$OUTPUT_DIR"
mkdir -p "$INPUT_DIR" "$OUTPUT_DIR"
# nnUNet expects CASENAME_0000.nii.gz (modality 0000)
cp "$PRE_PATH" "$INPUT_DIR/week7_pre_0000.nii.gz"

if [ -z "$RESULTS_FOLDER" ]; then
  echo "RESULTS_FOLDER not set. Set it to your nnUNet RESULTS_FOLDER (with STU-Net checkpoint)."
  exit 1
fi

# nnUNet_predict -t 101 is Task101_TotalSegmentator
NNUNET_PYTHON="${NNUNET_PYTHON:-/data1/julih/miniconda3/envs/julih_monai/bin/python3}"
echo "Running nnUNet_predict (STU-Net base) on Week7 sample..."
$NNUNET_PYTHON -m nnunet.inference.predict_simple -i "$INPUT_DIR" -o "$OUTPUT_DIR" -t 101 -m 3d_fullres -f 0 -tr STUNetTrainer_base -chk base_ep4k --mode fast --disable_tta 2>&1 || true
echo "If prediction succeeded, check: $OUTPUT_DIR"
