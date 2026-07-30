#!/bin/bash
# Legacy demo: SAM sample run + STU-Net nnUNet predict (segmentation), NOT pre→post CBF finetuning.
#
# For fair CAE3D comparison, train foundation backbones on pre→post instead:
#   third_party_foundation_3d/run_med3dvlm_week7_cvr.py
#   third_party_foundation_3d/run_sam_med3d_week7_cbf_regressor.py
# See: third_party_foundation_3d/FOUNDATION_PRE_TO_POST_FINETUNING.md

set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

echo "========== 1. SAM-Med3D on Week7 =========="
PYTHON="${PYTHON:-python3}"
if [ -n "$CONDA_PREFIX" ]; then
  PYTHON="$CONDA_PREFIX/bin/python"
fi
if [ -d "SAM-Med3D" ]; then
  cd SAM-Med3D
  $PYTHON "$ROOT/run_sam_med3d_week7.py" 2>&1 || {
    echo "SAM-Med3D run failed. Try: conda activate julih_monai && pip install medim torchio edt surface-distance && bash run_all_foundation_3d.sh"
  }
  cd "$ROOT"
else
  echo "SAM-Med3D not found at $ROOT/SAM-Med3D. Clone with: git clone https://github.com/uni-medical/SAM-Med3D.git"
fi

echo ""
echo "========== 2. STU-Net on Week7 =========="
# Use local stunet_results if RESULTS_FOLDER not set (we set up checkpoint there)
export RESULTS_FOLDER="${RESULTS_FOLDER:-$ROOT/stunet_results}"
export nnUNet_raw_data_base="${nnUNet_raw_data_base:-$ROOT/nnunet_data/raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-$ROOT/nnunet_data/preprocessed}"
mkdir -p "$nnUNet_raw_data_base" "$nnUNet_preprocessed"
if [ -f "$ROOT/stunet_results/nnUNet/3d_fullres/Task101_TotalSegmentator/STUNetTrainer_base__nnUNetPlansv2.1/fold_0/base_ep4k.model" ]; then
  bash "$ROOT/run_stunet_week7.sh" 2>&1 || echo "STU-Net predict failed."
else
  echo "STU-Net skipped: checkpoint not found at stunet_results/.../base_ep4k.model (run README_RUN.md steps to download)."
fi

echo ""
echo "Done. See README_RUN.md for install and manual steps."
