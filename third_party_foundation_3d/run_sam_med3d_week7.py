#!/usr/bin/env python3
"""
Run SAM-Med3D inference on one Week7 pre volume using brain mask as prompt GT.
Run from SAM-Med3D repo root with their env (medim, torch, etc.).
"""
import os
import sys
import json
import shutil

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_REPO_ROOT, "pyproject.toml")):
    _parent = os.path.dirname(_REPO_ROOT)
    if _parent == _REPO_ROOT:
        raise RuntimeError("Could not locate repository root (pyproject.toml not found)")
    _REPO_ROOT = _parent

SAM_ROOT = os.path.dirname(os.path.abspath(__file__))
WEEK7_SPLIT = os.path.join(_REPO_ROOT, "combined_subject_split.json")
BRAIN_MASK = os.path.join(_REPO_ROOT, "MNI152_T1_2mm_brain_mask_dil.nii.gz")
OUT_DIR = os.path.join(SAM_ROOT, "test_data", "week7_sam_run")
IMAGES_DIR = os.path.join(OUT_DIR, "imagesVa")
LABELS_DIR = os.path.join(OUT_DIR, "labelsVa")
PRED_DIR = os.path.join(OUT_DIR, "pred")


def main():
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(LABELS_DIR, exist_ok=True)
    os.makedirs(PRED_DIR, exist_ok=True)

    with open(WEEK7_SPLIT) as f:
        data = json.load(f)
    pairs = data.get("train", [])
    if not pairs:
        print("No Week7 train pairs in", WEEK7_SPLIT)
        sys.exit(1)
    pre_path = pairs[0].get("pre_path")
    if not pre_path or not os.path.isfile(pre_path):
        print("First pre_path missing or not a file:", pre_path)
        sys.exit(1)
    if not os.path.isfile(BRAIN_MASK):
        print("Brain mask not found:", BRAIN_MASK)
        sys.exit(1)

    img_dest = os.path.join(IMAGES_DIR, "week7_sample.nii.gz")
    gt_dest = os.path.join(LABELS_DIR, "week7_sample.nii.gz")
    shutil.copy2(pre_path, img_dest)
    shutil.copy2(BRAIN_MASK, gt_dest)
    out_path = os.path.join(PRED_DIR, "week7_sample.nii.gz")

    # When run from run_all_foundation_3d.sh we are in SAM-Med3D/ so cwd has utils/
    # SAM_ROOT is the dir containing this script (and usually SAM-Med3D/).
    sam_dir = os.path.join(SAM_ROOT, "SAM-Med3D")
    if os.path.isdir(sam_dir):
        sys.path.insert(0, sam_dir)
    else:
        sys.path.insert(0, SAM_ROOT)
    try:
        import medim
        from utils.infer_utils import validate_paired_img_gt
        from utils.metric_utils import compute_metrics, print_computed_metrics
    except ImportError as e:
        print("SAM-Med3D import error:", e)
        print("Install: pip install uv && uv pip install torch torchvision torchio medim edt surface-distance monai prefetch_generator")
        print("Then run from SAM-Med3D repo root.")
        sys.exit(1)

    ckpt_path = "https://huggingface.co/blueyo0/SAM-Med3D/blob/main/sam_med3d_turbo.pth"
    ckpt_local = os.path.join(SAM_ROOT, "SAM-Med3D", "ckpt", "sam_med3d_turbo.pth")
    if not os.path.isfile(ckpt_local):
        ckpt_local = os.path.join(SAM_ROOT, "ckpt", "sam_med3d_turbo.pth")
    if os.path.isfile(ckpt_local):
        ckpt_path = ckpt_local
    print("Loading SAM-Med3D (checkpoint:", ckpt_path, ")...")
    model = medim.create_model("SAM-Med3D", pretrained=True, checkpoint_path=ckpt_path)
    print("Running inference on Week7 sample...")
    validate_paired_img_gt(model, img_dest, gt_dest, out_path, num_clicks=1)
    print("Prediction saved to", out_path)
    metrics = compute_metrics(gt_path=gt_dest, pred_path=out_path, metrics=["dice"], classes=None)
    print_computed_metrics(metrics)


if __name__ == "__main__":
    main()
