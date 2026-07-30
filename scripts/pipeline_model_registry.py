"""
Canonical model roles for the Week7/8/11 ASL pre→post pipeline.

- Primary reconstruction (best hold-out MAE/PSNR on logged week11 aggregates):
    week7_unet3d  →  manuscript name "CAE3D (ours)"
- Diffusion flagship (near-matching metrics; generative path):
    Residual_3D_tips  →  "Residual_3D_tips" / residual diffusion with TIPS

Import this module in scripts instead of duplicating magic strings or orderings.
"""
from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_REPO_ROOT, "pyproject.toml")):
    _parent = os.path.dirname(_REPO_ROOT)
    if _parent == _REPO_ROOT:
        break
    _REPO_ROOT = _parent

ROOT = Path(_REPO_ROOT)

# --- Roles (code names as in results JSON / per-subject metrics) ---
PRIMARY_RECON_MODEL = "week7_unet3d"
DIFFUSION_FLAGSHIP_MODEL = "Residual_3D_tips"

# --- Checkpoints (may be absent on a fresh clone) ---
# week7_unet3d_best.pt: promoted 2026-03-30 from 50-ep retrain (see week9_stats/CAE3D_primary_checkpoint_record_20260330.md).
PRIMARY_RECON_CHECKPOINT = ROOT / "scripts" / "week7_results" / "week7_unet3d_best.pt"
DIFFUSION_FLAGSHIP_CHECKPOINT = (
    ROOT / "Diffusion_ResidualDiffusion_3D" / "residual_diffusion_3d_tips_week7_best.pt"
)

# --- Manuscript / table labels ---
PAPER_PRIMARY_NAME = "CAE3D (ours)"
PAPER_DIFFUSION_FLAGSHIP_NAME = "Residual_3D_tips (TIPS-stabilized residual diffusion)"

MODEL_DISPLAY_NAME = {
    PRIMARY_RECON_MODEL: f"{PAPER_PRIMARY_NAME} [{PRIMARY_RECON_MODEL}]",
    DIFFUSION_FLAGSHIP_MODEL: f"{PAPER_DIFFUSION_FLAGSHIP_NAME}",
    "UNet_3D": "CAE3D-ES / UNet_3D (MONAI)",
    "week7_resnet3d": "ResNet_3D (scripts)",
    "week7_unet2d": "CAE_2D / week7_unet2d",
    "FNO_3D": "FNO_3D",
    "Patch_3D": "Patch_3D",
    "Hybrid_3D": "Hybrid_3D",
    "Cold_3D": "Cold_3D",
    "DDPM_3D": "DDPM_3D",
    "Residual_3D": "Residual_3D (non-TIPS)",
}

# Order for markdown tables and Holm-style listings (best / flagship first).
REPORTING_MODEL_ORDER = [
    PRIMARY_RECON_MODEL,
    DIFFUSION_FLAGSHIP_MODEL,
    "UNet_3D",
    "FNO_3D",
    "Patch_3D",
    "week7_resnet3d",
    "week7_unet2d",
    "Hybrid_3D",
    "Cold_3D",
    "DDPM_3D",
    "Residual_3D",
]

# Default set for --best6_only style stats (subset with stable per-subject JSONs on test).
CORE_COMPARISON_MODELS = [
    PRIMARY_RECON_MODEL,
    DIFFUSION_FLAGSHIP_MODEL,
    "UNet_3D",
    "FNO_3D",
    "Patch_3D",
    "week7_resnet3d",
    "Cold_3D",
    "DDPM_3D",
]


def sort_models_by_reporting_order(names: list[str]) -> list[str]:
    """Stable sort: known order first, then remaining alphabetically."""
    order = {m: i for i, m in enumerate(REPORTING_MODEL_ORDER)}
    return sorted(names, key=lambda m: (order.get(m, 999), m))
