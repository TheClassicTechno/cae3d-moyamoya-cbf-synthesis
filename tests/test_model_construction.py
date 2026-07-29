"""Model construction + synthetic forward pass.

Skipped for now: model classes currently live as top-level scripts
(UNet_3D/model_3d.py, Diffusion_ResidualDiffusion_3D/model_3d_with_tips.py,
etc.) rather than importable modules under src/moyamoya_cvr/models/, so
there is nothing stable to import here yet. This is tracked as pending
reorganization work, not silently skipped without a reason.
"""
import pytest


@pytest.mark.skip(reason="model classes not yet importable as package modules (pending src/moyamoya_cvr/models/ migration)")
def test_cae3d_forward_pass_on_synthetic_input():
    pass
