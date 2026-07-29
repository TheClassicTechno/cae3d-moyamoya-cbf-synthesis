"""Missing-data behavior: scripts must fail informatively (or warn) when the restricted
dataset isn't present, rather than crash uninformatively or silently look successful.

These copy the target script into an isolated temp directory (with its own
pyproject.toml marker) rather than running it in place, because this
repository's actual checkout has the real (restricted) dataset present --
running in place would exercise the "data found" path, not the "data
missing" path this test is meant to cover.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _isolated_copy(*rel_paths):
    tmp = tempfile.mkdtemp()
    open(os.path.join(tmp, "pyproject.toml"), "w").close()  # repo-root marker only
    for rel in rel_paths:
        dst = os.path.join(tmp, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(os.path.join(REPO_ROOT, rel), dst)
    return tmp


def test_combined_subject_split_warns_on_zero_subjects():
    tmp = _isolated_copy("scripts/combined_subject_split.py")
    out_path = os.path.join(tmp, "split.json")
    result = subprocess.run(
        [sys.executable, os.path.join(tmp, "scripts", "combined_subject_split.py"), "--out", out_path],
        cwd=tmp,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "WARNING" in result.stderr and "0 subjects" in result.stderr
    with open(out_path) as f:
        data = json.load(f)
    assert data["n_train"] == 0


def test_render_region_masks_fails_clearly_without_brain_mask():
    tmp = _isolated_copy("scripts/render_region_masks_png.py")
    result = subprocess.run(
        [sys.executable, os.path.join(tmp, "scripts", "render_region_masks_png.py")],
        cwd=tmp,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "MNI152_T1_2mm_brain_mask_dil.nii.gz" in result.stderr
