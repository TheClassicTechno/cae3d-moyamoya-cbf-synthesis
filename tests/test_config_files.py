"""Config/metadata loading: pyproject.toml and CITATION.cff must parse and have required fields."""
import os
import tomllib

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_pyproject_toml_valid():
    with open(os.path.join(REPO_ROOT, "pyproject.toml"), "rb") as f:
        data = tomllib.load(f)
    assert data["project"]["name"] == "moyamoya-cvr"
    assert len(data["project"]["dependencies"]) > 0


def test_citation_cff_valid():
    yaml = pytest.importorskip("yaml", reason="pyyaml is a dev-only test dependency (pip install -e '.[dev]')")
    with open(os.path.join(REPO_ROOT, "CITATION.cff")) as f:
        data = yaml.safe_load(f)
    for required in ("cff-version", "message", "title", "authors"):
        assert required in data


def test_license_present():
    assert os.path.isfile(os.path.join(REPO_ROOT, "LICENSE"))
