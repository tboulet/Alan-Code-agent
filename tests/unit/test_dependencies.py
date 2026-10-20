"""Dependency pins that protect installs."""

import tomllib
from pathlib import Path

from packaging.requirements import Requirement

PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"


def _requirement(name):
    deps = tomllib.loads(PYPROJECT.read_text())["project"]["dependencies"]
    return next(Requirement(d) for d in deps if Requirement(d).name == name)


def test_malicious_litellm_releases_cannot_be_installed():
    """OSV MAL-2026-2144: litellm 1.82.7 and 1.82.8 were compromised releases."""
    spec = _requirement("litellm").specifier
    assert not spec.contains("1.82.7") and not spec.contains("1.82.8")
    assert spec.contains("1.82.6") and spec.contains("1.83.8")
