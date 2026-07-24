"""Proof-of-bug suite (adversarial testing round 001).

Every test here encodes the EXPECTED (design-intent) behaviour and FAILS
on the current code - see perso_dev/testing_001/FINDINGS.md for the
finding each test proves. They pass once the corresponding bug is fixed.

Deselect them from a normal run with:  pytest -m "not known_bug"
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(
    0, str(Path(__file__).parent.parent.parent / "perso_dev" / "testing_001")
)


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "known_bug: proof test for a documented, not-yet-fixed finding",
    )


def pytest_collection_modifyitems(config, items):
    for item in items:
        if "testing_001" in str(item.fspath):
            item.add_marker(pytest.mark.known_bug)
