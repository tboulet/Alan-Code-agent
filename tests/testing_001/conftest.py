"""Proof-of-bug suite (adversarial testing round 001).

Every test here encodes the expected design behaviour and is backed by the
self-contained instruments in this package.

Deselect them from a normal run with:  pytest -m "not known_bug"
"""

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "known_bug: proof test for a documented, not-yet-fixed finding",
    )


def pytest_collection_modifyitems(config, items):
    for item in items:
        if "testing_001" in str(item.fspath):
            item.add_marker(pytest.mark.known_bug)
