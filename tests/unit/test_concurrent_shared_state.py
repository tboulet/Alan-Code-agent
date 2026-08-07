"""Regression tests for project-scoped read-modify-write operations."""

import json
import time
from concurrent.futures import ThreadPoolExecutor

import alancode.backends.cw_probe as cw_probe
import alancode.permissions.project_rules as project_rules
from alancode.utils.atomic_io import atomic_write_json


def test_concurrent_context_window_cache_updates_are_not_lost(
    tmp_path, monkeypatch
):
    cache_path = tmp_path / "context_windows.json"
    monkeypatch.setattr(cw_probe, "_cache_file", lambda: cache_path)

    def slow_write(path, data, *, indent=2):
        time.sleep(0.005)
        atomic_write_json(path, data, indent=indent)

    monkeypatch.setattr(cw_probe, "atomic_write_json", slow_write)

    def save(index):
        cw_probe.save_cached_context_window(
            f"model-{index}", "http://local", 32_768 + index, "test"
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(save, range(24)))

    data = json.loads(cache_path.read_text())
    assert len(data) == 24
    for index in range(24):
        assert data[f"http://local|model-{index}"]["context_window"] == 32_768 + index


def test_concurrent_project_allow_rules_are_not_lost(tmp_path, monkeypatch):
    original_save = project_rules.save_project_allow_rules

    def slow_save(rules, cwd=None):
        time.sleep(0.005)
        original_save(rules, cwd)

    monkeypatch.setattr(project_rules, "save_project_allow_rules", slow_save)

    def add(index):
        project_rules.add_project_allow_rule(
            {
                "tool_name": "Bash",
                "rule_content": f"command-{index} *",
                "source": "test",
            },
            str(tmp_path),
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(add, range(24)))

    rules = project_rules.load_project_allow_rules(str(tmp_path))
    assert len(rules) == 24
    assert {rule["rule_content"] for rule in rules} == {
        f"command-{index} *" for index in range(24)
    }
