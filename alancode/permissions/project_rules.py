"""Project-scoped allow-rules store.

Allow rules accepted via "Allow always ..." prompts used to live in a
per-session state file (``.alan/sessions/<id>/state.json``), which meant
every new session started with no rules and the user had to re-approve
each command.  They now live in a single project-scoped file so rules
persist across sessions within the same project.

File: ``<project>/.alan/allow_rules.json``
Format:
    [
        {"tool_name": "Bash", "rule_content": "git *", "source": "session"},
        ...
    ]
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from alancode.settings import get_alan_dir

logger = logging.getLogger(__name__)

_FILENAME = "allow_rules.json"


def _rules_path(cwd: str | None = None) -> Path:
    return get_alan_dir(cwd) / _FILENAME


def load_project_allow_rules(cwd: str | None = None) -> list[dict[str, Any]]:
    """Read ``.alan/allow_rules.json``. Returns ``[]`` if missing or invalid."""
    path = _rules_path(cwd)
    if not path.exists():
        return []
    try:
        with open(path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to read %s: %s. Ignoring.", path, e)
        return []
    if not isinstance(data, list):
        logger.warning("Invalid rules format in %s. Ignoring.", path)
        return []
    return [r for r in data if isinstance(r, dict)]


def save_project_allow_rules(
    rules: list[dict[str, Any]], cwd: str | None = None
) -> None:
    """Write the full list of allow rules to ``.alan/allow_rules.json``."""
    from alancode.utils.atomic_io import atomic_write_json
    path = _rules_path(cwd)
    atomic_write_json(path, rules, indent=2)


def add_project_allow_rule(
    rule: dict[str, Any], cwd: str | None = None
) -> None:
    """Append a single rule and persist. Duplicates (same tool+content) are skipped."""
    existing = load_project_allow_rules(cwd)
    key = (rule.get("tool_name"), rule.get("rule_content"))
    if any((r.get("tool_name"), r.get("rule_content")) == key for r in existing):
        return
    existing.append(rule)
    save_project_allow_rules(existing, cwd)
