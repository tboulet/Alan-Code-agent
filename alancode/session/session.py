"""Session management — finding, creating, and resolving sessions."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)

# Fields that should NOT be written to session settings (ephemeral / per-invocation only)
_EPHEMERAL_FIELDS = {"api_key"}


# ── Path helpers ──────────────────────────────────────────────────────────────


def get_sessions_dir(cwd: str) -> Path:
    """Return ``.alan/sessions/`` for the given working directory."""
    return Path(cwd) / ".alan" / "sessions"


def get_session_dir(cwd: str, session_id: str) -> Path:
    """Return ``.alan/sessions/<session_id>/``."""
    return get_sessions_dir(cwd) / session_id


def generate_session_id() -> str:
    """Generate a new random session ID."""
    return uuid4().hex


# ── Session lookup ────────────────────────────────────────────────────────────


def find_session_by_prefix(cwd: str, prefix: str) -> str | None:
    """Find a session ID by prefix (min 3 chars). Returns None if 0 or >1 matches."""
    if len(prefix) < 3:
        return None
    sessions_dir = get_sessions_dir(cwd)
    if not sessions_dir.exists():
        return None
    matches = [d.name for d in sessions_dir.iterdir() if d.is_dir() and d.name.startswith(prefix)]
    return matches[0] if len(matches) == 1 else None


def get_last_session_id(cwd: str) -> str | None:
    """Find the most recent session ID by transcript modification time.

    Scans ``.alan/sessions/*/transcript.jsonl`` under *cwd*.

    Returns ``None`` if no matching session files exist.
    """
    sessions_dir = get_sessions_dir(cwd)
    if not sessions_dir.is_dir():
        return None

    # Collect all session dirs that have a transcript.jsonl
    candidates: list[tuple[str, float]] = []
    for session_dir in sessions_dir.iterdir():
        if not session_dir.is_dir():
            continue
        transcript = session_dir / "transcript.jsonl"
        if transcript.is_file():
            candidates.append((session_dir.name, transcript.stat().st_mtime))

    if not candidates:
        return None

    # Sort by mtime descending
    candidates.sort(key=lambda x: x[1], reverse=True)

    norm_cwd = os.path.normpath(cwd)
    for session_id, _ in candidates:
        transcript = sessions_dir / session_id / "transcript.jsonl"
        try:
            with open(transcript, "r", encoding="utf-8") as fh:
                first_line = fh.readline().strip()
            if not first_line:
                continue
            d = json.loads(first_line)
            meta = d.get("_metadata")
            if meta is None:
                continue
            session_cwd = os.path.normpath(meta.get("cwd", ""))
            if session_cwd == norm_cwd:
                return session_id
        except (json.JSONDecodeError, OSError, KeyError) as exc:
            # A corrupt transcript file would silently hide that session
            # from the listing and the user would see "no sessions" when
            # they know they have some. Log so they can find it.
            logger.warning(
                "Skipping session %s: transcript unreadable (%s)",
                session_id, exc,
            )
            continue

    return None


# ── Session settings ──────────────────────────────────────────────────────────


def get_session_settings_path(cwd: str, session_id: str) -> Path:
    """Return ``.alan/sessions/<session_id>/settings.json``."""
    return get_session_dir(cwd, session_id) / "settings.json"


def load_session_settings(cwd: str, session_id: str) -> dict[str, Any]:
    """Load settings for a specific session. Returns empty dict if not found."""
    path = get_session_settings_path(cwd, session_id)
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def save_session_settings(cwd: str, session_id: str, settings: dict[str, Any]) -> None:
    """Save settings snapshot for a session atomically."""
    from alancode.utils.atomic_io import atomic_write_json
    path = get_session_settings_path(cwd, session_id)
    to_write = {k: v for k, v in settings.items() if k not in _EPHEMERAL_FIELDS}
    atomic_write_json(path, to_write, indent=2)



# Note: Session state persistence (turn_count, cost, allow_rules, etc.)
# is handled by SessionState in session/state.py (disk-attached properties).
