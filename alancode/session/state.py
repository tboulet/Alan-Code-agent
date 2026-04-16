"""Disk-attached session state.

SessionState is the single source of truth for all persistent session data.
Every property read comes from an in-memory cache; every property write
flushes the cache to ``state.json`` on disk.  This guarantees crash-resilient
state without explicit save/restore calls.

Use :meth:`batch` to group multiple updates into a single disk write::

    with session.batch():
        session.turn_count += 1
        session.total_cost_usd += delta
"""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from alancode.utils.atomic_io import atomic_write_json

logger = logging.getLogger(__name__)


def _get_session_state_path(cwd: str, session_id: str) -> Path:
    """Return ``.alan/sessions/<session_id>/state.json``."""
    return Path(cwd) / ".alan" / "sessions" / session_id / "state.json"


class SessionState:
    """Disk-attached session state.

    Only ``session_id`` and ``cwd`` are plain attributes.  All other
    fields are properties backed by an in-memory cache that is flushed
    to ``.alan/sessions/<id>/state.json`` on every write.
    """

    def __init__(self, session_id: str, cwd: str) -> None:
        """Open (or create) the session state file and load its cache.

        Creates ``.alan/sessions/<session_id>/`` if it doesn't exist.
        Every property read comes from the in-memory ``_cache``; every
        setter calls ``_flush()`` to persist atomically.

        Args:
            session_id: Hex session identifier.
            cwd: Project working directory — `.alan/` lives here.
        """
        self.session_id = session_id
        self.cwd = cwd
        self._state_path = _get_session_state_path(cwd, session_id)
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, Any] = self._load_from_disk()
        self._batch_depth: int = 0

    # ── Disk I/O ──────────────────────────────────────────────────────────

    def _load_from_disk(self) -> dict[str, Any]:
        """Read ``state.json`` into a dict, or return ``{}`` on miss/error.

        Corrupt files are logged at WARNING and treated as empty — we
        prefer losing session state over crashing session start. The
        atomic-write discipline in ``_flush`` makes partial-write
        corruption vanishingly rare anyway.

        Returns:
            The parsed JSON object, or ``{}`` if the file is missing,
            unreadable, or not a dict.
        """
        if not self._state_path.exists():
            return {}
        try:
            with open(self._state_path) as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load session state %s: %s", self._state_path, e)
            return {}

    def _flush(self) -> None:
        """Write the cache to disk atomically (skipped inside a batch)."""
        if self._batch_depth > 0:
            return
        try:
            atomic_write_json(self._state_path, self._cache, indent=2)
        except OSError as e:
            logger.warning("Failed to write session state %s: %s", self._state_path, e)

    @contextmanager
    def batch(self):
        """Group multiple updates into a single disk write.

        Example::

            with session.batch():
                session.total_input_tokens += 100
                session.total_output_tokens += 50
                session.total_cost_usd += 0.01
            # single flush here
        """
        self._batch_depth += 1
        try:
            yield
        finally:
            self._batch_depth -= 1
            if self._batch_depth == 0:
                self._flush()

    # ── Internal helpers ──────────────────────────────────────────────────

    def _get(self, key: str, default: Any = None) -> Any:
        return self._cache.get(key, default)

    def _set(self, key: str, value: Any) -> None:
        self._cache[key] = value
        self._flush()

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def turn_count(self) -> int:
        return self._get("turn_count", 0)

    @turn_count.setter
    def turn_count(self, value: int) -> None:
        self._set("turn_count", value)

    @property
    def total_cost_usd(self) -> float:
        return self._get("total_cost_usd", 0.0)

    @total_cost_usd.setter
    def total_cost_usd(self, value: float) -> None:
        self._set("total_cost_usd", value)

    @property
    def total_input_tokens(self) -> int:
        return self._get("total_input_tokens", 0)

    @total_input_tokens.setter
    def total_input_tokens(self, value: int) -> None:
        self._set("total_input_tokens", value)

    @property
    def total_output_tokens(self) -> int:
        return self._get("total_output_tokens", 0)

    @total_output_tokens.setter
    def total_output_tokens(self, value: int) -> None:
        self._set("total_output_tokens", value)

    @property
    def total_cache_read_tokens(self) -> int:
        return self._get("total_cache_read_tokens", 0)

    @total_cache_read_tokens.setter
    def total_cache_read_tokens(self, value: int) -> None:
        self._set("total_cache_read_tokens", value)

    @property
    def total_cache_write_tokens(self) -> int:
        return self._get("total_cache_write_tokens", 0)

    @total_cache_write_tokens.setter
    def total_cache_write_tokens(self, value: int) -> None:
        self._set("total_cache_write_tokens", value)

    @property
    def cost_unknown(self) -> bool:
        return self._get("cost_unknown", False)

    @cost_unknown.setter
    def cost_unknown(self, value: bool) -> None:
        self._set("cost_unknown", value)

    # Last API call's reported usage. Persisted so a resumed session has
    # a usage-based floor for its first pre-call compaction estimate and
    # can display the "Conversation: N / M" figure before any new call
    # completes. Refreshed in-memory after every API call; saved to disk
    # at turn boundaries.
    @property
    def last_input_tokens(self) -> int:
        return self._get("last_input_tokens", 0)

    @last_input_tokens.setter
    def last_input_tokens(self, value: int) -> None:
        self._set("last_input_tokens", value)

    @property
    def last_output_tokens(self) -> int:
        return self._get("last_output_tokens", 0)

    @last_output_tokens.setter
    def last_output_tokens(self, value: int) -> None:
        self._set("last_output_tokens", value)

    @property
    def last_cache_read_tokens(self) -> int:
        return self._get("last_cache_read_tokens", 0)

    @last_cache_read_tokens.setter
    def last_cache_read_tokens(self, value: int) -> None:
        self._set("last_cache_read_tokens", value)

    @property
    def last_cache_write_tokens(self) -> int:
        return self._get("last_cache_write_tokens", 0)

    @last_cache_write_tokens.setter
    def last_cache_write_tokens(self, value: int) -> None:
        self._set("last_cache_write_tokens", value)

    @property
    def session_name(self) -> str:
        """User-assigned session name (via /name command). Empty if not set."""
        return self._get("session_name", "")

    @session_name.setter
    def session_name(self, value: str) -> None:
        self._set("session_name", value)

    # Allow rules are PROJECT-scoped (persist across sessions within a project).
    # Stored in .alan/allow_rules.json — see alancode/permissions/project_rules.py.
    @property
    def allow_rules(self) -> list[dict[str, Any]]:
        """Read the project-scoped allow rules list.

        On first access in a session, migrates any legacy session-scoped
        ``allow_rules`` entry from ``state.json`` into the project file
        so older sessions don't lose their rules.

        Returns:
            List of rule dicts (``tool_name``, ``rule_content``,
            ``source``). See :mod:`alancode.permissions.project_rules`.
        """
        from alancode.permissions.project_rules import load_project_allow_rules
        # One-time migration: if an older session had rules in state.json,
        # move them into the project file, then drop the session field.
        legacy = self._cache.get("allow_rules")
        if legacy:
            from alancode.permissions.project_rules import (
                load_project_allow_rules as _load,
                save_project_allow_rules as _save,
            )
            existing = _load(self.cwd)
            seen = {(r.get("tool_name"), r.get("rule_content")) for r in existing}
            for r in legacy:
                key = (r.get("tool_name"), r.get("rule_content"))
                if key not in seen:
                    existing.append(r)
                    seen.add(key)
            _save(existing, self.cwd)
            self._cache.pop("allow_rules", None)
            self._flush()
        return load_project_allow_rules(self.cwd)

    @allow_rules.setter
    def allow_rules(self, value: list[dict[str, Any]]) -> None:
        from alancode.permissions.project_rules import save_project_allow_rules
        save_project_allow_rules(value, self.cwd)

    def add_allow_rule(self, rule_dict: dict[str, Any]) -> None:
        """Append a single allow rule to the project-level store."""
        from alancode.permissions.project_rules import add_project_allow_rule
        add_project_allow_rule(rule_dict, self.cwd)

    # ── AGT (Agentic Git Tree) properties ────────────────────────────────

    @property
    def alan_commits(self) -> list[str]:
        """SHAs of commits made by the agent via GitCommit tool."""
        return list(self._get("alan_commits", []))

    @alan_commits.setter
    def alan_commits(self, value: list[str]) -> None:
        self._set("alan_commits", value)

    def add_alan_commit(self, sha: str) -> None:
        """Append a commit SHA and flush to disk."""
        commits = self._cache.get("alan_commits", [])
        commits.append(sha)
        self._set("alan_commits", commits)

    @property
    def conv_path(self) -> list[str]:
        """Ordered list of commit SHAs the agent visited (conversation path)."""
        return list(self._get("conv_path", []))

    @conv_path.setter
    def conv_path(self, value: list[str]) -> None:
        self._set("conv_path", value)

    def add_to_conv_path(self, sha: str) -> None:
        """Append a SHA to the conversation path and flush."""
        path = self._cache.get("conv_path", [])
        path.append(sha)
        self._set("conv_path", path)

    @property
    def compaction_markers(self) -> list[str]:
        """SHAs of HEAD at the time of each compaction."""
        return list(self._get("compaction_markers", []))

    @compaction_markers.setter
    def compaction_markers(self, value: list[str]) -> None:
        self._set("compaction_markers", value)

    def add_compaction_marker(self, sha: str) -> None:
        """Record that compaction happened at this commit."""
        markers = self._cache.get("compaction_markers", [])
        markers.append(sha)
        self._set("compaction_markers", markers)

    @property
    def session_root_sha(self) -> str:
        """SHA of HEAD when the session started."""
        return self._get("session_root_sha", "")

    @session_root_sha.setter
    def session_root_sha(self, value: str) -> None:
        self._set("session_root_sha", value)

    @property
    def agent_position_sha(self) -> str:
        """SHA of the commit the agent is currently on."""
        return self._get("agent_position_sha", "")

    @agent_position_sha.setter
    def agent_position_sha(self, value: str) -> None:
        self._set("agent_position_sha", value)

    @property
    def commit_message_indices(self) -> dict[str, int]:
        """Maps commit SHA → message list length at time of commit.

        Used by /convrevert to know exactly where to truncate messages.
        """
        return dict(self._get("commit_message_indices", {}))

    @commit_message_indices.setter
    def commit_message_indices(self, value: dict[str, int]) -> None:
        self._set("commit_message_indices", value)

    def record_commit_message_index(self, sha: str, message_count: int) -> None:
        """Record how many messages existed when this commit was made."""
        indices = self._cache.get("commit_message_indices", {})
        indices[sha] = message_count
        self._set("commit_message_indices", indices)
