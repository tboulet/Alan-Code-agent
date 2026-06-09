"""Project settings management (.alan/settings.json).

Implements the configuration priority chain:
1. CLI kwargs / AlanCodeAgent() constructor args  — Always win
2. Project settings (.alan/settings.json)         — Per-project defaults
3. Alan Code built-in defaults                    — Hardcoded fallback

On first use in a project, .alan/settings.json is generated with built-in defaults.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ── Built-in defaults (the ground truth) ─────────────────────────────────────
# Every configurable parameter must appear here. This is also used to
# initialize .alan/settings.json and to fill in missing fields on update.

SETTINGS_DEFAULTS: dict[str, Any] = {
    # Backend (transport) + model
    # "backend": which Alan transport speaks to the model.
    #   - "auto" : universal — LiteLLM, supports any provider via the
    #     model-string prefix (e.g. ``ollama/llama3``, ``openrouter/...``).
    #   - "anthropic-native" : direct Anthropic SDK. Unlocks cache_control
    #     breakpoints, native thinking, and native tool_use. The right
    #     choice for bare Claude model names.
    #   - "scripted" : deterministic provider used by tests.
    # When the user sets ``model`` without an explicit ``backend``, the
    # backend is inferred from the model string (see ``infer_backend``).
    "backend": "anthropic-native",
    "model": "claude-sonnet-4-6",
    "api_key": None,  # None = read from env var
    "base_url": None,  # None = use provider default. Set for local servers (e.g., http://localhost:8000/v1)
    "tool_call_format": None,  # Text-based tool call format: "hermes", "hermes_xml", "glm", "alan", or None (native)
    # Session
    "permission_mode": "edit",  # 'yolo', 'edit', 'safe'
    "max_iterations_per_turn": None,  # None = unlimited. Caps API calls per user message.
    "max_output_tokens": None,  # None = provider default
    # System prompt
    "custom_system_prompt": None,
    "append_system_prompt": None,
    # Memory
    "memory": "off",  # "on", "off", "intensive"
    # Verbose
    "verbose": False,
    # Hooks (lifecycle event hooks — see alancode/hooks/registry.py)
    "hooks": {},
    # Token / context management
    "compact_max_output_tokens": 20_000,  # Tokens reserved for compaction summary output
    "capped_default_max_tokens": 8_000,  # Default max_tokens (slot reservation optimization)
    "escalated_max_tokens": 64_000,  # Retry budget after capped default is hit
    "auto_compact_buffer_tokens": 13_000,  # Buffer below context window that triggers auto-compact
    "warning_threshold_buffer_tokens": 20_000,  # Remaining tokens to trigger warning
    "blocking_limit_buffer_tokens": 3_000,  # Hard floor: refuse to call API below this remaining
    "max_consecutive_compact_failures": 3,  # Circuit breaker for auto-compact retries
    "compaction_threshold_percent": 80,  # Percentage of context window that triggers compaction layers
    "max_compact_ptl_retries": 3,  # Max prompt-too-long retries during compaction summarize
    # Error recovery
    "max_output_tokens_recovery_limit": 3,  # Max multi-turn recovery attempts on output limit hit
    # Tool execution
    "max_tool_concurrency": 10,  # Max parallel read-only tool executions
    "tool_result_max_chars": 20_000,  # Per-tool-result size before truncation
    "compact_clear_keep_recent": 10,  # Number of recent tool results to preserve during Layer B (clear)
    # Thinking
    "thinking_budget_default": 10_000,  # Default thinking token budget (when model supports it)
    # Memory
    "memory_reminder_threshold": 10,  # Iterations between memory reminders (intensive mode)
    "max_scratchpad_sessions": 5,  # Max scratchpad session dirs to keep
    # Compaction layer toggles
    "compaction_truncate_enabled": True,
    "compaction_clear_enabled": True,
    "compaction_auto_enabled": True,
    # Incremental transcript persistence. When False (default), the transcript
    # is written only at turn boundaries (existing behavior). When True, the
    # transcript is also re-written periodically *during* a turn so the session
    # is never lost if the turn never completes (rate-limit stall, SIGTERM /
    # SIGKILL mid-turn). Opt-in for long-running batch/agentic consumers.
    "incremental_transcript": False,
    # Minimum seconds between mid-turn transcript flushes (debounce). Only used
    # when incremental_transcript is True.
    "incremental_transcript_interval_s": 3.0,
}

# Fields that should NOT be written to settings.json (ephemeral / per-invocation only)
_EPHEMERAL_FIELDS = {"api_key"}


def get_alan_dir(cwd: str | None = None) -> Path:
    """Get the .alan/ directory for the given working directory."""
    base = Path(cwd) if cwd else Path.cwd()
    return base / ".alan"


def get_settings_path(cwd: str | None = None) -> Path:
    """Get the path to .alan/settings.json."""
    return get_alan_dir(cwd) / "settings.json"


def load_settings(cwd: str | None = None) -> dict[str, Any]:
    """Load project settings from .alan/settings.json.

    If the file doesn't exist or is corrupt/invalid, returns empty dict.
    """
    path = get_settings_path(cwd)
    if not path.exists():
        return {}

    try:
        with open(path) as f:
            settings = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Failed to read {path}: {e}. Using defaults.")
        return {}

    if not isinstance(settings, dict):
        logger.warning(f"Invalid settings format in {path}. Using defaults.")
        return {}

    # Translate any legacy ``provider`` key into the new ``backend`` key
    # so older .alan/settings.json files keep loading without manual edits.
    if migrate_legacy_provider_key(settings):
        logger.info(
            "%s used the legacy 'provider' key — auto-migrated to 'backend'. "
            "Re-save settings to silence this notice (/settings backend=<value>).",
            path,
        )

    return settings


def save_settings(settings: dict[str, Any], cwd: str | None = None) -> None:
    """Write settings to .alan/settings.json.

    Creates .alan/ directory if needed. Excludes ephemeral fields.
    """
    path = get_settings_path(cwd)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Filter out ephemeral fields
    to_write = {k: v for k, v in settings.items() if k not in _EPHEMERAL_FIELDS}

    try:
        from alancode.utils.atomic_io import atomic_write_json
        atomic_write_json(path, to_write, indent=2)
        logger.debug(f"Settings saved to {path}")
    except OSError as e:
        logger.warning(f"Failed to write {path}: {e}")


def load_projects_settings_and_maybe_init(cwd: str | None = None) -> dict[str, Any]:
    """Ensure .alan/settings.json exists.

    If it doesn't exist, creates it with built-in defaults.
    If it exists, loads and returns it.
    """
    path = get_settings_path(cwd)
    if not path.exists():
        logger.info(f"Initializing {path} with default settings")
        defaults = {
            k: v for k, v in SETTINGS_DEFAULTS.items() if k not in _EPHEMERAL_FIELDS
        }
        save_settings(defaults, cwd)
        return dict(SETTINGS_DEFAULTS)

    return load_settings(cwd)


def coerce_value(raw: str) -> Any:
    """Auto-coerce a CLI string value to the appropriate Python type."""
    lower = raw.lower()
    if lower in ("true", "yes", "y"):
        return True
    if lower in ("false", "no"):
        return False
    if lower in ("null", "none", ""):
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


# ── Setting validators ──────────────────────────────────────────────────────
# Each entry is (check_fn, error_message).
# - check_fn(value) -> bool: returns True if valid
# - None values always pass (means "unset")
# - Keys without an entry are not validated.

_one_of = lambda *vals: (lambda v: v in vals, f"Must be one of: {', '.join(repr(v) for v in vals)}")
_is_str = (lambda v: isinstance(v, str), "Must be a string")
_is_bool = (lambda v: isinstance(v, bool), "Must be a boolean")
_is_pos_int = (lambda v: isinstance(v, int) and v > 0, "Must be a positive integer")
_is_pos_int_or_none = (lambda v: v is None or (isinstance(v, int) and v > 0), "Must be a positive integer or null")

SETTING_VALIDATORS: dict[str, tuple] = {
    "backend": _one_of("auto", "anthropic-native", "scripted"),
    "model": _is_str,
    "base_url": _is_str,
    "tool_call_format": _one_of("hermes", "hermes_xml", "glm", "alan"),
    "permission_mode": _one_of("yolo", "edit", "safe"),
    "max_iterations_per_turn": _is_pos_int_or_none,
    "max_output_tokens": _is_pos_int_or_none,
    "custom_system_prompt": _is_str,
    "append_system_prompt": _is_str,
    "memory": _one_of("on", "off", "intensive"),
    "verbose": _is_bool,
    "compact_max_output_tokens": _is_pos_int,
    "capped_default_max_tokens": _is_pos_int,
    "escalated_max_tokens": _is_pos_int,
    "auto_compact_buffer_tokens": _is_pos_int,
    "warning_threshold_buffer_tokens": _is_pos_int,
    "blocking_limit_buffer_tokens": _is_pos_int,
    "max_consecutive_compact_failures": _is_pos_int,
    "compaction_threshold_percent": (lambda v: isinstance(v, int) and 20 <= v <= 99, "Must be an integer between 20 and 99"),
    "max_compact_ptl_retries": _is_pos_int,
    "max_output_tokens_recovery_limit": _is_pos_int,
    "max_tool_concurrency": _is_pos_int,
    "tool_result_max_chars": _is_pos_int,
    "compact_clear_keep_recent": _is_pos_int,
    "thinking_budget_default": _is_pos_int,
    "memory_reminder_threshold": _is_pos_int,
    "max_scratchpad_sessions": _is_pos_int,
    "compaction_truncate_enabled": _is_bool,
    "compaction_clear_enabled": _is_bool,
    "compaction_auto_enabled": _is_bool,
    "incremental_transcript": _is_bool,
    "incremental_transcript_interval_s": (
        lambda v: isinstance(v, (int, float)) and v >= 0,
        "Must be a non-negative number",
    ),
}


def validate_setting(key: str, value: Any) -> str | None:
    """Validate a setting value against its validator.

    Returns an error message if invalid, or None if valid.
    None values always pass (they mean "unset").
    """
    entry = SETTING_VALIDATORS.get(key)
    if entry is None:
        return None  # no validator for this key
    check_fn, error_msg = entry
    if value is None:
        return None  # None always accepted
    if not check_fn(value):
        return f"Invalid value {value!r} for '{key}': {error_msg}"
    return None


# Settings that trigger backend (LLMProvider) recreation when changed mid-session.
BACKEND_SETTINGS: set[str] = {
    "backend",
    "model",
    "api_key",
    "base_url",
}

# Backwards-compatible alias — old name kept for one release for any
# external code (e.g. tests, downstream tools) that imported it.
PROVIDER_SETTINGS = BACKEND_SETTINGS


# ── Backend inference and legacy-key migration ──────────────────────────────


# Legacy --provider values → new backend names.
_LEGACY_PROVIDER_MAP: dict[str, str] = {
    "litellm": "auto",
    "anthropic": "anthropic-native",
    "scripted": "scripted",
}


def infer_backend(model: str | None) -> str:
    """Infer the backend from a model string.

    Rule: a bare Claude name (e.g. ``claude-sonnet-4-6``) routes through
    the native Anthropic SDK; everything else goes through the universal
    LiteLLM transport. The ``anthropic/claude-...`` prefix is the explicit
    escape hatch — it keeps the backend on ``auto`` so the user can route
    Claude through LiteLLM (e.g. for a LiteLLM Proxy).

    Returns ``"auto"`` when ``model`` is ``None`` or empty (used as a
    safe fallback during partial configuration).
    """
    if not model:
        return "auto"
    if "/" not in model and model.startswith("claude-"):
        return "anthropic-native"
    return "auto"


def migrate_legacy_provider_key(settings: dict[str, Any]) -> bool:
    """Translate the old ``provider`` key into the new ``backend`` key.

    Mutates *settings* in place. Returns ``True`` if anything changed
    (caller may want to log a deprecation notice).

    Mapping:
        ``provider="litellm"``   → ``backend="auto"``
        ``provider="anthropic"`` → ``backend="anthropic-native"``
        ``provider="scripted"``  → ``backend="scripted"``

    Any other legacy value is dropped silently; the caller is expected
    to surface a friendly error elsewhere.
    """
    if "provider" not in settings:
        return False

    old = settings.pop("provider")
    if "backend" in settings:
        # User already specified the new key; legacy value is stale.
        return True

    if isinstance(old, str):
        mapped = _LEGACY_PROVIDER_MAP.get(old.lower())
        if mapped is not None:
            settings["backend"] = mapped
            return True

    # Unknown legacy value — leave backend unset (caller will fall back
    # to the default or to inference).
    return True
