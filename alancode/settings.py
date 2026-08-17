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

from alancode.utils.atomic_io import atomic_write_json

logger = logging.getLogger(__name__)


# ── Built-in defaults (the ground truth) ─────────────────────────────────────
# Every configurable parameter must appear here. This is also used to
# initialize .alan/settings.json and to fill in missing fields on update.

SETTINGS_DEFAULTS: dict[str, Any] = {
    # Backend (transport) + model
    # "backend": which Alan transport speaks to the model.
    #   - "auto" : universal — LiteLLM, supports any backend via the
    #     model-string prefix (e.g. ``ollama/llama3``, ``openrouter/...``).
    #   - "anthropic-native" : direct Anthropic SDK. Unlocks cache_control
    #     breakpoints, native thinking, and native tool_use. The right
    #     choice for bare Claude model names.
    #   - "scripted" : deterministic backend used by tests.
    # When the user sets ``model`` without an explicit ``backend``, the
    # backend is inferred from the model string (see ``infer_backend``).
    "backend": "anthropic-native",
    "model": "claude-sonnet-4-6",
    "api_key": None,  # None = read from env var
    "base_url": None,  # None = use backend default. Set for local servers (e.g., http://localhost:8000/v1)
    "request_timeout": "auto",  # seconds; auto = 1h for custom endpoints, otherwise SDK default
    "tool_call_format": None,  # Text-based tool call format: "hermes", "hermes_xml", "glm", "alan", "meta_json", "bash_block", or None (native)
    # Session
    "permission_mode": "edit",  # 'yolo', 'edit', 'safe'
    "max_iterations_per_turn": None,  # None = unlimited. Caps API calls per user message.
    "max_output_tokens": None,  # None = backend default
    # System prompt
    "custom_system_prompt": None,
    "append_system_prompt": None,
    # Memory
    "memory": "off",  # "on", "off", "intensive"
    # Verbose
    "verbose": False,
    # Hooks (lifecycle event hooks — see alancode/hooks/registry.py)
    "hooks": {},
    # Token / context management. Budget keys accept "auto" (computed from
    # the model's context window - see alancode/budget.py for the DAG and
    # the auto formulas) or an explicit positive integer.
    "context_window": "auto",  # "auto" = resolve from registry/server/probe; int = trust this value
    "compact_max_output_tokens": "auto",  # Summarizer output budget (auto: min(20k, CW - T - m))
    "capped_default_max_tokens": 8_000,  # Default max_tokens (slot reservation optimization)
    "escalated_max_tokens": 64_000,  # Retry budget on length-truncation, overrides a lower max_output_tokens (clamped to the window)
    "auto_compact_buffer_tokens": 13_000,  # Buffer below context window that triggers auto-compact
    "warning_threshold_buffer_tokens": 20_000,  # Remaining tokens to trigger warning
    "max_consecutive_compact_failures": 3,  # Circuit breaker for auto-compact retries
    "compaction_threshold_percent": "auto",  # T as % of usable input (auto: 80)
    "max_compact_ptl_retries": 3,  # Max prompt-too-long retries during compaction summarize
    # Error recovery
    "max_output_tokens_recovery_limit": 3,  # Max multi-turn recovery attempts on output limit hit
    # Tool execution
    "max_tool_concurrency": 10,  # Max parallel read-only tool executions
    "tool_result_max_chars": "auto",  # Per-result cap (auto: min(10k, 10% of T in chars))
    # Thinking
    "thinking_budget_default": 10_000,  # Default thinking token budget (when model supports it)
    # Memory
    "memory_reminder_threshold": 10,  # Iterations between memory reminders (intensive mode)
    "max_scratchpad_sessions": 5,  # Max scratchpad session dirs to keep
    # Compaction layer toggles
    "compaction_truncate_enabled": True,
    "compaction_clear_enabled": True,
    "compaction_auto_enabled": True,
}

# Fields that should NOT be written to settings.json (ephemeral / per-invocation only)
_EPHEMERAL_FIELDS = {"api_key"}

# Settings removed by the budget redesign; stripped (with a notice) when
# found in an existing settings.json.
_DEPRECATED_KEYS = {"blocking_limit_buffer_tokens", "compact_clear_keep_recent"}


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
        logger.warning("Failed to read %s: %s. Using defaults.", path, e)
        return {}

    if not isinstance(settings, dict):
        logger.warning("Invalid settings format in %s. Using defaults.", path)
        return {}

    # Strip keys removed by the budget redesign (their behaviour is now
    # derived - see alancode/budget.py). Old settings.json files were
    # initialized with the full defaults, so they all carry these.
    for key in _DEPRECATED_KEYS & settings.keys():
        settings.pop(key)
        logger.info(
            "%s contains the removed setting '%s' (superseded by the "
            "auto-computed budget) - ignored.", path, key,
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
        atomic_write_json(path, to_write, indent=2)
        logger.debug("Settings saved to %s", path)
    except OSError as e:
        logger.warning("Failed to write %s: %s", path, e)


def load_projects_settings_and_maybe_init(cwd: str | None = None) -> dict[str, Any]:
    """Ensure .alan/settings.json exists.

    If it doesn't exist, creates it with built-in defaults.
    If it exists, loads and returns it.
    """
    path = get_settings_path(cwd)
    if not path.exists():
        logger.info("Initializing %s with default settings", path)
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

def _one_of(*vals):
    return lambda v: v in vals, f"Must be one of: {', '.join(repr(v) for v in vals)}"


_is_str = (lambda v: isinstance(v, str), "Must be a string")
_is_bool = (lambda v: isinstance(v, bool), "Must be a boolean")
_is_pos_int = (lambda v: isinstance(v, int) and v > 0, "Must be a positive integer")
_is_pos_int_or_none = (lambda v: v is None or (isinstance(v, int) and v > 0), "Must be a positive integer or null")


def _is_auto(v):
    return isinstance(v, str) and v.lower() == "auto"


_is_pos_int_or_auto = (
    lambda v: _is_auto(v) or (isinstance(v, int) and not isinstance(v, bool) and v > 0),
    'Must be a positive integer or "auto"',
)

SETTING_VALIDATORS: dict[str, tuple] = {
    "backend": _one_of("auto", "anthropic-native", "scripted"),
    "model": _is_str,
    "base_url": _is_str,
    "request_timeout": _is_pos_int_or_auto,
    "tool_call_format": _one_of("hermes", "hermes_xml", "glm", "alan", "meta_json", "bash_block"),
    "permission_mode": _one_of("yolo", "edit", "safe"),
    "max_iterations_per_turn": _is_pos_int_or_none,
    "max_output_tokens": (
        lambda v: v is None or _is_pos_int_or_auto[0](v),
        'Must be a positive integer, "auto", or null',
    ),
    "custom_system_prompt": _is_str,
    "append_system_prompt": _is_str,
    "memory": _one_of("on", "off", "intensive"),
    "verbose": _is_bool,
    "context_window": _is_pos_int_or_auto,
    "compact_max_output_tokens": _is_pos_int_or_auto,
    "capped_default_max_tokens": _is_pos_int,
    "escalated_max_tokens": _is_pos_int,
    "auto_compact_buffer_tokens": _is_pos_int,
    "warning_threshold_buffer_tokens": _is_pos_int,
    "max_consecutive_compact_failures": _is_pos_int,
    "compaction_threshold_percent": (
        lambda v: _is_auto(v) or (isinstance(v, int) and not isinstance(v, bool) and 1 <= v <= 99),
        'Must be an integer between 1 and 99, or "auto"',
    ),
    "max_compact_ptl_retries": _is_pos_int,
    "max_output_tokens_recovery_limit": _is_pos_int,
    "max_tool_concurrency": _is_pos_int,
    "tool_result_max_chars": _is_pos_int_or_auto,
    "thinking_budget_default": _is_pos_int,
    "memory_reminder_threshold": _is_pos_int,
    "max_scratchpad_sessions": _is_pos_int,
    "compaction_truncate_enabled": _is_bool,
    "compaction_clear_enabled": _is_bool,
    "compaction_auto_enabled": _is_bool,
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


# Settings that trigger backend (LLMBackend) recreation when changed mid-session.
BACKEND_SETTINGS: set[str] = {
    "backend",
    "model",
    "api_key",
    "base_url",
    "request_timeout",
    "context_window",
}

# ── Backend inference ────────────────────────────────────────────────────────


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
