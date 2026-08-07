"""Token-budget resolution - the single source of truth for context sizing.

Every quantity the agent uses to manage the context window derives from two
developer inputs (the model's context window and the configured output budget)
through an explicit dependency DAG::

    CW --+--> M = min(MOT_config | model_default, CW/4)   output budget
         |            |
         +-- m = max(500, 1% * CW)                        safety margin
                      |
                      v
          U = CW - M - m      max context size allowed (blocks inference past it)
                      |
                      +--> T = threshold_percent% * U     Layer C trigger
                      +--> G = (T + U) / 2                Layer B damage-control target
                      +--> r = min(10k chars, 10% * T * 3 chars)  per-result cap
                      +--> S = min(20k, CW - T - m)       summarizer output budget

Each parameter is computed automatically when its setting is ``"auto"`` (or
absent / ``None``) - the default - and honored verbatim when set explicitly.
Explicit values are validated against their parents; impossible combinations
raise :class:`ConfigError` at resolve time with a human explanation, instead
of failing later inside an API call.

"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from alancode.backends.base import ModelInfo

logger = logging.getLogger(__name__)


# ── Constants (auto-mode formulas) ───────────────────────────────────────────

MIN_CONTEXT_WINDOW = 8_192          # below this: ConfigError
WARN_CONTEXT_WINDOW = 16_384        # below this: loud warning (tight but usable)
MARGIN_FLOOR_TOKENS = 500           # safety margin: max(this, 1% of CW)
AUTO_OUTPUT_CW_FRACTION = 4         # auto M is capped at CW / 4
DEFAULT_THRESHOLD_PERCENT = 80      # T as a percentage of U
TOOL_RESULT_CAP_CHARS_MAX = 10_000  # absolute ceiling of the per-result cap
TOOL_RESULT_CAP_T_FRACTION = 0.10   # ...or 10% of T (in tokens), whichever is lower
CHARS_PER_TOKEN = 3                 # mirror of utils.tokens.CHARS_PER_TOKEN_FALLBACK
DEFAULT_SUMMARY_MAX_TOKENS = 20_000
MIN_SUMMARY_OUTPUT_TOKENS = 256      # below this legal ceiling, shrink input instead of calling


class ConfigError(ValueError):
    """An impossible or invalid budget configuration.

    Raised at resolve time so misconfiguration fails fast with an
    explanation, never later inside an API call.
    """


@dataclass(frozen=True)
class ContextBudget:
    """Resolved token budgets for one (model, settings) combination.

    All fields are token counts unless suffixed ``_chars``. Immutable -
    resolve once per turn and thread the object around.
    """

    context_window: int             # CW - trusted, post-resolution
    max_output_tokens: int          # M - per-call output budget (pre call-time clamp)
    safety_margin: int              # m - slack for token-estimate error
    max_context_size_allowed: int   # U = CW - M - m; inference is blocked past it
    threshold_compaction: int       # T - Layer C triggers at estimate >= T
    tool_result_clear_target: int   # G - Layer B clears down to (never below) this
    tool_result_cap_chars: int      # r - per-tool-result size cap, in characters
    max_tokens_for_summary: int     # S - summarizer output budget (pre call-time clamp)


def _setting(settings: dict | None, key: str):
    """Read *key* treating ``"auto"``/``None``/missing as "not set"."""
    if settings is None:
        return None
    value = settings.get(key)
    if value is None or (isinstance(value, str) and value.lower() == "auto"):
        return None
    return value


def _require_positive_int(value, key: str) -> int:
    """Validate an explicit setting is a positive integer."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(
            f"Setting '{key}' must be a positive integer or 'auto', "
            f"got {value!r}."
        )
    if value <= 0:
        raise ConfigError(f"Setting '{key}' must be > 0, got {value}.")
    return value


def resolve_context_budget(
    model_info: ModelInfo,
    settings: dict | None = None,
) -> ContextBudget:
    """Resolve all token budgets from *model_info* and *settings*.

    Settings read (each accepting ``"auto"`` / ``None`` / explicit int):

    - ``context_window``            - overrides ``model_info.context_window``
    - ``max_output_tokens``         - explicit M (auto: model default capped at CW/4)
    - ``compaction_threshold_percent`` - T as % of U (auto: 80; valid 1-99)
    - ``tool_result_max_chars``     - explicit r (auto: min(10k, 10% * T * 3))
    - ``compact_max_output_tokens`` - explicit S (auto: min(20k, CW - T - m))

    Raises:
        ConfigError: on any impossible combination, with an explanation.
    """
    settings = settings or {}

    # ── CW ──────────────────────────────────────────────────────────────
    cw_override = _setting(settings, "context_window")
    if cw_override is not None:
        cw = _require_positive_int(cw_override, "context_window")
    else:
        cw = model_info.context_window

    if cw < MIN_CONTEXT_WINDOW:
        raise ConfigError(
            f"Context window {cw} is below the supported minimum "
            f"{MIN_CONTEXT_WINDOW}. Alan's system prompt and tool schemas "
            f"cannot operate usefully in such a window. Use a larger model, "
            f"or override with the 'context_window' setting if the detected "
            f"value is wrong."
        )
    if cw < WARN_CONTEXT_WINDOW:
        logger.warning(
            "Context window %d is tight (< %d): expect frequent compaction "
            "and short responses.",
            cw,
            WARN_CONTEXT_WINDOW,
        )

    # ── m (safety margin) ───────────────────────────────────────────────
    margin = max(MARGIN_FLOOR_TOKENS, cw // 100)

    # ── M (output budget) ───────────────────────────────────────────────
    # Explicit config is honored verbatim (then validated via U > 0);
    # auto mode caps the model default at CW/4 so the output reservation
    # never dominates a small window.
    mot_cfg = _setting(settings, "max_output_tokens")
    if mot_cfg is not None:
        max_output = _require_positive_int(mot_cfg, "max_output_tokens")
        if max_output >= cw:
            raise ConfigError(
                f"max_output_tokens ({max_output}) must be smaller than the "
                f"context window ({cw})."
            )
    else:
        max_output = min(
            model_info.max_output_tokens, cw // AUTO_OUTPUT_CW_FRACTION
        )

    # ── U (max context size allowed; inference blocked past it) ─────────
    max_context_size = cw - max_output - margin
    if max_context_size <= 0:
        raise ConfigError(
            f"No usable input budget: context window {cw} - "
            f"max_output_tokens {max_output} - safety margin {margin} "
            f"= {max_context_size}. Lower max_output_tokens or use a "
            f"larger model."
        )

    # ── T (compaction threshold) ────────────────────────────────────────
    pct_cfg = _setting(settings, "compaction_threshold_percent")
    if pct_cfg is not None:
        pct = _require_positive_int(pct_cfg, "compaction_threshold_percent")
        if not (1 <= pct <= 99):
            raise ConfigError(
                f"compaction_threshold_percent must be between 1 and 99, "
                f"got {pct}."
            )
    else:
        pct = DEFAULT_THRESHOLD_PERCENT
    threshold = max_context_size * pct // 100

    # ── G (tool-result clear target) ────────────────────────────────────
    clear_target = (threshold + max_context_size) // 2

    # ── r (per-result cap, chars) ───────────────────────────────────────
    cap_cfg = _setting(settings, "tool_result_max_chars")
    if cap_cfg is not None:
        cap_chars = _require_positive_int(cap_cfg, "tool_result_max_chars")
    else:
        cap_chars = min(
            TOOL_RESULT_CAP_CHARS_MAX,
            int(TOOL_RESULT_CAP_T_FRACTION * threshold * CHARS_PER_TOKEN),
        )
        cap_chars = max(1, cap_chars)

    # ── S (summarizer output budget) ────────────────────────────────────
    s_cfg = _setting(settings, "compact_max_output_tokens")
    if s_cfg is not None:
        summary_max = _require_positive_int(s_cfg, "compact_max_output_tokens")
        if summary_max >= cw:
            raise ConfigError(
                f"compact_max_output_tokens ({summary_max}) must be smaller "
                f"than the context window ({cw})."
            )
    else:
        summary_max = min(DEFAULT_SUMMARY_MAX_TOKENS, cw - threshold - margin)
        summary_max = max(1, summary_max)

    budget = ContextBudget(
        context_window=cw,
        max_output_tokens=max_output,
        safety_margin=margin,
        max_context_size_allowed=max_context_size,
        threshold_compaction=threshold,
        tool_result_clear_target=clear_target,
        tool_result_cap_chars=cap_chars,
        max_tokens_for_summary=summary_max,
    )

    # ── Ordering invariant (defense in depth - should be unreachable) ───
    if not (0 < threshold < clear_target < max_context_size < cw):
        raise ConfigError(
            f"Budget ordering violated: T={threshold}, G={clear_target}, "
            f"U={max_context_size}, CW={cw}. This combination of explicit "
            f"settings is unusable."
        )

    return budget


def clamp_output_budget(
    budget: ContextBudget,
    estimated_input_tokens: int,
    requested_max_tokens: int | None = None,
) -> int:
    """Clamp a call's max_tokens so ``input + max_tokens + margin <= CW``.

    Whatever the requested budget (normal, escalated, or summarizer), the
    returned value never overflows the window given the input estimate.

    Returns 0 when no output fits at all - the caller must not make the
    call (the blocking limit should normally prevent ever reaching this).
    """
    requested = (
        requested_max_tokens
        if requested_max_tokens is not None
        else budget.max_output_tokens
    )
    available = (
        budget.context_window - estimated_input_tokens - budget.safety_margin
    )
    return max(0, min(requested, available))
