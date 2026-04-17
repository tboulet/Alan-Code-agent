"""Anthropic model registry — capabilities for all known Claude models.

Used by AnthropicProvider.get_model_info() to return accurate ModelInfo.
The lookup tries exact match first, then prefix match (so a dated model ID
like ``claude-sonnet-4-6-20260401`` matches ``claude-sonnet-4-6``).
"""

from alancode.providers.base import ModelInfo


ANTHROPIC_MODELS: dict[str, ModelInfo] = {
    # ── Active models ──────────────────────────────────────────────────────
    "claude-opus-4-7": ModelInfo(
        context_window=1_000_000,
        max_output_tokens=128_000,
        supports_thinking=True,
    ),
    "claude-opus-4-6": ModelInfo(
        context_window=1_000_000,
        max_output_tokens=128_000,
        supports_thinking=True,
    ),
    "claude-sonnet-4-6": ModelInfo(
        context_window=1_000_000,
        max_output_tokens=64_000,
        supports_thinking=True,
    ),
    "claude-sonnet-4-5": ModelInfo(
        context_window=200_000,
        max_output_tokens=64_000,
        supports_thinking=True,
    ),
    "claude-opus-4-5": ModelInfo(
        context_window=200_000,
        max_output_tokens=64_000,
        supports_thinking=True,
    ),
    "claude-opus-4-1": ModelInfo(
        context_window=200_000,
        max_output_tokens=32_000,
        supports_thinking=True,
    ),
    "claude-haiku-4-5": ModelInfo(
        context_window=200_000,
        max_output_tokens=64_000,
        supports_thinking=True,
    ),
    # ── Deprecated (still functional, retirement scheduled) ────────────────
    "claude-sonnet-4-20250514": ModelInfo(
        context_window=200_000,
        max_output_tokens=64_000,
        supports_thinking=True,
    ),
    "claude-opus-4-20250514": ModelInfo(
        context_window=200_000,
        max_output_tokens=32_000,
        supports_thinking=True,
    ),
    "claude-3-haiku-20240307": ModelInfo(
        context_window=200_000,
        max_output_tokens=4_096,
        supports_thinking=False,
    ),
    # ── Retired (included for backward compatibility) ──────────────────────
    "claude-3-7-sonnet-20250219": ModelInfo(
        context_window=200_000,
        max_output_tokens=128_000,
        supports_thinking=True,
    ),
    "claude-3-5-haiku-20241022": ModelInfo(
        context_window=200_000,
        max_output_tokens=8_192,
        supports_thinking=False,
    ),
    "claude-3-5-sonnet-20241022": ModelInfo(
        context_window=200_000,
        max_output_tokens=8_192,
        supports_thinking=False,
    ),
    "claude-3-5-sonnet-20240620": ModelInfo(
        context_window=200_000,
        max_output_tokens=8_192,
        supports_thinking=False,
    ),
    "claude-3-opus-20240229": ModelInfo(
        context_window=200_000,
        max_output_tokens=4_096,
        supports_thinking=False,
    ),
    "claude-3-sonnet-20240229": ModelInfo(
        context_window=200_000,
        max_output_tokens=4_096,
        supports_thinking=False,
    ),
    # ── Legacy (pre-Claude-3, no vision/tools) ─────────────────────────────
    "claude-2.1": ModelInfo(
        context_window=200_000,
        max_output_tokens=4_096,
        supports_thinking=False,
    ),
    "claude-2.0": ModelInfo(
        context_window=100_000,
        max_output_tokens=4_096,
        supports_thinking=False,
    ),
    "claude-instant-1.2": ModelInfo(
        context_window=100_000,
        max_output_tokens=4_096,
        supports_thinking=False,
    ),
}


# Aliases: map alternative model IDs to their canonical registry key.
ANTHROPIC_ALIASES: dict[str, str] = {
    "claude-haiku-4-5-20251001": "claude-haiku-4-5",
    "claude-sonnet-4-5-20250929": "claude-sonnet-4-5",
    "claude-opus-4-5-20251101": "claude-opus-4-5",
    "claude-opus-4-1-20250805": "claude-opus-4-1",
    "claude-sonnet-4-0": "claude-sonnet-4-20250514",
    "claude-opus-4-0": "claude-opus-4-20250514",
}


def lookup_anthropic_model(model: str) -> ModelInfo:
    """Look up a model by ID with alias resolution and prefix matching.

    Resolution order:
    1. Exact match in ANTHROPIC_MODELS
    2. Alias resolution via ANTHROPIC_ALIASES, then exact match
    3. Prefix match (e.g. ``claude-sonnet-4-6-20260401`` matches ``claude-sonnet-4-6``)
    4. Falls back to safe defaults (200K context, 8K output, no thinking)
    """
    # Exact match
    if model in ANTHROPIC_MODELS:
        return ANTHROPIC_MODELS[model]

    # Alias resolution
    canonical = ANTHROPIC_ALIASES.get(model)
    if canonical and canonical in ANTHROPIC_MODELS:
        return ANTHROPIC_MODELS[canonical]

    # Prefix match (handles dated suffixes like claude-sonnet-4-6-20260401)
    for key, info in ANTHROPIC_MODELS.items():
        if model.startswith(key):
            return info

    # Unknown model — safe defaults
    return ModelInfo()
