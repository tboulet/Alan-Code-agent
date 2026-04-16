"""API cost tracking per session.

CostTracker owns pricing logic (how much does a model cost?) and delegates
cumulative total storage to SessionState (disk-attached).  Per-model usage
and API duration are kept in-memory (display-only, not persisted).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from alancode.messages.types import Usage

if TYPE_CHECKING:
    from alancode.session.state import SessionState

# Pricing per million tokens for Anthropic models (hardcoded).
# Source: Anthropic pricing page + litellm registry cross-check.
# These are used by AnthropicProvider. LiteLLMProvider uses litellm's registry.
ANTHROPIC_PRICING: dict[str, dict[str, float]] = {
    # Current generation
    "claude-sonnet-4-6": {
        "input": 3.0,
        "output": 15.0,
        "cache_read": 0.30,
        "cache_write": 3.75,
    },
    "claude-opus-4-6": {
        "input": 5.0,
        "output": 25.0,
        "cache_read": 0.50,
        "cache_write": 6.25,
    },
    "claude-haiku-4-5": {
        "input": 1.0,
        "output": 5.0,
        "cache_read": 0.10,
        "cache_write": 1.25,
    },
    # Previous generation
    "claude-sonnet-4-5": {
        "input": 3.0,
        "output": 15.0,
        "cache_read": 0.30,
        "cache_write": 3.75,
    },
    "claude-opus-4-5": {
        "input": 5.0,
        "output": 25.0,
        "cache_read": 0.50,
        "cache_write": 6.25,
    },
    "claude-opus-4-1": {
        "input": 15.0,
        "output": 75.0,
        "cache_read": 1.50,
        "cache_write": 18.75,
    },
    "claude-opus-4": {
        "input": 15.0,
        "output": 75.0,
        "cache_read": 1.50,
        "cache_write": 18.75,
    },
    "claude-sonnet-4": {
        "input": 3.0,
        "output": 15.0,
        "cache_read": 0.30,
        "cache_write": 3.75,
    },
    "claude-3-haiku": {
        "input": 0.25,
        "output": 1.25,
        "cache_read": 0.03,
        "cache_write": 0.30,
    },
}

_PER_MILLION = 1_000_000.0


def _anthropic_cost(usage: Usage, model: str) -> float | None:
    """Calculate cost using Anthropic hardcoded pricing.

    Returns None if model is not an Anthropic model.
    Uses prefix matching (e.g. "claude-sonnet-4-6-20250514" matches "claude-sonnet-4-6").
    """
    prices = ANTHROPIC_PRICING.get(model)
    if prices is None:
        # Prefix match
        for key, p in ANTHROPIC_PRICING.items():
            if model.startswith(key):
                prices = p
                break
    if prices is None:
        return None
    return (
        usage.input_tokens * prices["input"]
        + usage.output_tokens * prices["output"]
        + usage.cache_read_input_tokens * prices.get("cache_read", 0.0)
        + usage.cache_creation_input_tokens * prices.get("cache_write", 0.0)
    ) / _PER_MILLION


def _litellm_cost(usage: Usage, model: str) -> float | None:
    """Calculate cost using litellm's model pricing registry.

    Returns None if litellm doesn't know the model or isn't available.
    """
    try:
        import litellm
        prompt_cost, completion_cost = litellm.cost_per_token(
            model=model,
            prompt_tokens=usage.input_tokens,
            completion_tokens=usage.output_tokens,
            cache_read_input_tokens=usage.cache_read_input_tokens,
            cache_creation_input_tokens=usage.cache_creation_input_tokens,
        )
        return prompt_cost + completion_cost
    except Exception:
        return None


class CostTracker:
    """Tracks API cost and usage.

    Pricing logic lives here.  Cumulative totals are stored in the
    ``SessionState`` passed at construction (disk-backed).  Per-model
    breakdowns and API duration are kept in-memory only.
    """

    def __init__(self, session: SessionState) -> None:
        """Bind to a ``SessionState`` for persistent cost totals.

        Args:
            session: The session whose disk-backed totals this tracker
                contributes to on every ``add_usage`` call.
        """
        self._session = session
        self.model_usage: dict[str, Usage] = {}
        self.total_api_duration_ms: float = 0.0

    def calculate_cost(self, usage: Usage, model: str) -> float | None:
        """Calculate the estimated USD cost for a single Usage record.

        Resolution order:
        1. Anthropic hardcoded pricing (accurate, includes cache pricing)
        2. litellm registry (covers hundreds of models)
        3. None (unknown model, self-hosted, etc.)
        """
        # Try Anthropic pricing first (most accurate for Anthropic models)
        cost = _anthropic_cost(usage, model)
        if cost is not None:
            return cost

        # Try litellm registry
        return _litellm_cost(usage, model)

    def add_usage(
        self, usage: Usage, model: str, duration_ms: float = 0.0
    ) -> None:
        """Record usage from a single API call.

        Updates the session state (disk-backed) and in-memory per-model tracking.
        """
        # Per-model tracking (in-memory only)
        if model not in self.model_usage:
            self.model_usage[model] = Usage()
        self.model_usage[model].accumulate(usage)
        self.total_api_duration_ms += duration_ms

        # Update session state totals (disk-backed, batched into single write)
        with self._session.batch():
            self._session.total_input_tokens += usage.input_tokens
            self._session.total_output_tokens += usage.output_tokens
            self._session.total_cache_read_tokens += usage.cache_read_input_tokens
            self._session.total_cache_write_tokens += usage.cache_creation_input_tokens

            cost = self.calculate_cost(usage, model)
            if cost is not None:
                self._session.total_cost_usd += cost
            else:
                self._session.cost_unknown = True

    def get_summary(self) -> dict:
        """Return a summary dict suitable for logging or display."""
        models: dict[str, dict] = {}
        for model_name, usage in self.model_usage.items():
            models[model_name] = {
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "cache_read_input_tokens": usage.cache_read_input_tokens,
                "cache_creation_input_tokens": usage.cache_creation_input_tokens,
                "cost_usd": self.calculate_cost(usage, model_name),
            }

        s = self._session
        return {
            "total_input_tokens": s.total_input_tokens,
            "total_output_tokens": s.total_output_tokens,
            "total_cache_read_tokens": s.total_cache_read_tokens,
            "total_cache_write_tokens": s.total_cache_write_tokens,
            "total_cost_usd": s.total_cost_usd,
            "total_api_duration_ms": self.total_api_duration_ms,
            "models": models,
        }
