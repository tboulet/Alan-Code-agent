"""Model facts from OpenRouter's public catalogue.

LiteLLM's registry lags new models: an ``openrouter/...`` model it does not
know had no context window (the session refused to start) and no price.
OpenRouter publishes both for every model it serves, without authentication,
at ``OPENROUTER_MODELS_URL``. The catalogue is fetched once per process; a
failed fetch is remembered so it is not retried on every call.
"""

from __future__ import annotations

import logging
from typing import Any

import requests

logger = logging.getLogger(__name__)

OPENROUTER_PREFIX = "openrouter/"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
CATALOGUE_TIMEOUT_SECONDS = 10

_catalogue: dict[str, dict[str, Any]] | None = None


def _fetch_catalogue() -> dict[str, dict[str, Any]]:
    resp = requests.get(OPENROUTER_MODELS_URL, timeout=CATALOGUE_TIMEOUT_SECONDS)
    resp.raise_for_status()
    return {entry["id"]: entry for entry in resp.json().get("data", []) if "id" in entry}


def model_entry(model: str) -> dict[str, Any] | None:
    """The catalogue entry of an ``openrouter/<id>`` model, or None."""
    global _catalogue
    if not model.startswith(OPENROUTER_PREFIX):
        return None
    if _catalogue is None:
        try:
            _catalogue = _fetch_catalogue()
        except Exception as exc:
            logger.warning("Could not read the OpenRouter model catalogue: %s", exc)
            _catalogue = {}
    return _catalogue.get(model[len(OPENROUTER_PREFIX):])


def context_window(model: str) -> int | None:
    entry = model_entry(model)
    if entry is None:
        return None
    return entry.get("context_length") or (entry.get("top_provider") or {}).get("context_length")


def max_output_tokens(model: str) -> int | None:
    entry = model_entry(model)
    if entry is None:
        return None
    return (entry.get("top_provider") or {}).get("max_completion_tokens")


def cost_usd(
    model: str,
    *,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> float | None:
    """Estimated cost of one call from the catalogue's per-token prices."""
    entry = model_entry(model)
    if entry is None or not entry.get("pricing"):
        return None
    pricing = entry["pricing"]
    prompt_tokens = input_tokens + cache_read_tokens + cache_write_tokens
    # Long prompts can be billed at a higher tier.
    for tier in sorted(pricing.get("overrides") or [], key=lambda t: t.get("min_prompt_tokens", 0)):
        if prompt_tokens >= tier.get("min_prompt_tokens", 0):
            pricing = {**pricing, **tier}
    try:
        prompt = float(pricing.get("prompt", 0))
        completion = float(pricing.get("completion", 0))
        cache_read = float(pricing.get("input_cache_read", prompt))
        cache_write = float(pricing.get("input_cache_write", prompt))
    except (TypeError, ValueError):
        return None
    return (
        input_tokens * prompt
        + output_tokens * completion
        + cache_read_tokens * cache_read
        + cache_write_tokens * cache_write
    )
