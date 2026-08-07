"""Context-window probing and caching for models with unknown limits.

Last-resort resolution rung for the context window (see the chain in
``LiteLLMBackend.get_model_info``): when the registry, the server metadata
endpoints and the known-models table all failed, actively probe the server.

Probing strategy (descending - cost-aware):

- Requests OVER the limit are rejected at validation: no tokens are
  processed, so they are free and near-instant.
- Requests UNDER the limit prefill the whole prompt: that is the only
  real cost (money on paid APIs, prefill time on local GPUs).

So we descend from a 1M-token ceiling, halving until the first success -
paying for exactly ONE large prefill - then optionally refine once inside
the bracket. The detected value is the backend's OWN count of what it
accepted (``usage.prompt_tokens`` of the successful probe), which is exact
and immune to our padding-tokenization error.

Guard: a server that accepts the 1M ceiling without ever erroring is
almost certainly a silent truncator (e.g. Ollama beyond ``num_ctx``) - its
"success" is meaningless, so the result is distrusted entirely.

Probed values are cached in ``~/.alan/context_windows.json`` keyed by
(api_base, model) so the prefill cost is paid once per model, ever.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from alancode.api.errors import is_prompt_too_long
from alancode.utils.atomic_io import atomic_write_json, interprocess_lock

logger = logging.getLogger(__name__)

PROBE_CEILING_TOKENS = 1_048_576   # 1M - above this we do not even look
PROBE_FLOOR_TOKENS = 8_192         # below MIN_CONTEXT_WINDOW probing is pointless
PROBE_TIMEOUT_SECONDS = 120        # per attempt (large prefills can be slow locally)


# ── Cache ────────────────────────────────────────────────────────────────────


def _cache_file() -> Path:
    """Path of the probe cache (function so tests can monkeypatch it)."""
    return Path.home() / ".alan" / "context_windows.json"


def _cache_key(model: str, api_base: str | None) -> str:
    return f"{api_base or 'default'}|{model}"


def load_cached_context_window(model: str, api_base: str | None) -> int | None:
    """Return a previously probed context window, or None."""
    path = _cache_file()
    try:
        if not path.is_file():
            return None
        data = json.loads(path.read_text())
        entry = data.get(_cache_key(model, api_base))
        if entry and isinstance(entry.get("context_window"), int):
            return entry["context_window"]
    except (OSError, json.JSONDecodeError) as exc:
        logger.debug("CW cache unreadable (%s); ignoring", exc)
    return None


def save_cached_context_window(
    model: str, api_base: str | None, value: int, method: str,
) -> None:
    """Persist a probed context window. Best-effort - never raises."""
    path = _cache_file()
    try:
        with interprocess_lock(path):
            data: dict = {}
            if path.is_file():
                try:
                    data = json.loads(path.read_text())
                except json.JSONDecodeError:
                    data = {}
            data[_cache_key(model, api_base)] = {
                "context_window": value,
                "method": method,
                "detected_at": datetime.now(timezone.utc).isoformat(),
            }
            atomic_write_json(path, data)
    except OSError as exc:
        logger.warning("Could not write CW cache %s: %s", path, exc)


# ── Probe ────────────────────────────────────────────────────────────────────


@dataclass
class ProbeResult:
    """Outcome of a context-window probe."""

    value: int | None          # detected CW, None when not trustworthy
    method: str                # "probed" | "ceiling_distrust" | "failed"
    detail: str = ""


async def _probe_attempt(
    model: str,
    n_tokens: int,
    *,
    api_key: str | None,
    api_base: str | None,
    timeout: float,
) -> tuple[str, int]:
    """One probe request of roughly *n_tokens* input, asking for 1 output token.

    Returns (outcome, accepted_tokens):
    - ("ok", usage.prompt_tokens) - the request was accepted;
    - ("too_long", 0)             - rejected as over the context limit;
    - ("fatal", 0)                - any other failure (auth, connection...):
                                    probing cannot continue.

    The padding is ``"a "`` repeated: roughly one token each in BPE
    tokenizers. Precision does not matter - on success the backend's own
    ``usage.prompt_tokens`` is the measurement.
    """
    import litellm

    padding = "a " * n_tokens
    try:
        response = await litellm.acompletion(
            model=model,
            messages=[{"role": "user", "content": padding}],
            max_tokens=1,
            api_key=api_key,
            api_base=api_base,
            timeout=timeout,
        )
    except Exception as exc:
        if is_prompt_too_long(str(exc)) or "ContextWindowExceeded" in type(exc).__name__:
            return ("too_long", 0)
        logger.debug("CW probe fatal error at %d tokens: %s", n_tokens, exc)
        return ("fatal", 0)

    accepted = 0
    usage = getattr(response, "usage", None)
    if usage is not None:
        accepted = getattr(usage, "prompt_tokens", 0) or 0
    if accepted <= 0:
        # Backend did not report usage - fall back to the nominal size.
        accepted = n_tokens
    return ("ok", accepted)


async def probe_context_window(
    model: str,
    *,
    api_key: str | None = None,
    api_base: str | None = None,
    ceiling: int = PROBE_CEILING_TOKENS,
    floor: int = PROBE_FLOOR_TOKENS,
    timeout: float = PROBE_TIMEOUT_SECONDS,
) -> ProbeResult:
    """Detect the model's context window by descending probe.

    See the module docstring for the strategy and its cost model.
    """
    sizes: list[int] = []
    size = ceiling
    while size >= floor:
        sizes.append(size)
        size //= 2

    for i, size in enumerate(sizes):
        outcome, accepted = await _probe_attempt(
            model, size, api_key=api_key, api_base=api_base, timeout=timeout,
        )

        if outcome == "fatal":
            return ProbeResult(
                None, "failed", f"probe request failed at {size} tokens"
            )

        if outcome == "too_long":
            continue

        # Success.
        if i == 0:
            # The ceiling itself was accepted: either a >1M-context model
            # (virtually always known to the registry, so we should not be
            # here) or - far more likely - a server that silently truncates
            # instead of erroring. Its acceptance proves nothing.
            return ProbeResult(
                None,
                "ceiling_distrust",
                f"server accepted the {ceiling}-token ceiling without error; "
                f"it likely truncates silently - probe results untrustworthy",
            )

        detected = accepted

        # One refinement inside the bracket (size, 2*size): costs at most
        # one more prefill, recovers up to half the bracket.
        mid = size * 3 // 2
        outcome2, accepted2 = await _probe_attempt(
            model, mid, api_key=api_key, api_base=api_base, timeout=timeout,
        )
        if outcome2 == "ok" and accepted2 > detected:
            detected = accepted2

        return ProbeResult(detected, "probed")

    return ProbeResult(
        None, "failed", f"even the {sizes[-1]}-token floor probe was rejected"
    )
