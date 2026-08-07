"""Token counting and context-window utilities.

We use two signals for token accounting, in this order of preference:

1. **Backend-reported ``usage``** — the exact token counts the API returned
   for the last call. Used directly for display (``/status``, one-liner).
2. **Pre-call estimate** — needed inside the compaction pipeline before an
   API call, because we can't ask the backend yet. Delegated to
   ``litellm.token_counter`` when LiteLLM is available (real model-specific
   tokenizer), otherwise a chars/3 heuristic.

To avoid under-budgeting in the compaction pre-check, we take the
``max`` of:

- ``usage_based``  = last call's input + output + tokens added since then
- ``full_estimate`` = a direct count of the full pre-call payload

A ``TokenEstimator`` / EMA-calibrated ratio used to live here and has
been removed — calibration was numerically broken (see commit notes).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


# ── Public aliases ───────────────────────────────────────────────────────────

MODEL_CONTEXT_WINDOW_DEFAULT = 200_000
MAX_OUTPUT_TOKENS_DEFAULT = 32_000

# Fallback ratio used only when no tokenizer is available.
# 3 chars/token is conservative for most models (English text + code).
CHARS_PER_TOKEN_FALLBACK = 3.0


# ── Raw counting primitives ──────────────────────────────────────────────────


def _chars_to_tokens(chars: int) -> int:
    """Chars -> tokens via the flat fallback ratio."""
    return max(1, int(chars / CHARS_PER_TOKEN_FALLBACK))


def rough_token_count(text: str) -> int:
    """Estimate tokens in a string via the chars/3 fallback."""
    return _chars_to_tokens(len(text))


def _content_block_tokens(block: Any) -> int:
    """Estimate tokens for a single content block (fallback heuristic)."""
    if isinstance(block, str):
        return rough_token_count(block)
    if hasattr(block, "text"):
        return rough_token_count(block.text)
    if hasattr(block, "thinking"):
        return rough_token_count(block.thinking)
    if hasattr(block, "content"):
        inner = block.content
        if isinstance(inner, str):
            return rough_token_count(inner)
        if isinstance(inner, list):
            return sum(_content_block_tokens(b) for b in inner)
    if hasattr(block, "input") and isinstance(block.input, dict):
        name_tokens = rough_token_count(getattr(block, "name", ""))
        input_tokens = rough_token_count(str(block.input))
        return name_tokens + input_tokens
    if hasattr(block, "summary"):
        return rough_token_count(block.summary)
    if hasattr(block, "data"):
        return rough_token_count(str(block.data))
    return 4


def estimate_message_tokens(messages: list) -> int:
    """Estimate tokens for a list of messages using the chars/3 heuristic.

    For a more accurate count that understands the model's tokenizer, use
    :func:`count_tokens_for_call`.
    """
    total = 0
    for msg in messages:
        total += 4  # per-message overhead
        content = getattr(msg, "content", None)
        if content is None:
            if hasattr(msg, "attachment"):
                att = msg.attachment
                total += rough_token_count(getattr(att, "content", ""))
                total += rough_token_count(getattr(att, "type", ""))
            elif hasattr(msg, "summary"):
                total += rough_token_count(msg.summary)
            elif hasattr(msg, "data") and isinstance(msg.data, dict):
                total += rough_token_count(str(msg.data))
            continue
        if isinstance(content, str):
            total += rough_token_count(content)
        elif isinstance(content, list):
            total += sum(_content_block_tokens(b) for b in content)
    return total


def count_message_chars(messages: list) -> int:
    """Count total characters in a message list."""
    total = 0
    for msg in messages:
        content = getattr(msg, "content", None)
        if content is None:
            if hasattr(msg, "attachment"):
                total += len(getattr(msg.attachment, "content", ""))
            elif hasattr(msg, "summary"):
                total += len(msg.summary)
            continue
        if isinstance(content, str):
            total += len(content)
        elif isinstance(content, list):
            total += sum(_count_block_chars(b) for b in content)
    return total


def _count_block_chars(block: Any) -> int:
    if isinstance(block, str):
        return len(block)
    if hasattr(block, "text"):
        return len(block.text)
    if hasattr(block, "thinking"):
        return len(block.thinking)
    if hasattr(block, "content"):
        inner = block.content
        if isinstance(inner, str):
            return len(inner)
        if isinstance(inner, list):
            return sum(_count_block_chars(b) for b in inner)
    if hasattr(block, "input") and isinstance(block.input, dict):
        return len(getattr(block, "name", "")) + len(str(block.input))
    return 0


# ── LiteLLM-backed counting for pre-call estimation ──────────────────────────


def _messages_for_litellm(messages: list) -> list[dict]:
    """Serialize our Message objects into the simple dict shape that
    ``litellm.token_counter`` expects (``role`` + ``content`` string).

    The function is forgiving — anything it can't serialize cleanly is
    skipped so we always produce *some* estimate. We're not trying for
    byte-perfect reproduction here; the caller takes ``max()`` with a
    usage-based count anyway.
    """
    out: list[dict] = []
    for msg in messages:
        role = getattr(msg, "role", None)
        if role is None:
            # Infer role from class name.
            cls = type(msg).__name__
            role = (
                "user" if "User" in cls
                else "assistant" if "Assistant" in cls
                else "system" if "System" in cls
                else "user"
            )
        content = getattr(msg, "content", "")
        if isinstance(content, list):
            # Flatten structured content to a single string of text.
            parts: list[str] = []
            for b in content:
                if hasattr(b, "text") and b.text:
                    parts.append(b.text)
                elif hasattr(b, "thinking") and b.thinking:
                    parts.append(b.thinking)
                elif hasattr(b, "input") and isinstance(b.input, dict):
                    parts.append(str(b.input))
                elif hasattr(b, "content"):
                    inner = b.content
                    if isinstance(inner, str):
                        parts.append(inner)
                    elif isinstance(inner, list):
                        for ib in inner:
                            if hasattr(ib, "text") and ib.text:
                                parts.append(ib.text)
            content = "\n".join(p for p in parts if p)
        elif not isinstance(content, str):
            content = str(content)
        out.append({"role": role, "content": content})
    return out


def count_tokens_for_call(
    model: str | None,
    messages: list,
    *,
    system: str | list[str] | None = None,
    tools: list | None = None,
) -> int:
    """Estimate token count for a prospective API call.

    Uses ``litellm.token_counter`` when LiteLLM is importable (real
    model-specific tokenizer for most mainstream and local models). Falls
    back to the chars/3 heuristic when it isn't or when the model is
    unrecognized.
    """
    # Build the prompt shape.
    msg_dicts = _messages_for_litellm(messages)

    if system:
        if isinstance(system, list):
            system_str = "\n\n".join(system)
        else:
            system_str = system
        if system_str:
            msg_dicts = [{"role": "system", "content": system_str}] + msg_dicts

    try:
        import litellm  # type: ignore
    except Exception:
        litellm = None  # type: ignore

    if litellm is not None and model:
        try:
            kwargs: dict[str, Any] = {"model": model, "messages": msg_dicts}
            if tools:
                kwargs["tools"] = tools
            return int(litellm.token_counter(**kwargs))
        except Exception as exc:
            logger.debug("litellm.token_counter failed (%s); using fallback", exc)

    # Fallback: chars/3 over messages + system + tools-as-str.
    total = estimate_message_tokens(messages)
    if system:
        system_str = "\n\n".join(system) if isinstance(system, list) else system
        total += rough_token_count(system_str)
    if tools:
        # Tools may be schema dicts or Tool objects — stringify defensively.
        total += rough_token_count(str(tools))
    return total


def predicted_next_call_tokens(
    model: str | None,
    messages: list,
    *,
    system: str | list[str] | None = None,
    tools: list | None = None,
    last_input_tokens: int = 0,
    last_output_tokens: int = 0,
    new_messages_since_last_call: list | None = None,
) -> int:
    """Estimate the token count of the upcoming API call.

    Returns ``max(usage_based, full_estimate)`` where:

    - ``usage_based`` = ``last_input_tokens + last_output_tokens + tokens of
      messages added since the last call``. This is close-to-exact when
      the backend populates ``usage``.
    - ``full_estimate`` = ``count_tokens_for_call(messages, ...)`` — a
      tokenizer-backed estimate of the whole upcoming payload.

    Taking the max protects against under-budgeting: if either side is
    wrong, the other caps it conservatively. When the backend doesn't
    populate ``usage`` (``last_input_tokens == 0``), we simply fall
    through to ``full_estimate``.
    """
    full_estimate = count_tokens_for_call(
        model, messages, system=system, tools=tools,
    )

    if last_input_tokens > 0:
        added = 0
        if new_messages_since_last_call:
            added = count_tokens_for_call(model, new_messages_since_last_call)
        usage_based = last_input_tokens + last_output_tokens + added
        return max(usage_based, full_estimate)

    return full_estimate


