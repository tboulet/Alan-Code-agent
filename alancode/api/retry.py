"""Retry logic with exponential backoff for LLM provider calls."""

import asyncio
import logging
import random
from typing import AsyncGenerator

from alancode.api.errors import (
    OverloadedError,
    PromptTooLongError,
    RateLimitError,
    classify_error,
    is_retryable_error,
)
from alancode.providers.base import (
    LLMProvider,
    ProviderStreamEvent,
    StreamError,
    ThinkingConfig,
    ToolSchema,
)

logger = logging.getLogger(__name__)

DEFAULT_MAX_RETRIES = 3
BASE_DELAY = 1.0  # seconds
MAX_DELAY = 60.0  # seconds


def _compute_delay(attempt: int, retry_after: float | None = None) -> float:
    """Compute the backoff delay for a given attempt.

    Uses exponential backoff with full jitter:
        delay = min(BASE_DELAY * 2^attempt + random(0, 1), MAX_DELAY)

    If the server provided a Retry-After hint, use the larger of
    the computed delay and that hint.
    """
    exp_delay = BASE_DELAY * (2 ** attempt) + random.random()
    delay = min(exp_delay, MAX_DELAY)
    if retry_after is not None and retry_after > delay:
        delay = min(retry_after, MAX_DELAY)
    return delay


def _extract_retry_after(error: Exception) -> float | None:
    """Extract a retry-after hint from a RateLimitError, if present."""
    if isinstance(error, RateLimitError):
        return error.retry_after
    return None


def _stream_error_to_exception(event: StreamError) -> Exception:
    """Convert a StreamError event into a typed exception."""
    msg = event.error
    etype = event.error_type
    status = event.status_code

    if etype == "overloaded" or status == 529:
        return OverloadedError(msg)
    if status == 429 or "rate limit" in msg.lower() or "too many requests" in msg.lower():
        return RateLimitError(msg)
    if "prompt" in msg.lower() and ("too long" in msg.lower() or "context" in msg.lower()):
        return PromptTooLongError(msg)
    return RuntimeError(msg)


async def stream_with_retry(
    provider: LLMProvider,
    messages: list[dict],
    system: list[str],
    tools: list[ToolSchema],
    *,
    model: str | None = None,
    max_tokens: int | None = None,
    thinking: ThinkingConfig | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    fallback_provider: LLMProvider | None = None,
    **kwargs,
) -> AsyncGenerator[ProviderStreamEvent, None]:
    """Stream with automatic retry on transient errors.

    Implements exponential backoff with jitter.  On persistent failures
    after exhausting all retries, optionally falls back to
    ``fallback_provider``.

    Non-retryable errors (e.g. prompt-too-long, invalid request) are
    raised immediately without consuming retry budget.

    Yields:
        ProviderStreamEvent instances from the underlying provider.

    Raises:
        The last encountered exception after all retries (and optional
        fallback) are exhausted.
    """
    # Initialize with a concrete exception so we never hit a `None` here.
    # Overwritten by the real error on any retryable failure.
    last_error: Exception = RuntimeError(
        "API call failed with no error detail recorded"
    )

    for attempt in range(max_retries + 1):
        try:
            stream = provider.stream(
                messages,
                system,
                tools,
                model=model,
                max_tokens=max_tokens,
                thinking=thinking,
                **kwargs,
            )
            # Buffer events so we can detect mid-stream errors before
            # yielding partial content on a retry-eligible failure.
            # However, for efficiency we yield eagerly and only retry
            # on errors that arrive *before* any content events.
            events_yielded = 0
            async for event in stream:
                # Detect StreamError events from the provider
                if isinstance(event, StreamError):
                    exc = _stream_error_to_exception(event)
                    if not is_retryable_error(exc):
                        raise exc
                    if events_yielded > 0:
                        # Already yielded content; cannot transparently
                        # retry. Re-raise so the caller can handle it.
                        raise exc
                    # Retryable error before any content -- break to retry loop
                    last_error = exc
                    break
                else:
                    yield event
                    events_yielded += 1
            else:
                # Stream completed normally (no break)
                return

            # If we broke out due to a retryable StreamError, fall through
            # to the retry logic below.

        except Exception as exc:
            last_error = exc
            category = classify_error(exc)

            if not is_retryable_error(exc):
                logger.error(
                    "Non-retryable error (category=%s): %s", category, exc
                )
                raise

        # We have a retryable error.  Log and back off.
        if attempt < max_retries:
            retry_after = _extract_retry_after(last_error)
            delay = _compute_delay(attempt, retry_after)
            logger.warning(
                "Retryable error on attempt %d/%d (category=%s): %s  "
                "Retrying in %.1fs...",
                attempt + 1,
                max_retries + 1,
                classify_error(last_error),
                last_error,
                delay,
            )
            await asyncio.sleep(delay)
        # else: will exit the loop

    # All retries exhausted.  Try fallback provider if available.
    if fallback_provider is not None:
        logger.warning(
            "All %d retries exhausted. Falling back to fallback provider.",
            max_retries + 1,
        )
        try:
            stream = fallback_provider.stream(
                messages,
                system,
                tools,
                model=model,
                max_tokens=max_tokens,
                thinking=thinking,
                **kwargs,
            )
            async for event in stream:
                if isinstance(event, StreamError):
                    raise _stream_error_to_exception(event)
                yield event
            return
        except Exception as fallback_exc:
            logger.error("Fallback provider also failed: %s", fallback_exc)
            # Raise the original error, chained with the fallback error
            raise last_error from fallback_exc

    # No fallback or fallback not configured -- raise the last error
    logger.error(
        "All %d retries exhausted. Raising last error: %s",
        max_retries + 1,
        last_error,
    )
    raise last_error
