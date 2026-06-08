"""API error classification and detection."""


class PromptTooLongError(Exception):
    """The prompt exceeded the model's context window."""

    def __init__(self, message: str, token_gap: int | None = None):
        super().__init__(message)
        self.token_gap = token_gap


class MaxOutputTokensError(Exception):
    """The response hit the max_output_tokens limit."""


class RateLimitError(Exception):
    """Rate limit exceeded."""

    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class OverloadedError(Exception):
    """API is overloaded (529)."""


# ── Detection helpers ──────────────────────────────────────────────────────


_PROMPT_TOO_LONG_PATTERNS = (
    # OpenAI / vLLM / NIM / generic OpenAI-compatible
    "context length",
    "context window",
    "maximum context",
    "context_length_exceeded",
    "exceeds the model",
    "exceeds the model's maximum",
    # Anthropic / Mistral / generic prose
    "prompt is too long",
    "prompt too long",
    "prompt is too large",
    # SGLang ("Input length (X) exceeds the maximum allowed length (Y)")
    "exceeds the maximum allowed length",
    "maximum allowed length",
    "input length",
    # TGI ("Input validation error: `inputs` tokens + `max_new_tokens` …")
    "input validation error",
    "inputs` tokens",
    # Ollama (truncation warnings — model still responds, but worth catching)
    "exceed context length",
    "truncating input messages which exceed",
    # Generic
    "token limit",
    "too many tokens",
    "requested token count exceeds",
)


def is_prompt_too_long(error_message: str) -> bool:
    """Return True if the error message indicates a prompt-too-long failure."""
    lower = error_message.lower()
    return any(pattern in lower for pattern in _PROMPT_TOO_LONG_PATTERNS)


_RETRYABLE_TYPES = (RateLimitError, OverloadedError, ConnectionError, TimeoutError)


def is_retryable_error(error: Exception) -> bool:
    """Return True if the error is transient and worth retrying.

    Retryable errors include rate limits, overloaded servers, connection
    errors, and timeouts.  Prompt-too-long and max-output-tokens errors
    are *not* retryable because re-sending the identical request will
    produce the same failure.
    """
    if isinstance(error, _RETRYABLE_TYPES):
        return True
    # Some providers wrap transient failures in generic exceptions.
    msg = str(error).lower()
    if any(kw in msg for kw in ("rate limit", "429", "529", "overloaded", "too many requests")):
        return True
    if any(kw in msg for kw in ("connection", "timeout", "timed out", "reset by peer")):
        return True
    return False


def classify_error(error: Exception) -> str:
    """Classify an exception into a human-readable category string.

    Categories:
        'prompt_too_long'   - context window exceeded
        'max_output_tokens' - output length limit hit
        'rate_limit'        - 429 / rate-limit
        'overloaded'        - 529 / server overloaded
        'connection'        - network-level failure
        'timeout'           - request timed out
        'unknown'           - anything else
    """
    if isinstance(error, PromptTooLongError):
        return "prompt_too_long"
    if isinstance(error, MaxOutputTokensError):
        return "max_output_tokens"
    if isinstance(error, RateLimitError):
        return "rate_limit"
    if isinstance(error, OverloadedError):
        return "overloaded"
    if isinstance(error, ConnectionError):
        return "connection"
    if isinstance(error, TimeoutError):
        return "timeout"

    # Heuristic fallback based on message text
    msg = str(error).lower()
    if is_prompt_too_long(msg):
        return "prompt_too_long"
    if "rate limit" in msg or "429" in msg or "too many requests" in msg:
        return "rate_limit"
    if "overloaded" in msg or "529" in msg:
        return "overloaded"
    if any(kw in msg for kw in ("connection", "reset by peer")):
        return "connection"
    if "timeout" in msg or "timed out" in msg:
        return "timeout"

    return "unknown"
