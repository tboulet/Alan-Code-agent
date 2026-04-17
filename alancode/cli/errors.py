"""Error classification and display for the CLI."""

from __future__ import annotations


def classify_error(error: Exception) -> tuple[str, str | None]:
    """Return (message, optional_hint) for display.

    The message always includes the original error text. The hint is an
    optional suggestion based on pattern matching — never a replacement
    for the real error.
    """
    msg = str(error)
    msg_lower = msg.lower()

    hint = None

    if "auth" in msg_lower or "api key" in msg_lower or "401" in msg:
        hint = "Check your API key and provider settings."

    elif "rate limit" in msg_lower or "429" in msg:
        hint = "Wait a moment and try again."

    elif "connection" in msg_lower or "timeout" in msg_lower or "network" in msg_lower:
        hint = "Check your internet connection."

    elif "tool calling" in msg_lower or "function calling" in msg_lower:
        hint = (
            "For models without native tool calling, use "
            "'/settings-project tool_call_format=hermes' to enable text-based tool calling."
        )

    elif "context" in msg_lower and ("long" in msg_lower or "exceeded" in msg_lower):
        hint = "Try /compact or /clear."

    return f"Error: {msg}", hint
