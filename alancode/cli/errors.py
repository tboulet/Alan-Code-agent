"""Error classification and display for the CLI."""

from __future__ import annotations


def classify_error(error: Exception) -> tuple[str, str | None]:
    """Classify an error into a category and return (message, hint).

    Returns:
        A tuple of (display_message, optional_hint).
    """
    msg = str(error)
    msg_lower = msg.lower()

    if "auth" in msg_lower or "api key" in msg_lower or "401" in msg:
        return "Authentication error: Check your API key and provider settings.", None

    if "rate limit" in msg_lower or "429" in msg:
        return "Rate limited. Wait a moment and try again.", None

    if "connection" in msg_lower or "timeout" in msg_lower or "network" in msg_lower:
        return f"Network error: {msg}", "Check your internet connection."

    if "tool calling" in msg_lower or "function calling" in msg_lower:
        return (
            f"Model error: {msg}",
            "If you believe this model supports tools, use "
            "'/settings-project force_supports_tools=true'. "
            "For text-based tool calling, use '/settings-project tool_call_format=hermes' or another format.",
        )

    if "context" in msg_lower and ("long" in msg_lower or "exceeded" in msg_lower):
        return "Context too long. Try /compact or /clear.", None

    return f"Error: {msg}", None
