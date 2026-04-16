"""Shared utilities for compaction modules."""

from alancode.messages.types import TextBlock


def text_length(content: str | list[TextBlock]) -> int:
    """Get the character length of a ToolResultBlock's content."""
    if isinstance(content, str):
        return len(content)
    return sum(len(block.text) for block in content)
