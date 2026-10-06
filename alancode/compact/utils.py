"""Shared utilities for compaction modules."""

from alancode.messages.types import ImageBlock, TextBlock


def text_length(content: str | list[TextBlock | ImageBlock]) -> int:
    """Get the character length of a ToolResultBlock's text content."""
    if isinstance(content, str):
        return len(content)
    return sum(len(block.text) for block in content if isinstance(block, TextBlock))
