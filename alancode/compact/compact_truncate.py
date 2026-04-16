"""Compaction Layer A — truncate oversized tool results.

Truncates individual tool results that exceed the size limit.
"""

from __future__ import annotations

import copy
import logging
from typing import Union

logger = logging.getLogger(__name__)

from alancode.messages.types import (
    Message,
    UserMessage,
    TextBlock,
    ToolResultBlock,
)

from alancode.utils.tokens import estimate_message_tokens

# Sentinel prefix so other compaction passes (and debugging) can tell this
# is synthetic truncation output rather than real tool data.
TRUNCATION_SENTINEL = "[ALAN-TRUNCATED]"
REPLACEMENT_MESSAGE = (
    TRUNCATION_SENTINEL
    + " Tool result truncated — {original_size} chars exceeded {max_size} limit."
)


from alancode.compact.utils import text_length as _text_length


def _truncate_tool_result_content(
    content: str | list[TextBlock],
    max_chars: int,
) -> str | list[TextBlock]:
    """Replace oversized tool result content with a truncation notice.

    Args:
        content: Original tool result content (string or list of TextBlocks).
        max_chars: Maximum allowed character count.

    Returns:
        A replacement string or single-element TextBlock list.
    """
    original_size = _text_length(content)
    replacement = REPLACEMENT_MESSAGE.format(
        original_size=original_size,
        max_size=max_chars,
    )

    if isinstance(content, str):
        return replacement
    # For list[TextBlock], replace with a single TextBlock
    return [TextBlock(text=replacement)]


def _process_tool_result_block(
    block: ToolResultBlock,
    max_chars: int,
) -> ToolResultBlock:
    """Return a copy of the block, truncating its content if it exceeds max_chars.

    Args:
        block: The tool result block to check.
        max_chars: Maximum allowed character count for the content.

    Returns:
        The original block if within limits, or a truncated copy.
    """
    if _text_length(block.content) <= max_chars:
        return block
    return ToolResultBlock(
        tool_use_id=block.tool_use_id,
        content=_truncate_tool_result_content(block.content, max_chars),
        is_error=block.is_error,
    )


def compaction_truncate_tool_results(
    messages: list[Message],
    *,
    max_chars: int | None = None,
    threshold_tokens: int | None = None,
    settings: dict | None = None,
) -> list[Message]:
    """Enforce per-message budget on tool result size (Layer A).

    If *threshold_tokens* is provided, only runs when estimated token count
    exceeds that threshold. Processes oldest tool results first and stops
    when the estimated token count drops below the threshold (or all
    oversized results have been processed).

    Returns a new list (does not mutate input).
    """
    if max_chars is None:
        max_chars = (settings or {}).get("tool_result_max_chars", 20_000)

    # No threshold gate for Layer A: ALWAYS truncate individual results
    # that exceed max_chars. The token estimation heuristic (chars/4) can
    # underestimate significantly, causing oversized results to slip through
    # the threshold gate and overflow the model's context window.

    # Collect indices of oversized tool results (oldest first — natural order)
    oversized: list[tuple[int, int]] = []  # (msg_idx, block_idx)
    for msg_idx, msg in enumerate(messages):
        if not isinstance(msg, UserMessage) or not isinstance(msg.content, list):
            continue
        for block_idx, block in enumerate(msg.content):
            if isinstance(block, ToolResultBlock) and _text_length(block.content) > max_chars:
                oversized.append((msg_idx, block_idx))

    if not oversized:
        return list(messages)

    # Build modified message list, processing oldest first
    # Track which messages need modification
    messages_to_modify: dict[int, set[int]] = {}
    for msg_idx, block_idx in oversized:
        messages_to_modify.setdefault(msg_idx, set()).add(block_idx)

    result: list[Message] = []
    for msg_idx, msg in enumerate(messages):
        if msg_idx not in messages_to_modify:
            result.append(msg)
            continue

        # Guard: only UserMessages with list content should be in
        # messages_to_modify. If the selection logic above ever lets a
        # different type slip through, leave the message untouched rather
        # than corrupting state (especially relevant under python -O
        # where `assert` would be stripped entirely).
        if not (isinstance(msg, UserMessage) and isinstance(msg.content, list)):
            logger.warning(
                "compact_truncate: unexpected message type in modification "
                "set (idx=%d, type=%s); skipping",
                msg_idx, type(msg).__name__,
            )
            result.append(msg)
            continue
        block_indices = messages_to_modify[msg_idx]

        new_content = []
        for block_idx, block in enumerate(msg.content):
            if block_idx in block_indices and isinstance(block, ToolResultBlock):
                new_content.append(_process_tool_result_block(block, max_chars))
            else:
                new_content.append(block)

        # Explicit construction of a new UserMessage: avoids the aliasing
        # hazards of copy.copy() (which shares list/dict fields with the
        # original) and makes exactly-which-fields-propagate visible.
        new_msg = UserMessage(
            content=new_content,
            tool_use_result=msg.tool_use_result,
            hide_in_ui=msg.hide_in_ui,
            hide_in_api=msg.hide_in_api,
            source_tool_assistant_uuid=getattr(msg, "source_tool_assistant_uuid", None),
            origin=getattr(msg, "origin", None),
            uuid=msg.uuid,
            timestamp=msg.timestamp,
        )
        result.append(new_msg)

    # If threshold-gated, check if we can stop early on next call
    # (for simplicity, we process all oversized in one pass — they are individually
    # cheap to truncate and the threshold is re-checked at the layer boundary)
    return result
