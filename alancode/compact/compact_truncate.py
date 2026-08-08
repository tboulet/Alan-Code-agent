"""Compaction Layer A — truncate oversized tool results.

Truncates individual tool results that exceed the size limit.
"""

from __future__ import annotations

import logging

from alancode.compact.utils import text_length as _text_length
from alancode.messages.types import (
    Message,
    UserMessage,
    TextBlock,
    ToolResultBlock,
)

logger = logging.getLogger(__name__)

# Sentinel prefix so other compaction passes (and debugging) can tell this
# is synthetic truncation output rather than real tool data.
TRUNCATION_SENTINEL = "[ALAN-TRUNCATED"

# Per-result cap used when no resolved budget is available (legacy path /
# "auto" setting without a ContextBudget). The budget's auto formula
# (min(10k, 10% of threshold)) supersedes this whenever a budget exists.
DEFAULT_TOOL_RESULT_MAX_CHARS = 20_000

# Head/tail split of the kept characters: starts carry structure (headers,
# first errors), tails carry conclusions and final state.
HEAD_FRACTION = 0.6

def _truncate_tool_result_content(
    content: str | list[TextBlock],
    max_chars: int,
) -> str | list[TextBlock]:
    """Truncate oversized tool result content middle-out.

    Keeps the first 60% and last 40% of the character budget, with a
    sentinel in the middle stating what was elided, e.g.::

        [ALAN-TRUNCATED: middle 78% of output elided (84,213 of 108,000 chars)]

    Args:
        content: Original tool result content (string or list of TextBlocks).
        max_chars: Maximum allowed character count for the kept content.

    Returns:
        A truncated string or single-element TextBlock list.
    """
    text = (
        content
        if isinstance(content, str)
        else "".join(b.text for b in content)
    )
    original_size = len(text)

    # The cap includes the sentinel and its surrounding newlines. Reserve
    # enough room for the longest sentinel this result can produce, then
    # divide the remaining content budget between the head and tail.
    max_sentinel = (
        f"{TRUNCATION_SENTINEL}: middle 100% of output elided "
        f"({original_size:,} of {original_size:,} chars)]"
    )
    content_budget = max(0, max_chars - len(max_sentinel) - 2)
    head_len = int(content_budget * HEAD_FRACTION)
    tail_len = content_budget - head_len
    elided = original_size - head_len - tail_len
    elided_pct = round(elided * 100 / original_size) if original_size else 0

    sentinel = (
        f"{TRUNCATION_SENTINEL}: middle {elided_pct}% of output elided "
        f"({elided:,} of {original_size:,} chars)]"
    )
    if len(sentinel) + 2 > max_chars:
        truncated = sentinel[:max_chars]
    else:
        tail = text[original_size - tail_len:] if tail_len else ""
        truncated = text[:head_len] + "\n" + sentinel + "\n" + tail

    if isinstance(content, str):
        return truncated
    return [TextBlock(text=truncated)]


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
    settings: dict | None = None,
) -> tuple[list[Message], int]:
    """Enforce a per-result size cap on tool results (Layer A).

    Always on - every individual tool result whose content exceeds
    *max_chars* is truncated middle-out (head + tail kept, sentinel in
    between), regardless of total conversation size.

    Returns (new_messages, truncated_count). The count lets the caller
    know its usage-based token estimates are stale (content was removed
    since the last API call measured it).
    """
    if max_chars is None:
        max_chars = (settings or {}).get(
            "tool_result_max_chars", DEFAULT_TOOL_RESULT_MAX_CHARS
        )
        if not isinstance(max_chars, int):
            # "auto" without a resolved budget: legacy default
            max_chars = DEFAULT_TOOL_RESULT_MAX_CHARS

    # Collect indices of oversized tool results (oldest first — natural order)
    oversized: list[tuple[int, int]] = []  # (msg_idx, block_idx)
    for msg_idx, msg in enumerate(messages):
        if not isinstance(msg, UserMessage) or not isinstance(msg.content, list):
            continue
        for block_idx, block in enumerate(msg.content):
            if isinstance(block, ToolResultBlock) and _text_length(block.content) > max_chars:
                oversized.append((msg_idx, block_idx))

    if not oversized:
        return list(messages), 0

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

    return result, len(oversized)
