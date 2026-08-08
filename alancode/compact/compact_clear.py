"""Compaction Layer B — clear old tool result content.

Clears content from old tool results while preserving the message structure.
"""

from __future__ import annotations

from alancode.messages.types import (
    AssistantMessage,
    Message,
    UserMessage,
    ToolResultBlock,
    ToolUseBlock,
)
from alancode.utils.tokens import rough_token_count, estimate_message_tokens

COMPACTABLE_TOOLS = {"Bash", "Read", "Grep", "Glob", "WebSearch", "WebFetch", "Edit", "Write"}
CLEARED_MESSAGE = "[Old tool result content cleared]"

def _estimate_block_tokens(block: ToolResultBlock) -> int:
    """Estimate token count for a tool result block's content.

    Args:
        block: A ToolResultBlock whose content may be a string or list of TextBlocks.

    Returns:
        Approximate token count using the chars/3 heuristic (see utils.tokens).
    """
    if isinstance(block.content, str):
        return rough_token_count(block.content)
    return sum(rough_token_count(tb.text) for tb in block.content)


def _find_tool_name_for_result(messages: list[Message], tool_use_id: str) -> str | None:
    """Walk backwards through messages to find the tool name for a given tool_use_id.

    The ToolUseBlock (in an AssistantMessage) has the tool name; the ToolResultBlock
    (in a UserMessage) only has the tool_use_id.
    """
    for msg in reversed(messages):
        if not isinstance(msg, AssistantMessage):
            continue
        for block in msg.content:
            if isinstance(block, ToolUseBlock) and block.id == tool_use_id:
                return block.name
    return None


def _collect_tool_result_indices(
    messages: list[Message],
    compactable_tools: set[str],
) -> list[tuple[int, int, str]]:
    """Collect (message_index, block_index, tool_use_id) for all compactable tool results.

    Returns them in order of appearance (oldest first).
    """
    indices: list[tuple[int, int, str]] = []

    for msg_idx, msg in enumerate(messages):
        if not isinstance(msg, UserMessage) or not isinstance(msg.content, list):
            continue
        for block_idx, block in enumerate(msg.content):
            if not isinstance(block, ToolResultBlock):
                continue
            tool_name = _find_tool_name_for_result(messages, block.tool_use_id)
            if tool_name is not None and tool_name in compactable_tools:
                indices.append((msg_idx, block_idx, block.tool_use_id))

    return indices


def compaction_clear_tool_results(
    messages: list[Message],
    *,
    clear_target_tokens: int,
    compactable_tools: set[str] = COMPACTABLE_TOOLS,
) -> tuple[list[Message], int]:
    """Clear old tool result content down to a token target (Layer B).

    Recent results are protected by the target itself: clearing stops at
    G, and with the per-result cap at 10% of T a single clear can never
    jump the estimate from above G to below T.

    Returns (new_messages, tokens_saved).
    Only clears tool results from compactable tools.
    """
    # Gate: nothing to do at or below the target.
    current_tokens = estimate_message_tokens(messages)
    if current_tokens <= clear_target_tokens:
        return list(messages), 0

    # Find all compactable tool result locations (oldest first).
    tool_result_indices = _collect_tool_result_indices(messages, compactable_tools)

    if not tool_result_indices:
        return list(messages), 0

    # Clear oldest first, stopping as soon as the estimate reaches the target.
    indices_to_clear_list: list[tuple[int, int]] = []
    running_saved = 0
    for msg_idx, block_idx, _ in tool_result_indices:
        msg = messages[msg_idx]
        if not isinstance(msg, UserMessage) or not isinstance(msg.content, list):
            continue
        block = msg.content[block_idx]
        if isinstance(block, ToolResultBlock):
            old_tokens = _estimate_block_tokens(block)
            new_tokens = rough_token_count(CLEARED_MESSAGE)
            saved = max(0, old_tokens - new_tokens)
            indices_to_clear_list.append((msg_idx, block_idx))
            running_saved += saved
            if current_tokens - running_saved <= clear_target_tokens:
                break
    indices_to_clear = set(indices_to_clear_list)

    if not indices_to_clear:
        return list(messages), 0

    # Track which messages need modification
    messages_to_modify: dict[int, set[int]] = {}
    for msg_idx, block_idx in indices_to_clear:
        messages_to_modify.setdefault(msg_idx, set()).add(block_idx)

    tokens_saved = 0
    new_messages: list[Message] = []

    for msg_idx, msg in enumerate(messages):
        if msg_idx not in messages_to_modify:
            new_messages.append(msg)
            continue

        # This message has tool results to clear
        if not isinstance(msg, UserMessage) or not isinstance(msg.content, list):
            continue

        block_indices_to_clear = messages_to_modify[msg_idx]
        new_content = []

        for block_idx, block in enumerate(msg.content):
            if block_idx not in block_indices_to_clear or not isinstance(block, ToolResultBlock):
                new_content.append(block)
                continue

            # Calculate tokens saved before clearing
            old_tokens = _estimate_block_tokens(block)
            new_tokens = rough_token_count(CLEARED_MESSAGE)
            tokens_saved += max(0, old_tokens - new_tokens)

            # Replace with cleared version
            new_content.append(
                ToolResultBlock(
                    tool_use_id=block.tool_use_id,
                    content=CLEARED_MESSAGE,
                    is_error=block.is_error,
                )
            )

        # Explicit construction: avoid aliasing mutable fields of `msg`.
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
        new_messages.append(new_msg)

    return new_messages, tokens_saved
