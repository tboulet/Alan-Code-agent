"""Message serialization — convert message dataclasses to API dict format.

Two serialization targets:
- **OpenAI format** (``messages_to_openai_dicts``) — the universal default.
  Used by LiteLLM and any OpenAI-compatible provider.
- **Anthropic format** (``message_to_anthropic_dict``) — used by AnthropicProvider.

The query loop and compaction produce OpenAI-format dicts. Each provider
translates if needed.
"""

from __future__ import annotations

import json
from typing import Any

from alancode.messages.types import (
    AssistantMessage,
    ImageBlock,
    RedactedThinkingBlock,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)


# ── Anthropic format (used by AnthropicProvider) ────────────────────────────


def block_to_anthropic_dict(block: Any) -> dict[str, Any]:
    """Convert a content block to Anthropic API dict format."""
    if isinstance(block, TextBlock):
        return {"type": "text", "text": block.text}
    if isinstance(block, ToolUseBlock):
        return {
            "type": "tool_use",
            "id": block.id,
            "name": block.name,
            "input": block.input,
        }
    if isinstance(block, ToolResultBlock):
        content = block.content
        if isinstance(content, list):
            content = [block_to_anthropic_dict(b) for b in content]
        return {
            "type": "tool_result",
            "tool_use_id": block.tool_use_id,
            "content": content,
            "is_error": block.is_error,
        }
    if isinstance(block, ThinkingBlock):
        d: dict[str, Any] = {"type": "thinking", "thinking": block.thinking}
        if block.signature:
            d["signature"] = block.signature
        return d
    if isinstance(block, RedactedThinkingBlock):
        return {"type": "redacted_thinking", "data": block.data}
    if isinstance(block, ImageBlock):
        return {"type": "image", "source": block.source}
    return {"type": "unknown"}


def message_to_anthropic_dict(msg: UserMessage | AssistantMessage) -> dict[str, Any]:
    """Convert a message to Anthropic API dict format.

    Anthropic format:
    - User: ``{"role": "user", "content": [{"type": "tool_result", ...}, ...]}``
    - Assistant: ``{"role": "assistant", "content": [{"type": "tool_use", ...}, ...]}``
    """
    if isinstance(msg, UserMessage):
        if isinstance(msg.content, str):
            return {"role": "user", "content": msg.content}
        return {
            "role": "user",
            "content": [block_to_anthropic_dict(b) for b in msg.content],
        }
    # AssistantMessage
    return {
        "role": "assistant",
        "content": [block_to_anthropic_dict(b) for b in msg.content],
    }


# ── OpenAI format (universal default) ───────────────────────────────────────


def messages_to_openai_dicts(
    messages: list[UserMessage | AssistantMessage],
) -> list[dict[str, Any]]:
    """Convert a list of messages to OpenAI API dict format.

    One internal message may produce multiple OpenAI dicts:
    - A UserMessage with tool_result blocks becomes multiple ``role: "tool"``
      messages plus an optional ``role: "user"`` message for remaining text.
    - An AssistantMessage with tool_use blocks becomes one message with
      ``content`` (text) and ``tool_calls`` (structured tool invocations).

    OpenAI format:
    - User: ``{"role": "user", "content": "text"}``
    - Assistant: ``{"role": "assistant", "content": "text", "tool_calls": [...]}``
    - Tool result: ``{"role": "tool", "tool_call_id": "...", "content": "..."}``
    """
    result: list[dict[str, Any]] = []

    for msg in messages:
        if isinstance(msg, AssistantMessage):
            result.extend(_assistant_to_openai(msg))
        elif isinstance(msg, UserMessage):
            result.extend(_user_to_openai(msg))
        else:
            # Pass through unknown message types
            result.append({"role": "user", "content": str(msg)})

    return result


def _assistant_to_openai(msg: AssistantMessage) -> list[dict[str, Any]]:
    """Convert an AssistantMessage to OpenAI format.

    Splits content into text (``content``) and tool calls (``tool_calls``).
    """
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []

    for block in msg.content:
        if isinstance(block, TextBlock):
            text_parts.append(block.text)
        elif isinstance(block, ToolUseBlock):
            tool_calls.append({
                "id": block.id,
                "type": "function",
                "function": {
                    "name": block.name,
                    "arguments": json.dumps(block.input) if isinstance(block.input, dict) else str(block.input),
                },
            })
        # ThinkingBlock, RedactedThinkingBlock — not included in OpenAI format

    d: dict[str, Any] = {
        "role": "assistant",
        "content": "\n".join(text_parts) if text_parts else None,
    }
    if tool_calls:
        d["tool_calls"] = tool_calls

    return [d]


def _user_to_openai(msg: UserMessage) -> list[dict[str, Any]]:
    """Convert a UserMessage to OpenAI format.

    A UserMessage with tool_result blocks is split into:
    - ``role: "tool"`` messages (one per tool result)
    - ``role: "user"`` message for any remaining text content
    """
    if isinstance(msg.content, str):
        return [{"role": "user", "content": msg.content}]

    result: list[dict[str, Any]] = []
    tool_results = [b for b in msg.content if isinstance(b, ToolResultBlock)]
    other_blocks = [b for b in msg.content if not isinstance(b, ToolResultBlock)]

    # Emit tool result messages first (role=tool)
    for tr in tool_results:
        tr_content = tr.content
        if isinstance(tr_content, list):
            tr_content = "\n".join(
                b.text if isinstance(b, TextBlock) else str(b)
                for b in tr_content
            )
        result.append({
            "role": "tool",
            "tool_call_id": tr.tool_use_id,
            "content": str(tr_content),
        })

    # Emit remaining user content (if any)
    if other_blocks:
        text_parts = []
        for b in other_blocks:
            if isinstance(b, TextBlock):
                text_parts.append(b.text)
            else:
                text_parts.append(str(b))
        if text_parts:
            result.append({"role": "user", "content": "\n".join(text_parts)})

    return result
