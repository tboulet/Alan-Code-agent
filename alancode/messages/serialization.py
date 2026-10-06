"""Message serialization — convert message dataclasses to API dict format.

Two serialization targets:
- **OpenAI format** (``messages_to_openai_dicts``) — the universal default.
  Used by LiteLLM and any OpenAI-compatible backend.
- **Anthropic format** (``message_to_anthropic_dict``) — used by AnthropicBackend.

The query loop and compaction produce OpenAI-format dicts. Each backend
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


# ── Anthropic format (used by AnthropicBackend) ────────────────────────────


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

IMAGE_PLACEHOLDER = "[image]"
TOOL_IMAGES_NOTE = "Image(s) returned by the tool call above:"


def messages_to_openai_dicts(
    messages: list[UserMessage | AssistantMessage],
    *,
    include_thinking: bool = False,
    text_dialect: bool = False,
    include_images: bool = True,
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

    ``include_thinking`` renders each ThinkingBlock back into the assistant
    content as inline ``<think>...</think>`` text (the ``persist_thinking``
    setting), so models whose state lives in their reasoning can re-see it.

    ``text_dialect`` replays a tool call as the markup the model wrote and its
    result as an ordinary user message. A structured ``tool_calls`` entry is
    rendered by the server's chat template into that model's NATIVE markup,
    carrying ids alancode minted - so a model taught a text dialect sees a
    different one in its own history and imitates it.

    An image in a tool result is sent as an ``image_url`` part of a user
    message right after the results, since a ``role: "tool"`` message carries
    text only. ``include_images=False`` sends ``IMAGE_PLACEHOLDER`` instead.
    """
    result: list[dict[str, Any]] = []

    for msg in messages:
        if isinstance(msg, AssistantMessage):
            result.extend(_assistant_to_openai(
                msg, include_thinking=include_thinking, text_dialect=text_dialect,
            ))
        elif isinstance(msg, UserMessage):
            result.extend(_user_to_openai(
                msg, text_dialect=text_dialect, include_images=include_images,
            ))
        else:
            # Pass through unknown message types
            result.append({"role": "user", "content": str(msg)})

    return result


def _assistant_to_openai(
    msg: AssistantMessage, *, include_thinking: bool = False,
    text_dialect: bool = False,
) -> list[dict[str, Any]]:
    """Convert an AssistantMessage to OpenAI format.

    Splits content into text (``content``) and tool calls (``tool_calls``).
    """
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []

    for block in msg.content:
        if isinstance(block, TextBlock):
            text_parts.append(block.text)
        elif isinstance(block, ToolUseBlock):
            if text_dialect and block.raw_text:
                text_parts.append(block.raw_text)
                continue
            tool_calls.append({
                "id": block.id,
                "type": "function",
                "function": {
                    "name": block.name,
                    "arguments": json.dumps(block.input) if isinstance(block.input, dict) else str(block.input),
                },
            })
        elif isinstance(block, ThinkingBlock) and include_thinking and block.thinking:
            text_parts.append(f"<think>{block.thinking}</think>")
        # ThinkingBlock (when not persisted), RedactedThinkingBlock - not
        # included in OpenAI format

    d: dict[str, Any] = {
        "role": "assistant",
        # The OpenAI API permits null content when tool_calls are present,
        # but strict compatible servers such as Ollama reject JSON null.
        # An empty string is accepted by both and preserves the same meaning.
        "content": "\n".join(text_parts) if text_parts else "",
    }
    if tool_calls:
        d["tool_calls"] = tool_calls

    return [d]


def _image_part(block: ImageBlock) -> dict[str, Any]:
    source = block.source
    url = source.get("url") or (
        f"data:{source.get('media_type', '')};base64,{source.get('data', '')}"
    )
    return {"type": "image_url", "image_url": {"url": url}}


def _split_text_and_images(
    blocks: list[Any], *, include_images: bool,
) -> tuple[str, list[dict[str, Any]]]:
    """Join the text of ``blocks``; return their images as image_url parts,
    or mark them in the text when images are not sent."""
    texts: list[str] = []
    images: list[dict[str, Any]] = []
    for b in blocks:
        if isinstance(b, TextBlock):
            texts.append(b.text)
        elif isinstance(b, ImageBlock):
            if include_images:
                images.append(_image_part(b))
            else:
                texts.append(IMAGE_PLACEHOLDER)
        else:
            texts.append(str(b))
    return "\n".join(texts), images


def _user_content(text: str, images: list[dict[str, Any]]) -> str | list[dict[str, Any]]:
    if not images:
        return text
    return ([{"type": "text", "text": text}] if text else []) + images


def _user_to_openai(
    msg: UserMessage, *, text_dialect: bool = False, include_images: bool = True,
) -> list[dict[str, Any]]:
    """Convert a UserMessage to OpenAI format.

    A UserMessage with tool_result blocks is split into:
    - ``role: "tool"`` messages (one per tool result)
    - ``role: "user"`` message for any remaining text content

    Under ``text_dialect`` the results become ordinary user content instead:
    a ``role: "tool"`` entry references a tool_calls id, and the assistant
    side emits none, so a strict server would reject the orphan.
    """
    if isinstance(msg.content, str):
        return [{"role": "user", "content": msg.content}]

    result: list[dict[str, Any]] = []
    tool_results = [b for b in msg.content if isinstance(b, ToolResultBlock)]
    other_blocks = [b for b in msg.content if not isinstance(b, ToolResultBlock)]

    dialect_texts: list[str] = []
    result_images: list[dict[str, Any]] = []
    for tr in tool_results:
        tr_content = tr.content
        if isinstance(tr_content, list):
            tr_content, images = _split_text_and_images(
                tr_content, include_images=include_images,
            )
            result_images.extend(images)
        if text_dialect:
            dialect_texts.append(str(tr_content))
            continue
        result.append({
            "role": "tool",
            "tool_call_id": tr.tool_use_id,
            "content": str(tr_content),
        })
    if text_dialect and (dialect_texts or result_images):
        result.append({
            "role": "user",
            "content": _user_content("\n".join(dialect_texts), result_images),
        })
    elif result_images:
        result.append({
            "role": "user",
            "content": _user_content(TOOL_IMAGES_NOTE, result_images),
        })

    # Emit remaining user content (if any)
    if other_blocks:
        text, images = _split_text_and_images(
            other_blocks, include_images=include_images,
        )
        if text or images:
            result.append({"role": "user", "content": _user_content(text, images)})

    return result
