"""Message normalization for API submission.

Transforms the internal message list (which may contain system messages,
attachments, virtual display-only messages, and progress events) into the
strict user/assistant alternation required by the Claude API.
"""

from __future__ import annotations

import logging
from copy import deepcopy

from alancode.messages.types import (
    AssistantMessage,
    AttachmentMessage,
    Message,
    ProgressMessage,
    SystemMessage,
    SystemMessageSubtype,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserContentBlock,
    UserMessage,
)

logger = logging.getLogger(__name__)


# ── Public API ──────────────────────────────────────────────────────────────


def normalize_messages_for_api(
    messages: list[Message],
) -> list[UserMessage | AssistantMessage]:
    """Transform internal messages to API-ready format.

    Steps:
        1. Filter out virtual messages (``hide_in_api=True``).
        2. Filter out system messages — except ``local_command``, which is
           converted to a UserMessage.
        3. Convert attachment messages to UserMessages carrying the
           attachment content as text.
        4. Filter out progress messages.
        5. Merge consecutive same-role messages (required for Bedrock
           compatibility and to satisfy API alternation constraints).
        6. Return only UserMessage and AssistantMessage instances.
    """
    result: list[UserMessage | AssistantMessage] = []

    for msg in messages:
        converted = _convert_message(msg)
        if converted is None:
            continue

        # Merge consecutive same-role messages
        if result and _same_role(result[-1], converted):
            if isinstance(result[-1], UserMessage) and isinstance(converted, UserMessage):
                result[-1] = merge_user_messages(result[-1], converted)
            elif isinstance(result[-1], AssistantMessage) and isinstance(converted, AssistantMessage):
                result[-1] = _merge_assistant_messages(result[-1], converted)
        else:
            result.append(converted)

    # Drop orphan tool_result blocks whose tool_use_id has no matching
    # tool_use in any preceding assistant message. After the merge pass
    # it's possible for an orphan to sneak in (the pairing helper runs
    # before merging). Any orphan tool_result would cause the API to
    # reject the request with 400.
    _drop_orphan_tool_results(result)
    _answer_unanswered_tool_uses(result)

    return result


INTERRUPTED_TOOL_RESULT = (
    "This tool call was interrupted before it returned a result (the session "
    "was stopped). It may or may not have run: check its effects before "
    "repeating it."
)


def _answer_unanswered_tool_uses(
    messages: list[UserMessage | AssistantMessage],
) -> None:
    """Give every tool_use without a tool_result an error result. Mutates
    ``messages`` in place.

    A session killed while a tool ran is saved ending on the call; resumed,
    the request carried a tool_use with no result, which strict APIs reject.
    """
    i = 0
    while i < len(messages):
        msg = messages[i]
        if not isinstance(msg, AssistantMessage):
            i += 1
            continue
        tool_use_ids = [b.id for b in msg.content if isinstance(b, ToolUseBlock)]
        nxt = messages[i + 1] if i + 1 < len(messages) else None
        answered: set[str] = set()
        if isinstance(nxt, UserMessage) and isinstance(nxt.content, list):
            answered = {
                b.tool_use_id for b in nxt.content if isinstance(b, ToolResultBlock)
            }
        missing = [
            ToolResultBlock(
                tool_use_id=tool_use_id,
                content=INTERRUPTED_TOOL_RESULT,
                is_error=True,
            )
            for tool_use_id in tool_use_ids
            if tool_use_id not in answered
        ]
        if missing:
            logger.warning(
                "Answered %d tool call(s) left without a result", len(missing),
            )
            if isinstance(nxt, UserMessage):
                # Results first: the API wants them ahead of other content.
                answered_msg = deepcopy(nxt)
                answered_msg.content = missing + _to_content_list(nxt.content)
                messages[i + 1] = answered_msg
            else:
                messages.insert(i + 1, UserMessage(content=missing))
        i += 1


def _drop_orphan_tool_results(
    messages: list[UserMessage | AssistantMessage],
) -> None:
    """Strip tool_result blocks whose tool_use_id never appeared in a
    prior assistant tool_use. Mutates ``messages`` in place.
    """
    known_tool_use_ids: set[str] = set()
    for msg in messages:
        if isinstance(msg, AssistantMessage):
            for block in msg.content:
                if isinstance(block, ToolUseBlock):
                    known_tool_use_ids.add(block.id)
            continue
        # UserMessage: drop orphan tool_results
        if not isinstance(msg.content, list):
            continue
        kept = []
        dropped = 0
        for block in msg.content:
            if (
                isinstance(block, ToolResultBlock)
                and block.tool_use_id
                and block.tool_use_id not in known_tool_use_ids
            ):
                dropped += 1
                continue
            kept.append(block)
        if dropped:
            logger.warning(
                "Dropped %d orphan tool_result block(s) during normalization",
                dropped,
            )
            msg.content = kept


def merge_user_messages(a: UserMessage, b: UserMessage) -> UserMessage:
    """Merge two consecutive user messages into one.

    Content lists are concatenated.  If either message has plain-string
    content it is first wrapped in a :class:`TextBlock`.
    """
    merged = deepcopy(a)
    merged.content = _to_content_list(a.content) + _to_content_list(b.content)
    return merged


def get_text_content(message: UserMessage | AssistantMessage) -> str:
    """Extract the concatenated text content from a message.

    Returns the joined text of all :class:`TextBlock` items (for list
    content) or the raw string (for string content).
    """
    content = message.content
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        if isinstance(block, TextBlock):
            parts.append(block.text)
    return "".join(parts)


# ── Internal helpers ────────────────────────────────────────────────────────


def _convert_message(msg: Message) -> UserMessage | AssistantMessage | None:
    """Convert or filter a single message.

    Returns ``None`` when the message should be dropped entirely.
    """
    # 1. Filter progress messages
    if isinstance(msg, ProgressMessage):
        return None

    # 2. Filter virtual messages
    if isinstance(msg, (UserMessage, AssistantMessage)) and msg.hide_in_api:
        return None

    # 3. Handle system messages
    if isinstance(msg, SystemMessage):
        return _convert_system_message(msg)

    # 4. Handle attachment messages
    if isinstance(msg, AttachmentMessage):
        return _convert_attachment_message(msg)

    # 5. Pass through user and assistant messages as-is
    if isinstance(msg, (UserMessage, AssistantMessage)):
        return msg

    return None


def _convert_system_message(msg: SystemMessage) -> UserMessage | None:
    """Convert a SystemMessage if it should be kept, otherwise drop it.

    Only ``local_command`` system messages are converted to user messages
    so the model can see their output.  All other system messages (compact
    boundaries, informational, etc.) are filtered out.
    """
    if msg.subtype == SystemMessageSubtype.LOCAL_COMMAND:
        return UserMessage(
            content=msg.content,
            uuid=msg.uuid,
            timestamp=msg.timestamp,
            hide_in_ui=True,
        )
    return None


def _convert_attachment_message(msg: AttachmentMessage) -> UserMessage:
    """Convert an AttachmentMessage to a UserMessage.

    The attachment content is surfaced as a text block so the model can
    read it.  A small header is prepended to give the model context about
    the attachment type.
    """
    attachment = msg.attachment
    if attachment.content:
        text = f"[Attachment: {attachment.type}]\n{attachment.content}"
    else:
        text = f"[Attachment: {attachment.type}]"
    return UserMessage(
        content=text,
        uuid=msg.uuid,
        timestamp=msg.timestamp,
        hide_in_ui=True,
    )


def _merge_assistant_messages(
    a: AssistantMessage, b: AssistantMessage,
) -> AssistantMessage:
    """Merge two consecutive assistant messages."""
    merged = deepcopy(a)
    merged.content = list(a.content) + list(b.content)
    return merged


def _same_role(
    a: UserMessage | AssistantMessage,
    b: UserMessage | AssistantMessage,
) -> bool:
    """Check whether two messages share the same role."""
    return type(a) is type(b)


def _to_content_list(
    content: str | list[UserContentBlock],
) -> list[UserContentBlock]:
    """Ensure content is a list of content blocks."""
    if isinstance(content, str):
        return [TextBlock(text=content)]
    return list(content)
