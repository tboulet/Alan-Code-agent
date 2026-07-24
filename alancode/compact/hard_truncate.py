"""Deterministic last-resort compaction when LLM summarization fails."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import uuid

from alancode.messages.factory import (
    create_compact_boundary_message,
    create_user_message,
)
from alancode.messages.types import (
    AssistantMessage,
    Message,
    SystemMessage,
    ToolResultBlock,
    UserMessage,
    is_compact_boundary,
)
from alancode.utils.tokens import estimate_message_tokens


def _is_plain_user_text(message: Message) -> bool:
    """Whether a message can safely start an API conversation."""
    if not isinstance(message, UserMessage):
        return False
    if isinstance(message.content, str):
        return True
    return not any(
        isinstance(block, ToolResultBlock)
        for block in message.content
    )


def hard_truncate_messages(
    messages: list[Message],
    target_tokens: int,
) -> tuple[list[Message], int]:
    """Keep the valuable head and recent tail while meeting a token target.

    A prior compact summary, or otherwise the opening user request, is kept
    as the head. Old messages immediately after it are discarded first.
    The retained tail is adjusted to begin at a structurally valid message,
    so tool results are not separated from their assistant tool calls.

    Returns ``(retained_messages, dropped_count)``.
    """
    if not messages:
        return [], 0

    head_end = 0
    if isinstance(messages[0], SystemMessage):
        head_end = 1
        while head_end < len(messages) and getattr(
            messages[head_end], "is_compact_summary", False
        ):
            head_end += 1
    elif _is_plain_user_text(messages[0]):
        head_end = 1

    head = list(messages[:head_end])
    tail = list(messages[head_end:])
    dropped = 0

    if head and estimate_message_tokens(head) > target_tokens:
        dropped += len(head)
        head = []

    while tail and estimate_message_tokens(head + tail) > target_tokens:
        tail.pop(0)
        dropped += 1

    while tail and not (
        _is_plain_user_text(tail[0])
        or (head and isinstance(tail[0], AssistantMessage))
    ):
        tail.pop(0)
        dropped += 1

    return head + tail, dropped


@dataclass
class HardTruncationResult:
    """Durable replacement history produced by hard truncation."""

    boundary_message: SystemMessage
    retained_messages: list[Message]
    notice_message: UserMessage
    dropped_count: int

    @property
    def messages(self) -> list[Message]:
        return [
            self.boundary_message,
            *self.retained_messages,
            self.notice_message,
        ]


def _clone_for_replay(messages: list[Message]) -> list[Message]:
    """Copy retained messages with fresh identities for post-boundary replay."""
    replayed_originals = [
        message for message in messages if not is_compact_boundary(message)
    ]
    clones = deepcopy(replayed_originals)
    uuid_map = {}

    for original, clone in zip(replayed_originals, clones):
        old_uuid = getattr(original, "uuid", None)
        if old_uuid is None:
            continue
        new_uuid = uuid.uuid4()
        clone.uuid = new_uuid
        uuid_map[old_uuid] = new_uuid

    for clone in clones:
        if isinstance(clone, UserMessage):
            source_uuid = clone.source_tool_assistant_uuid
            if source_uuid in uuid_map:
                clone.source_tool_assistant_uuid = uuid_map[source_uuid]

    return clones


def build_hard_truncation_result(
    messages: list[Message],
    *,
    target_tokens: int,
    failures: int,
) -> HardTruncationResult:
    """Select and package a fallback history that survives future turns."""
    pre_fallback_tokens = estimate_message_tokens(messages)
    retained, dropped = hard_truncate_messages(messages, target_tokens)
    boundary = create_compact_boundary_message(
        trigger="auto",
        pre_tokens=pre_fallback_tokens,
        messages_summarized=dropped,
    )
    notice = create_user_message(
        f"Summarization failed {failures} times consecutively. "
        f"{dropped} older message(s) were hard-truncated from the "
        "context to keep the session alive. Earlier conversation "
        "details are gone - re-read files if something is missing.",
        hide_in_ui=False,
    )
    return HardTruncationResult(
        boundary_message=boundary,
        retained_messages=_clone_for_replay(retained),
        notice_message=notice,
        dropped_count=dropped,
    )
