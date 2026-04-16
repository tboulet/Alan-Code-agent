"""Factory functions for creating messages.

Provides convenient constructors for all message types, handling defaults
and common patterns (e.g. synthetic error messages, interruption messages).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from alancode.messages.types import (
    AssistantMessage,
    Attachment,
    AttachmentMessage,
    CompactMetadata,
    CompactClearMetadata,
    SystemMessage,
    SystemMessageSubtype,
    TextBlock,
    ToolResultBlock,
    Usage,
    UserMessage,
    MessageOrigin,
)


# ── Sentinel constants ──────────────────────────────────────────────────────

SYNTHETIC_MODEL = "<synthetic>"

INTERRUPT_MESSAGE = "[Request interrupted by user]"
INTERRUPT_MESSAGE_FOR_TOOL_USE = "[Request interrupted by user for tool use]"

CANCEL_MESSAGE = (
    "The user doesn't want to take this action right now. "
    "STOP what you are doing and wait for the user to tell you how to proceed."
)
REJECT_MESSAGE = (
    "The user doesn't want to proceed with this tool use. The tool use was rejected. "
    "STOP what you are doing and wait for the user to tell you how to proceed."
)


# ── Helpers ─────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── User messages ───────────────────────────────────────────────────────────


def create_user_message(
    content: str | list,
    *,
    hide_in_ui: bool = False,
    hide_in_api: bool = False,
    is_compact_summary: bool = False,
    tool_use_result: Any = None,
    source_tool_assistant_uuid: UUID | None = None,
    permission_mode: str | None = None,
    origin: MessageOrigin | None = None,
    uuid: UUID | None = None,
    timestamp: str | None = None,
) -> UserMessage:
    """Create a UserMessage with sensible defaults."""
    return UserMessage(
        content=content,
        uuid=uuid or uuid4(),
        timestamp=timestamp or _now_iso(),
        hide_in_ui=hide_in_ui,
        hide_in_api=hide_in_api,
        is_compact_summary=is_compact_summary,
        tool_use_result=tool_use_result,
        source_tool_assistant_uuid=source_tool_assistant_uuid,
        permission_mode=permission_mode,
        origin=origin,
    )


def create_user_interruption_message(*, tool_use: bool = False) -> UserMessage:
    """Create a user message indicating the request was interrupted.

    Args:
        tool_use: If True, indicates the interruption was to perform tool use.
    """
    text = INTERRUPT_MESSAGE_FOR_TOOL_USE if tool_use else INTERRUPT_MESSAGE
    return create_user_message(text)


def create_tool_result_message(
    tool_use_id: str,
    content: str,
    *,
    is_error: bool = False,
    source_tool_assistant_uuid: UUID | None = None,
) -> UserMessage:
    """Create a UserMessage containing a single ToolResultBlock.

    This is how tool execution results are fed back to the model.
    """
    result_block = ToolResultBlock(
        tool_use_id=tool_use_id,
        content=content,
        is_error=is_error,
    )
    return create_user_message(
        [result_block],
        source_tool_assistant_uuid=source_tool_assistant_uuid,
    )


# ── Assistant messages ──────────────────────────────────────────────────────


def create_assistant_message(
    content: str | list,
    *,
    usage: Usage | None = None,
    hide_in_api: bool = False,
) -> AssistantMessage:
    """Create an AssistantMessage.

    If *content* is a plain string it is wrapped in a single TextBlock.
    """
    if isinstance(content, str):
        content = [TextBlock(text=content)]
    return AssistantMessage(
        content=content,
        usage=usage or Usage(),
        hide_in_api=hide_in_api,
    )


def create_assistant_error_message(
    content: str,
    *,
    api_error: str | None = None,
    error_details: str | None = None,
) -> AssistantMessage:
    """Create a synthetic assistant error message.

    Sets ``is_api_error_message=True`` and ``model=SYNTHETIC_MODEL`` so
    downstream code can distinguish real model output from error placeholders.
    """
    msg = create_assistant_message(content)
    msg.model = SYNTHETIC_MODEL
    msg.is_api_error_message = True
    msg.api_error = api_error
    msg.error_details = error_details
    return msg


# ── System messages ─────────────────────────────────────────────────────────


def create_system_message(
    content: str,
    level: str = "info",
) -> SystemMessage:
    """Create an informational SystemMessage."""
    return SystemMessage(
        content=content,
        subtype=SystemMessageSubtype.INFORMATIONAL,
        level=level,
    )


def create_compact_boundary_message(
    trigger: str,
    pre_tokens: int,
    *,
    messages_summarized: int | None = None,
    user_context: str | None = None,
) -> SystemMessage:
    """Create a compaction boundary marker."""
    return SystemMessage(
        content="",
        subtype=SystemMessageSubtype.COMPACT_BOUNDARY,
        compact_metadata=CompactMetadata(
            trigger=trigger,  # type: ignore[arg-type]
            pre_tokens=pre_tokens,
            user_context=user_context,
            messages_summarized=messages_summarized,
        ),
    )


def create_compact_clear_boundary_message(
    trigger: str,
    pre_tokens: int,
    tokens_saved: int,
    compacted_tool_ids: list[str],
    cleared_attachment_uuids: list[str],
) -> SystemMessage:
    """Create a Layer B (clear) boundary marker."""
    return SystemMessage(
        content="",
        subtype=SystemMessageSubtype.COMPACT_CLEAR_BOUNDARY,
        compact_clear_metadata=CompactClearMetadata(
            trigger=trigger,  # type: ignore[arg-type]
            pre_tokens=pre_tokens,
            tokens_saved=tokens_saved,
            compacted_tool_ids=compacted_tool_ids,
            cleared_attachment_uuids=cleared_attachment_uuids,
        ),
    )


# ── Attachment messages ─────────────────────────────────────────────────────


def create_attachment_message(
    attachment_type: str,
    *,
    content: str = "",
    metadata: dict[str, Any] | None = None,
) -> AttachmentMessage:
    """Create an AttachmentMessage wrapping an Attachment."""
    return AttachmentMessage(
        attachment=Attachment(
            type=attachment_type,
            content=content,
            metadata=metadata or {},
        ),
    )
