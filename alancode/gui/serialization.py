"""Event serialization — converts agent events to OutputEvent.

This is the SINGLE place where internal message/event types become
JSON-serializable :class:`OutputEvent` instances.  Both CLI and GUI
frontends receive the same format.

Reuses ``transcript.message_to_dict()`` for content block serialization
to avoid duplicating block-to-dict logic.
"""

from __future__ import annotations

from typing import Any

from alancode.gui.protocol import OutputEvent
from alancode.messages.types import (
    AssistantMessage,
    AttachmentMessage,
    Message,
    ProgressMessage,
    RequestStartEvent,
    StreamEvent,
    SystemMessage,
    UserMessage,
)
from alancode.session.transcript import message_to_dict


def agent_event_to_output(event: StreamEvent | Message) -> OutputEvent:
    """Convert an agent-yielded event to a serializable OutputEvent.

    Handles all event types produced by ``query_events_async()``
    and ``query_loop()``.
    """
    if isinstance(event, RequestStartEvent):
        return OutputEvent(type="request_start", data={}, original=event)

    if isinstance(event, AssistantMessage):
        data = message_to_dict(event)
        etype = "assistant_delta" if event.hide_in_api else "assistant_message"
        return OutputEvent(type=etype, data=data, original=event)

    if isinstance(event, UserMessage):
        data = message_to_dict(event)
        return OutputEvent(type="user_message", data=data, original=event)

    if isinstance(event, SystemMessage):
        data = message_to_dict(event)
        return OutputEvent(type="system_message", data=data, original=event)

    if isinstance(event, AttachmentMessage):
        data = message_to_dict(event)
        return OutputEvent(type="attachment_message", data=data, original=event)

    if isinstance(event, ProgressMessage):
        data = message_to_dict(event)
        return OutputEvent(type="progress_message", data=data, original=event)

    # Fallback for unknown types
    return OutputEvent(type="unknown", data={"repr": repr(event)}, original=event)


def cost_summary_event(
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int,
    cache_write_tokens: int,
    cost_usd: float,
    cost_unknown: bool,
) -> OutputEvent:
    """Create a cost summary OutputEvent (emitted after each turn)."""
    return OutputEvent(
        type="cost_summary",
        data={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_read_tokens": cache_read_tokens,
            "cache_write_tokens": cache_write_tokens,
            "cost_usd": cost_usd,
            "cost_unknown": cost_unknown,
        },
    )


def local_output_event(text: str, style: str = "default") -> OutputEvent:
    """Create a local output event for slash command results.

    Used for non-agent output that should appear in both CLI and GUI
    (help tables, status panels, diff output, etc.).
    """
    return OutputEvent(
        type="local_output",
        data={"text": text, "style": style},
    )
