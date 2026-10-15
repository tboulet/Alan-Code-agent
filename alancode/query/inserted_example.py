"""An example Bash call placed at the start of a session (setting
``inserted_example_bash_use``): the model sees its tool-call format used
once, with a real result, before it writes anything."""

from __future__ import annotations

import uuid
from typing import AsyncGenerator

from alancode.messages.factory import SYNTHETIC_MODEL
from alancode.messages.types import AssistantMessage, Message, TextBlock, ToolUseBlock
from alancode.tools.base import Tool, ToolUseContext
from alancode.tools.execution import PermissionCallback
from alancode.tools.orchestration import run_tools
from alancode.tools.text_tool_parser import get_format

EXAMPLE_TOOL_NAME = "Bash"
EXAMPLE_COMMAND = "ls -la"
EXAMPLE_PREFACE = "Let me look at the working directory first."


def example_bash_message(tool_call_format: str | None) -> AssistantMessage:
    """The example turn, in the session's tool-call format."""
    arguments = {"command": EXAMPLE_COMMAND}
    raw_text = (
        get_format(tool_call_format).render_call(EXAMPLE_TOOL_NAME, arguments)
        if tool_call_format
        else None
    )
    call = ToolUseBlock(
        id=f"call_{uuid.uuid4().hex[:24]}",
        name=EXAMPLE_TOOL_NAME,
        input=arguments,
        raw_text=raw_text,
    )
    return AssistantMessage(
        content=[TextBlock(text=EXAMPLE_PREFACE), call],
        model=SYNTHETIC_MODEL,
        stop_reason="end_turn",
    )


async def inserted_example_events(
    tools: list[Tool],
    context: ToolUseContext,
    tool_call_format: str | None,
    permission_callback: PermissionCallback | None = None,
) -> AsyncGenerator[Message, None]:
    """Yield the example turn and the real result of running it."""
    message = example_bash_message(tool_call_format)
    yield message
    async for update in run_tools(
        message.tool_use_blocks, tools, context,
        permission_callback=permission_callback,
    ):
        if update.message:
            yield update.message
