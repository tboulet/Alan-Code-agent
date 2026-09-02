"""A custom tool relying on an inert v1 field gets told, once."""

import pytest

from alancode.tools import execution
from alancode.tools.base import Tool, ToolResult, ToolUseContext
from alancode.messages.types import ToolUseBlock


class PlainTool(Tool):
    @property
    def name(self):
        return "Plain"

    @property
    def description(self):
        return "returns a plain result"

    @property
    def input_schema(self):
        return {"type": "object", "properties": {}}

    async def call(self, args, context):
        return ToolResult(data="ok")


class ExtraMessagesTool(PlainTool):
    @property
    def name(self):
        return "ExtraMessages"

    async def call(self, args, context):
        return ToolResult(data="ok", new_messages=[{"role": "user", "content": "hi"}])


class SmallCapTool(PlainTool):
    @property
    def name(self):
        return "SmallCap"

    @property
    def max_result_size_chars(self):
        return 100


@pytest.fixture(autouse=True)
def reset_warned():
    execution._inert_field_warned.clear()
    yield
    execution._inert_field_warned.clear()


async def _run(tool):
    return await execution.run_tool_use(
        ToolUseBlock(id="t1", name=tool.name, input={}),
        tool,
        ToolUseContext(cwd=".", messages=[], settings={}),
    )


@pytest.mark.asyncio
async def test_plain_tool_warns_about_nothing(caplog):
    with caplog.at_level("WARNING"):
        await _run(PlainTool())
    assert caplog.records == []


@pytest.mark.asyncio
async def test_new_messages_is_reported(caplog):
    with caplog.at_level("WARNING"):
        await _run(ExtraMessagesTool())
    assert "ToolResult.new_messages" in caplog.text


@pytest.mark.asyncio
async def test_overridden_result_cap_is_reported(caplog):
    with caplog.at_level("WARNING"):
        await _run(SmallCapTool())
    assert "Tool.max_result_size_chars" in caplog.text


@pytest.mark.asyncio
async def test_warning_fires_once_per_tool(caplog):
    tool = ExtraMessagesTool()
    with caplog.at_level("WARNING"):
        await _run(tool)
        await _run(tool)
        await _run(tool)
    assert len(caplog.records) == 1


@pytest.mark.asyncio
async def test_the_result_itself_is_unaffected():
    message = await _run(ExtraMessagesTool())
    assert "ok" in str(message.content)
