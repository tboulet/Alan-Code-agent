"""Missing-parameter errors must not teach one format's syntax to every session."""

import inspect

import pytest

from alancode.tools.builtin.bash import BashTool
from alancode.tools.builtin.file_read import FileReadTool
from alancode.tools.builtin.glob_tool import GlobTool
from alancode.tools.base import ToolUseContext
from alancode.tools.registry import get_enabled_tools

# GLM writes arguments as <arg_key>K</arg_key><arg_value>V</arg_value>. Under
# bash_block or auto that markup is wrong, and a model that follows it never
# recovers - it just re-emits an unparseable call.
FOREIGN_MARKUP = ("<arg_key>", "<arg_value>", "<parameter=", "```bash")


def _ctx():
    return ToolUseContext(cwd=".", messages=[], settings={})


@pytest.mark.asyncio
async def test_empty_args_error_names_the_parameter_without_prescribing_markup():
    result = await BashTool().call({}, _ctx())
    assert result.is_error
    assert "'command' parameter is required" in result.data
    assert "Got parameters: []" in result.data
    for markup in FOREIGN_MARKUP:
        assert markup not in result.data


@pytest.mark.asyncio
async def test_other_tools_do_not_prescribe_markup_either():
    for tool in (FileReadTool(), GlobTool()):
        result = await tool.call({}, _ctx())
        assert result.is_error
        for markup in FOREIGN_MARKUP:
            assert markup not in result.data, f"{tool.name} still prescribes {markup}"


def test_no_builtin_tool_source_hardcodes_a_tool_call_markup():
    # Catches the pattern coming back anywhere in the builtin tool set.
    for tool in get_enabled_tools():
        try:
            source = inspect.getsource(type(tool))
        except (OSError, TypeError):
            continue
        assert "<arg_key>" not in source, f"{tool.name} hardcodes GLM argument markup"
