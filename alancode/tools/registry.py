"""Tool registry — manages the pool of available tools."""

from alancode.tools.base import Tool
from alancode.tools.builtin import ALL_BUILTIN_TOOLS


def get_all_builtin_tools() -> list[Tool]:
    """Return all built-in tools."""
    return list(ALL_BUILTIN_TOOLS)


def get_enabled_tools(tools: list[Tool] | None = None) -> list[Tool]:
    """Filter to only enabled tools."""
    all_tools = tools or get_all_builtin_tools()
    return [t for t in all_tools if t.is_enabled()]


def find_tool_by_name(tools: list[Tool], name: str) -> Tool | None:
    """Find a tool by name or alias."""
    for t in tools:
        if t.matches_name(name):
            return t
    return None


def tools_to_schemas(tools: list[Tool]) -> list[dict]:
    """Convert tools to API schema format."""
    return [t.to_schema() for t in tools]
