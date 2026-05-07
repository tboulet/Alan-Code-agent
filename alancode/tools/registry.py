"""Tool registry — manages the pool of available tools."""

from alancode.tools.base import Tool
from alancode.tools.builtin import ALL_BUILTIN_TOOLS

# Tools excluded by default in programmatic mode: external network
# (WebFetch), git mutation (GitCommit), and user prompts (AskUser).
PROGRAMMATIC_EXCLUDED_TOOL_NAMES = frozenset({
    "WebFetch", "GitCommit", "AskUserQuestion",
})


def get_all_builtin_tools() -> list[Tool]:
    """Return all built-in tools."""
    return list(ALL_BUILTIN_TOOLS)


def get_enabled_tools(tools: list[Tool] | None = None) -> list[Tool]:
    """Filter to only enabled tools."""
    all_tools = tools or get_all_builtin_tools()
    return [t for t in all_tools if t.is_enabled()]


def get_programmatic_tool_set() -> list[Tool]:
    """Return the tool set used by default in programmatic mode."""
    return [
        t for t in get_enabled_tools()
        if t.name not in PROGRAMMATIC_EXCLUDED_TOOL_NAMES
    ]


def find_tool_by_name(tools: list[Tool], name: str) -> Tool | None:
    """Find a tool by name or alias."""
    for t in tools:
        if t.matches_name(name):
            return t
    return None


def tools_to_schemas(tools: list[Tool]) -> list[dict]:
    """Convert tools to API schema format."""
    return [t.to_schema() for t in tools]
