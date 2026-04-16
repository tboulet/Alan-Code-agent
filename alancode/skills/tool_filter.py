"""Tool restriction for skills — filter available tools based on allowed-tools.

When a skill specifies ``allowed-tools`` in its frontmatter, only
matching tools should be available during skill execution.
"""

import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from alancode.tools.base import Tool

logger = logging.getLogger(__name__)

# Pattern to match tool restriction with args, e.g. "Bash(git:*)"
_PATTERN_RE = re.compile(r"^(\w+)\((.+)\)$")

# Map friendly names in allowed-tools to actual tool class names
_TOOL_NAME_ALIASES: dict[str, set[str]] = {
    "Bash": {"Bash"},
    "Read": {"Read", "FileRead"},
    "Write": {"Write", "FileWrite"},
    "Edit": {"Edit", "FileEdit"},
    "Glob": {"Glob"},
    "Grep": {"Grep"},
    "WebFetch": {"WebFetch"},
    "AskUser": {"AskUser", "AskUserQuestion"},
    "Skill": {"Skill"},
    "GitCommit": {"GitCommit"},
}


def _matches_tool_name(tool_name: str, pattern_name: str) -> bool:
    """Check if a tool name matches a pattern name (with alias resolution)."""
    aliases = _TOOL_NAME_ALIASES.get(pattern_name)
    if aliases:
        return tool_name in aliases
    # Fallback: direct name match
    return tool_name == pattern_name


def filter_tools_for_skill(
    all_tools: list["Tool"],
    allowed_patterns: list[str],
) -> list["Tool"]:
    """Filter tools to only those matching allowed-tools patterns.

    Patterns are simple tool-name matches:
    - ``"Bash"`` — allow BashTool
    - ``"Read"`` — allow FileReadTool
    - ``"Edit"`` — allow FileEditTool
    - etc.

    Pattern-based restrictions like ``Bash(git:*)`` are parsed — only
    the tool-name part is used for filtering; the argument restriction
    is logged but not enforced at the tool level.

    The Skill tool is always included so the model can invoke other skills.
    """
    if not allowed_patterns:
        return list(all_tools)

    # Parse patterns into plain tool names
    allowed_names: set[str] = set()
    for pattern in allowed_patterns:
        m = _PATTERN_RE.match(pattern)
        if m:
            tool_name = m.group(1)
            restriction = m.group(2)
            logger.debug(
                "Tool pattern %r: allowing %s with restriction %r (restriction logged, not enforced)",
                pattern, tool_name, restriction,
            )
            allowed_names.add(tool_name)
        else:
            allowed_names.add(pattern)

    # Always include Skill tool
    allowed_names.add("Skill")

    filtered = []
    for tool in all_tools:
        for allowed in allowed_names:
            if _matches_tool_name(tool.name, allowed):
                filtered.append(tool)
                break

    return filtered
