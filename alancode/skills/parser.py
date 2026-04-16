"""SKILL.md parser — YAML frontmatter + markdown body extraction.

Skills are markdown files with YAML frontmatter that define reusable
prompt templates. This module handles parsing them into SkillDefinition
objects.
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

# Regex to split YAML frontmatter from markdown body.
# Matches: ---\n<yaml>\n---\n<body>
_FRONTMATTER_RE = re.compile(
    r"\A\s*---\s*\n(.*?)\n---\s*\n(.*)",
    re.DOTALL,
)


@dataclass
class SkillDefinition:
    """A parsed skill definition from a SKILL.md file."""

    name: str  # skill name (used in /skill <name>)
    description: str  # trigger phrases / summary for model
    body: str  # markdown body (the prompt template)
    source_path: str  # absolute path to SKILL.md (or "<builtin>")
    allowed_tools: list[str] | None = None  # tool restriction patterns (None = all)
    argument_hint: str | None = None  # e.g. "[environment]"
    when_to_use: str | None = None  # detailed guidance for model auto-invoke
    context: str = "inline"  # "inline" only for now
    version: str | None = None


def parse_skill_file(path: str) -> SkillDefinition | None:
    """Parse a SKILL.md file into a SkillDefinition.

    Returns None if the file cannot be read or has invalid/missing frontmatter.
    Logs warnings on parse errors but never raises.
    """
    try:
        content = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        logger.warning("Failed to read skill file %s: %s", path, exc)
        return None

    return parse_skill_content(content, source_path=path)


def parse_skill_content(content: str, *, source_path: str = "<string>") -> SkillDefinition | None:
    """Parse skill content (frontmatter + body) into a SkillDefinition.

    Useful for testing without files on disk.
    """
    match = _FRONTMATTER_RE.match(content)
    if not match:
        # Surface visibly: a missing frontmatter is a common first-author mistake.
        logger.warning(
            "Skill %s: no YAML frontmatter found. "
            "The file must start with a `---` block defining at least "
            "`name:` and `description:`.",
            source_path,
        )
        return None

    yaml_text = match.group(1)
    body = match.group(2).strip()

    try:
        meta = yaml.safe_load(yaml_text)
    except yaml.YAMLError as exc:
        # Surface visibly — first-time skill authors spend hours otherwise.
        logger.warning(
            "Skill %s failed to load: invalid YAML frontmatter (%s)",
            source_path, exc,
        )
        return None

    if not isinstance(meta, dict):
        logger.warning(
            "Skill %s failed to load: frontmatter is not a mapping "
            "(got %s). The `---` block should contain key: value pairs.",
            source_path, type(meta).__name__,
        )
        return None

    # Required fields
    name = meta.get("name")
    description = meta.get("description")
    if not name or not description:
        logger.warning(
            "Skill %s missing required fields (name=%r, description=%r). "
            "Add both to the frontmatter.",
            source_path, name, description,
        )
        return None

    # Optional fields: allowed_tools must be list[str] if present.
    # Accept a single string as a convenience (common YAML shorthand),
    # but reject dicts / nested lists / non-string items that would have
    # silently turned into str([...]) and filtered nothing.
    allowed_tools = meta.get("allowed-tools") or meta.get("allowed_tools")
    if allowed_tools is not None:
        if isinstance(allowed_tools, str):
            allowed_tools = [allowed_tools]
        elif isinstance(allowed_tools, list) and all(
            isinstance(x, str) for x in allowed_tools
        ):
            pass  # Already the correct shape.
        else:
            logger.warning(
                "Skill %s: `allowed-tools` must be a list of tool-name strings "
                "(or a single tool name), got %r. Skill ignored.",
                source_path, allowed_tools,
            )
            return None

    return SkillDefinition(
        name=str(name),
        description=str(description),
        body=body,
        source_path=source_path,
        allowed_tools=allowed_tools,
        argument_hint=meta.get("argument-hint") or meta.get("argument_hint"),
        when_to_use=meta.get("when_to_use") or meta.get("when-to-use"),
        context=meta.get("context", "inline"),
        version=meta.get("version"),
    )
