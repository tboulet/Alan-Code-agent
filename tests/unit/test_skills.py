"""Tests for the skill system — parser, discovery, registry, tool filter."""

import os
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from alancode.skills.parser import SkillDefinition, parse_skill_content, parse_skill_file
from alancode.skills.discovery import discover_skills, _scan_skills_dir
from alancode.skills.registry import SkillRegistry
from alancode.skills.tool_filter import filter_tools_for_skill


# ── Parser tests ───────────────────────────────────────────────────────────


VALID_SKILL = textwrap.dedent("""\
    ---
    name: deploy
    description: Use when user asks to deploy the application
    allowed-tools: [Bash, Read]
    argument-hint: "[environment]"
    when_to_use: "When the user wants to deploy"
    version: 0.1.0
    ---

    # Deploy Application

    Deploy to $ARGUMENTS environment.

    ## Steps

    1. Run tests
    2. Build
    3. Deploy
""")


def test_parse_valid_skill():
    skill = parse_skill_content(VALID_SKILL, source_path="test.md")
    assert skill is not None
    assert skill.name == "deploy"
    assert skill.description == "Use when user asks to deploy the application"
    assert skill.allowed_tools == ["Bash", "Read"]
    assert skill.argument_hint == "[environment]"
    assert skill.when_to_use == "When the user wants to deploy"
    assert skill.context == "inline"
    assert "Deploy to $ARGUMENTS" in skill.body
    assert skill.source_path == "test.md"


def test_parse_minimal_skill():
    content = textwrap.dedent("""\
        ---
        name: greet
        description: Greet the user
        ---

        Hello $ARGUMENTS!
    """)
    skill = parse_skill_content(content)
    assert skill is not None
    assert skill.name == "greet"
    assert skill.description == "Greet the user"
    assert skill.allowed_tools is None
    assert skill.argument_hint is None
    assert skill.when_to_use is None
    assert skill.body == "Hello $ARGUMENTS!"


def test_parse_no_frontmatter():
    content = "# Just markdown\nNo frontmatter here."
    skill = parse_skill_content(content)
    assert skill is None


def test_parse_invalid_yaml():
    content = textwrap.dedent("""\
        ---
        name: broken
        description: [invalid yaml
        ---

        Body.
    """)
    skill = parse_skill_content(content)
    assert skill is None


def test_parse_missing_required_fields():
    content = textwrap.dedent("""\
        ---
        name: incomplete
        ---

        Body without description.
    """)
    skill = parse_skill_content(content)
    assert skill is None


def test_parse_frontmatter_not_mapping():
    content = textwrap.dedent("""\
        ---
        - just a list
        ---

        Body.
    """)
    skill = parse_skill_content(content)
    assert skill is None


def test_parse_allowed_tools_single_string():
    """allowed-tools as a single string should be normalized to a list."""
    content = textwrap.dedent("""\
        ---
        name: simple
        description: Simple skill
        allowed-tools: Bash
        ---

        Body.
    """)
    skill = parse_skill_content(content)
    assert skill is not None
    assert skill.allowed_tools == ["Bash"]


def test_parse_underscore_fields():
    """Support both hyphenated and underscored field names."""
    content = textwrap.dedent("""\
        ---
        name: compat
        description: Compatibility test
        allowed_tools: [Read, Write]
        argument_hint: "[file]"
        when_to_use: "When user wants compatibility"
        ---

        Body.
    """)
    skill = parse_skill_content(content)
    assert skill is not None
    assert skill.allowed_tools == ["Read", "Write"]
    assert skill.argument_hint == "[file]"
    assert skill.when_to_use == "When user wants compatibility"


def test_parse_skill_file_nonexistent():
    skill = parse_skill_file("/nonexistent/SKILL.md")
    assert skill is None


def test_parse_skill_file_on_disk(tmp_path):
    skill_file = tmp_path / "SKILL.md"
    skill_file.write_text(VALID_SKILL)
    skill = parse_skill_file(str(skill_file))
    assert skill is not None
    assert skill.name == "deploy"
    assert skill.source_path == str(skill_file)


# ── Discovery tests ───────────────────────────────────────────────────────


def test_scan_skills_dir_empty(tmp_path):
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    result = _scan_skills_dir(skills_dir)
    assert result == {}


def test_scan_skills_dir_nonexistent(tmp_path):
    result = _scan_skills_dir(tmp_path / "nonexistent")
    assert result == {}


def test_scan_skills_dir_with_skills(tmp_path):
    skills_dir = tmp_path / "skills"
    deploy_dir = skills_dir / "deploy"
    deploy_dir.mkdir(parents=True)
    (deploy_dir / "SKILL.md").write_text(VALID_SKILL)

    greet_dir = skills_dir / "greet"
    greet_dir.mkdir()
    (greet_dir / "SKILL.md").write_text(textwrap.dedent("""\
        ---
        name: greet
        description: Greet someone
        ---

        Hello $ARGUMENTS!
    """))

    result = _scan_skills_dir(skills_dir)
    assert len(result) == 2
    assert "deploy" in result
    assert "greet" in result


def test_scan_skills_dir_skips_invalid(tmp_path):
    skills_dir = tmp_path / "skills"
    bad_dir = skills_dir / "bad"
    bad_dir.mkdir(parents=True)
    (bad_dir / "SKILL.md").write_text("No frontmatter here")

    result = _scan_skills_dir(skills_dir)
    assert result == {}


def test_discover_skills_includes_builtin(tmp_path):
    """Built-in skills should always be present."""
    skills = discover_skills(str(tmp_path))
    assert "create" in skills
    assert skills["create"].source_path == "<builtin>"


def test_discover_skills_project_overrides_builtin(tmp_path):
    """Project skills override built-in skills with the same name."""
    skills_dir = tmp_path / ".alan" / "skills" / "create"
    skills_dir.mkdir(parents=True)
    (skills_dir / "SKILL.md").write_text(textwrap.dedent("""\
        ---
        name: create
        description: Custom create skill
        ---

        Custom body.
    """))

    skills = discover_skills(str(tmp_path))
    assert skills["create"].description == "Custom create skill"
    assert skills["create"].source_path != "<builtin>"


def test_discover_skills_project_and_user(tmp_path, monkeypatch):
    """Project skills override user skills."""
    # User skill
    user_home = tmp_path / "home"
    user_skills = user_home / ".alan" / "skills" / "deploy"
    user_skills.mkdir(parents=True)
    (user_skills / "SKILL.md").write_text(textwrap.dedent("""\
        ---
        name: deploy
        description: User deploy
        ---

        User deploy body.
    """))

    # Project skill (same name)
    project = tmp_path / "project"
    proj_skills = project / ".alan" / "skills" / "deploy"
    proj_skills.mkdir(parents=True)
    (proj_skills / "SKILL.md").write_text(textwrap.dedent("""\
        ---
        name: deploy
        description: Project deploy
        ---

        Project deploy body.
    """))

    monkeypatch.setattr(Path, "home", lambda: user_home)
    skills = discover_skills(str(project))
    assert skills["deploy"].description == "Project deploy"


# ── Registry tests ────────────────────────────────────────────────────────


def test_registry_basic(tmp_path):
    registry = SkillRegistry(str(tmp_path))
    # Should have built-in "create"
    assert "create" in registry
    assert len(registry) >= 1
    assert registry.get("nonexistent") is None


def test_registry_expand(tmp_path):
    skills_dir = tmp_path / ".alan" / "skills" / "greet"
    skills_dir.mkdir(parents=True)
    (skills_dir / "SKILL.md").write_text(textwrap.dedent("""\
        ---
        name: greet
        description: Greet someone
        ---

        Hello $ARGUMENTS! Welcome.
    """))

    registry = SkillRegistry(str(tmp_path))
    result = registry.expand("greet", "World")
    assert result == "Hello World! Welcome."


def test_registry_expand_no_args(tmp_path):
    skills_dir = tmp_path / ".alan" / "skills" / "greet"
    skills_dir.mkdir(parents=True)
    (skills_dir / "SKILL.md").write_text(textwrap.dedent("""\
        ---
        name: greet
        description: Greet someone
        ---

        Hello $ARGUMENTS! Welcome.
    """))

    registry = SkillRegistry(str(tmp_path))
    result = registry.expand("greet")
    assert result == "Hello ! Welcome."


def test_registry_expand_unknown(tmp_path):
    registry = SkillRegistry(str(tmp_path))
    assert registry.expand("nonexistent") is None


def test_registry_list_all(tmp_path):
    registry = SkillRegistry(str(tmp_path))
    skills = registry.list_all()
    assert isinstance(skills, list)
    # Should be sorted by name
    names = [s.name for s in skills]
    assert names == sorted(names)


def test_registry_reload(tmp_path):
    registry = SkillRegistry(str(tmp_path))
    initial_count = len(registry)

    # Add a new skill
    skills_dir = tmp_path / ".alan" / "skills" / "new-skill"
    skills_dir.mkdir(parents=True)
    (skills_dir / "SKILL.md").write_text(textwrap.dedent("""\
        ---
        name: new-skill
        description: A new skill
        ---

        New skill body.
    """))

    registry.reload(str(tmp_path))
    assert len(registry) == initial_count + 1
    assert "new-skill" in registry


# ── Tool filter tests ─────────────────────────────────────────────────────


class _MockTool:
    """Minimal mock for tool filtering tests."""

    def __init__(self, name: str):
        self.name = name


def test_filter_empty_patterns():
    tools = [_MockTool("Bash"), _MockTool("Read")]
    result = filter_tools_for_skill(tools, [])
    assert len(result) == 2


def test_filter_basic_names():
    tools = [_MockTool("Bash"), _MockTool("Read"), _MockTool("Edit"), _MockTool("Glob")]
    result = filter_tools_for_skill(tools, ["Bash", "Read"])
    names = {t.name for t in result}
    assert names == {"Bash", "Read"}


def test_filter_always_includes_skill():
    tools = [_MockTool("Bash"), _MockTool("Skill"), _MockTool("Read")]
    result = filter_tools_for_skill(tools, ["Bash"])
    names = {t.name for t in result}
    assert "Skill" in names
    assert "Bash" in names


def test_filter_with_pattern_syntax():
    """Patterns like Bash(git:*) should still allow the Bash tool."""
    tools = [_MockTool("Bash"), _MockTool("Read"), _MockTool("Edit")]
    result = filter_tools_for_skill(tools, ["Bash(git:*)", "Read"])
    names = {t.name for t in result}
    assert names == {"Bash", "Read"}


def test_filter_aliases():
    """FileRead should match Read alias."""
    tools = [_MockTool("FileRead"), _MockTool("FileEdit")]
    result = filter_tools_for_skill(tools, ["Read"])
    assert len(result) == 1
    assert result[0].name == "FileRead"
