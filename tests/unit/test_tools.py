"""Test tool registry, orchestration, and permissions."""

import pytest

from alancode.tools.base import Tool, ToolResult, ToolUseContext
from alancode.tools.registry import (
    get_all_builtin_tools,
    get_enabled_tools,
    find_tool_by_name,
    tools_to_schemas,
)
from alancode.tools.orchestration import partition_tool_calls
from alancode.messages.types import ToolUseBlock
from alancode.permissions.context import (
    PermissionBehavior,
    PermissionMode,
    PermissionResult,
    PermissionRule,
    ToolPermissionContext,
)
from alancode.permissions.pipeline import check_permissions, check_rule_match

from tests.conftest import EchoTool, MutateTool


# ---------------------------------------------------------------------------
# Tool registry tests
# ---------------------------------------------------------------------------


class TestToolRegistry:
    def test_get_all_builtin_tools(self):
        tools = get_all_builtin_tools()
        # 7 tools + AskUserQuestion + GitCommit = 9 (WebSearch disabled)
        assert len(tools) == 9

    def test_builtin_tools_are_classes(self):
        tools = get_all_builtin_tools()
        for t in tools:
            # Each entry should be a Tool subclass (class object)
            assert isinstance(t, type) or isinstance(t, Tool)

    def test_find_tool_by_name_primary(self):
        echo = EchoTool()
        mutate = MutateTool()
        tools = [echo, mutate]
        found = find_tool_by_name(tools, "Echo")
        assert found is echo

    def test_find_tool_by_name_alias(self):
        echo = EchoTool()
        tools = [echo]
        found = find_tool_by_name(tools, "EchoAlias")
        assert found is echo

    def test_find_tool_by_name_not_found(self):
        echo = EchoTool()
        tools = [echo]
        found = find_tool_by_name(tools, "NonExistent")
        assert found is None

    def test_tools_to_schemas(self):
        echo = EchoTool()
        schemas = tools_to_schemas([echo])
        assert len(schemas) == 1
        schema = schemas[0]
        assert schema["name"] == "Echo"
        assert schema["description"] == "Echoes the input text"
        assert "properties" in schema["input_schema"]

    def test_tool_matches_name(self):
        echo = EchoTool()
        assert echo.matches_name("Echo") is True
        assert echo.matches_name("echo") is True  # alias
        assert echo.matches_name("EchoAlias") is True
        assert echo.matches_name("Other") is False

    def test_tool_to_schema(self):
        echo = EchoTool()
        schema = echo.to_schema()
        assert schema["name"] == "Echo"
        assert "input_schema" in schema


# ---------------------------------------------------------------------------
# Tool orchestration (partitioning) tests
# ---------------------------------------------------------------------------


class TestPartitionToolCalls:
    def test_empty_tool_calls(self):
        batches = partition_tool_calls([], [])
        assert batches == []

    def test_single_read_only_tool(self):
        echo = EchoTool()
        block = ToolUseBlock(id="t1", name="Echo", input={"text": "hi"})
        batches = partition_tool_calls([block], [echo])
        assert len(batches) == 1
        assert batches[0].is_concurrent is True
        assert len(batches[0].blocks) == 1

    def test_single_mutating_tool(self):
        mutate = MutateTool()
        block = ToolUseBlock(id="t1", name="Mutate", input={"target": "x"})
        batches = partition_tool_calls([block], [mutate])
        assert len(batches) == 1
        assert batches[0].is_concurrent is False

    def test_consecutive_read_only_grouped(self):
        echo = EchoTool()
        blocks = [
            ToolUseBlock(id="t1", name="Echo", input={"text": "a"}),
            ToolUseBlock(id="t2", name="Echo", input={"text": "b"}),
            ToolUseBlock(id="t3", name="Echo", input={"text": "c"}),
        ]
        batches = partition_tool_calls(blocks, [echo])
        assert len(batches) == 1
        assert batches[0].is_concurrent is True
        assert len(batches[0].blocks) == 3

    def test_mixed_read_write_partitioning(self):
        echo = EchoTool()
        mutate = MutateTool()
        blocks = [
            ToolUseBlock(id="t1", name="Echo", input={"text": "a"}),
            ToolUseBlock(id="t2", name="Echo", input={"text": "b"}),
            ToolUseBlock(id="t3", name="Mutate", input={"target": "x"}),
            ToolUseBlock(id="t4", name="Echo", input={"text": "c"}),
        ]
        batches = partition_tool_calls(blocks, [echo, mutate])
        assert len(batches) == 3
        # First batch: two read-only calls (concurrent)
        assert batches[0].is_concurrent is True
        assert len(batches[0].blocks) == 2
        # Second batch: one mutating call (serial)
        assert batches[1].is_concurrent is False
        assert len(batches[1].blocks) == 1
        # Third batch: one read-only call (concurrent)
        assert batches[2].is_concurrent is True
        assert len(batches[2].blocks) == 1

    def test_unknown_tool_treated_as_mutating(self):
        blocks = [
            ToolUseBlock(id="t1", name="UnknownTool", input={}),
        ]
        batches = partition_tool_calls(blocks, [])
        assert len(batches) == 1
        assert batches[0].is_concurrent is False


# ---------------------------------------------------------------------------
# Permission tests
# ---------------------------------------------------------------------------


class TestPermissions:
    def test_deny_rule_blocks(self):
        echo = EchoTool()
        deny_rule = PermissionRule(
            tool_name="Echo",
            behavior=PermissionBehavior.DENY,
            source="test",
        )
        result = check_rule_match([deny_rule], echo, {"text": "anything"})
        assert result is not None
        assert result.behavior == PermissionBehavior.DENY

    def test_allow_rule_permits(self):
        echo = EchoTool()
        allow_rule = PermissionRule(
            tool_name="Echo",
            behavior=PermissionBehavior.ALLOW,
            source="test",
        )
        result = check_rule_match([allow_rule], echo, {"text": "anything"})
        assert result is not None
        assert result.behavior == PermissionBehavior.ALLOW

    def test_rule_no_match_different_tool(self):
        echo = EchoTool()
        rule = PermissionRule(
            tool_name="OtherTool",
            behavior=PermissionBehavior.DENY,
        )
        result = check_rule_match([rule], echo, {"text": "anything"})
        assert result is None

    def test_rule_content_matching(self):
        """e.g., 'Bash(git *)' matches command starting with 'git '."""
        echo = EchoTool()
        rule = PermissionRule(
            tool_name="Echo",
            rule_content="git *",
            behavior=PermissionBehavior.ALLOW,
        )
        # Should match because "text" value starts with "git "
        result = check_rule_match([rule], echo, {"text": "git push"})
        assert result is not None

        # Should NOT match a different prefix
        result = check_rule_match([rule], echo, {"text": "npm install"})
        assert result is None

    @pytest.mark.asyncio
    async def test_bypass_mode_allows(self):
        echo = EchoTool()
        context = ToolUseContext(cwd="/tmp", messages=[])
        perm_context = ToolPermissionContext(mode=PermissionMode.YOLO)
        result = await check_permissions(echo, {"text": "hi"}, context, perm_context)
        assert result.behavior == PermissionBehavior.ALLOW

    @pytest.mark.asyncio
    async def test_safe_mode_allows_read_tools(self):
        echo = EchoTool()  # permission_level = "read"
        context = ToolUseContext(cwd="/tmp", messages=[])
        perm_context = ToolPermissionContext(mode=PermissionMode.SAFE)
        result = await check_permissions(echo, {"text": "hi"}, context, perm_context)
        assert result.behavior == PermissionBehavior.ALLOW

    @pytest.mark.asyncio
    async def test_safe_mode_asks_for_write_tools(self):
        mutate = MutateTool()  # permission_level = "write"
        context = ToolUseContext(cwd="/tmp", messages=[])
        perm_context = ToolPermissionContext(mode=PermissionMode.SAFE)
        result = await check_permissions(mutate, {"target": "x"}, context, perm_context)
        # No allow rules, no prompt callback -> ASK
        assert result.behavior == PermissionBehavior.ASK

    @pytest.mark.asyncio
    async def test_edit_mode_allows_write_tools(self):
        mutate = MutateTool()  # permission_level = "write"
        context = ToolUseContext(cwd="/tmp", messages=[])
        perm_context = ToolPermissionContext(mode=PermissionMode.EDIT)
        result = await check_permissions(mutate, {"target": "x"}, context, perm_context)
        assert result.behavior == PermissionBehavior.ALLOW

    @pytest.mark.asyncio
    async def test_safe_mode_with_allow_rule(self):
        mutate = MutateTool()
        context = ToolUseContext(cwd="/tmp", messages=[])
        allow_rule = PermissionRule(
            tool_name="Mutate",
            behavior=PermissionBehavior.ALLOW,
            source="test",
        )
        perm_context = ToolPermissionContext(
            mode=PermissionMode.SAFE,
            allow_rules=[allow_rule],
        )
        result = await check_permissions(mutate, {"target": "x"}, context, perm_context)
        assert result.behavior == PermissionBehavior.ALLOW

    @pytest.mark.asyncio
    async def test_deny_rule_in_pipeline(self):
        echo = EchoTool()
        context = ToolUseContext(cwd="/tmp", messages=[])
        deny_rule = PermissionRule(
            tool_name="Echo",
            behavior=PermissionBehavior.DENY,
            source="settings",
        )
        perm_context = ToolPermissionContext(
            mode=PermissionMode.YOLO,  # Even bypass won't override deny
            deny_rules=[deny_rule],
        )
        result = await check_permissions(echo, {"text": "hi"}, context, perm_context)
        assert result.behavior == PermissionBehavior.DENY

    @pytest.mark.asyncio
    async def test_prompt_callback_allow(self):
        mutate = MutateTool()  # write-level, needs permission in safe mode
        context = ToolUseContext(cwd="/tmp", messages=[])
        perm_context = ToolPermissionContext(mode=PermissionMode.SAFE)

        async def always_allow(name, desc, inp):
            return PermissionBehavior.ALLOW

        result = await check_permissions(
            mutate, {"target": "x"}, context, perm_context, prompt_user=always_allow,
        )
        assert result.behavior == PermissionBehavior.ALLOW

    @pytest.mark.asyncio
    async def test_prompt_callback_deny(self):
        mutate = MutateTool()  # write-level, needs permission in safe mode
        context = ToolUseContext(cwd="/tmp", messages=[])
        perm_context = ToolPermissionContext(mode=PermissionMode.SAFE)

        async def always_deny(name, desc, inp):
            return PermissionBehavior.DENY

        result = await check_permissions(
            mutate, {"target": "x"}, context, perm_context, prompt_user=always_deny,
        )
        assert result.behavior == PermissionBehavior.DENY

    @pytest.mark.asyncio
    async def test_background_agent_denies_without_prompt(self):
        mutate = MutateTool()  # write-level, needs permission in safe mode
        context = ToolUseContext(cwd="/tmp", messages=[])
        perm_context = ToolPermissionContext(
            mode=PermissionMode.SAFE,
            should_avoid_prompts=True,
        )
        result = await check_permissions(mutate, {"target": "x"}, context, perm_context)
        assert result.behavior == PermissionBehavior.DENY
