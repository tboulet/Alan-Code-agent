"""Extended permission tests (alancode/permissions/)."""

import pytest

from alancode.permissions.context import (
    PermissionBehavior,
    PermissionMode,
    PermissionRule,
    ToolPermissionContext,
)
from alancode.permissions.pipeline import (
    _mode_allows,
    check_permissions,
    check_rule_match,
)
from alancode.tools.base import Tool, ToolResult, ToolUseContext


# ---------------------------------------------------------------------------
# Helpers — minimal tool implementations for testing
# ---------------------------------------------------------------------------


class ReadTool(Tool):
    """A read-only tool."""

    @property
    def name(self):
        return "ReadTool"

    @property
    def description(self):
        return "Reads something"

    @property
    def input_schema(self):
        return {"type": "object", "properties": {}, "required": []}

    async def call(self, args, context):
        return ToolResult(data="read")

    def permission_level(self, args):
        return "read"


class WriteTool(Tool):
    """A write-level tool."""

    @property
    def name(self):
        return "WriteTool"

    @property
    def description(self):
        return "Writes something"

    @property
    def input_schema(self):
        return {"type": "object", "properties": {}, "required": []}

    async def call(self, args, context):
        return ToolResult(data="write")

    def permission_level(self, args):
        return "write"


class ExecTool(Tool):
    """An exec-level tool (like Bash)."""

    @property
    def name(self):
        return "ExecTool"

    @property
    def aliases(self):
        return ["Bash"]

    @property
    def description(self):
        return "Executes something"

    @property
    def input_schema(self):
        return {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        }

    async def call(self, args, context):
        return ToolResult(data="exec")

    def permission_level(self, args):
        return "exec"


# ---------------------------------------------------------------------------
# _mode_allows matrix
# ---------------------------------------------------------------------------


class TestModeAllowsMatrix:
    """Verify the full yolo/edit/safe x read/write/exec permission matrix."""

    # YOLO mode: everything allowed
    def test_yolo_allows_read(self):
        assert _mode_allows(PermissionMode.YOLO, "read") is True

    def test_yolo_allows_write(self):
        assert _mode_allows(PermissionMode.YOLO, "write") is True

    def test_yolo_allows_exec(self):
        assert _mode_allows(PermissionMode.YOLO, "exec") is True

    # EDIT mode: read + write allowed, exec denied
    def test_edit_allows_read(self):
        assert _mode_allows(PermissionMode.EDIT, "read") is True

    def test_edit_allows_write(self):
        assert _mode_allows(PermissionMode.EDIT, "write") is True

    def test_edit_denies_exec(self):
        assert _mode_allows(PermissionMode.EDIT, "exec") is False

    # SAFE mode: only read allowed
    def test_safe_allows_read(self):
        assert _mode_allows(PermissionMode.SAFE, "read") is True

    def test_safe_denies_write(self):
        assert _mode_allows(PermissionMode.SAFE, "write") is False

    def test_safe_denies_exec(self):
        assert _mode_allows(PermissionMode.SAFE, "exec") is False


class TestModeAllowsIntegration:
    """Test the full pipeline behavior for each mode x level combination."""

    @pytest.mark.asyncio
    async def test_yolo_allows_exec_tool(self):
        tool = ExecTool()
        ctx = ToolUseContext(cwd="/tmp", messages=[])
        perm_ctx = ToolPermissionContext(mode=PermissionMode.YOLO)
        result = await check_permissions(tool, {"command": "rm -rf /"}, ctx, perm_ctx)
        assert result.behavior == PermissionBehavior.ALLOW

    @pytest.mark.asyncio
    async def test_edit_asks_for_exec_tool(self):
        tool = ExecTool()
        ctx = ToolUseContext(cwd="/tmp", messages=[])
        perm_ctx = ToolPermissionContext(mode=PermissionMode.EDIT)
        result = await check_permissions(tool, {"command": "ls"}, ctx, perm_ctx)
        # No prompt callback, no allow rules -> ASK
        assert result.behavior == PermissionBehavior.ASK

    @pytest.mark.asyncio
    async def test_safe_asks_for_write_tool(self):
        tool = WriteTool()
        ctx = ToolUseContext(cwd="/tmp", messages=[])
        perm_ctx = ToolPermissionContext(mode=PermissionMode.SAFE)
        result = await check_permissions(tool, {}, ctx, perm_ctx)
        assert result.behavior == PermissionBehavior.ASK

    @pytest.mark.asyncio
    async def test_safe_allows_read_tool(self):
        tool = ReadTool()
        ctx = ToolUseContext(cwd="/tmp", messages=[])
        perm_ctx = ToolPermissionContext(mode=PermissionMode.SAFE)
        result = await check_permissions(tool, {}, ctx, perm_ctx)
        assert result.behavior == PermissionBehavior.ALLOW


# ---------------------------------------------------------------------------
# Rule word-boundary matching
# ---------------------------------------------------------------------------


class TestRuleWordBoundary:

    def test_git_star_matches_git_push(self):
        """Rule 'git *' should match command 'git push'."""
        tool = ExecTool()
        rule = PermissionRule(
            tool_name="ExecTool",
            rule_content="git *",
            behavior=PermissionBehavior.ALLOW,
        )
        result = check_rule_match([rule], tool, {"command": "git push"})
        assert result is not None

    def test_git_star_does_not_match_gitconfig(self):
        """Rule 'git *' should NOT match 'gitconfig' — word boundary matters."""
        tool = ExecTool()
        rule = PermissionRule(
            tool_name="ExecTool",
            rule_content="git *",
            behavior=PermissionBehavior.ALLOW,
        )
        result = check_rule_match([rule], tool, {"command": "gitconfig"})
        assert result is None

    def test_git_star_does_not_match_github(self):
        """Rule 'git *' should NOT match 'github-cli'."""
        tool = ExecTool()
        rule = PermissionRule(
            tool_name="ExecTool",
            rule_content="git *",
            behavior=PermissionBehavior.ALLOW,
        )
        result = check_rule_match([rule], tool, {"command": "github-cli"})
        assert result is None

    def test_git_star_matches_exact_git(self):
        """Rule 'git *' should match the exact command 'git' (no args)."""
        tool = ExecTool()
        rule = PermissionRule(
            tool_name="ExecTool",
            rule_content="git *",
            behavior=PermissionBehavior.ALLOW,
        )
        # "git *" stripped of trailing "* " becomes "git"
        # The check is: value == rule_pattern or value.startswith(rule_pattern + " ")
        result = check_rule_match([rule], tool, {"command": "git"})
        assert result is not None  # exact match "git" == "git"

    def test_blanket_rule_matches_any_input(self):
        """A rule with rule_content=None should match any input."""
        tool = ExecTool()
        rule = PermissionRule(
            tool_name="ExecTool",
            rule_content=None,
            behavior=PermissionBehavior.ALLOW,
        )
        result = check_rule_match([rule], tool, {"command": "anything at all"})
        assert result is not None

    def test_rule_does_not_match_wrong_tool(self):
        """A rule for ExecTool should not match ReadTool."""
        tool = ReadTool()
        rule = PermissionRule(
            tool_name="ExecTool",
            rule_content=None,
            behavior=PermissionBehavior.ALLOW,
        )
        result = check_rule_match([rule], tool, {})
        assert result is None


# ---------------------------------------------------------------------------
# validate_input is NOT called from the pipeline
# ---------------------------------------------------------------------------


class TestValidateInputNotCalledInPipeline:
    """Ensure check_permissions does NOT call tool.validate_input.

    Per the code comment:
        'Note: tool.validate_input() is called in run_tool_use() before
         check_permissions(), so we don't duplicate it here.'
    """

    @pytest.mark.asyncio
    async def test_validate_input_not_called(self):
        """validate_input should not be invoked during permission checking."""
        call_tracker = {"called": False}

        class InstrumentedTool(ReadTool):
            def validate_input(self, args, context):
                call_tracker["called"] = True
                return None

        tool = InstrumentedTool()
        ctx = ToolUseContext(cwd="/tmp", messages=[])
        perm_ctx = ToolPermissionContext(mode=PermissionMode.YOLO)

        await check_permissions(tool, {}, ctx, perm_ctx)
        assert call_tracker["called"] is False, (
            "validate_input should not be called from check_permissions"
        )
