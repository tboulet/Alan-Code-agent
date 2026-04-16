"""Test hooks system."""

import asyncio
import json
import os
import stat
import sys
import tempfile

import pytest

from alancode.hooks.registry import (
    HookConfig,
    HookResult,
    HookType,
    execute_hook,
    load_hooks_from_settings,
    run_hooks,
    run_pre_tool_hooks,
    run_post_tool_hooks,
)


# ---------------------------------------------------------------------------
# Helper: build a hook command from a Python one-liner
# ---------------------------------------------------------------------------

def _py_cmd(code: str) -> str:
    """Build a shell command that runs a Python one-liner."""
    return f'{sys.executable} -c "{code}"'


# ---------------------------------------------------------------------------
# TestHookConfig -- loading from settings dict
# ---------------------------------------------------------------------------


class TestHookConfig:

    def test_load_empty_hooks(self):
        result = load_hooks_from_settings({})
        assert result == {}

    def test_load_empty_hooks_key(self):
        result = load_hooks_from_settings({"hooks": {}})
        assert result == {}

    def test_load_pre_tool_hook(self):
        settings = {
            "hooks": {
                "PreToolUse": [
                    {"command": "echo allow"}
                ]
            }
        }
        result = load_hooks_from_settings(settings)
        assert HookType.PRE_TOOL_USE in result
        assert len(result[HookType.PRE_TOOL_USE]) == 1
        assert result[HookType.PRE_TOOL_USE][0].command == "echo allow"
        assert result[HookType.PRE_TOOL_USE][0].tools is None  # matches all tools

    def test_load_with_tool_filter(self):
        settings = {
            "hooks": {
                "PreToolUse": [
                    {"command": "check.py", "tools": ["Bash", "Write"]}
                ]
            }
        }
        result = load_hooks_from_settings(settings)
        hook = result[HookType.PRE_TOOL_USE][0]
        assert hook.tools == ["Bash", "Write"]

    def test_load_invalid_hook_type(self):
        """Unknown hook type name is skipped without error."""
        settings = {
            "hooks": {
                "NonExistent": [{"command": "echo nope"}],
                "PreToolUse": [{"command": "echo yes"}],
            }
        }
        result = load_hooks_from_settings(settings)
        assert HookType.PRE_TOOL_USE in result
        assert len(result) == 1  # NonExistent was skipped

    def test_load_shorthand_string(self):
        """A hook entry can be a plain string (command shorthand)."""
        settings = {
            "hooks": {
                "SessionStart": ["echo hello"]
            }
        }
        result = load_hooks_from_settings(settings)
        assert HookType.SESSION_START in result
        assert result[HookType.SESSION_START][0].command == "echo hello"

    def test_load_multiple_hooks_same_type(self):
        settings = {
            "hooks": {
                "PostToolUse": [
                    {"command": "cmd1"},
                    {"command": "cmd2"},
                ]
            }
        }
        result = load_hooks_from_settings(settings)
        assert len(result[HookType.POST_TOOL_USE]) == 2

    def test_load_missing_command_skipped(self):
        settings = {
            "hooks": {
                "PreToolUse": [
                    {"tools": ["Bash"]},  # missing 'command'
                    {"command": "valid.py"},
                ]
            }
        }
        result = load_hooks_from_settings(settings)
        assert len(result[HookType.PRE_TOOL_USE]) == 1

    def test_load_non_dict_hooks_value(self):
        """If hooks value is not a dict, it should be ignored."""
        result = load_hooks_from_settings({"hooks": "invalid"})
        assert result == {}

    def test_load_non_list_hook_entries(self):
        """If hook entries for a type are not a list, skip."""
        settings = {"hooks": {"PreToolUse": "not a list"}}
        result = load_hooks_from_settings(settings)
        assert result == {}


# ---------------------------------------------------------------------------
# TestExecuteHook -- running individual hooks
# ---------------------------------------------------------------------------


class TestExecuteHook:

    @pytest.mark.asyncio
    async def test_hook_allow(self):
        """Hook that outputs {"action": "allow"} on stdout."""
        cmd = _py_cmd("import json; print(json.dumps({'action': 'allow'}))")
        hook = HookConfig(command=cmd)
        result = await execute_hook(HookType.PRE_TOOL_USE, hook, {"tool_name": "Bash"})

        assert result.action == "allow"
        assert result.exit_code == 0

    @pytest.mark.asyncio
    async def test_hook_deny(self):
        """Hook that outputs {"action": "deny", "message": "blocked"}."""
        cmd = _py_cmd(
            "import json; print(json.dumps({'action': 'deny', 'message': 'blocked'}))"
        )
        hook = HookConfig(command=cmd)
        result = await execute_hook(HookType.PRE_TOOL_USE, hook, {"tool_name": "Bash"})

        assert result.action == "deny"
        assert result.message == "blocked"
        assert result.exit_code == 0

    @pytest.mark.asyncio
    async def test_hook_timeout(self):
        """Hook that sleeps too long gets killed."""
        cmd = _py_cmd("import time; time.sleep(60)")
        hook = HookConfig(command=cmd, timeout=1)
        result = await execute_hook(HookType.PRE_TOOL_USE, hook, {})

        assert result.exit_code == -1
        assert "timed out" in result.stderr.lower()
        # On timeout of a PreToolUse hook, fall back to "ask" — a broken
        # safety-critical hook must not silently allow. (PostToolUse still
        # falls back to "allow" since it's informational only.)
        assert result.action == "ask"

    @pytest.mark.asyncio
    async def test_hook_crash(self):
        """Hook that exits with non-zero code on PreToolUse -> deny."""
        cmd = _py_cmd("import sys; sys.exit(1)")
        hook = HookConfig(command=cmd)
        result = await execute_hook(HookType.PRE_TOOL_USE, hook, {})

        assert result.exit_code == 1
        assert result.action == "deny"

    @pytest.mark.asyncio
    async def test_hook_crash_post_tool(self):
        """Non-zero exit on PostToolUse does NOT auto-deny (only PreToolUse does)."""
        cmd = _py_cmd("import sys; sys.exit(1)")
        hook = HookConfig(command=cmd)
        result = await execute_hook(HookType.POST_TOOL_USE, hook, {})

        assert result.exit_code == 1
        # PostToolUse ignores exit code for deny logic
        assert result.action == "allow"

    @pytest.mark.asyncio
    async def test_hook_receives_payload(self):
        """Hook reads stdin and includes a marker in stdout."""
        # Read stdin, parse JSON, echo back the tool_name
        code = (
            "import sys, json; "
            "data = json.load(sys.stdin); "
            "print(json.dumps({'action': 'allow', 'message': data.get('tool_name', 'none')}))"
        )
        cmd = _py_cmd(code)
        hook = HookConfig(command=cmd)
        result = await execute_hook(
            HookType.PRE_TOOL_USE, hook, {"tool_name": "MyTool"}
        )

        assert result.action == "allow"
        assert result.message == "MyTool"

    @pytest.mark.asyncio
    async def test_hook_invalid_json_stdout(self):
        """Hook that outputs non-JSON on stdout -> treated as allow."""
        cmd = _py_cmd("print('not json')")
        hook = HookConfig(command=cmd)
        result = await execute_hook(HookType.PRE_TOOL_USE, hook, {})

        assert result.action == "allow"
        assert result.exit_code == 0

    @pytest.mark.asyncio
    async def test_hook_ask_action(self):
        """Hook that returns action=ask."""
        cmd = _py_cmd("import json; print(json.dumps({'action': 'ask'}))")
        hook = HookConfig(command=cmd)
        result = await execute_hook(HookType.PRE_TOOL_USE, hook, {})

        assert result.action == "ask"


# ---------------------------------------------------------------------------
# TestRunPreToolHooks -- the aggregation layer
# ---------------------------------------------------------------------------


class TestRunPreToolHooks:

    @pytest.mark.asyncio
    async def test_no_hooks_returns_none(self):
        """When no hooks configured, returns None (all allowed)."""
        result = await run_pre_tool_hooks("Bash", {"command": "ls"}, settings={})
        assert result is None

    @pytest.mark.asyncio
    async def test_deny_hook_blocks(self):
        cmd = _py_cmd(
            "import json; print(json.dumps({'action': 'deny', 'message': 'nope'}))"
        )
        settings = {
            "hooks": {
                "PreToolUse": [{"command": cmd}]
            }
        }
        result = await run_pre_tool_hooks("Bash", {"command": "rm -rf /"}, settings=settings)
        assert result is not None
        assert result.action == "deny"
        assert result.message == "nope"

    @pytest.mark.asyncio
    async def test_tool_filter_matches(self):
        """Hook with tools=["Bash"] runs for Bash tool calls."""
        cmd = _py_cmd(
            "import json; print(json.dumps({'action': 'deny', 'message': 'bash blocked'}))"
        )
        settings = {
            "hooks": {
                "PreToolUse": [{"command": cmd, "tools": ["Bash"]}]
            }
        }
        result = await run_pre_tool_hooks("Bash", {"command": "ls"}, settings=settings)
        assert result is not None
        assert result.action == "deny"

    @pytest.mark.asyncio
    async def test_tool_filter_skips(self):
        """Hook with tools=["Bash"] does NOT run for Read tool calls."""
        cmd = _py_cmd(
            "import json; print(json.dumps({'action': 'deny', 'message': 'should not see'}))"
        )
        settings = {
            "hooks": {
                "PreToolUse": [{"command": cmd, "tools": ["Bash"]}]
            }
        }
        result = await run_pre_tool_hooks("Read", {"file_path": "/tmp/x"}, settings=settings)
        assert result is None  # hook was filtered out

    @pytest.mark.asyncio
    async def test_allow_hook_returns_none(self):
        """When all hooks return allow, result is None."""
        cmd = _py_cmd("import json; print(json.dumps({'action': 'allow'}))")
        settings = {
            "hooks": {
                "PreToolUse": [{"command": cmd}]
            }
        }
        result = await run_pre_tool_hooks("Bash", {"command": "ls"}, settings=settings)
        assert result is None


# ---------------------------------------------------------------------------
# TestRunPostToolHooks
# ---------------------------------------------------------------------------


class TestRunPostToolHooks:

    @pytest.mark.asyncio
    async def test_post_tool_hooks_fire_and_forget(self):
        """Post-tool hooks run without returning a blocking result."""
        cmd = _py_cmd("print('logged')")
        settings = {
            "hooks": {
                "PostToolUse": [{"command": cmd}]
            }
        }
        # Should not raise; return type is None
        await run_post_tool_hooks(
            "Bash", {"command": "ls"}, "output text",
            settings=settings,
        )

    @pytest.mark.asyncio
    async def test_post_tool_failure_hooks(self):
        """PostToolUseFailure hooks run when is_error=True."""
        cmd = _py_cmd("print('failure logged')")
        settings = {
            "hooks": {
                "PostToolUseFailure": [{"command": cmd}],
                "PostToolUse": [{"command": cmd}],
            }
        }
        # Should run both PostToolUseFailure and PostToolUse hooks
        await run_post_tool_hooks(
            "Bash", {"command": "bad"}, "error output",
            is_error=True, settings=settings,
        )


# ---------------------------------------------------------------------------
# TestRunHooks -- general
# ---------------------------------------------------------------------------


class TestRunHooks:

    @pytest.mark.asyncio
    async def test_run_hooks_returns_list(self):
        cmd = _py_cmd("import json; print(json.dumps({'action': 'allow'}))")
        settings = {
            "hooks": {
                "PreToolUse": [{"command": cmd}]
            }
        }
        results = await run_hooks(
            HookType.PRE_TOOL_USE, {"tool_name": "X"}, settings=settings,
        )
        assert isinstance(results, list)
        assert len(results) == 1
        assert results[0].action == "allow"

    @pytest.mark.asyncio
    async def test_run_hooks_empty_settings(self):
        results = await run_hooks(HookType.SESSION_START, {}, settings={})
        assert results == []
