"""BashTool — execute shell commands via asyncio subprocess."""

import asyncio
from typing import Any

from alancode.tools.base import Tool, ToolResult, ToolUseContext


class BashTool(Tool):
    """Execute shell commands via asyncio subprocess."""

    @property
    def name(self) -> str:
        return "Bash"

    @property
    def description(self) -> str:
        return (
            "Executes a given bash command and returns its output.\n\n"
            "The working directory is set to the project root for each call. "
            "stdout and stderr are combined in the output.\n\n"
            "IMPORTANT: Avoid using this tool to run cat, head, tail, sed, awk, "
            "or echo commands unless explicitly instructed. Instead, use the "
            "appropriate dedicated tool (Read, Edit, Write, Glob, Grep) as they "
            "provide a better experience.\n\n"
            "For quick Python snippets, use: python3 -c '<code>'"
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": (
                        "The command to execute in a bash shell. "
                        "Use '&&' to chain sequential commands. "
                        "Always quote file paths that contain spaces."
                    ),
                },
                "timeout": {
                    "type": "integer",
                    "description": (
                        "Optional timeout in milliseconds (default 120000, i.e. 2 minutes). "
                        "The command will be killed if it exceeds this duration."
                    ),
                },
                "purpose": {
                    "type": "string",
                    "description": (
                        "A clear, concise one-line summary of what this "
                        "command does. Shown to the user before approval."
                    ),
                },
            },
            "required": ["command"],
        }

    def permission_level(self, args: dict[str, Any]) -> str:
        return "exec"

    async def call(self, args: dict[str, Any], context: ToolUseContext) -> ToolResult:
        command = args.get("command", "")
        if not command.strip():
            given_keys = list(args.keys())
            return ToolResult(
                data=f"Error: 'command' parameter is required but was not provided. "
                     f"Got parameters: {given_keys}. "
                     f"Use <arg_key>command</arg_key><arg_value>YOUR_COMMAND</arg_value>",
                is_error=True,
            )

        timeout_ms = args.get("timeout", 120_000)
        if not isinstance(timeout_ms, (int, float)) or timeout_ms <= 0:
            timeout_ms = 120_000
        timeout_s = timeout_ms / 1000.0

        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=context.cwd,
            )
        except Exception as exc:
            return ToolResult(data=f"Failed to start process: {exc}", is_error=True)

        try:
            stdout, _ = await asyncio.wait_for(process.communicate(), timeout=timeout_s)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            return ToolResult(
                data=f"Command timed out after {timeout_ms}ms and was killed.",
                is_error=True,
            )
        except Exception as exc:
            return ToolResult(data=f"Error during execution: {exc}", is_error=True)

        output = stdout.decode("utf-8", errors="replace") if stdout else ""
        exit_code = process.returncode

        # Trim trailing whitespace but keep structure
        output = output.rstrip()

        if exit_code == 0:
            return ToolResult(data=output if output else "(no output)")
        else:
            text = output + (f"\n\nExit code: {exit_code}" if output else f"Exit code: {exit_code}")
            return ToolResult(data=text, is_error=(exit_code != 0))
