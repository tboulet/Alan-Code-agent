"""FileReadTool — read file contents with line numbers."""

import os
from typing import Any

from alancode.tools.base import Tool, ToolResult, ToolUseContext


class FileReadTool(Tool):
    """Read file contents with line numbers (cat -n format)."""

    @property
    def name(self) -> str:
        return "Read"

    @property
    def description(self) -> str:
        return (
            "Reads a file from the local filesystem. You can access any file "
            "directly by using this tool. Assume this tool is able to read all "
            "files on the machine.\n\n"
            "Usage:\n"
            "- The file_path parameter must be an absolute path, not a relative path\n"
            "- By default, it reads up to 2000 lines starting from the beginning of the file\n"
            "- When you already know which part of the file you need, only read that part\n"
            "- Results are returned using cat -n format, with line numbers starting at 1\n"
            "- This tool can only read files, not directories. To read a directory, "
            "use the Bash tool with 'ls'.\n"
            "- If you read a file that exists but has empty contents you will receive "
            "a warning in place of file contents."
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "The absolute path to the file to read (must be absolute, not relative).",
                },
                "offset": {
                    "type": "integer",
                    "description": (
                        "The line number to start reading from (0-indexed). "
                        "Only provide if the file is too large to read at once."
                    ),
                    "minimum": 0,
                },
                "limit": {
                    "type": "integer",
                    "description": (
                        "The number of lines to read. Default 2000. "
                        "Only provide if the file is too large to read at once."
                    ),
                    "exclusiveMinimum": 0,
                },
            },
            "required": ["file_path"],
        }

    def permission_level(self, args: dict[str, Any]) -> str:
        return "read"

    async def call(self, args: dict[str, Any], context: ToolUseContext) -> ToolResult:
        file_path = args.get("file_path", "")
        offset = args.get("offset", 0)
        limit = args.get("limit", 2000)

        if not file_path:
            given_keys = list(args.keys())
            return ToolResult(
                data=f"Error: 'file_path' parameter is required but was not provided. "
                     f"Got parameters: {given_keys}. "
                     f"Use <arg_key>file_path</arg_key><arg_value>/absolute/path/to/file</arg_value>",
                is_error=True,
            )

        # Resolve relative paths against cwd and follow symlinks
        if not os.path.isabs(file_path):
            file_path = os.path.join(context.cwd, file_path)
        file_path = os.path.realpath(file_path)

        if not os.path.exists(file_path):
            return ToolResult(data=f"Error: file not found: {file_path}", is_error=True)

        if os.path.isdir(file_path):
            return ToolResult(
                data=f"Error: {file_path} is a directory, not a file. Use Bash with 'ls' to list directories.",
                is_error=True,
            )

        # Check for binary files
        try:
            with open(file_path, "rb") as f:
                chunk = f.read(8192)
            if b"\x00" in chunk:
                return ToolResult(
                    data=f"Error: {file_path} appears to be a binary file.",
                    is_error=True,
                )
        except PermissionError:
            return ToolResult(data=f"Error: permission denied reading {file_path}", is_error=True)
        except Exception as exc:
            return ToolResult(data=f"Error reading file: {exc}", is_error=True)

        # Read the file
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                all_lines = f.readlines()
        except Exception as exc:
            return ToolResult(data=f"Error reading file: {exc}", is_error=True)

        total_lines = len(all_lines)
        if total_lines == 0:
            return ToolResult(data=f"(empty file: {file_path})")

        if offset >= total_lines:
            return ToolResult(
                data=f"Warning: offset {offset} exceeds file length ({total_lines} lines).",
                is_error=True,
            )

        selected = all_lines[offset : offset + limit]

        # Format with line numbers (1-based, matching cat -n)
        result_lines = []
        for i, line in enumerate(selected, start=offset + 1):
            # Right-align line number in 6 chars, tab, then content (no trailing newline)
            result_lines.append(f"{i:>6}\t{line.rstrip()}")

        output = "\n".join(result_lines)

        # Warn if truncated
        remaining = total_lines - (offset + len(selected))
        if remaining > 0:
            output += f"\n\n... ({remaining} more lines not shown. Use offset={offset + limit} to continue.)"

        return ToolResult(data=output)
