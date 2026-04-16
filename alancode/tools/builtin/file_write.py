"""FileWriteTool — create or overwrite a file."""

import difflib
import os
from typing import Any

from alancode.tools.base import Tool, ToolResult, ToolUseContext


class FileWriteTool(Tool):
    """Create or overwrite a file on the local filesystem."""

    @property
    def name(self) -> str:
        return "Write"

    @property
    def description(self) -> str:
        return (
            "Writes a file to the local filesystem.\n\n"
            "Usage:\n"
            "- This tool will overwrite the existing file if there is one at the "
            "provided path.\n"
            "- If this is an existing file, you MUST use the Read tool first to "
            "read the file's contents. This tool will fail if you did not read the "
            "file first.\n"
            "- Prefer the Edit tool for modifying existing files -- it only sends "
            "the diff. Only use this tool to create new files or for complete rewrites.\n"
            "- NEVER create documentation files (*.md) or README files unless "
            "explicitly requested by the user."
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "The absolute path to the file to write (must be absolute, not relative).",
                },
                "content": {
                    "type": "string",
                    "description": "The content to write to the file.",
                },
            },
            "required": ["file_path", "content"],
        }

    def permission_level(self, args: dict[str, Any]) -> str:
        return "write"

    async def call(self, args: dict[str, Any], context: ToolUseContext) -> ToolResult:
        file_path = args.get("file_path", "")
        content = args.get("content", "")

        if not file_path:
            given_keys = list(args.keys())
            return ToolResult(
                data=f"Error: 'file_path' parameter is required but was not provided. "
                     f"Got parameters: {given_keys}. "
                     f"Use <arg_key>file_path</arg_key><arg_value>/absolute/path/to/file</arg_value> "
                     f"and <arg_key>content</arg_key><arg_value>FILE_CONTENT</arg_value>",
                is_error=True,
            )

        # Resolve relative paths against cwd and follow symlinks
        if not os.path.isabs(file_path):
            file_path = os.path.join(context.cwd, file_path)
        file_path = os.path.realpath(file_path)

        # Create parent directories if needed
        parent = os.path.dirname(file_path)
        try:
            if parent:
                os.makedirs(parent, exist_ok=True)
        except OSError as exc:
            return ToolResult(data=f"Error creating directory {parent}: {exc}", is_error=True)

        existed = os.path.exists(file_path)
        old_content = ""
        if existed:
            try:
                with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                    old_content = f.read()
            except Exception:
                old_content = ""

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
        except PermissionError:
            return ToolResult(data=f"Error: permission denied writing to {file_path}", is_error=True)
        except Exception as exc:
            return ToolResult(data=f"Error writing file: {exc}", is_error=True)

        line_count = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
        verb = "Overwrote" if existed else "Created"
        summary = f"{verb} {file_path} ({line_count} lines, {len(content)} chars)"

        diff_text = _make_write_diff(old_content, content, file_path, existed)
        if diff_text:
            return ToolResult(data=f"[ALAN-DIFF]\n{diff_text}\n{summary}")
        return ToolResult(data=summary)


def _make_write_diff(
    old: str, new: str, path: str, existed: bool, context: int = 3,
) -> str:
    """Return a unified diff for the write.

    - On overwrite: diff of old vs new content.
    - On creation: diff from empty — the whole file is shown as '+' lines.
    """
    old_lines = old.splitlines(keepends=True) if existed else []
    new_lines = new.splitlines(keepends=True)
    diff_iter = difflib.unified_diff(
        old_lines, new_lines,
        fromfile=path if existed else "/dev/null",
        tofile=path,
        n=context,
    )
    return "".join(diff_iter).rstrip("\n")
