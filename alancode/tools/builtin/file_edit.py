"""FileEditTool — perform string replacement in files."""

import difflib
import os
from typing import Any

from alancode.tools.base import Tool, ToolResult, ToolUseContext


class FileEditTool(Tool):
    """Perform exact string replacements in files."""

    @property
    def name(self) -> str:
        return "Edit"

    @property
    def description(self) -> str:
        return (
            "Performs exact string replacements in files.\n\n"
            "Usage:\n"
            "- You must use your Read tool at least once in the conversation "
            "before editing. This tool will error if you attempt an edit without "
            "reading the file.\n"
            "- ALWAYS prefer editing existing files in the codebase. NEVER write "
            "new files unless explicitly required.\n"
            "- The edit will FAIL if old_string is not unique in the file. Either "
            "provide a larger string with more surrounding context to make it "
            "unique, or use replace_all to change every instance of old_string.\n"
            "- Use replace_all for replacing and renaming strings across the file."
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "The absolute path to the file to modify (must be absolute, not relative).",
                },
                "old_string": {
                    "type": "string",
                    "description": (
                        "The text to replace. Must be an exact match of the file content "
                        "including whitespace and indentation."
                    ),
                },
                "new_string": {
                    "type": "string",
                    "description": "The text to replace it with (must be different from old_string).",
                },
                "replace_all": {
                    "type": "boolean",
                    "description": (
                        "Replace all occurrences of old_string (default false). "
                        "If false and old_string appears multiple times, the edit fails."
                    ),
                    "default": False,
                },
            },
            "required": ["file_path", "old_string", "new_string"],
        }

    def permission_level(self, args: dict[str, Any]) -> str:
        return "write"

    async def call(self, args: dict[str, Any], context: ToolUseContext) -> ToolResult:
        file_path = args.get("file_path", "")
        old_string = args.get("old_string", "")
        new_string = args.get("new_string", "")
        replace_all = args.get("replace_all", False)

        if not file_path:
            given_keys = list(args.keys())
            return ToolResult(
                data=f"Error: 'file_path' parameter is required but was not provided. "
                     f"Got parameters: {given_keys}. "
                     f"Use <arg_key>file_path</arg_key><arg_value>/absolute/path/to/file</arg_value> "
                     f"<arg_key>old_string</arg_key><arg_value>TEXT_TO_FIND</arg_value> "
                     f"<arg_key>new_string</arg_key><arg_value>REPLACEMENT_TEXT</arg_value>",
                is_error=True,
            )
        if not old_string:
            given_keys = list(args.keys())
            return ToolResult(
                data=f"Error: 'old_string' parameter is required but was not provided. "
                     f"Got parameters: {given_keys}. "
                     f"Use <arg_key>file_path</arg_key><arg_value>/absolute/path/to/file</arg_value> "
                     f"<arg_key>old_string</arg_key><arg_value>TEXT_TO_FIND</arg_value> "
                     f"<arg_key>new_string</arg_key><arg_value>REPLACEMENT_TEXT</arg_value>",
                is_error=True,
            )
        if old_string == new_string:
            return ToolResult(data="Error: old_string and new_string are identical.", is_error=True)

        # Resolve relative paths and follow symlinks
        if not os.path.isabs(file_path):
            file_path = os.path.join(context.cwd, file_path)
        file_path = os.path.realpath(file_path)

        if not os.path.exists(file_path):
            return ToolResult(data=f"Error: file not found: {file_path}", is_error=True)

        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except Exception as exc:
            return ToolResult(data=f"Error reading file: {exc}", is_error=True)

        count = content.count(old_string)

        if count == 0:
            preview = old_string[:120].replace("\n", "\\n")
            return ToolResult(
                data=(
                    f"Error: old_string not found in {file_path}.\n"
                    f"  Searched for ({len(old_string)} chars): \"{preview}\"\n"
                    f"  replace_all={replace_all}\n"
                    f"Hint: read the file first to get the exact content, "
                    f"including whitespace and indentation."
                ),
                is_error=True,
            )

        if not replace_all and count > 1:
            preview = old_string[:80].replace("\n", "\\n")
            return ToolResult(
                data=(
                    f"Error: old_string appears {count} times in {file_path}.\n"
                    f"  Searched for: \"{preview}\"\n"
                    f"  replace_all={replace_all}\n"
                    f"Provide more surrounding context to make the match unique, "
                    f"or set replace_all=true to replace all occurrences."
                ),
                is_error=True,
            )

        new_content = content.replace(old_string, new_string) if replace_all else content.replace(old_string, new_string, 1)

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(new_content)
        except Exception as exc:
            return ToolResult(data=f"Error writing file: {exc}", is_error=True)

        replacements = count if replace_all else 1
        diff_text = _make_diff(content, new_content, file_path)
        summary = f"Successfully replaced {replacements} occurrence(s) in {file_path}."
        return ToolResult(data=f"[ALAN-DIFF]\n{diff_text}\n{summary}")


def _make_diff(old: str, new: str, path: str, context: int = 3) -> str:
    """Return a unified diff for old vs new. Trailing newline trimmed."""
    old_lines = old.splitlines(keepends=True)
    new_lines = new.splitlines(keepends=True)
    diff_iter = difflib.unified_diff(
        old_lines, new_lines,
        fromfile=path, tofile=path,
        n=context,
    )
    return "".join(diff_iter).rstrip("\n")
