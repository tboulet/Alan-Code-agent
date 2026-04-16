"""GrepTool — content search using regex (via ripgrep, grep, or pure Python fallback)."""

import asyncio
import fnmatch
import os
import pathlib
import re
import shutil
from typing import Any

from alancode.tools.base import Tool, ToolResult, ToolUseContext

_DEFAULT_HEAD_LIMIT = 250


class GrepTool(Tool):
    """Search file contents using regex patterns via ripgrep, grep, or pure Python fallback."""

    @property
    def name(self) -> str:
        return "Grep"

    @property
    def description(self) -> str:
        return (
            "Search file contents using regex patterns. Supports ripgrep syntax "
            "when available, with fallback to grep or pure Python.\n\n"
            "Usage:\n"
            "- Use for searching code, configuration, and patterns across the codebase\n"
            "- Supports full regex syntax (e.g., 'log.*Error', 'function\\s+\\w+')\n"
            "- Filter files with the glob parameter (e.g., '*.py', '*.{ts,tsx}')\n"
            "- Output modes: 'files_with_matches' (default, just paths), "
            "'content' (matching lines), 'count' (match counts)"
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The regular expression pattern to search for in file contents.",
                },
                "path": {
                    "type": "string",
                    "description": (
                        "File or directory to search in. Defaults to current working directory. "
                        "Use this for directory scoping (e.g. 'src/' to search only in src/)."
                    ),
                },
                "glob": {
                    "type": "string",
                    "description": (
                        "Glob pattern to filter files (e.g. '*.py', '*.{ts,tsx}'). "
                        "Do NOT include directory prefixes here -- use 'path' for that."
                    ),
                },
                "output_mode": {
                    "type": "string",
                    "enum": ["content", "files_with_matches", "count"],
                    "description": (
                        "Output mode: 'content' shows matching lines with context, "
                        "'files_with_matches' shows only file paths (default), "
                        "'count' shows match counts per file."
                    ),
                },
                "context": {
                    "type": "integer",
                    "description": (
                        "Number of lines to show before and after each match. "
                        "Only applies when output_mode is 'content'."
                    ),
                },
            },
            "required": ["pattern"],
        }

    def permission_level(self, args: dict[str, Any]) -> str:
        return "read"

    async def call(self, args: dict[str, Any], context: ToolUseContext) -> ToolResult:
        pattern = args.get("pattern", "")
        search_path = args.get("path", "") or context.cwd
        file_glob = args.get("glob", "")
        output_mode = args.get("output_mode", "files_with_matches")
        ctx_lines = args.get("context", None)

        if not pattern:
            given_keys = list(args.keys())
            return ToolResult(
                data=f"Error: 'pattern' parameter is required but was not provided. "
                     f"Got parameters: {given_keys}. "
                     f"Use <arg_key>pattern</arg_key><arg_value>YOUR_REGEX_PATTERN</arg_value>",
                is_error=True,
            )

        if not os.path.isabs(search_path):
            search_path = os.path.join(context.cwd, search_path)
        # Normalize: /foo/bar/. → /foo/bar
        search_path = os.path.normpath(search_path)

        if not os.path.exists(search_path):
            return ToolResult(data=f"Error: path not found: {search_path}", is_error=True)

        # Try rg first, then grep, then fallback to Python
        rg_path = shutil.which("rg")
        grep_path = shutil.which("grep")

        if rg_path:
            result = await self._run_rg(rg_path, pattern, search_path, file_glob, output_mode, ctx_lines)
        elif grep_path:
            result = await self._run_grep(grep_path, pattern, search_path, file_glob, output_mode, ctx_lines)
        else:
            result = await self._python_fallback(pattern, search_path, file_glob, output_mode, ctx_lines)

        return result

    async def _run_rg(
        self, rg: str, pattern: str, path: str, file_glob: str,
        mode: str, ctx_lines: int | None,
    ) -> ToolResult:
        """Build and run a ripgrep command with the given search parameters."""
        cmd = [rg, "--no-heading", "-n"]

        if mode == "files_with_matches":
            cmd.append("-l")
        elif mode == "count":
            cmd.append("-c")

        if ctx_lines is not None and mode == "content":
            cmd.extend(["-C", str(ctx_lines)])

        if file_glob:
            cmd.extend(["--glob", file_glob])

        cmd.extend(["--", pattern, path])
        return await self._run_subprocess(cmd)

    async def _run_grep(
        self, grep_bin: str, pattern: str, path: str, file_glob: str,
        mode: str, ctx_lines: int | None,
    ) -> ToolResult:
        """Build and run a GNU grep command with the given search parameters."""
        cmd = [grep_bin, "-rn", "-E"]

        if mode == "files_with_matches":
            cmd.append("-l")
        elif mode == "count":
            cmd.append("-c")

        if ctx_lines is not None and mode == "content":
            cmd.extend(["-C", str(ctx_lines)])

        if file_glob:
            # GNU grep's --include doesn't support **/ (double-star recursion).
            # Since -r already recurses, strip any leading **/ prefixes.
            # e.g., "**/*.py" → "*.py", "**/src/**/*.ts" → "*.ts"
            clean_glob = file_glob
            while clean_glob.startswith("**/"):
                clean_glob = clean_glob[3:]
            # If there's still a **/ in the middle, take the last segment
            if "**/" in clean_glob:
                clean_glob = clean_glob.rsplit("**/", 1)[-1]
            cmd.extend(["--include", clean_glob])

        cmd.extend(["--", pattern, path])
        return await self._run_subprocess(cmd)

    async def _run_subprocess(self, cmd: list[str]) -> ToolResult:
        """Execute a search subprocess and return its output as a ToolResult."""
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
        except asyncio.TimeoutError:
            return ToolResult(data="Search timed out after 30s.", is_error=True)
        except Exception as exc:
            return ToolResult(data=f"Error running search: {exc}", is_error=True)

        output = stdout.decode("utf-8", errors="replace").rstrip()

        if proc.returncode == 1 and not output:
            return ToolResult(data="No matches found.")

        if proc.returncode not in (0, 1):
            err = stderr.decode("utf-8", errors="replace").rstrip()
            return ToolResult(data=f"Search error (exit {proc.returncode}): {err}", is_error=True)

        # Truncate to head limit
        lines = output.split("\n")
        if len(lines) > _DEFAULT_HEAD_LIMIT:
            output = "\n".join(lines[:_DEFAULT_HEAD_LIMIT])
            output += f"\n\n(Truncated — showing {_DEFAULT_HEAD_LIMIT} of {len(lines)} lines.)"

        return ToolResult(data=output if output else "No matches found.")

    async def _python_fallback(
        self, pattern: str, path: str, file_glob: str,
        mode: str, ctx_lines: int | None,
    ) -> ToolResult:
        """Pure-Python fallback when neither rg nor grep is available.

        Wall-clock capped at 30s. On a large tree this matters — without
        the cap the agent hangs until the full filesystem walk completes.
        """
        try:
            return await asyncio.wait_for(
                self._python_fallback_impl(
                    pattern, path, file_glob, mode, ctx_lines,
                ),
                timeout=30.0,
            )
        except asyncio.TimeoutError:
            return ToolResult(
                data="Python-fallback search timed out after 30s. "
                     "Install `ripgrep` (`rg`) or GNU `grep` for faster search, "
                     "or narrow the pattern / path.",
                is_error=True,
            )

    async def _python_fallback_impl(
        self, pattern: str, path: str, file_glob: str,
        mode: str, ctx_lines: int | None,
    ) -> ToolResult:
        try:
            regex = re.compile(pattern)
        except re.error as exc:
            return ToolResult(data=f"Invalid regex: {exc}", is_error=True)

        base = pathlib.Path(path)
        files = [base] if base.is_file() else sorted(base.rglob("*"))

        results: list[str] = []
        for fp in files:
            if not fp.is_file():
                continue
            if file_glob and not fnmatch.fnmatch(fp.name, file_glob):
                continue
            try:
                text = fp.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue

            if mode == "files_with_matches":
                if regex.search(text):
                    results.append(str(fp))
            elif mode == "count":
                cnt = len(regex.findall(text))
                if cnt:
                    results.append(f"{fp}:{cnt}")
            else:
                for i, line in enumerate(text.splitlines(), 1):
                    if regex.search(line):
                        results.append(f"{fp}:{i}:{line}")

            if len(results) >= _DEFAULT_HEAD_LIMIT:
                break
            # Cooperative checkpoint so asyncio.wait_for can interrupt us.
            await asyncio.sleep(0)

        if not results:
            return ToolResult(data="No matches found.")

        output = "\n".join(results[:_DEFAULT_HEAD_LIMIT])
        if len(results) > _DEFAULT_HEAD_LIMIT:
            output += f"\n\n(Truncated to {_DEFAULT_HEAD_LIMIT} results.)"
        return ToolResult(data=output)
