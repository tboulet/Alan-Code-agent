"""GlobTool — file pattern matching."""

import asyncio
import os
import pathlib
from typing import Any

from alancode.tools.base import Tool, ToolResult, ToolUseContext

_MAX_RESULTS = 1000
_DEFAULT_TIMEOUT_MS = 30_000  # 30 seconds


class GlobTool(Tool):
    """Find files by glob pattern, sorted by modification time."""

    @property
    def name(self) -> str:
        return "Glob"

    @property
    def description(self) -> str:
        return (
            "Fast file pattern matching tool that works with any codebase size. "
            "Supports glob patterns like '**/*.py' or 'src/**/*.ts'. "
            "Returns matching file paths sorted by modification time (newest first). "
            "Use this tool when you need to find files by name or extension patterns."
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The glob pattern to match files against (e.g., '**/*.py', 'src/**/*.ts').",
                },
                "path": {
                    "type": "string",
                    "description": (
                        "The directory to search in. If not specified, the current working "
                        "directory will be used. Must be a valid directory path if provided."
                    ),
                },
                "timeout": {
                    "type": "integer",
                    "description": (
                        "Optional timeout in milliseconds (default 30000). "
                        "The search will stop if it exceeds this duration."
                    ),
                },
            },
            "required": ["pattern"],
        }

    def permission_level(self, args: dict[str, Any]) -> str:
        return "read"

    async def call(self, args: dict[str, Any], context: ToolUseContext) -> ToolResult:
        """Execute the glob search with timeout protection."""
        pattern = args.get("pattern", "")
        search_path = args.get("path", "") or context.cwd
        timeout_ms = args.get("timeout", _DEFAULT_TIMEOUT_MS)
        if not isinstance(timeout_ms, (int, float)) or timeout_ms <= 0:
            timeout_ms = _DEFAULT_TIMEOUT_MS
        timeout_s = timeout_ms / 1000.0

        if not pattern:
            given_keys = list(args.keys())
            return ToolResult(
                data=f"Error: 'pattern' parameter is required but was not provided. "
                     f"Got parameters: {given_keys}. "
                     f"Use <arg_key>pattern</arg_key><arg_value>YOUR_PATTERN</arg_value>",
                is_error=True,
            )

        if not os.path.isabs(search_path):
            search_path = os.path.join(context.cwd, search_path)

        if not os.path.isdir(search_path):
            return ToolResult(data=f"Error: directory not found: {search_path}", is_error=True)

        base = pathlib.Path(search_path)

        # Scan an extra chunk beyond _MAX_RESULTS so the caller can tell
        # the difference between "exactly _MAX_RESULTS files" and "more
        # than _MAX_RESULTS files exist but I only kept the first ones".
        scan_cap = _MAX_RESULTS * 2

        def _run_glob():
            matches = []
            for p in base.glob(pattern):
                if p.is_file():
                    try:
                        mtime = p.stat().st_mtime
                    except OSError:
                        mtime = 0.0
                    matches.append((str(p), mtime))
                    if len(matches) >= scan_cap:
                        break
            return matches

        try:
            matches = await asyncio.wait_for(
                asyncio.to_thread(_run_glob),
                timeout=timeout_s,
            )
        except asyncio.TimeoutError:
            return ToolResult(
                data=f"Glob search timed out after {timeout_ms}ms. Try a more specific pattern.",
                is_error=True,
            )
        except Exception as exc:
            return ToolResult(data=f"Error during glob: {exc}", is_error=True)

        # Sort by modification time, newest first
        matches.sort(key=lambda x: x[1], reverse=True)

        if not matches:
            return ToolResult(data=f"No files matched pattern '{pattern}' in {search_path}")

        total_seen = len(matches)
        truncated = total_seen > _MAX_RESULTS
        hit_scan_cap = total_seen >= scan_cap
        matches = matches[:_MAX_RESULTS]
        paths = [m[0] for m in matches]

        output = "\n".join(paths)
        if truncated:
            if hit_scan_cap:
                # We stopped scanning at scan_cap — real match count may be
                # higher than we can report. Tell the model so it narrows
                # the pattern instead of treating scan_cap as the truth.
                output += (
                    f"\n\n(Showing first {_MAX_RESULTS} of {scan_cap}+ matches. "
                    f"Narrow the pattern — the real count may be higher.)"
                )
            else:
                output += (
                    f"\n\n(Showing first {_MAX_RESULTS} of {total_seen} matches.)"
                )

        return ToolResult(data=output)
