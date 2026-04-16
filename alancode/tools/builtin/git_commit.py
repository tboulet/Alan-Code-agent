"""GitCommit tool — stage and commit files to git.

Creates a commit tracked as an "Alan commit" in the session state,
enabling AGT (Agentic Git Tree) visualization and navigation.
"""

from __future__ import annotations

import os
import subprocess
from typing import Any

from alancode.tools.base import Tool, ToolResult, ToolUseContext


class GitCommitTool(Tool):
    """Stage files and create a git commit."""

    @property
    def name(self) -> str:
        return "GitCommit"

    @property
    def description(self) -> str:
        return (
            "Stage and commit files to git with a commit message. "
            "If no files are specified, all changes are staged (git add -A). "
            "The commit is tracked in the session for history visualization."
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "The commit message.",
                },
                "files": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Files to stage before committing. "
                        "If omitted, all changes are staged (git add -A)."
                    ),
                },
                "allow_empty": {
                    "type": "boolean",
                    "description": (
                        "Allow creating a commit with no file changes "
                        "(e.g., when only memory was updated). Default false."
                    ),
                },
            },
            "required": ["message"],
        }

    def permission_level(self, args: dict[str, Any]) -> str:
        return "write"

    async def call(self, args: dict[str, Any], context: ToolUseContext) -> ToolResult:
        cwd = context.cwd
        message = args.get("message", "")
        files = args.get("files", [])
        allow_empty = args.get("allow_empty", False)

        if not message:
            return ToolResult(data="Error: commit message is required.", is_error=True)

        # Check git repo
        from alancode.utils.env import is_git_repo
        if not is_git_repo(cwd):
            return ToolResult(data="Error: not a git repository.", is_error=True)

        # Check for merge conflicts
        merge_head = os.path.join(cwd, ".git", "MERGE_HEAD")
        if os.path.exists(merge_head):
            return ToolResult(
                data="Error: merge in progress. Resolve conflicts first.",
                is_error=True,
            )

        # Stage files
        if files:
            for f in files:
                result = self._run_git(cwd, "add", f)
                if result.returncode != 0:
                    return ToolResult(
                        data=f"Error staging {f}: {result.stderr.strip()}",
                        is_error=True,
                    )
        else:
            result = self._run_git(cwd, "add", "-A")
            if result.returncode != 0:
                return ToolResult(
                    data=f"Error staging files: {result.stderr.strip()}",
                    is_error=True,
                )

        # Check if there's anything to commit
        status = self._run_git(cwd, "status", "--porcelain")
        if not allow_empty and not status.stdout.strip():
            return ToolResult(
                data="Nothing to commit — working tree clean.",
                is_error=True,
            )

        # Commit
        commit_cmd = ["commit", "-m", message]
        if allow_empty:
            commit_cmd.append("--allow-empty")
        result = self._run_git(cwd, *commit_cmd)
        if result.returncode != 0:
            return ToolResult(
                data=f"Error committing: {result.stderr.strip()}",
                is_error=True,
            )

        # Get the new commit SHA
        sha_result = self._run_git(cwd, "rev-parse", "HEAD")
        if sha_result.returncode != 0:
            return ToolResult(
                data="Commit succeeded but failed to get SHA.",
                is_error=True,
            )
        new_sha = sha_result.stdout.strip()
        short_sha = new_sha[:7]

        # Get branch name
        branch_result = self._run_git(cwd, "symbolic-ref", "--short", "HEAD")
        branch = branch_result.stdout.strip() if branch_result.returncode == 0 else "detached"

        # Update session state (AGT tracking)
        state = context.session_state
        if state is not None:
            try:
                with state.batch():
                    state.add_alan_commit(new_sha)
                    state.add_to_conv_path(new_sha)
                    state.agent_position_sha = new_sha
                    # Record message count so /convrevert can truncate precisely
                    state.record_commit_message_index(
                        new_sha, len(context.messages),
                    )
            except Exception:
                pass  # AGT tracking is non-critical

        return ToolResult(
            data=f"Committed {short_sha} on {branch}: {message}",
        )

    @staticmethod
    def _run_git(cwd: str, *args: str) -> subprocess.CompletedProcess:
        """Run a git command in the given directory."""
        return subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=30,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        )
