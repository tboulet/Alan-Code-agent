"""Git test helpers — temporary repos for AGT integration tests.

Provides fixtures and helper functions for creating temporary git
repositories with controlled commit histories.  Repos are created in
/tmp/ and cleaned up after each test.

Usage in tests::

    from tests.integration.git_helpers import GitTestRepo

    def test_something(tmp_path):
        repo = GitTestRepo(tmp_path / "myrepo")
        repo.init()
        repo.write_file("main.py", "print('hello')")
        sha1 = repo.commit("Initial commit")

        repo.write_file("main.py", "print('updated')")
        sha2 = repo.commit("Second commit")

        assert repo.log_hashes() == [sha2, sha1]
        assert repo.current_branch() == "main"
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any


class GitTestRepo:
    """A temporary git repository for testing.

    All git operations use explicit ``--git-dir`` and ``--work-tree``
    to avoid interfering with the parent repo.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.git_dir = self.path / ".git"

    # ── Setup ────────────────────────────────────────────────────────

    def init(self, initial_branch: str = "main") -> None:
        """Initialize a new git repo with an initial commit."""
        self.path.mkdir(parents=True, exist_ok=True)
        self._run("git", "init", "-b", initial_branch)
        self._run("git", "config", "user.email", "test@alancode.dev")
        self._run("git", "config", "user.name", "Test User")
        # Create .alan/ directory (gitignored)
        alan_dir = self.path / ".alan"
        alan_dir.mkdir(exist_ok=True)
        (self.path / ".gitignore").write_text(".alan/\n")
        self._run("git", "add", ".gitignore")
        self._run("git", "commit", "-m", "Initial: add .gitignore")

    # ── File operations ──────────────────────────────────────────────

    def write_file(self, rel_path: str, content: str) -> Path:
        """Write a file (creating parent dirs as needed)."""
        full = self.path / rel_path
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(content)
        return full

    def read_file(self, rel_path: str) -> str:
        """Read a file's contents."""
        return (self.path / rel_path).read_text()

    def delete_file(self, rel_path: str) -> None:
        """Delete a file."""
        (self.path / rel_path).unlink()

    def file_exists(self, rel_path: str) -> bool:
        """Check if a file exists."""
        return (self.path / rel_path).exists()

    # ── Git operations ───────────────────────────────────────────────

    def commit(
        self,
        message: str,
        files: list[str] | None = None,
        add_all: bool = True,
    ) -> str:
        """Stage files and commit. Returns the commit SHA."""
        if files:
            for f in files:
                self._run("git", "add", f)
        elif add_all:
            self._run("git", "add", "-A")
        self._run("git", "commit", "-m", message, "--allow-empty")
        return self.head_sha()

    def head_sha(self) -> str:
        """Return the full SHA of HEAD."""
        return self._run("git", "rev-parse", "HEAD").strip()

    def short_sha(self, sha: str | None = None) -> str:
        """Return the short SHA (7 chars)."""
        target = sha or "HEAD"
        return self._run("git", "rev-parse", "--short", target).strip()

    def log_hashes(self, max_count: int = 50, all_branches: bool = False) -> list[str]:
        """Return list of commit SHAs (newest first)."""
        cmd = ["git", "log", f"--max-count={max_count}", "--format=%H"]
        if all_branches:
            cmd.append("--all")
        output = self._run(*cmd)
        return [line.strip() for line in output.strip().split("\n") if line.strip()]

    def log_oneline(self, max_count: int = 20, all_branches: bool = False) -> list[str]:
        """Return list of 'sha message' strings (newest first)."""
        cmd = ["git", "log", f"--max-count={max_count}", "--oneline"]
        if all_branches:
            cmd.append("--all")
        return [line.strip() for line in self._run(*cmd).strip().split("\n") if line.strip()]

    def log_graph(self, max_count: int = 30) -> str:
        """Return git log --graph output (for debugging)."""
        return self._run(
            "git", "log", "--all", "--oneline", "--graph",
            f"--max-count={max_count}",
        )

    def parent_sha(self, sha: str | None = None, nth: int = 1) -> str | None:
        """Return the nth parent of a commit (None if no parent)."""
        target = sha or "HEAD"
        try:
            return self._run("git", "rev-parse", f"{target}~{nth}").strip()
        except subprocess.CalledProcessError:
            return None

    def commit_message(self, sha: str | None = None) -> str:
        """Return the commit message for a SHA."""
        target = sha or "HEAD"
        return self._run("git", "log", "-1", "--format=%s", target).strip()

    def commit_timestamp(self, sha: str | None = None) -> str:
        """Return the ISO timestamp of a commit."""
        target = sha or "HEAD"
        return self._run("git", "log", "-1", "--format=%aI", target).strip()

    def commit_author(self, sha: str | None = None) -> str:
        """Return the author name of a commit."""
        target = sha or "HEAD"
        return self._run("git", "log", "-1", "--format=%an", target).strip()

    # ── Branch operations ────────────────────────────────────────────

    def current_branch(self) -> str | None:
        """Return current branch name, or None if detached HEAD."""
        try:
            branch = self._run("git", "symbolic-ref", "--short", "HEAD").strip()
            return branch
        except subprocess.CalledProcessError:
            return None  # Detached HEAD

    def create_branch(self, name: str, start: str | None = None) -> None:
        """Create a new branch at the given start point (or HEAD)."""
        cmd = ["git", "branch", name]
        if start:
            cmd.append(start)
        self._run(*cmd)

    def checkout(self, target: str) -> None:
        """Checkout a branch or commit."""
        self._run("git", "checkout", target)

    def checkout_new_branch(self, name: str, start: str | None = None) -> None:
        """Create and checkout a new branch."""
        cmd = ["git", "checkout", "-b", name]
        if start:
            cmd.append(start)
        self._run(*cmd)

    def branches(self) -> list[str]:
        """Return list of all branch names."""
        output = self._run("git", "branch", "--format=%(refname:short)")
        return [b.strip() for b in output.strip().split("\n") if b.strip()]

    def merge(self, branch: str, message: str | None = None) -> str:
        """Merge a branch into the current branch. Returns merge commit SHA."""
        cmd = ["git", "merge", branch, "--no-ff"]
        if message:
            cmd.extend(["-m", message])
        self._run(*cmd)
        return self.head_sha()

    # ── Status ───────────────────────────────────────────────────────

    def status_porcelain(self) -> str:
        """Return git status --porcelain output."""
        return self._run("git", "status", "--porcelain")

    def is_dirty(self) -> bool:
        """Check if working tree has uncommitted changes."""
        return bool(self.status_porcelain().strip())

    def diff(self, ref1: str = "HEAD", ref2: str | None = None) -> str:
        """Return diff between two refs (or HEAD vs working tree)."""
        cmd = ["git", "diff", ref1]
        if ref2:
            cmd.append(ref2)
        return self._run(*cmd)

    def diff_stat(self, ref1: str = "HEAD", ref2: str | None = None) -> str:
        """Return diff --stat between two refs."""
        cmd = ["git", "diff", "--stat", ref1]
        if ref2:
            cmd.append(ref2)
        return self._run(*cmd)

    # ── Destructive operations (used carefully in tests) ─────────────

    def reset_hard(self, target: str) -> None:
        """Reset current branch to target (destructive)."""
        self._run("git", "reset", "--hard", target)

    def clean_untracked(self) -> None:
        """Remove untracked files."""
        self._run("git", "clean", "-fd")

    # ── Alan-specific helpers ────────────────────────────────────────

    def alan_dir(self) -> Path:
        """Return the .alan/ directory path."""
        d = self.path / ".alan"
        d.mkdir(exist_ok=True)
        return d

    def memory_dir(self) -> Path:
        """Return the .alan/memory/ directory path."""
        d = self.alan_dir() / "memory"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def write_memory(self, filename: str, content: str) -> Path:
        """Write a memory file."""
        f = self.memory_dir() / filename
        f.write_text(content)
        return f

    def read_memory(self, filename: str) -> str:
        """Read a memory file."""
        return (self.memory_dir() / filename).read_text()

    def memory_snapshot_dir(self, commit_sha: str) -> Path:
        """Return memory snapshot path for a commit."""
        d = self.alan_dir() / "memory_snapshots" / commit_sha
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ── Scenario builders ────────────────────────────────────────────

    def build_linear_history(self, n: int, prefix: str = "file") -> list[str]:
        """Create n commits with one file each. Returns list of SHAs (oldest first)."""
        shas = []
        for i in range(n):
            self.write_file(f"{prefix}_{i}.py", f"# File {i}\nprint({i})\n")
            sha = self.commit(f"Add {prefix}_{i}.py")
            shas.append(sha)
        return shas

    def build_branching_history(self) -> dict[str, Any]:
        """Create a repo with branches for testing tree layout.

        Creates::

            main:    c1 -- c2 -- c3 -- c5 (merge)
                               \\       /
            feature:            c4 ---

        Returns dict with all SHAs and branch names.
        """
        self.write_file("base.py", "# base\n")
        c1 = self.commit("c1: base")

        self.write_file("main.py", "# main\n")
        c2 = self.commit("c2: main file")

        # Create feature branch
        self.checkout_new_branch("feature")
        self.write_file("feature.py", "# feature\n")
        c4 = self.commit("c4: feature file")

        # Back to main, add another commit
        self.checkout("main")
        self.write_file("utils.py", "# utils\n")
        c3 = self.commit("c3: utils file")

        # Merge feature into main
        c5 = self.merge("feature", "c5: merge feature")

        return {
            "c1": c1, "c2": c2, "c3": c3, "c4": c4, "c5": c5,
            "branches": ["main", "feature"],
        }

    # ── Internal ─────────────────────────────────────────────────────

    def _run(self, *cmd: str, check: bool = True) -> str:
        """Run a command in the repo directory."""
        result = subprocess.run(
            cmd,
            cwd=str(self.path),
            capture_output=True,
            text=True,
            check=check,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        )
        return result.stdout
