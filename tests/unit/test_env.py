"""Test environment detection utilities (alancode/utils/env.py)."""

import os
import tempfile

from alancode.utils.env import (
    get_os_version,
    get_platform,
    get_shell,
    is_git_repo,
)


# ---------------------------------------------------------------------------
# get_platform
# ---------------------------------------------------------------------------


class TestGetPlatform:

    def test_returns_string(self):
        result = get_platform()
        assert isinstance(result, str)

    def test_returns_known_platform(self):
        """On any CI or dev machine, should return one of the standard identifiers."""
        result = get_platform()
        assert result in ("linux", "darwin", "win32", "freebsd", "openbsd", "sunos"), (
            f"Unexpected platform: {result}"
        )


# ---------------------------------------------------------------------------
# get_shell
# ---------------------------------------------------------------------------


class TestGetShell:

    def test_returns_string(self):
        result = get_shell()
        assert isinstance(result, str)

    def test_returns_known_or_unknown(self):
        """Should return a recognized shell name or 'unknown'."""
        known = {"bash", "zsh", "fish", "sh", "dash", "ksh", "tcsh", "csh", "unknown"}
        result = get_shell()
        assert result in known, f"Unexpected shell: {result}"


# ---------------------------------------------------------------------------
# get_os_version
# ---------------------------------------------------------------------------


class TestGetOsVersion:

    def test_returns_string(self):
        result = get_os_version()
        assert isinstance(result, str)

    def test_contains_system_name(self):
        """Should start with the system name (Linux, Darwin, Windows)."""
        result = get_os_version()
        assert any(
            result.startswith(s) for s in ("Linux", "Darwin", "Windows")
        ), f"Unexpected OS version: {result}"


# ---------------------------------------------------------------------------
# is_git_repo
# ---------------------------------------------------------------------------


class TestIsGitRepo:

    def test_in_git_directory(self):
        """The project root (this repo) is a git repo."""
        project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        assert is_git_repo(project_root) is True

    def test_not_in_git_directory(self):
        """A fresh temp directory is not a git repo."""
        with tempfile.TemporaryDirectory() as tmpdir:
            assert is_git_repo(tmpdir) is False

    def test_default_cwd(self):
        """Calling with None uses the current working directory; should not crash."""
        # Just verify it returns a bool without error
        result = is_git_repo(None)
        assert isinstance(result, bool)


