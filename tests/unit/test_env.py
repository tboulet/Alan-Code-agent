"""Test environment detection utilities (alancode/utils/env.py)."""

import os
import subprocess
import tempfile

import pytest

from alancode.utils.env import (
    get_git_status,
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


# ---------------------------------------------------------------------------
# get_git_status
# ---------------------------------------------------------------------------


class TestGetGitStatus:

    def test_in_git_repo_returns_string(self):
        """In the project repo, get_git_status should return a non-None string."""
        project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        status = get_git_status(project_root)
        assert status is not None
        assert isinstance(status, str)

    def test_contains_expected_sections(self):
        """The status string should contain branch info and status sections."""
        project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        status = get_git_status(project_root)
        if status is None:
            pytest.skip("Not running inside a git repo")
        assert "Current branch:" in status
        assert "Main branch" in status
        assert "Status:" in status
        assert "Recent commits:" in status

    def test_not_in_git_repo_returns_none(self):
        """In a non-git directory, get_git_status should return None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            status = get_git_status(tmpdir)
            assert status is None

    def test_fresh_git_repo(self):
        """A freshly initialized git repo with no commits."""
        with tempfile.TemporaryDirectory() as tmpdir:
            subprocess.run(
                ["git", "init"], cwd=tmpdir, capture_output=True, check=True
            )
            subprocess.run(
                ["git", "config", "user.email", "test@test.com"],
                cwd=tmpdir,
                capture_output=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "Test"],
                cwd=tmpdir,
                capture_output=True,
            )
            assert is_git_repo(tmpdir) is True
            status = get_git_status(tmpdir)
            # May be None or a string depending on whether git rev-parse HEAD works;
            # the function should not crash either way
            if status is not None:
                assert isinstance(status, str)
