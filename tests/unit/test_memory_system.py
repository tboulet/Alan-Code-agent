"""Tests for the memory system: directory structure, prompt building, settings migration, scratchpad."""

import json
import os
import tempfile
import time

import pytest

from alancode.memory.memdir import (
    MEMORY_SUBDIRS,
    cleanup_old_scratchpads,
    ensure_memory_structure,
    get_memory_dir,
    get_scratchpad_dir,
    load_memory_index,
)
from alancode.memory.prompt import (
    build_memory_section,
    get_memory_instructions_intensive,
    get_memory_instructions_off,
    get_memory_instructions_on,
    get_save_command_prompt,
)
from alancode.settings import SETTINGS_DEFAULTS, load_settings, save_settings


# ---------------------------------------------------------------------------
# Directory structure
# ---------------------------------------------------------------------------


class TestMemoryDirectoryStructure:

    def test_get_memory_dir_returns_alan_memory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = get_memory_dir(tmpdir)
            assert result.parts[-2:] == (".alan", "memory")
            assert str(result).startswith(tmpdir)

    def test_ensure_memory_structure_creates_all_subdirs(self):
        from alancode.memory.memdir import PROJECT_MEMORY_SUBDIRS, GLOBAL_MEMORY_SUBDIRS, get_global_memory_dir
        with tempfile.TemporaryDirectory() as tmpdir:
            mem_dir = ensure_memory_structure(tmpdir)
            assert mem_dir.is_dir()
            # Project memory subdirs
            for subdir in PROJECT_MEMORY_SUBDIRS:
                assert (mem_dir / subdir).is_dir(), f"Missing project subdir: {subdir}"
            # Global memory subdirs
            global_dir = get_global_memory_dir()
            for subdir in GLOBAL_MEMORY_SUBDIRS:
                assert (global_dir / subdir).is_dir(), f"Missing global subdir: {subdir}"

    def test_ensure_memory_structure_idempotent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ensure_memory_structure(tmpdir)
            ensure_memory_structure(tmpdir)
            mem_dir = get_memory_dir(tmpdir)
            assert mem_dir.is_dir()

    def test_memory_subdirs_include_workflow(self):
        assert "workflow" in MEMORY_SUBDIRS


# ---------------------------------------------------------------------------
# Scratchpad
# ---------------------------------------------------------------------------


class TestScratchpad:

    def test_get_scratchpad_dir_path(self):
        result = get_scratchpad_dir("/tmp/proj", "abc123")
        assert str(result).endswith(os.path.join(".alan", "sessions", "abc123", "scratchpad"))

    def test_cleanup_old_scratchpads_removes_oldest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sessions_root = os.path.join(tmpdir, ".alan", "sessions")
            os.makedirs(sessions_root)

            # Create 7 session dirs with scratchpad subdirs
            for i in range(7):
                scratch = os.path.join(sessions_root, f"session_{i}", "scratchpad")
                os.makedirs(scratch)
                os.utime(scratch, (1000 + i, 1000 + i))

            cleanup_old_scratchpads(tmpdir, max_sessions=3)

            remaining = [
                d for d in os.listdir(sessions_root)
                if os.path.isdir(os.path.join(sessions_root, d, "scratchpad"))
            ]
            assert len(remaining) == 3
            # The 3 newest should remain (session_4, session_5, session_6)
            for name in remaining:
                assert name in ("session_4", "session_5", "session_6")

    def test_cleanup_noop_when_under_limit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sessions_root = os.path.join(tmpdir, ".alan", "sessions")
            os.makedirs(sessions_root)
            os.makedirs(os.path.join(sessions_root, "s1", "scratchpad"))
            os.makedirs(os.path.join(sessions_root, "s2", "scratchpad"))

            cleanup_old_scratchpads(tmpdir, max_sessions=5)
            remaining = [
                d for d in os.listdir(sessions_root)
                if os.path.isdir(os.path.join(sessions_root, d, "scratchpad"))
            ]
            assert len(remaining) == 2

    def test_cleanup_noop_when_no_scratchpad_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Should not raise
            cleanup_old_scratchpads(tmpdir, max_sessions=5)


# ---------------------------------------------------------------------------
# Memory index loading
# ---------------------------------------------------------------------------


class TestLoadMemoryIndex:

    def test_returns_none_when_no_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = load_memory_index(cwd=tmpdir)
            assert result is None

    def test_returns_none_for_empty_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mem_dir = ensure_memory_structure(tmpdir)
            (mem_dir / "MEMORY.md").write_text("", encoding="utf-8")
            result = load_memory_index(cwd=tmpdir)
            assert result is None

    def test_returns_formatted_content(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mem_dir = ensure_memory_structure(tmpdir)
            (mem_dir / "MEMORY.md").write_text(
                "- [User profile](user/profile.md) -- ML practitioner\n",
                encoding="utf-8",
            )
            result = load_memory_index(cwd=tmpdir)
            assert result is not None
            assert "Your project memory index" in result
            assert "ML practitioner" in result

    def test_explicit_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "custom_memory.md")
            with open(path, "w") as f:
                f.write("# Custom memory\nSome content\n")
            result = load_memory_index(memory_path=path)
            assert result is not None
            assert "Custom memory" in result


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------


class TestPromptBuilding:

    def test_off_mode(self):
        result = get_memory_instructions_off()
        assert "disabled" in result.lower()
        assert "/memory" in result

    def test_on_mode_contains_key_sections(self):
        result = get_memory_instructions_on("/tmp/.alan/memory")
        assert "Types of memory" in result
        assert "What NOT to save" in result
        assert "When to save" in result
        assert "How to save" in result
        assert "When to access" in result
        assert "Before recommending from memory" in result
        assert "/tmp/.alan/memory" in result

    def test_on_mode_save_only_on_request(self):
        result = get_memory_instructions_on("/tmp/.alan/memory")
        assert "ONLY when the user explicitly asks" in result

    def test_intensive_mode_proactive(self):
        result = get_memory_instructions_intensive("/tmp/.alan/memory")
        assert "proactively" in result.lower()

    def test_intensive_mode_contains_all_sections(self):
        result = get_memory_instructions_intensive("/tmp/.alan/memory")
        assert "Types of memory" in result
        assert "workflow" in result

    def test_build_memory_section_off(self):
        result = build_memory_section("off", "/tmp", None)
        assert "disabled" in result.lower()

    def test_build_memory_section_on_without_index(self):
        result = build_memory_section("on", "/tmp/.alan/memory", None)
        assert "Memory" in result
        assert "Your memory index" not in result

    def test_build_memory_section_on_with_index(self):
        index = "## Your memory index (MEMORY.md)\n\n- [Profile](user/p.md)"
        result = build_memory_section("on", "/tmp/.alan/memory", index)
        assert "Your memory index" in result
        assert "Profile" in result

    def test_build_memory_section_intensive(self):
        result = build_memory_section("intensive", "/tmp/.alan/memory", None)
        assert "proactively" in result.lower()

    def test_save_command_prompt(self):
        result = get_save_command_prompt()
        assert "memory" in result.lower()
        assert "Write" in result

    def test_five_memory_types_present(self):
        result = get_memory_instructions_on("/tmp/.alan/memory")
        for t in ("user", "feedback", "project", "reference", "workflow"):
            assert f"<name>{t}</name>" in result


# ---------------------------------------------------------------------------
# Settings migration
# ---------------------------------------------------------------------------


class TestSettingsDefaults:

    def test_new_defaults_have_memory(self):
        assert "memory" in SETTINGS_DEFAULTS
        assert "memory_enabled" not in SETTINGS_DEFAULTS
        assert "memory_mode" not in SETTINGS_DEFAULTS

    def test_has_memory_fields(self):
        assert "memory_reminder_threshold" in SETTINGS_DEFAULTS
        assert SETTINGS_DEFAULTS["memory_reminder_threshold"] == 10
        assert "max_scratchpad_sessions" in SETTINGS_DEFAULTS
        assert SETTINGS_DEFAULTS["max_scratchpad_sessions"] == 5

    def test_has_compaction_toggles(self):
        assert SETTINGS_DEFAULTS["compaction_truncate_enabled"] is True
        assert SETTINGS_DEFAULTS["compaction_clear_enabled"] is True
        assert SETTINGS_DEFAULTS["compaction_auto_enabled"] is True
