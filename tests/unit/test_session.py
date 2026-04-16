"""Test session utilities (alancode/session/) and disk-attached SessionState."""

import json
import os
import tempfile
import time

import pytest

from alancode.session.session import (
    find_session_by_prefix,
    generate_session_id,
    get_last_session_id,
    get_session_dir,
    get_sessions_dir,
    load_session_settings,
    save_session_settings,
)
from alancode.session.state import SessionState


# ---------------------------------------------------------------------------
# generate_session_id
# ---------------------------------------------------------------------------


class TestGenerateSessionId:

    def test_returns_string(self):
        sid = generate_session_id()
        assert isinstance(sid, str)

    def test_length_is_32(self):
        """uuid4().hex is always 32 hex chars."""
        sid = generate_session_id()
        assert len(sid) == 32

    def test_uniqueness(self):
        """Two generated IDs should not collide."""
        ids = {generate_session_id() for _ in range(100)}
        assert len(ids) == 100


# ---------------------------------------------------------------------------
# find_session_by_prefix
# ---------------------------------------------------------------------------


class TestFindSessionByPrefix:

    def _create_session(self, cwd: str, session_id: str) -> None:
        """Helper: create a session directory."""
        session_dir = get_session_dir(cwd, session_id)
        session_dir.mkdir(parents=True, exist_ok=True)

    def test_exact_match(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sid = "abcdef1234567890abcdef1234567890"
            self._create_session(tmpdir, sid)
            result = find_session_by_prefix(tmpdir, sid)
            assert result == sid

    def test_prefix_match(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sid = "abcdef1234567890abcdef1234567890"
            self._create_session(tmpdir, sid)
            result = find_session_by_prefix(tmpdir, "abcdef")
            assert result == sid

    def test_ambiguous_returns_none(self):
        """When two sessions share a prefix, return None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sid1 = "abcdef1234567890abcdef1234567890"
            sid2 = "abcdef9876543210abcdef9876543210"
            self._create_session(tmpdir, sid1)
            self._create_session(tmpdir, sid2)
            result = find_session_by_prefix(tmpdir, "abcdef")
            assert result is None

    def test_not_found_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._create_session(tmpdir, "abcdef1234567890abcdef1234567890")
            result = find_session_by_prefix(tmpdir, "zzz999")
            assert result is None

    def test_minimum_length_enforced(self):
        """Prefix shorter than 3 characters returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sid = "abcdef1234567890abcdef1234567890"
            self._create_session(tmpdir, sid)
            assert find_session_by_prefix(tmpdir, "ab") is None
            assert find_session_by_prefix(tmpdir, "a") is None
            assert find_session_by_prefix(tmpdir, "") is None

    def test_no_sessions_dir(self):
        """If the sessions directory doesn't exist, return None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = find_session_by_prefix(tmpdir, "abc")
            assert result is None


# ---------------------------------------------------------------------------
# get_last_session_id
# ---------------------------------------------------------------------------


class TestGetLastSessionId:

    def _create_session_with_transcript(
        self, cwd: str, session_id: str, metadata_cwd: str | None = None
    ) -> None:
        """Helper: create a session dir with a minimal transcript.jsonl."""
        session_dir = get_session_dir(cwd, session_id)
        session_dir.mkdir(parents=True, exist_ok=True)
        transcript_path = session_dir / "transcript.jsonl"
        meta = {"_metadata": {"cwd": metadata_cwd or cwd}}
        with open(transcript_path, "w") as f:
            f.write(json.dumps(meta) + "\n")

    def test_no_sessions_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = get_last_session_id(tmpdir)
            assert result is None

    def test_no_sessions_dir_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Don't create any .alan directory
            result = get_last_session_id(tmpdir)
            assert result is None

    def test_single_session_returned(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sid = generate_session_id()
            self._create_session_with_transcript(tmpdir, sid)
            result = get_last_session_id(tmpdir)
            assert result == sid

    def test_most_recent_session_returned(self):
        """The session with the most recently modified transcript wins."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sid_old = generate_session_id()
            self._create_session_with_transcript(tmpdir, sid_old)

            # Ensure the second session has a later mtime
            time.sleep(0.05)

            sid_new = generate_session_id()
            self._create_session_with_transcript(tmpdir, sid_new)

            result = get_last_session_id(tmpdir)
            assert result == sid_new

    def test_different_cwd_not_matched(self):
        """Session from a different cwd should not be returned."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sid = generate_session_id()
            # Transcript metadata says cwd is "/some/other/path"
            self._create_session_with_transcript(tmpdir, sid, metadata_cwd="/some/other/path")
            result = get_last_session_id(tmpdir)
            assert result is None

    def test_empty_transcript_skipped(self):
        """Sessions with empty transcripts are skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sid = generate_session_id()
            session_dir = get_session_dir(tmpdir, sid)
            session_dir.mkdir(parents=True, exist_ok=True)
            (session_dir / "transcript.jsonl").write_text("")

            result = get_last_session_id(tmpdir)
            assert result is None


# ---------------------------------------------------------------------------
# save/load session settings roundtrip
# ---------------------------------------------------------------------------


class TestSessionSettings:

    def test_save_and_load_roundtrip(self):
        """save_session_settings -> load_session_settings preserves values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sid = generate_session_id()
            original = {"model": "test-model", "verbose": True, "max_iterations_per_turn": 10}
            save_session_settings(tmpdir, sid, original)

            loaded = load_session_settings(tmpdir, sid)
            assert loaded["model"] == "test-model"
            assert loaded["verbose"] is True
            assert loaded["max_iterations_per_turn"] == 10

    def test_ephemeral_fields_excluded(self):
        """api_key should not appear in saved session settings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sid = generate_session_id()
            settings = {"model": "test-model", "api_key": "sk-secret-12345"}
            save_session_settings(tmpdir, sid, settings)

            loaded = load_session_settings(tmpdir, sid)
            assert "api_key" not in loaded

    def test_load_missing_returns_empty_dict(self):
        """load_session_settings for a non-existent session returns {}."""
        with tempfile.TemporaryDirectory() as tmpdir:
            loaded = load_session_settings(tmpdir, "nonexistent-session-id")
            assert loaded == {}

    def test_save_creates_directory(self):
        """save_session_settings creates the session directory if needed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sid = generate_session_id()
            save_session_settings(tmpdir, sid, {"model": "test"})
            assert get_session_dir(tmpdir, sid).exists()

    def test_overwrite_existing(self):
        """A second save overwrites the first."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sid = generate_session_id()
            save_session_settings(tmpdir, sid, {"model": "first"})
            save_session_settings(tmpdir, sid, {"model": "second"})

            loaded = load_session_settings(tmpdir, sid)
            assert loaded["model"] == "second"


# ---------------------------------------------------------------------------
# SessionState (disk-attached)
# ---------------------------------------------------------------------------


class TestSessionState:

    def test_initial_defaults(self, tmp_path):
        """New SessionState starts with zero/empty defaults."""
        s = SessionState(session_id="test-1", cwd=str(tmp_path))
        assert s.turn_count == 0
        assert s.total_cost_usd == 0.0
        assert s.total_input_tokens == 0
        assert s.total_output_tokens == 0
        assert s.allow_rules == []
        assert s.cost_unknown is False

    def test_write_through_persistence(self, tmp_path):
        """Setting a property immediately persists to disk."""
        s = SessionState(session_id="test-2", cwd=str(tmp_path))
        s.turn_count = 5
        s.total_cost_usd = 1.23

        # Load fresh from same path — should see the data
        s2 = SessionState(session_id="test-2", cwd=str(tmp_path))
        assert s2.turn_count == 5
        assert s2.total_cost_usd == pytest.approx(1.23)

    def test_increment_pattern(self, tmp_path):
        """x += delta reads, adds, writes correctly."""
        s = SessionState(session_id="test-3", cwd=str(tmp_path))
        s.total_input_tokens = 100
        s.total_input_tokens += 50
        assert s.total_input_tokens == 150

        # Verify on disk
        s2 = SessionState(session_id="test-3", cwd=str(tmp_path))
        assert s2.total_input_tokens == 150

    def test_batch_single_write(self, tmp_path):
        """batch() groups multiple updates into a single disk write."""
        s = SessionState(session_id="test-4", cwd=str(tmp_path))
        with s.batch():
            s.total_input_tokens = 500
            s.total_output_tokens = 200
            s.total_cost_usd = 0.05

        s2 = SessionState(session_id="test-4", cwd=str(tmp_path))
        assert s2.total_input_tokens == 500
        assert s2.total_output_tokens == 200
        assert s2.total_cost_usd == pytest.approx(0.05)

    def test_nested_batch(self, tmp_path):
        """Nested batch() only flushes on the outermost exit."""
        s = SessionState(session_id="test-5", cwd=str(tmp_path))
        with s.batch():
            s.turn_count = 1
            with s.batch():
                s.turn_count = 2
            # Inner batch exits but outer still active — no flush yet
            # (we can't directly test "no flush" without mocking, but
            # the final value should be correct)
            s.turn_count = 3

        assert s.turn_count == 3
        s2 = SessionState(session_id="test-5", cwd=str(tmp_path))
        assert s2.turn_count == 3

    def test_allow_rules_roundtrip(self, tmp_path):
        """Allow rules are persisted and restored."""
        s = SessionState(session_id="test-6", cwd=str(tmp_path))
        s.add_allow_rule({"tool_name": "Bash", "rule_content": "ls *", "source": "session"})
        s.add_allow_rule({"tool_name": "Bash", "rule_content": "cat *", "source": "session"})

        s2 = SessionState(session_id="test-6", cwd=str(tmp_path))
        assert len(s2.allow_rules) == 2
        assert s2.allow_rules[0]["rule_content"] == "ls *"
        assert s2.allow_rules[1]["rule_content"] == "cat *"

    def test_allow_rules_getter_returns_copy(self, tmp_path):
        """allow_rules getter returns a copy, not a reference to the cache."""
        s = SessionState(session_id="test-7", cwd=str(tmp_path))
        s.add_allow_rule({"tool_name": "Bash", "rule_content": "ls *", "source": "session"})
        rules = s.allow_rules
        rules.append({"tool_name": "Bash", "rule_content": "EXTRA", "source": "test"})
        # Cache should not be modified by mutating the returned list
        assert len(s.allow_rules) == 1

    def test_corrupt_state_file_recovers(self, tmp_path):
        """If state.json is corrupt, SessionState starts with defaults."""
        s = SessionState(session_id="test-9", cwd=str(tmp_path))
        # Write garbage to state file
        s._state_path.write_text("not valid json{{{")

        s2 = SessionState(session_id="test-9", cwd=str(tmp_path))
        assert s2.turn_count == 0
        assert s2.total_cost_usd == 0.0
