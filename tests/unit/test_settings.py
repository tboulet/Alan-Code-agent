"""Test settings system."""

import json
import os
import tempfile

import pytest

from alancode.settings import (
    SETTINGS_DEFAULTS,
    coerce_value,
    load_projects_settings_and_maybe_init,
    load_settings,
    save_settings,
    validate_setting,
)


# ---------------------------------------------------------------------------
# TestCoercion -- coerce_value
# ---------------------------------------------------------------------------


class TestCoercion:

    def test_true(self):
        assert coerce_value("true") is True

    def test_true_yes(self):
        assert coerce_value("yes") is True

    def test_true_case_insensitive(self):
        assert coerce_value("True") is True
        assert coerce_value("TRUE") is True

    def test_false(self):
        assert coerce_value("false") is False

    def test_false_no(self):
        assert coerce_value("no") is False

    def test_null(self):
        assert coerce_value("null") is None

    def test_none(self):
        assert coerce_value("none") is None

    def test_empty_string(self):
        assert coerce_value("") is None

    def test_int(self):
        assert coerce_value("42") == 42
        assert isinstance(coerce_value("42"), int)

    def test_negative_int(self):
        assert coerce_value("-7") == -7

    def test_float(self):
        assert coerce_value("3.14") == 3.14
        assert isinstance(coerce_value("3.14"), float)

    def test_string(self):
        assert coerce_value("hello") == "hello"

    def test_string_with_spaces(self):
        assert coerce_value("hello world") == "hello world"

    def test_model_name(self):
        """Model names like 'gpt-4o' should stay as strings."""
        assert coerce_value("gpt-4o") == "gpt-4o"


# ---------------------------------------------------------------------------
# TestSettings -- file creation and migration
# ---------------------------------------------------------------------------


class TestSettings:

    def test_ensure_creates_file(self):
        """ensure_settings creates .alan/settings.json if missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = load_projects_settings_and_maybe_init(tmpdir)
            path = os.path.join(tmpdir, ".alan", "settings.json")
            assert os.path.exists(path)
            # Should have the default keys
            assert "model" in settings
            assert "permission_mode" in settings

    def test_ensure_idempotent(self):
        """Calling ensure_settings twice doesn't corrupt the file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            s1 = load_projects_settings_and_maybe_init(tmpdir)
            s2 = load_projects_settings_and_maybe_init(tmpdir)
            # First call returns BUILTIN_DEFAULTS (includes ephemeral api_key),
            # second loads from disk (ephemeral fields excluded). Compare
            # only the on-disk keys.
            non_ephemeral_1 = {k for k in s1 if k != "api_key"}
            non_ephemeral_2 = {k for k in s2 if k != "api_key"}
            assert non_ephemeral_1 == non_ephemeral_2

    def test_load_nonexistent_returns_empty(self):
        """load_settings on a dir without .alan/ returns empty dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = load_settings(tmpdir)
            assert result == {}

    def test_load_returns_file_contents_as_is(self):
        """load_settings returns the JSON contents without modification."""
        with tempfile.TemporaryDirectory() as tmpdir:
            alan_dir = os.path.join(tmpdir, ".alan")
            os.makedirs(alan_dir)
            path = os.path.join(alan_dir, "settings.json")
            with open(path, "w") as f:
                json.dump({"model": "test-model", "max_iterations_per_turn": 5}, f)

            settings = load_settings(tmpdir)
            assert settings == {"model": "test-model", "max_iterations_per_turn": 5}

    def test_ephemeral_not_saved(self):
        """api_key should not appear in the saved file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = dict(SETTINGS_DEFAULTS)
            settings["api_key"] = "sk-secret-12345"
            save_settings(settings, tmpdir)

            path = os.path.join(tmpdir, ".alan", "settings.json")
            with open(path) as f:
                on_disk = json.load(f)

            assert "api_key" not in on_disk

    def test_save_and_load_roundtrip(self):
        """save_settings -> load_settings preserves values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original = dict(SETTINGS_DEFAULTS)
            original["model"] = "test-model"
            original["verbose"] = True
            save_settings(original, tmpdir)

            loaded = load_settings(tmpdir)
            assert loaded["model"] == "test-model"
            assert loaded["verbose"] is True

    def test_corrupt_json_returns_empty(self):
        """Corrupt settings.json doesn't crash, returns empty."""
        with tempfile.TemporaryDirectory() as tmpdir:
            alan_dir = os.path.join(tmpdir, ".alan")
            os.makedirs(alan_dir)
            path = os.path.join(alan_dir, "settings.json")
            with open(path, "w") as f:
                f.write("{corrupt json!!!")

            result = load_settings(tmpdir)
            assert result == {}


# ---------------------------------------------------------------------------
# TestValidation -- validate_setting
# ---------------------------------------------------------------------------


class TestValidation:

    def test_valid_enum(self):
        assert validate_setting("memory", "on") is None
        assert validate_setting("memory", "off") is None
        assert validate_setting("memory", "intensive") is None

    def test_invalid_enum(self):
        error = validate_setting("memory", "yooo")
        assert error is not None
        assert "yooo" in error

    def test_valid_type(self):
        assert validate_setting("model", "gpt-4o") is None

    def test_invalid_type(self):
        error = validate_setting("model", 123)
        assert error is not None

    def test_none_accepted_for_typed(self):
        """None is always accepted (means 'unset')."""
        assert validate_setting("model", None) is None

    def test_no_validator(self):
        """Keys without a validator always pass."""
        assert validate_setting("max_tool_concurrency", 999) is None

    def test_permission_mode(self):
        assert validate_setting("permission_mode", "yolo") is None
        assert validate_setting("permission_mode", "edit") is None
        assert validate_setting("permission_mode", "safe") is None
        error = validate_setting("permission_mode", "bypass")
        assert error is not None


# ---------------------------------------------------------------------------
# TestBuiltinDefaults -- sanity checks
# ---------------------------------------------------------------------------


class TestBuiltinDefaults:

    def test_flat_structure(self):
        """Settings should be a flat dict (no 'advanced' nesting)."""
        assert "advanced" not in SETTINGS_DEFAULTS

    def test_known_keys_present(self):
        assert "model" in SETTINGS_DEFAULTS
        assert "permission_mode" in SETTINGS_DEFAULTS
        assert "max_iterations_per_turn" in SETTINGS_DEFAULTS
        assert "verbose" in SETTINGS_DEFAULTS
        assert "hooks" in SETTINGS_DEFAULTS

    def test_tuning_keys_present(self):
        """All tuning knobs are top-level."""
        assert "max_tool_concurrency" in SETTINGS_DEFAULTS
        assert "thinking_budget_default" in SETTINGS_DEFAULTS
        assert "compact_max_output_tokens" in SETTINGS_DEFAULTS
        assert "memory_reminder_threshold" in SETTINGS_DEFAULTS
        assert "compaction_threshold_percent" in SETTINGS_DEFAULTS
        assert "max_consecutive_compact_failures" in SETTINGS_DEFAULTS
        assert "max_compact_ptl_retries" in SETTINGS_DEFAULTS
