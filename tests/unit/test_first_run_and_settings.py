"""Tests for first-run setup, restart-required settings, and defaults."""

import json
import os
import tempfile

import pytest

from alancode.settings import (
    SETTINGS_DEFAULTS,
    BACKEND_SETTINGS,
    infer_backend,
    migrate_legacy_provider_key,
    validate_setting,
    load_settings,
    save_settings,
    get_settings_path,
    load_projects_settings_and_maybe_init,
)


class TestDefaults:
    """Verify the new defaults."""

    def test_default_backend_is_anthropic_native(self):
        assert SETTINGS_DEFAULTS["backend"] == "anthropic-native"

    def test_default_model_is_claude_sonnet(self):
        assert SETTINGS_DEFAULTS["model"] == "claude-sonnet-4-6"

    def test_default_permission_mode_is_edit(self):
        assert SETTINGS_DEFAULTS["permission_mode"] == "edit"

    def test_default_tool_call_format_is_none(self):
        assert SETTINGS_DEFAULTS["tool_call_format"] is None


class TestBackendSettings:
    """Verify backend-related settings trigger LLMProvider recreation."""

    def test_backend_triggers_recreation(self):
        assert "backend" in BACKEND_SETTINGS

    def test_model_triggers_recreation(self):
        assert "model" in BACKEND_SETTINGS

    def test_api_key_triggers_recreation(self):
        assert "api_key" in BACKEND_SETTINGS

    def test_memory_does_not_trigger_recreation(self):
        assert "memory" not in BACKEND_SETTINGS

    def test_permission_mode_does_not_trigger_recreation(self):
        assert "permission_mode" not in BACKEND_SETTINGS


class TestBackendInference:
    """The model-string inference rule (see settings.infer_backend)."""

    def test_bare_claude_name_picks_native(self):
        assert infer_backend("claude-sonnet-4-6") == "anthropic-native"
        assert infer_backend("claude-opus-4-7") == "anthropic-native"

    def test_anthropic_prefix_picks_auto(self):
        # Explicit anthropic/ prefix means "via LiteLLM" — escape hatch.
        assert infer_backend("anthropic/claude-sonnet-4-6") == "auto"

    def test_bare_non_claude_picks_auto(self):
        assert infer_backend("gpt-4o") == "auto"
        assert infer_backend("gemini-2.5-pro") == "auto"

    def test_other_provider_prefix_picks_auto(self):
        assert infer_backend("ollama/llama3.1") == "auto"
        assert infer_backend("openrouter/google/gemini-2.5-pro") == "auto"

    def test_none_or_empty_falls_back_to_auto(self):
        assert infer_backend(None) == "auto"
        assert infer_backend("") == "auto"


class TestLegacyProviderMigration:
    """The migrate_legacy_provider_key helper handles old .alan/settings.json."""

    def test_legacy_litellm_becomes_auto(self):
        d = {"provider": "litellm", "model": "gpt-4o"}
        changed = migrate_legacy_provider_key(d)
        assert changed is True
        assert d == {"backend": "auto", "model": "gpt-4o"}

    def test_legacy_anthropic_becomes_native(self):
        d = {"provider": "anthropic"}
        migrate_legacy_provider_key(d)
        assert d == {"backend": "anthropic-native"}

    def test_legacy_scripted_passes_through(self):
        d = {"provider": "scripted"}
        migrate_legacy_provider_key(d)
        assert d == {"backend": "scripted"}

    def test_no_provider_key_is_noop(self):
        d = {"model": "claude-sonnet-4-6"}
        assert migrate_legacy_provider_key(d) is False
        assert d == {"model": "claude-sonnet-4-6"}

    def test_existing_backend_wins_over_legacy(self):
        d = {"provider": "litellm", "backend": "anthropic-native"}
        migrate_legacy_provider_key(d)
        assert d == {"backend": "anthropic-native"}


class TestUpdateSessionSettingRejectsRestartRequired:
    """Verify update_session_setting handles backend-related keys correctly."""

    def test_backend_change_recreates_provider(self):
        """Backend can be changed mid-session — the LLMProvider is recreated."""
        from alancode.agent import AlanCodeAgent
        from alancode.providers.scripted_provider import ScriptedProvider

        provider = ScriptedProvider()
        agent = AlanCodeAgent(backend=provider, cwd="/tmp/test_prov_change", permission_mode="yolo")

        # Change to scripted (always works, no API key needed)
        error = agent.update_session_setting("backend", "scripted")
        assert error is None
        assert agent._settings["backend"] == "scripted"

    def test_legacy_provider_key_still_accepted(self):
        """update_session_setting('provider', 'scripted') routes to backend."""
        from alancode.agent import AlanCodeAgent
        from alancode.providers.scripted_provider import ScriptedProvider

        provider = ScriptedProvider()
        agent = AlanCodeAgent(backend=provider, cwd="/tmp/test_legacy_set", permission_mode="yolo")

        error = agent.update_session_setting("provider", "scripted")
        assert error is None
        assert agent._settings["backend"] == "scripted"

    def test_model_change_reinfers_backend(self):
        """Changing only the model promotes the backend per the inference rule."""
        from alancode.agent import AlanCodeAgent
        from alancode.providers.scripted_provider import ScriptedProvider

        provider = ScriptedProvider()
        agent = AlanCodeAgent(
            backend=provider, model="gpt-4o",
            cwd="/tmp/test_reinfer", permission_mode="yolo",
        )
        # Switch to a bare Claude name — inference should promote.
        error = agent.update_session_setting("model", "claude-sonnet-4-6")
        assert error is None
        # Backend recreation will use the new inferred name, but the
        # actual LLMProvider creation will fail because no real Anthropic
        # client is available in tests — we only assert on the setting.
        # (The error path returns a string, not None, if recreation fails.)
        # In CI we accept either: the recreation may or may not succeed
        # depending on whether anthropic is importable.
        assert agent._settings["backend"] == "anthropic-native"

    def test_tool_call_format_change_succeeds(self):
        from alancode.agent import AlanCodeAgent
        from alancode.providers.scripted_provider import ScriptedProvider

        provider = ScriptedProvider()
        agent = AlanCodeAgent(backend=provider, cwd="/tmp/test_tcf", permission_mode="yolo")

        error = agent.update_session_setting("tool_call_format", "hermes")
        assert error is None
        assert agent._settings["tool_call_format"] == "hermes"

    def test_allow_model_change(self):
        from alancode.agent import AlanCodeAgent
        from alancode.providers.scripted_provider import ScriptedProvider

        provider = ScriptedProvider()
        agent = AlanCodeAgent(backend=provider, cwd="/tmp/test_allow1", permission_mode="yolo")

        error = agent.update_session_setting("model", "gpt-4o")
        assert error is None
        assert agent._model == "gpt-4o"

    def test_allow_memory_change(self):
        from alancode.agent import AlanCodeAgent
        from alancode.providers.scripted_provider import ScriptedProvider

        provider = ScriptedProvider()
        agent = AlanCodeAgent(backend=provider, cwd="/tmp/test_allow2", permission_mode="yolo")

        error = agent.update_session_setting("memory", "off")
        assert error is None
        assert agent._memory_mode == "off"

    def test_allow_permission_mode_change(self):
        from alancode.agent import AlanCodeAgent
        from alancode.providers.scripted_provider import ScriptedProvider

        provider = ScriptedProvider()
        agent = AlanCodeAgent(backend=provider, cwd="/tmp/test_allow3", permission_mode="yolo")

        error = agent.update_session_setting("permission_mode", "safe")
        assert error is None
        assert agent._permission_mode == "safe"


class TestUpdateProjectSetting:
    """Verify that update_project_setting allows all keys (including restart-required)."""

    def test_allow_backend_change_in_project(self):
        from alancode.agent import AlanCodeAgent
        from alancode.providers.scripted_provider import ScriptedProvider

        with tempfile.TemporaryDirectory() as tmpdir:
            provider = ScriptedProvider()
            agent = AlanCodeAgent(backend=provider, cwd=tmpdir, permission_mode="yolo")

            error = agent.update_project_setting("backend", "auto")
            assert error is None

            settings = load_settings(tmpdir)
            assert settings["backend"] == "auto"

    def test_legacy_provider_key_translated_in_project(self):
        """update_project_setting('provider', 'anthropic') writes backend='anthropic-native'."""
        from alancode.agent import AlanCodeAgent
        from alancode.providers.scripted_provider import ScriptedProvider

        with tempfile.TemporaryDirectory() as tmpdir:
            provider = ScriptedProvider()
            agent = AlanCodeAgent(backend=provider, cwd=tmpdir, permission_mode="yolo")

            error = agent.update_project_setting("provider", "anthropic")
            assert error is None

            settings = load_settings(tmpdir)
            assert settings.get("backend") == "anthropic-native"
            assert "provider" not in settings

    def test_allow_model_change_in_project(self):
        from alancode.agent import AlanCodeAgent
        from alancode.providers.scripted_provider import ScriptedProvider

        with tempfile.TemporaryDirectory() as tmpdir:
            provider = ScriptedProvider()
            agent = AlanCodeAgent(backend=provider, cwd=tmpdir, permission_mode="yolo")

            error = agent.update_project_setting("model", "gpt-4o")
            assert error is None

            settings = load_settings(tmpdir)
            assert settings["model"] == "gpt-4o"


class TestFirstRunDetection:
    """Test the first-run detection logic."""

    def test_settings_path_does_not_exist_initially(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            assert not get_settings_path(tmpdir).exists()

    def test_settings_created_after_agent_init(self):
        from alancode.agent import AlanCodeAgent
        from alancode.providers.scripted_provider import ScriptedProvider

        with tempfile.TemporaryDirectory() as tmpdir:
            provider = ScriptedProvider()
            agent = AlanCodeAgent(backend=provider, cwd=tmpdir, permission_mode="yolo")

            # Agent init creates settings.json via load_projects_settings_and_maybe_init
            assert get_settings_path(tmpdir).exists()

    def test_settings_has_correct_defaults(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = load_projects_settings_and_maybe_init(tmpdir)
            assert settings["backend"] == "anthropic-native"
            assert settings["model"] == "claude-sonnet-4-6"
            assert settings["permission_mode"] == "edit"

    def test_legacy_settings_file_is_migrated(self):
        """An old .alan/settings.json with the 'provider' key loads cleanly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from pathlib import Path

            alan_dir = Path(tmpdir) / ".alan"
            alan_dir.mkdir()
            (alan_dir / "settings.json").write_text(json.dumps({
                "provider": "litellm",
                "model": "openrouter/anthropic/claude-sonnet-4",
            }))

            settings = load_settings(tmpdir)
            assert settings.get("backend") == "auto"
            assert "provider" not in settings
            assert settings["model"] == "openrouter/anthropic/claude-sonnet-4"

    def test_second_run_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # First run creates defaults
            load_projects_settings_and_maybe_init(tmpdir)

            # Modify a setting
            settings = load_settings(tmpdir)
            settings["model"] = "custom-model"
            save_settings(settings, tmpdir)

            # Second run should load, not overwrite
            settings2 = load_projects_settings_and_maybe_init(tmpdir)
            assert settings2["model"] == "custom-model"


class TestAPIKeyDetection:
    """Test the API key detection in main.py."""

    def test_detect_anthropic_key(self):
        from alancode.cli.main import _detect_api_keys

        old = os.environ.get("ANTHROPIC_API_KEY")
        try:
            os.environ["ANTHROPIC_API_KEY"] = "sk-test-key"
            detections = _detect_api_keys()
            anthropic_found = any(d["model"] == "claude-sonnet-4-6" for d in detections)
            assert anthropic_found
        finally:
            if old:
                os.environ["ANTHROPIC_API_KEY"] = old
            else:
                os.environ.pop("ANTHROPIC_API_KEY", None)

    def test_detect_openai_key(self):
        from alancode.cli.main import _detect_api_keys

        old = os.environ.get("OPENAI_API_KEY")
        try:
            os.environ["OPENAI_API_KEY"] = "sk-test-key"
            detections = _detect_api_keys()
            openai_found = any(d["model"] == "gpt-4o" for d in detections)
            assert openai_found
        finally:
            if old:
                os.environ["OPENAI_API_KEY"] = old
            else:
                os.environ.pop("OPENAI_API_KEY", None)

    def test_no_keys_returns_empty(self):
        from alancode.cli.main import _detect_api_keys

        # Save and clear all keys
        saved = {}
        for key in ("ANTHROPIC_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY",
                    "GEMINI_API_KEY", "GOOGLE_API_KEY"):
            saved[key] = os.environ.pop(key, None)
        try:
            detections = _detect_api_keys()
            assert detections == []
        finally:
            for key, val in saved.items():
                if val:
                    os.environ[key] = val


class TestValidatorCoverage:
    """Ensure all restart-required settings have validators."""

    def test_all_backend_settings_have_validators(self):
        from alancode.settings import SETTING_VALIDATORS
        for key in BACKEND_SETTINGS:
            if key == "api_key":
                continue  # api_key accepts any value
            assert key in SETTING_VALIDATORS, f"'{key}' is a backend setting but has no validator"
