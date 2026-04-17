"""Tests for first-run setup, restart-required settings, and defaults."""

import json
import os
import tempfile

import pytest

from alancode.settings import (
    SETTINGS_DEFAULTS,
    PROVIDER_SETTINGS,
    validate_setting,
    load_settings,
    save_settings,
    get_settings_path,
    load_projects_settings_and_maybe_init,
)


class TestDefaults:
    """Verify the new defaults."""

    def test_default_provider_is_litellm(self):
        assert SETTINGS_DEFAULTS["provider"] == "litellm"

    def test_default_model_is_claude_sonnet(self):
        assert SETTINGS_DEFAULTS["model"] == "anthropic/claude-sonnet-4-6"

    def test_default_permission_mode_is_edit(self):
        assert SETTINGS_DEFAULTS["permission_mode"] == "edit"

    def test_default_tool_call_format_is_none(self):
        assert SETTINGS_DEFAULTS["tool_call_format"] is None


class TestProviderSettings:
    """Verify provider-related settings trigger recreation."""

    def test_provider_triggers_recreation(self):
        assert "provider" in PROVIDER_SETTINGS

    def test_model_triggers_recreation(self):
        assert "model" in PROVIDER_SETTINGS

    def test_api_key_triggers_recreation(self):
        assert "api_key" in PROVIDER_SETTINGS

    def test_memory_does_not_trigger_recreation(self):
        assert "memory" not in PROVIDER_SETTINGS

    def test_permission_mode_does_not_trigger_recreation(self):
        assert "permission_mode" not in PROVIDER_SETTINGS


class TestUpdateSessionSettingRejectsRestartRequired:
    """Verify that update_session_setting rejects restart-required keys."""

    def test_provider_change_recreates_provider(self):
        """Provider can be changed mid-session — the provider object is recreated."""
        from alancode.agent import AlanCodeAgent
        from alancode.providers.scripted_provider import ScriptedProvider

        provider = ScriptedProvider()
        agent = AlanCodeAgent(provider=provider, cwd="/tmp/test_prov_change", permission_mode="yolo")

        # Change to scripted (always works, no API key needed)
        error = agent.update_session_setting("provider", "scripted")
        assert error is None
        assert agent._settings["provider"] == "scripted"

    def test_tool_call_format_change_succeeds(self):
        from alancode.agent import AlanCodeAgent
        from alancode.providers.scripted_provider import ScriptedProvider

        provider = ScriptedProvider()
        agent = AlanCodeAgent(provider=provider, cwd="/tmp/test_tcf", permission_mode="yolo")

        error = agent.update_session_setting("tool_call_format", "hermes")
        assert error is None
        assert agent._settings["tool_call_format"] == "hermes"

    def test_allow_model_change(self):
        from alancode.agent import AlanCodeAgent
        from alancode.providers.scripted_provider import ScriptedProvider

        provider = ScriptedProvider()
        agent = AlanCodeAgent(provider=provider, cwd="/tmp/test_allow1", permission_mode="yolo")

        error = agent.update_session_setting("model", "gpt-4o")
        assert error is None
        assert agent._model == "gpt-4o"

    def test_allow_memory_change(self):
        from alancode.agent import AlanCodeAgent
        from alancode.providers.scripted_provider import ScriptedProvider

        provider = ScriptedProvider()
        agent = AlanCodeAgent(provider=provider, cwd="/tmp/test_allow2", permission_mode="yolo")

        error = agent.update_session_setting("memory", "off")
        assert error is None
        assert agent._memory_mode == "off"

    def test_allow_permission_mode_change(self):
        from alancode.agent import AlanCodeAgent
        from alancode.providers.scripted_provider import ScriptedProvider

        provider = ScriptedProvider()
        agent = AlanCodeAgent(provider=provider, cwd="/tmp/test_allow3", permission_mode="yolo")

        error = agent.update_session_setting("permission_mode", "safe")
        assert error is None
        assert agent._permission_mode == "safe"


class TestUpdateProjectSetting:
    """Verify that update_project_setting allows all keys (including restart-required)."""

    def test_allow_provider_change_in_project(self):
        from alancode.agent import AlanCodeAgent
        from alancode.providers.scripted_provider import ScriptedProvider

        with tempfile.TemporaryDirectory() as tmpdir:
            provider = ScriptedProvider()
            agent = AlanCodeAgent(provider=provider, cwd=tmpdir, permission_mode="yolo")

            error = agent.update_project_setting("provider", "litellm")
            assert error is None

            settings = load_settings(tmpdir)
            assert settings["provider"] == "litellm"

    def test_allow_model_change_in_project(self):
        from alancode.agent import AlanCodeAgent
        from alancode.providers.scripted_provider import ScriptedProvider

        with tempfile.TemporaryDirectory() as tmpdir:
            provider = ScriptedProvider()
            agent = AlanCodeAgent(provider=provider, cwd=tmpdir, permission_mode="yolo")

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
            agent = AlanCodeAgent(provider=provider, cwd=tmpdir, permission_mode="yolo")

            # Agent init creates settings.json via load_projects_settings_and_maybe_init
            assert get_settings_path(tmpdir).exists()

    def test_settings_has_correct_defaults(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = load_projects_settings_and_maybe_init(tmpdir)
            assert settings["provider"] == "litellm"
            assert settings["model"] == "anthropic/claude-sonnet-4-6"
            assert settings["permission_mode"] == "edit"

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
            anthropic_found = any(d["model"] == "anthropic/claude-sonnet-4-6" for d in detections)
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
            openai_found = any(d["model"] == "openai/gpt-4o" for d in detections)
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
        for key in ("ANTHROPIC_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY"):
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

    def test_all_provider_settings_have_validators(self):
        from alancode.settings import SETTING_VALIDATORS
        for key in PROVIDER_SETTINGS:
            if key == "api_key":
                continue  # api_key accepts any value
            assert key in SETTING_VALIDATORS, f"'{key}' is a provider setting but has no validator"
