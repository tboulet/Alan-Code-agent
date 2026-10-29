"""The CLI names a missing API key at startup instead of a traceback later."""

import pytest

from alancode.cli.main import missing_api_key_message


@pytest.fixture
def no_keys(monkeypatch):
    for key in ("ANTHROPIC_API_KEY", "OPENROUTER_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(key, raising=False)


def test_default_claude_model_without_its_key(no_keys):
    """Plain `alancode` without ANTHROPIC_API_KEY printed the Anthropic SDK's
    'Could not resolve authentication method' traceback on the first message."""
    message = missing_api_key_message({"backend": "anthropic-native", "model": "claude-haiku-5-5"})
    assert "ANTHROPIC_API_KEY" in message and "--api-key" in message


def test_litellm_model_names_its_provider_key(no_keys):
    message = missing_api_key_message(
        {"backend": "auto", "model": "openrouter/anthropic/claude-haiku-5.5"}
    )
    assert "OPENROUTER_API_KEY" in message


@pytest.mark.parametrize("settings", [
    {"backend": "anthropic-native", "model": "claude-haiku-5-5", "api_key": "sk-x"},
    {"backend": "auto", "model": "openai/local", "base_url": "http://localhost:8000/v1"},
    {"backend": "auto", "model": "ollama/llama3.1"},
    {"backend": "scripted", "model": "scripted-model"},
])
def test_no_complaint_when_nothing_is_missing(no_keys, settings):
    assert missing_api_key_message(settings) is None


def test_key_in_environment_is_enough(no_keys, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    assert missing_api_key_message(
        {"backend": "auto", "model": "openrouter/anthropic/claude-haiku-5.5"}
    ) is None
