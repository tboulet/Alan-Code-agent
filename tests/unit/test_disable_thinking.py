"""disable_thinking: forward the chat-template switch only where it exists."""

import pytest

from alancode.agent import _create_backend_from_settings
from alancode.settings import SETTINGS_DEFAULTS, validate_setting


def _settings(**overrides):
    s = dict(SETTINGS_DEFAULTS)
    s.update(overrides)
    return s


def test_default_is_off_and_sends_nothing():
    backend = _create_backend_from_settings(
        _settings(backend="auto", model="openai/gpt-4o")
    )
    assert "chat_template_kwargs" not in backend._extra_kwargs


def test_enabled_reaches_the_litellm_transport():
    backend = _create_backend_from_settings(
        _settings(backend="auto", model="openai/gpt-4o", disable_thinking=True)
    )
    assert backend._extra_kwargs["chat_template_kwargs"] == {
        "enable_thinking": False
    }


def test_explicit_caller_kwargs_win():
    backend = _create_backend_from_settings(
        _settings(backend="auto", model="openai/gpt-4o", disable_thinking=True),
        chat_template_kwargs={"enable_thinking": True},
    )
    assert backend._extra_kwargs["chat_template_kwargs"] == {
        "enable_thinking": True
    }


def test_native_anthropic_backend_is_left_alone():
    # AnthropicBackend has no chat template; forwarding the switch would be an
    # unexpected keyword argument rather than a no-op.
    backend = _create_backend_from_settings(
        _settings(
            backend="anthropic-native",
            model="claude-sonnet-4",
            api_key="test-key",
            disable_thinking=True,
        )
    )
    assert not hasattr(backend, "_extra_kwargs") or (
        "chat_template_kwargs" not in getattr(backend, "_extra_kwargs", {})
    )


@pytest.mark.parametrize("value", [True, False])
def test_validator_accepts_bools(value):
    assert validate_setting("disable_thinking", value) is None
    assert validate_setting("no_verbalize_warning", value) is None


@pytest.mark.parametrize("value", ["yes", 1, None])
def test_validator_rejects_non_bools(value):
    assert validate_setting("disable_thinking", value) is not None
    assert validate_setting("no_verbalize_warning", value) is not None
