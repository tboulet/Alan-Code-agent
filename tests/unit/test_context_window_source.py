import pytest
"""The resolved window is useless in a log without knowing if it was a guess."""

from alancode.agent import AlanCodeAgent


def _agent(**kw):
    return AlanCodeAgent(programmatic=True, cwd="/tmp", **kw)


def test_unknown_served_model_refuses():
    # A served name litellm has never seen: registry, server probe and cache all
    # miss. Guessing here is invisible in the results, so it must refuse.
    from alancode.budget import ConfigError

    agent = _agent(model="openai/no-such-served-model", base_url="http://127.0.0.1:9/v1")
    with pytest.raises(ConfigError, match="no-such-served-model"):
        agent.context_window


def test_unknown_served_model_uses_a_named_fallback():
    agent = _agent(
        model="openai/no-such-served-model",
        base_url="http://127.0.0.1:9/v1",
        context_window_fallback=32_768,
    )
    assert agent.context_window == 32_768
    assert agent.context_window_source == "fallback"


def test_explicit_context_window_reports_override():
    agent = _agent(
        model="openai/no-such-served-model",
        base_url="http://127.0.0.1:9/v1",
        context_window=160_000,
    )
    assert agent.context_window == 160_000
    assert agent.context_window_source == "override"


def test_a_known_model_is_not_a_fallback():
    agent = _agent(model="claude-sonnet-4-6")
    assert agent.context_window_source != "fallback"
    assert agent.context_window > 32_768
