"""OpenRouter's catalogue fills in what LiteLLM's registry does not know."""

import pytest

from alancode.api.cost_tracker import CostTracker
from alancode.backends import openrouter_models
from alancode.backends.litellm_backend import LiteLLMBackend
from alancode.budget import ConfigError
from alancode.messages.types import Usage

NEW_MODEL = "openrouter/google/gemini-3.8-flash"
ENTRY = {
    "id": "google/gemini-3.8-flash",
    "context_length": 1_048_576,
    "pricing": {
        "prompt": "0.000001",
        "completion": "0.000004",
        "input_cache_read": "0.0000001",
        "overrides": [{"min_prompt_tokens": 200_000, "prompt": "0.000002", "completion": "0.000008"}],
    },
    "top_provider": {"context_length": 1_048_576, "max_completion_tokens": 65_536},
}


@pytest.fixture
def catalogue(monkeypatch):
    calls = []

    def fetch():
        calls.append(1)
        return {ENTRY["id"]: ENTRY}

    monkeypatch.setattr(openrouter_models, "_catalogue", None)
    monkeypatch.setattr(openrouter_models, "_fetch_catalogue", fetch)
    return calls


def _backend(model):
    return LiteLLMBackend(model=model)


def test_new_openrouter_model_gets_its_window(catalogue, monkeypatch):
    """gemini-3.8-flash, unknown to LiteLLM, refused to start: 'Context
    window could not be determined'."""
    monkeypatch.setattr("alancode.backends.litellm_backend.cw_probe.load_cached_context_window", lambda *a: None)
    info = _backend(NEW_MODEL).get_model_info()
    assert info.context_window == 1_048_576
    assert info.max_output_tokens == 65_536
    assert info.cw_source == "server"


def test_model_absent_from_the_catalogue_still_refuses(catalogue, monkeypatch):
    monkeypatch.setattr("alancode.backends.litellm_backend.cw_probe.load_cached_context_window", lambda *a: None)
    with pytest.raises(ConfigError):
        _backend("openrouter/nobody/unknown-model").get_model_info()


def test_catalogue_is_fetched_once(catalogue):
    for _ in range(3):
        openrouter_models.context_window(NEW_MODEL)
    assert len(catalogue) == 1


def test_a_failed_fetch_is_not_retried_and_answers_none(monkeypatch):
    calls = []

    def broken():
        calls.append(1)
        raise OSError("offline")

    monkeypatch.setattr(openrouter_models, "_catalogue", None)
    monkeypatch.setattr(openrouter_models, "_fetch_catalogue", broken)
    assert openrouter_models.context_window(NEW_MODEL) is None
    assert openrouter_models.context_window(NEW_MODEL) is None
    assert len(calls) == 1


def test_non_openrouter_models_never_fetch(catalogue):
    assert openrouter_models.context_window("ollama/llama3.1") is None
    assert catalogue == []


def test_cost_of_a_model_litellm_cannot_price(catalogue):
    """claude-haiku-5.5 through OpenRouter showed $0.0000: LiteLLM had no price."""
    tracker = CostTracker(session=None)
    usage = Usage(input_tokens=10_000, output_tokens=1_000, cache_read_input_tokens=5_000)
    cost = tracker.calculate_cost(usage, NEW_MODEL)
    assert cost == pytest.approx(10_000 * 1e-6 + 1_000 * 4e-6 + 5_000 * 1e-7)


def test_long_prompts_use_the_higher_price_tier(catalogue):
    cost = openrouter_models.cost_usd(NEW_MODEL, input_tokens=250_000, output_tokens=1_000)
    assert cost == pytest.approx(250_000 * 2e-6 + 1_000 * 8e-6)


@pytest.mark.parametrize("model, window", [
    ("gemini/gemini-3.8-flash", 1_048_576),
    ("anthropic/claude-sonnet-5-5", 1_000_000),
    ("openai/gpt-5.6-terra", 1_050_000),
])
def test_current_families_resolve_outside_openrouter(model, window, monkeypatch):
    """Direct provider routes do not read OpenRouter's catalogue; the
    known-models table covers current families LiteLLM's registry lacks."""
    monkeypatch.setattr("alancode.backends.litellm_backend.cw_probe.load_cached_context_window", lambda *a: None)
    info = LiteLLMBackend(model=model).get_model_info()
    assert info.context_window == window
