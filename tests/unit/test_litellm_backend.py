"""Focused tests for the LiteLLM streaming adapter."""

from types import SimpleNamespace

import pytest

from alancode.backends.base import (
    StreamError,
    StreamMessageDelta,
    StreamMessageStart,
    StreamToolUseStop,
)
from alancode.backends.litellm_backend import LiteLLMBackend


def _chunk(*, content=None, tool_calls=None, finish_reason=None, usage=None):
    delta = SimpleNamespace(
        content=content,
        reasoning_content=None,
        reasoning=None,
        tool_calls=tool_calls,
    )
    choice = SimpleNamespace(delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], usage=usage)


def _tool_delta(*, index=0, call_id=None, name=None, arguments=None):
    function = SimpleNamespace(name=name, arguments=arguments)
    return SimpleNamespace(
        index=index,
        id=call_id,
        function=function,
    )


class FakeLiteLLM:
    def __init__(self, chunks=(), error=None):
        self.chunks = list(chunks)
        self.error = error
        self.calls = []

    def get_model_info(self, _model):
        return {
            "max_input_tokens": 131_072,
            "max_output_tokens": 16_384,
        }

    async def acompletion(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error

        async def response():
            for chunk in self.chunks:
                yield chunk

        return response()


async def _events(backend):
    return [event async for event in backend.stream([], [], [])]


@pytest.mark.asyncio
async def test_custom_endpoint_uses_long_auto_timeout(monkeypatch):
    fake = FakeLiteLLM([_chunk(content="ok", finish_reason="stop")])
    monkeypatch.setattr(
        "alancode.backends.litellm_backend._load_litellm",
        lambda: fake,
    )
    backend = LiteLLMBackend(
        model="openai/local",
        api_base="http://localhost:9876/v1",
    )

    events = await _events(backend)

    assert fake.calls[0]["timeout"] == 3_600
    assert fake.calls[0]["api_base"] == "http://localhost:9876/v1"
    assert isinstance(events[0], StreamMessageStart)


@pytest.mark.asyncio
async def test_explicit_request_timeout_wins(monkeypatch):
    fake = FakeLiteLLM([_chunk(finish_reason="stop")])
    monkeypatch.setattr(
        "alancode.backends.litellm_backend._load_litellm",
        lambda: fake,
    )
    backend = LiteLLMBackend(
        model="openai/local",
        api_base="http://localhost:9876/v1",
        request_timeout=90,
    )

    await _events(backend)

    assert fake.calls[0]["timeout"] == 90


def test_request_timeout_accepts_case_insensitive_auto():
    backend = LiteLLMBackend(
        model="openai/local",
        api_base="http://localhost:9876/v1",
        request_timeout="AUTO",
    )

    assert backend._request_timeout == 3_600


@pytest.mark.parametrize("invalid", [0, -1, True, "forever"])
def test_request_timeout_rejects_invalid_direct_values(invalid):
    with pytest.raises(ValueError, match="request_timeout"):
        LiteLLMBackend(model="openai/local", request_timeout=invalid)


@pytest.mark.asyncio
async def test_tool_call_is_finalized_without_finish_reason(monkeypatch):
    chunks = [
        _chunk(tool_calls=[_tool_delta(
            call_id="call_1",
            name="Echo",
            arguments='{\"text\":',
        )]),
        _chunk(tool_calls=[_tool_delta(arguments='\"hello\"}')]),
    ]
    fake = FakeLiteLLM(chunks)
    monkeypatch.setattr(
        "alancode.backends.litellm_backend._load_litellm",
        lambda: fake,
    )

    events = await _events(LiteLLMBackend(model="openai/local"))

    stops = [e for e in events if isinstance(e, StreamToolUseStop)]
    deltas = [e for e in events if isinstance(e, StreamMessageDelta)]
    assert len(stops) == 1
    assert stops[0].name == "Echo"
    assert stops[0].input == {"text": "hello"}
    assert deltas[0].stop_reason == "tool_use"


@pytest.mark.asyncio
async def test_malformed_tool_arguments_are_visible(monkeypatch):
    fake = FakeLiteLLM([
        _chunk(
            tool_calls=[_tool_delta(
                call_id="call_1",
                name="Echo",
                arguments="not-json",
            )],
            finish_reason="tool_calls",
        ),
    ])
    monkeypatch.setattr(
        "alancode.backends.litellm_backend._load_litellm",
        lambda: fake,
    )

    events = await _events(LiteLLMBackend(model="openai/local"))

    errors = [e for e in events if isinstance(e, StreamError)]
    assert len(errors) == 1
    assert errors[0].error_type == "invalid_tool_call"
    assert "invalid JSON" in errors[0].error
    assert not any(isinstance(e, StreamToolUseStop) for e in events)


@pytest.mark.asyncio
async def test_pre_stream_503_has_status_and_no_message_start(monkeypatch):
    class TemporaryFailure(Exception):
        status_code = 503

    fake = FakeLiteLLM(error=TemporaryFailure("service unavailable"))
    monkeypatch.setattr(
        "alancode.backends.litellm_backend._load_litellm",
        lambda: fake,
    )

    events = await _events(LiteLLMBackend(model="openai/local"))

    assert len(events) == 1
    assert isinstance(events[0], StreamError)
    assert events[0].error_type == "server_error"
    assert events[0].status_code == 503
