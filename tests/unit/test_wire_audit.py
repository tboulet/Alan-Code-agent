"""ALANCODE_WIRE_LOG: prove what was actually handed to the provider SDK."""

import json
import logging
from types import SimpleNamespace

import pytest

from alancode.backends import wire
from alancode.backends.litellm_backend import LiteLLMBackend


@pytest.fixture(autouse=True)
def reset_wire_state():
    """The audit logger is a process-level singleton; unbind it per test."""
    yield
    logger = logging.getLogger("alancode.wire")
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    wire._wire_logger = None
    wire._setup_failed = False


def test_disabled_by_default(monkeypatch, tmp_path):
    monkeypatch.delenv(wire.WIRE_LOG_ENV, raising=False)
    wire.log_wire_request("litellm", {"model": "gpt-4o"})
    assert list(tmp_path.iterdir()) == []


def test_enabled_writes_one_json_line_per_request(monkeypatch, tmp_path):
    path = tmp_path / "wire.jsonl"
    monkeypatch.setenv(wire.WIRE_LOG_ENV, str(path))
    wire.log_wire_request("litellm", {"model": "gpt-4o", "max_tokens": 100})
    wire.log_wire_request("anthropic", {"model": "claude-sonnet-4-6"})

    lines = path.read_text().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["provider"] == "litellm"
    assert first["request"]["model"] == "gpt-4o"
    assert first["request"]["max_tokens"] == 100
    assert json.loads(lines[1])["provider"] == "anthropic"


def test_credentials_are_redacted(monkeypatch, tmp_path):
    path = tmp_path / "wire.jsonl"
    monkeypatch.setenv(wire.WIRE_LOG_ENV, str(path))
    wire.log_wire_request(
        "litellm",
        {"model": "gpt-4o", "api_key": "sk-secret", "headers": {"x-key": "sk-secret"}},
    )
    written = path.read_text()
    assert "sk-secret" not in written
    request = json.loads(written)["request"]
    assert request["api_key"] == wire.REDACTED_PLACEHOLDER
    assert request["headers"] == wire.REDACTED_PLACEHOLDER


def test_unserializable_values_do_not_break_the_call(monkeypatch, tmp_path):
    path = tmp_path / "wire.jsonl"
    monkeypatch.setenv(wire.WIRE_LOG_ENV, str(path))
    wire.log_wire_request("litellm", {"model": "gpt-4o", "callback": object()})
    assert "callback" in json.loads(path.read_text())["request"]


def test_unopenable_path_disables_the_audit_without_raising(monkeypatch, tmp_path):
    monkeypatch.setenv(wire.WIRE_LOG_ENV, str(tmp_path / "missing-dir" / "wire.jsonl"))
    wire.log_wire_request("litellm", {"model": "gpt-4o"})
    assert wire._setup_failed is True


@pytest.mark.asyncio
async def test_the_litellm_call_site_logs_the_real_kwargs(monkeypatch, tmp_path):
    """The point of the audit is the shaped request, not a hand-built dict."""
    path = tmp_path / "wire.jsonl"
    monkeypatch.setenv(wire.WIRE_LOG_ENV, str(path))

    class FakeLiteLLM:
        def get_model_info(self, _model):
            return {"max_input_tokens": 131_072, "max_output_tokens": 16_384}

        async def acompletion(self, **kwargs):
            async def response():
                delta = SimpleNamespace(
                    content="ok", reasoning_content=None, reasoning=None,
                    tool_calls=None,
                )
                yield SimpleNamespace(
                    choices=[SimpleNamespace(delta=delta, finish_reason="stop")],
                    usage=None,
                )

            return response()

    monkeypatch.setattr(
        "alancode.backends.litellm_backend._load_litellm", lambda: FakeLiteLLM(),
    )
    backend = LiteLLMBackend(
        model="openai/local",
        api_base="http://localhost:9876/v1",
        api_key="sk-secret",
    )
    [event async for event in backend.stream([], ["be helpful"], [])]

    logged = json.loads(path.read_text())
    assert logged["provider"] == "litellm"
    assert logged["request"]["stream"] is True
    assert logged["request"]["api_base"] == "http://localhost:9876/v1"
    assert logged["request"]["api_key"] == wire.REDACTED_PLACEHOLDER
    assert "sk-secret" not in path.read_text()
