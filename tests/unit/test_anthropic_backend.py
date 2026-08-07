"""Focused tests for Anthropic backend construction."""

from types import SimpleNamespace
import sys

import pytest

from alancode.backends.anthropic_backend import AnthropicBackend


class FakeAsyncAnthropic:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.__class__.instances.append(self)

    async def close(self):
        return None


@pytest.fixture(autouse=True)
def fake_anthropic(monkeypatch):
    FakeAsyncAnthropic.instances.clear()
    monkeypatch.setitem(
        sys.modules,
        "anthropic",
        SimpleNamespace(AsyncAnthropic=FakeAsyncAnthropic),
    )


def test_custom_endpoint_uses_long_auto_timeout():
    AnthropicBackend(base_url="http://localhost:9876")

    assert FakeAsyncAnthropic.instances[-1].kwargs["timeout"] == 3_600


def test_explicit_request_timeout_wins():
    AnthropicBackend(
        base_url="http://localhost:9876",
        request_timeout=90,
    )

    assert FakeAsyncAnthropic.instances[-1].kwargs["timeout"] == 90


def test_default_anthropic_endpoint_keeps_sdk_timeout():
    AnthropicBackend(request_timeout="auto")

    assert "timeout" not in FakeAsyncAnthropic.instances[-1].kwargs


@pytest.mark.parametrize("invalid", [0, -1, True, "forever"])
def test_request_timeout_rejects_invalid_values(invalid):
    with pytest.raises(ValueError, match="request_timeout"):
        AnthropicBackend(request_timeout=invalid)
