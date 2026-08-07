"""Regression tests for retry behavior around streamed metadata."""

import pytest

from alancode.api.errors import ServerError
from alancode.api.retry import stream_with_retry
from alancode.backends.base import (
    LLMBackend,
    ModelInfo,
    StreamError,
    StreamMessageStart,
    StreamTextDelta,
)


class RetryBackend(LLMBackend):
    def __init__(self, failures: int, *, partial_content: bool = False):
        self.calls = 0
        self.failures = failures
        self.partial_content = partial_content

    async def stream(self, messages, system, tools, **kwargs):
        self.calls += 1
        yield StreamMessageStart(model="test", request_id=str(self.calls))
        if self.calls <= self.failures:
            if self.partial_content:
                yield StreamTextDelta(text="partial")
            yield StreamError(
                error="server took too long",
                error_type="timeout",
            )
            return
        yield StreamTextDelta(text="ok")

    def get_model_info(self, model=None):
        return ModelInfo()


@pytest.mark.asyncio
async def test_message_start_does_not_disable_startup_retry(monkeypatch):
    backend = RetryBackend(failures=2)

    async def no_sleep(_delay):
        return None

    monkeypatch.setattr("alancode.api.retry.asyncio.sleep", no_sleep)
    events = [
        event
        async for event in stream_with_retry(
            backend, [], [], [], max_retries=2,
        )
    ]

    assert backend.calls == 3
    assert isinstance(events[-1], StreamTextDelta)
    assert events[-1].text == "ok"


@pytest.mark.asyncio
async def test_partial_content_is_never_replayed(monkeypatch):
    backend = RetryBackend(failures=1, partial_content=True)

    async def no_sleep(_delay):
        return None

    monkeypatch.setattr("alancode.api.retry.asyncio.sleep", no_sleep)
    with pytest.raises(TimeoutError) as exc_info:
        _ = [
            event
            async for event in stream_with_retry(
                backend, [], [], [], max_retries=2,
            )
        ]

    assert backend.calls == 1
    assert exc_info.value.alan_response_content_yielded is True


@pytest.mark.asyncio
async def test_http_5xx_stream_error_is_retryable(monkeypatch):
    class ServerFailureBackend(RetryBackend):
        async def stream(self, messages, system, tools, **kwargs):
            self.calls += 1
            if self.calls == 1:
                yield StreamError(
                    error="temporary failure",
                    error_type="server_error",
                    status_code=503,
                )
                return
            yield StreamTextDelta(text="recovered")

    backend = ServerFailureBackend(failures=0)

    async def no_sleep(_delay):
        return None

    monkeypatch.setattr("alancode.api.retry.asyncio.sleep", no_sleep)
    events = [
        event
        async for event in stream_with_retry(
            backend, [], [], [], max_retries=1,
        )
    ]

    assert backend.calls == 2
    assert events[-1].text == "recovered"


def test_server_error_keeps_http_status():
    error = ServerError("temporary", status_code=502)
    assert error.status_code == 502
