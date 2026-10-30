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


class OversizedLineBackend(LLMBackend):
    """Raises the oversized-line ValueError until asked not to stream."""

    def __init__(self, *, partial_content: bool = False):
        self.calls: list[bool] = []
        self.partial_content = partial_content

    async def stream(self, messages, system, tools, *, disable_stream=False, **kwargs):
        self.calls.append(disable_stream)
        yield StreamMessageStart(model="test")
        if not disable_stream:
            if self.partial_content:
                yield StreamTextDelta(text="partial")
            raise ValueError(
                "Separator is not found, and chunk exceed the limit"
            )
        yield StreamTextDelta(text="ok")

    def get_model_info(self, model=None):
        return ModelInfo()


@pytest.mark.asyncio
async def test_oversized_sse_line_falls_back_to_unstreamed():
    # The buffer limit belongs to litellm's HTTP session, not to alancode, so
    # the only recovery available here is to re-issue the request unstreamed.
    backend = OversizedLineBackend()
    events = [e async for e in stream_with_retry(backend, [], [], [])]

    assert backend.calls == [False, True]
    assert any(
        isinstance(e, StreamTextDelta) and e.text == "ok" for e in events
    )


@pytest.mark.asyncio
async def test_oversized_sse_line_after_content_is_not_replayed():
    # The no-replay boundary outranks the fallback: re-issuing a request whose
    # response was already partly emitted would duplicate it.
    backend = OversizedLineBackend(partial_content=True)
    with pytest.raises(ValueError, match="Separator is not found"):
        async for _ in stream_with_retry(backend, [], [], []):
            pass

    assert backend.calls == [False]


@pytest.mark.asyncio
async def test_rate_limit_waits_span_a_per_minute_quota(monkeypatch):
    """Albert answers 429 once a model's tokens-per-minute quota is spent and
    sends no Retry-After; three waits of about 1, 2 and 4 s all fell inside
    the same minute, so the turn ended on the third 429."""
    class QuotaBackend(RetryBackend):
        async def stream(self, messages, system, tools, **kwargs):
            self.calls += 1
            if self.calls <= 3:
                yield StreamError(error="Rate limit exceeded", status_code=429)
                return
            yield StreamTextDelta(text="ok")

    backend = QuotaBackend(failures=0)
    waits = []

    async def record_sleep(delay):
        waits.append(delay)

    monkeypatch.setattr("alancode.api.retry.asyncio.sleep", record_sleep)
    events = [e async for e in stream_with_retry(backend, [], [], [], max_retries=3)]

    assert events[-1].text == "ok"
    assert sum(waits) >= 60
    assert waits[0] >= 15


@pytest.mark.asyncio
async def test_other_transient_errors_keep_short_waits(monkeypatch):
    backend = RetryBackend(failures=1)
    waits = []

    async def record_sleep(delay):
        waits.append(delay)

    monkeypatch.setattr("alancode.api.retry.asyncio.sleep", record_sleep)
    _ = [e async for e in stream_with_retry(backend, [], [], [], max_retries=2)]
    assert waits and waits[0] < 3
