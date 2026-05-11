"""Tests for ``RemoteScriptedProvider`` (HTTP-driven impersonation backend)."""

from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.request

import pytest

from alancode.providers.base import (
    StreamMessageDelta,
    StreamMessageStart,
    StreamMessageStop,
    StreamTextDelta,
    StreamToolUseStart,
    StreamToolUseStop,
    ToolSchema,
)
from alancode.providers.remote_scripted_provider import RemoteScriptedProvider


# ── HTTP helpers ────────────────────────────────────────────────────────────


def _http_get(url: str, timeout: float = 5.0) -> tuple[int, dict | None]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            body = resp.read()
            data = json.loads(body) if body else None
            return resp.status, data
    except urllib.error.HTTPError as e:
        return e.code, None


def _http_post_json(url: str, body: dict, timeout: float = 5.0) -> tuple[int, dict | None]:
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
            return resp.status, data
    except urllib.error.HTTPError as e:
        return e.code, None


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def provider():
    p = RemoteScriptedProvider()
    yield p
    p.shutdown()


# ── Basic lifecycle ─────────────────────────────────────────────────────────


def test_server_health_endpoint(provider):
    code, data = _http_get(f"http://127.0.0.1:{provider._port}/api/health")
    assert code == 200
    assert data == {"ok": True}


def test_session_endpoint_unbound(provider):
    code, data = _http_get(f"http://127.0.0.1:{provider._port}/api/session")
    assert code == 200
    assert data["session_id"] is None
    assert data["cwd"] is None
    assert data["port"] == provider._port


def test_session_endpoint_after_bind(provider, tmp_path):
    provider.set_session_context(session_id="abcdef0123", cwd=str(tmp_path))
    code, data = _http_get(f"http://127.0.0.1:{provider._port}/api/session")
    assert code == 200
    assert data["session_id"] == "abcdef0123"
    assert data["cwd"] == str(tmp_path)


def test_pending_returns_204_when_idle(provider):
    code, _ = _http_get(f"http://127.0.0.1:{provider._port}/api/pending")
    assert code == 204


def test_respond_returns_409_when_no_pending(provider):
    code, _ = _http_post_json(
        f"http://127.0.0.1:{provider._port}/api/respond",
        {"text": "hi"},
    )
    assert code == 409


def test_two_providers_pick_distinct_ports():
    a = RemoteScriptedProvider()
    b = RemoteScriptedProvider()
    try:
        assert a._port != b._port
    finally:
        a.shutdown()
        b.shutdown()


# ── End-to-end stream cycle ─────────────────────────────────────────────────


def _drive_one_stream(
    provider: RemoteScriptedProvider,
    response_body: dict,
    *,
    messages: list | None = None,
    tools: list[ToolSchema] | None = None,
) -> list:
    """Run a single ``stream()`` cycle: kick off the provider, poll until
    ``/api/pending`` returns the payload, POST the response, collect events."""
    events: list = []

    async def run() -> None:
        async def consume() -> None:
            async for e in provider.stream(
                messages=messages or [{"role": "user", "content": "hi"}],
                system=["sys"],
                tools=tools or [],
                model="remote",
            ):
                events.append(e)

        consume_task = asyncio.create_task(consume())

        # Wait until the provider has a pending call.
        for _ in range(50):
            await asyncio.sleep(0.02)
            code, _ = _http_get(
                f"http://127.0.0.1:{provider._port}/api/pending"
            )
            if code == 200:
                break
        else:
            consume_task.cancel()
            raise AssertionError("pending payload never appeared")

        code, _ = _http_post_json(
            f"http://127.0.0.1:{provider._port}/api/respond", response_body
        )
        assert code == 200

        await consume_task

    asyncio.run(run())
    return events


def test_stream_text_only(provider):
    events = _drive_one_stream(provider, {"text": "hello world"})
    types = [type(e).__name__ for e in events]
    assert types[0] == "StreamMessageStart"
    assert types[-1] == "StreamMessageStop"
    text_events = [e for e in events if isinstance(e, StreamTextDelta)]
    assert len(text_events) == 1
    assert text_events[0].text == "hello world"

    deltas = [e for e in events if isinstance(e, StreamMessageDelta)]
    assert deltas[0].stop_reason == "end_turn"


def test_stream_tool_call(provider):
    events = _drive_one_stream(
        provider,
        {
            "tool_calls": [
                {"name": "Bash", "input": {"command": "ls"}},
            ],
        },
    )
    starts = [e for e in events if isinstance(e, StreamToolUseStart)]
    stops = [e for e in events if isinstance(e, StreamToolUseStop)]
    assert len(starts) == 1 and len(stops) == 1
    assert starts[0].name == "Bash"
    assert stops[0].input == {"command": "ls"}

    delta = [e for e in events if isinstance(e, StreamMessageDelta)][0]
    assert delta.stop_reason == "tool_use"


def test_stream_text_plus_tool_calls(provider):
    events = _drive_one_stream(
        provider,
        {
            "text": "running ls.",
            "tool_calls": [{"name": "Bash", "input": {"command": "ls"}}],
        },
    )
    text_events = [e for e in events if isinstance(e, StreamTextDelta)]
    tool_starts = [e for e in events if isinstance(e, StreamToolUseStart)]
    assert text_events[0].text == "running ls."
    assert tool_starts[0].name == "Bash"


def test_pending_includes_serialized_call(provider):
    """The pending payload must expose system, messages, tools — everything
    the external caller needs to act as the LLM."""
    events: list = []

    async def run() -> None:
        async def consume() -> None:
            async for e in provider.stream(
                messages=[{"role": "user", "content": "ping"}],
                system=["sys-line"],
                tools=[
                    ToolSchema(
                        name="Bash",
                        description="run shell",
                        input_schema={"type": "object"},
                    )
                ],
                model="remote",
            ):
                events.append(e)

        consume_task = asyncio.create_task(consume())
        for _ in range(50):
            await asyncio.sleep(0.02)
            code, data = _http_get(
                f"http://127.0.0.1:{provider._port}/api/pending"
            )
            if code == 200:
                payload = data
                break
        else:
            consume_task.cancel()
            raise AssertionError("pending payload never appeared")

        assert payload["system"] == ["sys-line"]
        assert payload["messages"] == [{"role": "user", "content": "ping"}]
        assert payload["tools"][0]["name"] == "Bash"
        assert payload["model"] == "remote"

        _http_post_json(
            f"http://127.0.0.1:{provider._port}/api/respond", {"text": "ok"}
        )
        await consume_task

    asyncio.run(run())


def test_disk_mirror_when_bound(provider, tmp_path):
    """Pending payload is mirrored to
    ``<cwd>/.alan/sessions/<sid>/remote_inbox.json`` once bound."""
    provider.set_session_context(session_id="deadbeef" + "0" * 24, cwd=str(tmp_path))
    expected = (
        tmp_path / ".alan" / "sessions" / ("deadbeef" + "0" * 24) / "remote_inbox.json"
    )

    async def run() -> None:
        async def consume() -> None:
            async for _ in provider.stream(
                messages=[{"role": "user", "content": "hi"}],
                system=["s"],
                tools=[],
                model="remote",
            ):
                pass

        consume_task = asyncio.create_task(consume())
        for _ in range(50):
            await asyncio.sleep(0.02)
            if expected.exists():
                break
        else:
            consume_task.cancel()
            raise AssertionError("disk mirror file never appeared")

        data = json.loads(expected.read_text())
        assert data["session_id"] == "deadbeef" + "0" * 24
        assert data["model"] == "remote"

        _http_post_json(
            f"http://127.0.0.1:{provider._port}/api/respond", {"text": "ok"}
        )
        await consume_task

    asyncio.run(run())


def test_no_disk_mirror_when_unbound(provider, tmp_path):
    """Without ``set_session_context``, no inbox file is written."""

    async def run() -> None:
        async def consume() -> None:
            async for _ in provider.stream(
                messages=[{"role": "user", "content": "hi"}],
                system=["s"],
                tools=[],
                model="remote",
            ):
                pass

        consume_task = asyncio.create_task(consume())
        for _ in range(20):
            await asyncio.sleep(0.02)
            code, _ = _http_get(
                f"http://127.0.0.1:{provider._port}/api/pending"
            )
            if code == 200:
                break
        _http_post_json(
            f"http://127.0.0.1:{provider._port}/api/respond", {"text": "ok"}
        )
        await consume_task

    asyncio.run(run())
    # No .alan/ created since we never bound a session.
    assert not (tmp_path / ".alan").exists()


def test_response_error_yields_stream_error(provider):
    from alancode.providers.base import StreamError

    events = _drive_one_stream(provider, {"error": "boom"})
    errors = [e for e in events if isinstance(e, StreamError)]
    assert len(errors) == 1
    assert errors[0].error == "boom"


# ── Backend resolution wiring ───────────────────────────────────────────────


def test_resolve_backend_picks_remote_provider():
    from alancode.agent import _resolve_backend

    p = _resolve_backend("scripted", model="remote")
    try:
        assert isinstance(p, RemoteScriptedProvider)
    finally:
        p.shutdown()


def test_resolve_backend_other_model_still_uses_classic_scripted():
    from alancode.agent import _resolve_backend
    from alancode.providers.scripted_provider import ScriptedProvider

    p = _resolve_backend("scripted", model="some-other-model")
    assert isinstance(p, ScriptedProvider)
    assert not isinstance(p, RemoteScriptedProvider)
