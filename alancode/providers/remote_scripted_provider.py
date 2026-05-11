"""Remote-controlled scripted provider.

A test/impersonation backend that hosts a small HTTP server and waits for an
external caller (human or another agent) to post the assistant response.

Selected via::

    AlanCodeAgent(backend="scripted", model="remote")

or on the CLI::

    alancode --backend scripted --model remote

Endpoints (host: ``127.0.0.1``, port chosen at startup and printed to stdout):

- ``GET  /api/health``      — ``{"ok": true}`` once the server is up.
- ``GET  /api/session``     — session metadata (id, cwd, model).
- ``GET  /api/pending``     — current pending LLM call payload, or ``204``.
- ``POST /api/respond``     — submit assistant response, unblocks ``stream()``.

The same pending payload is also mirrored to disk at
``<cwd>/.alan/sessions/<session_id>/remote_inbox.json`` once the agent has
bound its session context (see ``set_session_context``). That mirror is
read-only — writes go through ``POST /api/respond``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import socket
import threading
import uuid as _uuid
from concurrent.futures import Future as ConcurrentFuture
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, AsyncGenerator

from alancode.providers.base import (
    LLMProvider,
    ModelInfo,
    ProviderStreamEvent,
    StreamError,
    StreamMessageDelta,
    StreamMessageStart,
    StreamMessageStop,
    StreamTextDelta,
    StreamThinkingDelta,
    StreamToolUseInputDelta,
    StreamToolUseStart,
    StreamToolUseStop,
    ThinkingConfig,
    ToolSchema,
)

logger = logging.getLogger(__name__)


_DEFAULT_PORT = 8430
_MAX_PORT_ATTEMPTS = 20


def _find_available_port(start: int = _DEFAULT_PORT, attempts: int = _MAX_PORT_ATTEMPTS) -> int:
    for port in range(start, start + attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError(
        f"No available port found in range {start}-{start + attempts - 1}"
    )


def _serialize_tool(tool: ToolSchema) -> dict[str, Any]:
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": tool.input_schema,
    }


def _serialize_thinking(t: ThinkingConfig | None) -> dict[str, Any] | None:
    if t is None:
        return None
    return {"type": t.type, "budget_tokens": t.budget_tokens}


class RemoteScriptedProvider(LLMProvider):
    """Provider that delegates every LLM call to an external HTTP caller.

    On each ``stream()``, the call payload is exposed via ``GET /api/pending``
    and the method blocks until ``POST /api/respond`` is received. The response
    JSON is translated into the same provider-stream-event sequence the
    regular ``ScriptedProvider`` emits.
    """

    def __init__(self, *, port: int | None = None) -> None:
        self._port = port if port is not None else _find_available_port()
        self._session_id: str | None = None
        self._cwd: str | None = None
        self._model: str | None = None
        self._call_count = 0

        # Cross-thread comm: the HTTP handler thread completes this future
        # to deliver the response back to the awaiting ``stream()`` coroutine.
        self._pending_payload: dict[str, Any] | None = None
        self._pending_lock = threading.Lock()
        self._pending_future: ConcurrentFuture[dict[str, Any]] | None = None

        self._server = self._build_server()
        self._server_thread = threading.Thread(
            target=self._server.serve_forever,
            name=f"remote-scripted-{self._port}",
            daemon=True,
        )
        self._server_thread.start()
        logger.info(
            "remote-scripted provider listening at http://127.0.0.1:%d", self._port,
        )
        print(
            f"[remote-scripted] LLM endpoint: http://127.0.0.1:{self._port}",
            flush=True,
        )

    # ── Public bind hook (called by AlanCodeAgent after session init) ───

    def set_session_context(self, *, session_id: str, cwd: str) -> None:
        """Bind the provider to the agent's session for the on-disk mirror."""
        self._session_id = session_id
        self._cwd = cwd
        print(
            f"[remote-scripted] bound to session {session_id[:8]} (cwd={cwd})",
            flush=True,
        )

    # ── LLMProvider interface ─────────────────────────────────────────────

    async def stream(
        self,
        messages: list[dict[str, Any]],
        system: list[str],
        tools: list[ToolSchema],
        *,
        model: str | None = None,
        max_tokens: int | None = None,
        thinking: ThinkingConfig | None = None,
        stop_sequences: list[str] | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[ProviderStreamEvent, None]:
        turn = self._call_count
        self._call_count += 1
        self._model = model

        request_id = f"remote-req-{turn}-{_uuid.uuid4().hex[:8]}"
        payload: dict[str, Any] = {
            "request_id": request_id,
            "turn": turn,
            "model": model,
            "max_tokens": max_tokens,
            "thinking": _serialize_thinking(thinking),
            "stop_sequences": stop_sequences,
            "system": system,
            "messages": messages,
            "tools": [_serialize_tool(t) for t in tools],
            "session_id": self._session_id,
            "cwd": self._cwd,
        }

        future: ConcurrentFuture[dict[str, Any]] = ConcurrentFuture()
        with self._pending_lock:
            self._pending_payload = payload
            self._pending_future = future
        self._mirror_to_disk(payload)

        try:
            response = await asyncio.wrap_future(future)
        finally:
            with self._pending_lock:
                self._pending_payload = None
                self._pending_future = None

        # Translate response → stream events
        if isinstance(response, dict) and response.get("error"):
            yield StreamError(
                error=str(response["error"]),
                error_type=str(response.get("error_type", "api_error")),
                status_code=response.get("status_code"),
            )
            return

        yield StreamMessageStart(model=model or "remote", request_id=request_id)

        thinking_text = response.get("thinking") if isinstance(response, dict) else None
        if thinking_text:
            yield StreamThinkingDelta(thinking=str(thinking_text))

        text_content = response.get("text") if isinstance(response, dict) else None
        if text_content:
            yield StreamTextDelta(text=str(text_content))

        tool_calls = response.get("tool_calls") if isinstance(response, dict) else None
        if tool_calls:
            for tc in tool_calls:
                tool_id = tc.get("id") or f"toolu_{_uuid.uuid4().hex[:16]}"
                tool_name = tc["name"]
                tool_input = tc.get("input", {})
                yield StreamToolUseStart(id=tool_id, name=tool_name)
                yield StreamToolUseInputDelta(
                    id=tool_id, partial_json=json.dumps(tool_input),
                )
                yield StreamToolUseStop(
                    id=tool_id, name=tool_name, input=tool_input,
                )

        stop_reason = response.get("stop_reason") if isinstance(response, dict) else None
        if stop_reason is None:
            stop_reason = "tool_use" if tool_calls else "end_turn"
        usage = response.get("usage") if isinstance(response, dict) else None
        if not isinstance(usage, dict):
            usage = {"input_tokens": 0, "output_tokens": 0}

        yield StreamMessageDelta(stop_reason=stop_reason, usage=usage)
        yield StreamMessageStop()

    def get_model_info(self, model: str | None = None) -> ModelInfo:
        return ModelInfo(context_window=200_000, max_output_tokens=8_192)

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def shutdown(self) -> None:
        """Stop the HTTP server and join its thread."""
        try:
            self._server.shutdown()
        except Exception:
            pass
        try:
            self._server.server_close()
        except Exception:
            pass

    def __del__(self) -> None:
        try:
            self.shutdown()
        except Exception:
            pass

    # ── Internals ─────────────────────────────────────────────────────────

    def _mirror_to_disk(self, payload: dict[str, Any]) -> None:
        if not self._cwd or not self._session_id:
            return
        target = (
            Path(self._cwd) / ".alan" / "sessions" / self._session_id
            / "remote_inbox.json"
        )
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload, default=str, indent=2))
            tmp.replace(target)
        except OSError as exc:
            logger.warning("Could not mirror pending payload to %s: %s", target, exc)

    def _build_server(self) -> ThreadingHTTPServer:
        provider = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt: str, *args: Any) -> None:
                logger.debug("[remote-scripted http] " + fmt, *args)

            def _send_json(self, code: int, body: Any) -> None:
                data = json.dumps(body, default=str).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _send_empty(self, code: int) -> None:
                self.send_response(code)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def do_GET(self) -> None:
                if self.path == "/api/health":
                    self._send_json(200, {"ok": True})
                    return
                if self.path == "/api/session":
                    self._send_json(200, {
                        "session_id": provider._session_id,
                        "cwd": provider._cwd,
                        "model": provider._model,
                        "port": provider._port,
                        "calls_served": provider._call_count,
                    })
                    return
                if self.path == "/api/pending":
                    with provider._pending_lock:
                        payload = provider._pending_payload
                    if payload is None:
                        self._send_empty(204)
                    else:
                        self._send_json(200, payload)
                    return
                self._send_json(404, {"error": f"unknown path {self.path}"})

            def do_POST(self) -> None:
                if self.path != "/api/respond":
                    self._send_json(404, {"error": f"unknown path {self.path}"})
                    return
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length) if length else b""
                try:
                    body = json.loads(raw) if raw else {}
                except json.JSONDecodeError as e:
                    self._send_json(400, {"error": f"invalid JSON: {e}"})
                    return
                with provider._pending_lock:
                    fut = provider._pending_future
                if fut is None or fut.done():
                    self._send_json(409, {"error": "no pending call to respond to"})
                    return
                fut.set_result(body if isinstance(body, dict) else {})
                self._send_json(200, {"accepted": True})

        return ThreadingHTTPServer(("127.0.0.1", self._port), Handler)
