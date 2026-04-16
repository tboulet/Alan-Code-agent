"""GUIUI — browser-based SessionUI implementation.

Runs a FastAPI/uvicorn server and communicates with the browser
via WebSocket.  All I/O goes through the browser — no terminal input.

Usage::

    ui = GUIUI(agent, cwd="/path/to/project")
    await ui.start()       # Starts the server, prints URL
    await run_session(agent, ui)
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
from typing import TYPE_CHECKING, Any, Set

from rich.console import Console

from alancode.gui.base import SessionUI
from alancode.gui.serialization import agent_event_to_output

if TYPE_CHECKING:
    from fastapi import WebSocket

    from alancode.agent import AlanCodeAgent
    from alancode.messages.types import Message, StreamEvent, Usage

logger = logging.getLogger(__name__)


class GUIUI(SessionUI):
    """Browser GUI: all I/O via WebSocket.

    Parameters
    ----------
    agent : AlanCodeAgent
        The agent (for session info, abort, LLM perspective).
    cwd : str
        Working directory (for URL path and session listing).
    """

    # The chat panel renders the full conversation itself on resume.
    renders_conversation = True

    def __init__(self, agent: AlanCodeAgent, cwd: str = "") -> None:
        self._agent = agent
        self._cwd = cwd
        self._connections: Set[WebSocket] = set()
        self._pending_input: asyncio.Future[str] | None = None
        self._event_history: list[dict] = []  # For replay on reconnect
        self._console_instance = _GUIConsole(self)
        self.llm_perspective: list[dict] | None = None
        self.llm_system_prompt: str = ""
        self._last_tree_data: dict | None = None

    # ── Server lifecycle ──────────────────────────────────────────────────

    async def start(self, port: int | None = None) -> str:
        """Start the FastAPI server. Returns the URL."""
        from alancode.gui.server import start_gui_server

        url, self._server, self._server_task = await start_gui_server(
            gui_ui=self, cwd=self._cwd, port=port,
        )
        return url

    async def stop(self) -> None:
        """Request a clean shutdown of the uvicorn server."""
        server = getattr(self, "_server", None)
        task = getattr(self, "_server_task", None)
        if server is None or task is None:
            return
        # Close all live websockets so uvicorn's graceful shutdown can
        # complete immediately instead of waiting for browser disconnects.
        for ws in list(self._connections):
            try:
                await ws.close()
            except Exception:
                pass
        self._connections.clear()
        server.should_exit = True
        try:
            await asyncio.wait_for(task, timeout=3.0)
        except asyncio.TimeoutError:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        except (asyncio.CancelledError, Exception):
            pass

    # ── WebSocket connection management ───────────────────────────────────

    def add_connection(self, ws: WebSocket) -> None:
        self._connections.add(ws)
        logger.info("GUI client connected (%d total)", len(self._connections))

    def remove_connection(self, ws: WebSocket) -> None:
        self._connections.discard(ws)
        logger.info("GUI client disconnected (%d total)", len(self._connections))

    async def send_to_all(self, msg: str) -> None:
        """Send a text message to all connected browsers."""
        dead: set[WebSocket] = set()
        for ws in self._connections:
            try:
                await ws.send_text(msg)
            except Exception:
                dead.add(ws)
        self._connections -= dead

    async def _send_event(self, event_type: str, data: dict) -> None:
        """Send a structured event to all browsers."""
        msg = json.dumps({
            "kind": "event",
            "event": {"type": event_type, "data": data},
        }, default=str)
        self._event_history.append(json.loads(msg))
        await self.send_to_all(msg)

    # ── Replay (for clients connecting mid-session) ───────────────────────

    async def send_history(self, ws: WebSocket) -> None:
        """Send full event history to a newly connected client.

        Starts with a ``reset`` event so the browser discards any DOM it
        kept from a previous (now-dead) session before re-rendering — this
        is what prevents duplicated messages across ``alancode`` restarts.
        """
        await ws.send_text(json.dumps({
            "kind": "event",
            "event": {"type": "reset", "data": {}},
        }))
        for entry in self._event_history:
            await ws.send_text(json.dumps(entry, default=str))

        # Send LLM perspective if available
        if self.llm_perspective:
            await ws.send_text(json.dumps({
                "kind": "event",
                "event": {
                    "type": "llm_perspective",
                    "data": {
                        "messages": self.llm_perspective,
                        "system_prompt": self.llm_system_prompt,
                    },
                },
            }, default=str))

        # Send git tree data if available
        if self._last_tree_data:
            await ws.send_text(json.dumps({
                "kind": "event",
                "event": {
                    "type": "git_tree_update",
                    "data": self._last_tree_data,
                },
            }, default=str))

        # If we're currently waiting for input, notify the new client
        if self._pending_input and not self._pending_input.done():
            await ws.send_text(json.dumps({
                "kind": "input_request",
                "request": {"type": "prompt", "question": "> ", "options": []},
            }))

    # ── SessionUI: Input ──────────────────────────────────────────────────

    # Time to wait for a browser to connect before giving up on input.
    # Long enough for a human to open the URL; short enough not to hang
    # indefinitely on a closed firewall or wrong URL.
    BROWSER_CONNECT_TIMEOUT = 120.0

    async def _wait_for_browser_or_fail(self) -> None:
        """Block until at least one browser is connected, or timeout.

        On timeout, raises ``TimeoutError`` with actionable guidance so
        the caller can surface it to the user instead of hanging forever.
        """
        if self._connections:
            return
        start = asyncio.get_running_loop().time()
        while not self._connections:
            if asyncio.get_running_loop().time() - start > self.BROWSER_CONNECT_TIMEOUT:
                raise TimeoutError(
                    "No browser connected after "
                    f"{int(self.BROWSER_CONNECT_TIMEOUT)}s. Open the GUI "
                    "URL printed at startup, or run `alancode` without "
                    "`--gui` for the terminal UI."
                )
            await asyncio.sleep(0.1)

    async def get_input(self, prompt: str = "\n> ") -> str:
        """Wait for user input from the browser."""
        await self._wait_for_browser_or_fail()

        await self.send_to_all(json.dumps({
            "kind": "input_request",
            "request": {"type": "prompt", "question": prompt, "options": []},
        }))

        loop = asyncio.get_running_loop()
        self._pending_input = loop.create_future()
        try:
            return await self._pending_input
        finally:
            self._pending_input = None

    async def ask_user(self, question: str, options: list[str]) -> str:
        """Ask the user a question with options via the browser."""
        await self._wait_for_browser_or_fail()

        await self.send_to_all(json.dumps({
            "kind": "input_request",
            "request": {
                "type": "ask",
                "question": question,
                "options": options,
            },
        }))

        loop = asyncio.get_running_loop()
        self._pending_input = loop.create_future()
        try:
            return await self._pending_input
        finally:
            self._pending_input = None

    def submit_input(self, value: str) -> bool:
        """Called by the WebSocket handler when the browser sends input."""
        if self._pending_input is None or self._pending_input.done():
            return False
        self._pending_input.set_result(value)
        return True

    # ── SessionUI: Agent event output ─────────────────────────────────────

    async def on_agent_event(self, event: StreamEvent | Message) -> None:
        output = agent_event_to_output(event)
        await self._send_event(output.type, output.data)

    async def on_cost(
        self, usage: Usage, cost_usd: float, cost_unknown: bool,
        conversation_tokens: int = 0, context_window: int = 0,
    ) -> None:
        await self._send_event("cost_summary", {
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cache_read_tokens": usage.cache_read_input_tokens,
            "cache_write_tokens": usage.cache_creation_input_tokens,
            "cost_usd": cost_usd,
            "cost_unknown": cost_unknown,
            "conversation_tokens": conversation_tokens,
            "context_window": context_window,
        })

    # ── SessionUI: Lifecycle ──────────────────────────────────────────────

    def on_agent_start(self) -> None:
        asyncio.ensure_future(self.send_to_all(json.dumps({"kind": "agent_start"})))

    def on_agent_done(self) -> None:
        asyncio.ensure_future(self.send_to_all(json.dumps({"kind": "agent_done"})))

    def reset_stream_state(self, assume_thinking: bool = False) -> None:
        pass  # GUI doesn't have a streaming state machine

    # ── SessionUI: Console ────────────────────────────────────────────────

    @property
    def console(self) -> Console:
        return self._console_instance

    # ── LLM Perspective ───────────────────────────────────────────────────

    def set_llm_perspective(
        self,
        api_messages: list[dict],
        system_prompt: list[str] | None = None,
    ) -> None:
        """Store LLM perspective and send to browser."""
        self.llm_perspective = api_messages
        self.llm_system_prompt = "\n\n".join(system_prompt) if system_prompt else ""
        asyncio.ensure_future(self._send_event("llm_perspective", {
            "messages": api_messages,
            "system_prompt": self.llm_system_prompt,
        }))

    # ── Initial data ───────────────────────────────────────────────────

    def on_initial_conversation(self, messages: list) -> None:
        """Send existing conversation to browser chat panel."""
        async def _send():
            for msg in messages[-100:]:  # Cap at 100 messages
                try:
                    output = agent_event_to_output(msg)
                    await self._send_event(output.type, output.data)
                except Exception:
                    pass
        asyncio.ensure_future(_send())

    def on_initial_system_prompt(self, system_prompt: str) -> None:
        """Send system prompt to LLM Perspective panel."""
        self.llm_system_prompt = system_prompt
        asyncio.ensure_future(self._send_event("llm_perspective", {
            "messages": [],
            "system_prompt": system_prompt,
        }))

    # ── Git Tree (AGT) ───────────────────────────────────────────────────

    def on_git_tree_update(self, tree_data: dict) -> None:
        """Send git tree layout to all browsers."""
        self._last_tree_data = tree_data
        asyncio.ensure_future(self._send_event("git_tree_update", tree_data))

    # ── Handle incoming WebSocket messages ────────────────────────────────

    async def handle_ws_message(self, data: dict) -> None:
        """Process a message from the browser.

        Message kinds:
        - ``input_response`` / ``prompt``: user input (when waiting)
        - ``inject``: "btw" message injected mid-turn (appended at next iteration)
        - ``abort``: stop the agent (equivalent to Ctrl+C)
        """
        kind = data.get("kind", "")

        if kind in ("input_response", "prompt"):
            value = data.get("value") or data.get("text", "")
            self.submit_input(str(value))

        elif kind == "inject":
            # "BTW" message — injected into the agent's queue, picked up
            # at the start of the next loop iteration.
            value = data.get("text", "")
            if value:
                self._agent.inject_message(str(value))
                await self._send_event("system_message", {
                    "content": f"Message queued: {value[:80]}",
                    "level": "info",
                    "subtype": "informational",
                    "hide_in_ui": False,
                })

        elif kind == "abort":
            self._agent.abort()

        else:
            logger.debug("Unknown WS message kind: %s", kind)


# ── GUIConsole ────────────────────────────────────────────────────────────


class _GUIConsole(Console):
    """Rich Console subclass that sends output to the browser ONLY.

    Does NOT write to the terminal.  Captures Rich output as plain
    text and sends it via WebSocket.
    """

    def __init__(self, gui_ui: GUIUI) -> None:
        self._buf = io.StringIO()
        super().__init__(file=self._buf, width=120, no_color=True)
        self._gui_ui = gui_ui

    def print(self, *objects: Any, **kwargs: Any) -> None:
        self._buf.truncate(0)
        self._buf.seek(0)
        super().print(*objects, **kwargs)
        text = self._buf.getvalue().rstrip()
        self._buf.truncate(0)
        self._buf.seek(0)

        if text:
            try:
                asyncio.ensure_future(
                    self._gui_ui._send_event("local_output", {"text": text})
                )
            except RuntimeError:
                pass
