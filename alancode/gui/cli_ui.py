"""CLIUI — terminal-based SessionUI implementation.

Uses Rich for display and prompt_toolkit for input.
This is the default UI when ``--gui`` is not passed.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from rich.console import Console

from alancode.cli.display import (
    _reset_stream_state,
    display_event,
    display_replay,
)
from alancode.gui.base import SessionUI
from alancode.messages.types import Message, StreamEvent, Usage


class CLIUI(SessionUI):
    """Terminal UI: Rich console + prompt_toolkit input."""

    # The CLI now replays the tail of the conversation on resume
    # (see on_initial_conversation below), so the REPL can skip the
    # "Last exchange" text summary.
    renders_conversation = True

    def __init__(self) -> None:
        self._console = Console()

        # Set up prompt-toolkit with persistent history.
        # Enter = submit, Alt+Enter (Esc then Enter) = newline.
        from prompt_toolkit.key_binding import KeyBindings

        kb = KeyBindings()

        @kb.add("escape", "enter")
        def _insert_newline(event):
            event.current_buffer.insert_text("\n")

        history_path = Path.home() / ".alan" / "history"
        history_path.parent.mkdir(parents=True, exist_ok=True)
        self._session: PromptSession[str] = PromptSession(
            history=FileHistory(str(history_path)),
            key_bindings=kb,
        )
        # Separate session for permission prompts — no history, no custom
        # keybindings, but same prompt_toolkit machinery so Ctrl+C unwinds
        # cleanly instead of orphaning a blocked input() against stdin.
        self._ask_session: PromptSession[str] = PromptSession()

    # ── Input ─────────────────────────────────────────────────────────────

    async def get_input(self, prompt: str = "\n> ") -> str:
        loop = asyncio.get_running_loop()
        text = await loop.run_in_executor(
            None, lambda: self._session.prompt(prompt)
        )
        return text.strip()

    async def ask_user(self, question: str, options: list[str]) -> str:
        from alancode.cli.user_input import ask_user_cli

        return await ask_user_cli(question, options, session=self._ask_session)

    # ── Agent event output ────────────────────────────────────────────────

    def on_initial_conversation(self, messages: list) -> None:
        """Replay the tail of a resumed conversation to the terminal."""
        display_replay(messages, self._console, limit=100)

    async def on_agent_event(self, event: StreamEvent | Message) -> None:
        display_event(event, self._console)

    async def on_cost(
        self, usage: Usage, cost_usd: float, cost_unknown: bool,
        conversation_tokens: int = 0, context_window: int = 0,
    ) -> None:
        # Session cost
        parts = [f"  [dim]Session: {usage.total_input:,} in + {usage.output_tokens:,} out"]
        if not cost_unknown:
            parts.append(f"= ${cost_usd:.4f} (estimated)")
        # Conversation tokens
        if context_window > 0 and conversation_tokens > 0:
            pct = conversation_tokens * 100 // context_window
            parts.append(
                f"| Conversation: {conversation_tokens:,} / {context_window:,} ({pct}%)"
            )
        self._console.print(" ".join(parts) + "[/dim]")

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def reset_stream_state(self, assume_thinking: bool = False) -> None:
        _reset_stream_state(assume_thinking=assume_thinking)

    # ── Console ───────────────────────────────────────────────────────────

    @property
    def console(self) -> Console:
        return self._console
