"""SessionUI — abstract interface for all session I/O.

Both CLI and GUI implement this.  The session loop (``run_session``)
is completely UI-agnostic: it works identically with either implementation.

Slash commands receive ``ui.console`` (a Rich Console or GUIConsole) and
call ``console.print()`` — no changes needed to any slash command code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rich.console import Console

    from alancode.messages.types import Message, StreamEvent, Usage


class SessionUI(ABC):
    """Abstract interface for session I/O.

    Implementations: :class:`CLIUI` (terminal) and :class:`GUIUI` (browser).
    """

    # True if this UI re-renders the full conversation itself (e.g. via
    # ``on_initial_conversation``), so the REPL can skip the textual
    # "Session resumed / Last exchange" summary and avoid duplicating info.
    renders_conversation: bool = False

    # ── Input ─────────────────────────────────────────────────────────────

    @abstractmethod
    async def get_input(self, prompt: str = "\n> ") -> str:
        """Wait for user input (main prompt or free text).

        Returns the text entered by the user (stripped).
        Raises ``EOFError`` on Ctrl+D / disconnect.
        """
        ...

    @abstractmethod
    async def ask_user(self, question: str, options: list[str]) -> str:
        """Ask the user a question with options (permissions, AskUserQuestion).

        Returns the selected option text or custom user input.
        """
        ...

    # ── Agent event output ────────────────────────────────────────────────

    @abstractmethod
    async def on_agent_event(self, event: StreamEvent | Message) -> None:
        """Display an event from the agent (streaming delta, tool call, etc.)."""
        ...

    @abstractmethod
    async def on_cost(
        self,
        usage: Usage,
        cost_usd: float,
        cost_unknown: bool,
        conversation_tokens: int = 0,
        context_window: int = 0,
    ) -> None:
        """Display cost summary after a turn.

        Parameters
        ----------
        usage : Usage
            Cumulative session token usage.
        cost_usd : float
            Cumulative estimated cost.
        cost_unknown : bool
            True if pricing is unavailable.
        conversation_tokens : int
            Current conversation size in estimated tokens.
        context_window : int
            Model's context window size.
        """
        ...

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def on_agent_start(self) -> None:
        """Called when the agent starts processing a turn.

        GUI uses this to disable input and show the Stop button.
        CLI is a no-op.
        """

    def on_agent_done(self) -> None:
        """Called when the agent finishes processing a turn.

        GUI uses this to re-enable input.
        CLI is a no-op.
        """

    def reset_stream_state(self, assume_thinking: bool = False) -> None:
        """Reset streaming display state before a new turn.

        CLI resets the ``<think>``/``<tool_call>`` tag filter state machine.
        GUI is a no-op.
        """

    # ── Initial data (sent once at session start) ─────────────────────────

    def on_initial_conversation(self, messages: list) -> None:
        """Send existing conversation messages to the UI at session start.

        GUI renders them in the chat panel. CLI is a no-op (already on screen).
        """

    def on_initial_system_prompt(self, system_prompt: str) -> None:
        """Send the system prompt to the LLM Perspective panel at start.

        GUI displays it. CLI is a no-op.
        """

    # ── Git Tree (AGT) ──────────────────────────────────────────────────────

    def on_git_tree_update(self, tree_data: dict) -> None:
        """Called when the git tree layout should be refreshed.

        GUI sends the data to the browser via WebSocket.
        CLI and ScriptedUI are no-ops.
        """

    # ── Console (for slash commands) ──────────────────────────────────────

    @property
    @abstractmethod
    def console(self) -> Console:
        """Rich Console used by slash commands for output.

        CLI: real Rich Console (beautiful tables, syntax highlighting).
        GUI: GUIConsole (renders to text, sends via WebSocket).
        """
        ...
