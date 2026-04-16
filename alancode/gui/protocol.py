"""Frontend protocol — types shared between bridge and all frontends.

Every frontend (CLI, WebSocket GUI, future frontends) implements the
:class:`Frontend` ABC.  All data flowing through the bridge uses
:class:`OutputEvent` (agent → frontends) and :class:`InputRequest`
(frontends → agent).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


@dataclass
class OutputEvent:
    """A serializable event flowing from the agent to frontends.

    The ``type`` field determines how the frontend renders the event.
    The ``data`` dict carries the payload (already JSON-serializable).
    """

    type: str
    """Event type. Values include:

    - ``"assistant_delta"`` — streaming text/thinking chunk (hide_in_api=True)
    - ``"assistant_message"`` — final assembled assistant message
    - ``"user_message"`` — user message (tool results, system reminders)
    - ``"system_message"`` — system-level info/warning/error
    - ``"attachment_message"`` — contextual injection
    - ``"progress_message"`` — real-time tool progress
    - ``"request_start"`` — new API call starting
    - ``"cost_summary"`` — token usage and cost after a turn
    - ``"llm_perspective"`` — snapshot of messages sent to the API
    - ``"local_output"`` — slash command output (Rich renderables as text)
    - ``"welcome"`` — session welcome banner
    - ``"error"`` — error display
    """

    data: dict[str, Any]
    """Event payload (JSON-serializable). Structure depends on ``type``."""

    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    original: Any = field(default=None, repr=False)
    """Original agent event object (not serialized). Used by CLIFrontend
    to delegate to display.py which expects Message dataclasses."""


@dataclass
class InputRequest:
    """A request for user input, broadcast to all frontends.

    The first frontend to respond via ``bridge.submit_input()`` wins.
    """

    id: str = field(default_factory=lambda: uuid4().hex[:12])
    """Unique request ID for correlation."""

    type: str = "prompt"
    """Request type:

    - ``"prompt"`` — main REPL input (free text)
    - ``"ask"`` — question with options (permissions, AskUserQuestion tool)
    - ``"confirm"`` — yes/no confirmation
    """

    question: str = "> "
    """The prompt text or question to display."""

    options: list[str] = field(default_factory=list)
    """Available options. Empty list means free-text input."""


class Frontend(ABC):
    """Protocol that every UI frontend must implement.

    Frontends are registered on a :class:`FrontendBridge` and receive
    all output events and input requests.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name for this frontend (e.g., ``"cli"``, ``"ws"``)."""
        ...

    @abstractmethod
    async def on_event(self, event: OutputEvent) -> None:
        """Handle an output event (display it, send over WebSocket, etc.)."""
        ...

    @abstractmethod
    async def on_input_request(self, request: InputRequest) -> None:
        """A new input request is pending.  Show the prompt to the user."""
        ...

    @abstractmethod
    async def on_input_resolved(self, source: str, value: str) -> None:
        """Another frontend responded to the input request.

        The implementor should cancel any local prompt and optionally
        display a notification that input was provided elsewhere.
        """
        ...
