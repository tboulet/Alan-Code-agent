"""Tool system base types.

Every tool in Alan Code implements the Tool ABC.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from alancode.messages.types import AssistantMessage, Message


@dataclass
class ToolResult:
    """Result of a tool execution."""
    data: Any  # The tool's output (usually str)
    is_error: bool = False
    # Additional messages to inject after the result
    new_messages: list = field(default_factory=list)


@dataclass
class ToolUseContext:
    """Context passed to every tool execution.
    Carries all state needed for tool calls."""
    cwd: str
    messages: list  # Current conversation history
    settings: dict = None  # type: ignore[assignment]  # Full settings dict (for hooks, etc.)
    abort_signal: Any = None  # asyncio.Event or similar
    agent_id: str | None = None  # Non-null for subagents
    verbose: bool = False
    ask_user_callback: Callable[[str, list[str]], Awaitable[str]] | None = None
    session_state: Any = None  # SessionState instance (for AGT tools)


class Tool(ABC):
    """Abstract base class for all tools.

    Subclasses declare ``name``, ``description``, ``input_schema`` and
    implement ``call``. The agent loop finds tools via the registry, sends
    their schemas to the model, then dispatches tool_use blocks back to
    ``call``.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Canonical tool name. Used in schemas and tool_use blocks."""
        ...

    @property
    def aliases(self) -> list[str]:
        """Alternative names the tool also responds to.

        Returns:
            List of extra names. Default: empty list.
        """
        return []

    @property
    @abstractmethod
    def description(self) -> str:
        """Prose description shown to the model in the tool schema."""
        ...

    @property
    @abstractmethod
    def input_schema(self) -> dict[str, Any]:
        """JSON Schema (OpenAI-compatible) for the tool's input parameters."""
        ...

    @abstractmethod
    async def call(self, args: dict[str, Any], context: ToolUseContext) -> ToolResult:
        """Execute the tool with the given arguments.

        Args:
            args: The validated input dict the model provided.
            context: Session-scoped state (cwd, messages, abort signal, etc.).

        Returns:
            A :class:`ToolResult` — use ``is_error=True`` to signal failure
            to the model without raising an exception.
        """
        ...

    def permission_level(self, args: dict[str, Any]) -> Literal["read", "write", "exec"]:
        """Permission level for this invocation.

        - ``"read"``  — read-only, can run concurrently, always allowed
        - ``"write"`` — mutates files, runs serially, needs permission in ``safe`` mode
        - ``"exec"``  — arbitrary execution (Bash), runs serially, needs permission
          in both ``edit`` and ``safe`` modes

        Args:
            args: The same args that would be passed to :meth:`call`.

        Returns:
            One of ``"read"``, ``"write"``, ``"exec"``. Default: ``"write"``.
        """
        return "write"

    def is_enabled(self) -> bool:
        """Whether this tool is currently available.

        Returns:
            ``True`` to expose the tool to the model, ``False`` to hide it.
        """
        return True

    def validate_input(self, args: dict[str, Any], context: ToolUseContext) -> str | None:
        """Validate input beyond JSON Schema.

        Use for semantic checks the schema can't express (e.g. "file_path
        must exist", "options list must have ≥1 item").

        Args:
            args: The input dict from the model.
            context: Session state.

        Returns:
            A human-readable error message (sent back to the model as the
            tool result), or ``None`` if the input is valid.
        """
        return None

    @property
    def max_result_size_chars(self) -> int | float:
        """Maximum result size in characters before disk persistence.

        Returns:
            Size cap. Use ``float('inf')`` to disable disk persistence.
        """
        return 50_000

    def matches_name(self, name: str) -> bool:
        """Check if this tool responds to ``name`` (primary or alias).

        Args:
            name: The name the model used in its tool_use block.

        Returns:
            ``True`` if the name matches.
        """
        return name == self.name or name in self.aliases

    def to_schema(self) -> dict[str, Any]:
        """Convert to API tool schema format.

        Injects ``additionalProperties: false`` into the input_schema if
        the tool doesn't already set it. This makes the API reject calls
        that include unknown fields instead of silently dropping them —
        the model gets a clear error and self-corrects next turn.
        A tool can override by setting ``additionalProperties: true`` in
        its own schema if it genuinely accepts open-ended input.
        """
        schema = dict(self.input_schema)
        if schema.get("type") == "object" and "additionalProperties" not in schema:
            schema["additionalProperties"] = False
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": schema,
        }
