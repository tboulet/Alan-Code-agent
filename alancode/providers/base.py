"""LLM Provider abstraction layer.

The agentic loop only interacts with LLMProvider.
Each implementation translates its native API format into StreamEvents.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncGenerator, Any


# ── Stream events — yielded by LLMProvider.stream() ─────────────────────────


@dataclass
class StreamTextDelta:
    """Incremental text content."""
    text: str
    type: str = "text_delta"


@dataclass
class StreamToolUseStart:
    """Start of a tool call."""
    id: str
    name: str
    type: str = "tool_use_start"


@dataclass
class StreamToolUseInputDelta:
    """Incremental tool input JSON."""
    id: str
    partial_json: str
    type: str = "input_json_delta"


@dataclass
class StreamToolUseStop:
    """End of a tool call with complete input."""
    id: str
    name: str
    input: dict[str, Any]
    type: str = "tool_use_stop"


@dataclass
class StreamThinkingDelta:
    """Incremental thinking content."""
    thinking: str
    type: str = "thinking_delta"


@dataclass
class StreamMessageStart:
    """Start of a new message from the model."""
    model: str
    request_id: str | None = None
    usage: dict[str, int] | None = None
    type: str = "message_start"


@dataclass
class StreamMessageDelta:
    """Message-level metadata update (stop_reason, usage)."""
    stop_reason: str | None = None
    usage: dict[str, int] | None = None
    type: str = "message_delta"


@dataclass
class StreamMessageStop:
    """End of the message."""
    type: str = "message_stop"


@dataclass
class StreamError:
    """An error during streaming."""
    error: str
    error_type: str = "api_error"  # 'api_error', 'overloaded', 'invalid_request'
    status_code: int | None = None
    type: str = "error"


# Union of all stream events
ProviderStreamEvent = (
    StreamTextDelta
    | StreamToolUseStart
    | StreamToolUseInputDelta
    | StreamToolUseStop
    | StreamThinkingDelta
    | StreamMessageStart
    | StreamMessageDelta
    | StreamMessageStop
    | StreamError
)


# ── Model & tool configuration ──────────────────────────────────────────────


@dataclass
class ModelInfo:
    """Information about a model's capabilities."""
    context_window: int = 200_000
    max_output_tokens: int = 8_192
    supports_thinking: bool = False


@dataclass
class ToolSchema:
    """Tool definition sent to the provider."""
    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass
class ThinkingConfig:
    """Thinking mode configuration."""
    type: str = "disabled"  # 'disabled', 'adaptive', 'budget'
    budget_tokens: int | None = None


# ── Abstract base class ─────────────────────────────────────────────────────


class LLMProvider(ABC):
    """Abstract interface for any LLM backend.

    Implementations must translate their native streaming API into a sequence
    of ``ProviderStreamEvent`` dataclasses so the agentic loop can remain
    provider-agnostic.
    """

    @abstractmethod
    async def stream(
        self,
        messages: list[dict[str, Any]],  # API-format messages
        system: list[str],
        tools: list[ToolSchema],
        *,
        model: str | None = None,
        max_tokens: int | None = None,
        thinking: ThinkingConfig | None = None,
        stop_sequences: list[str] | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[ProviderStreamEvent, None]:
        """Stream a response from the LLM.

        Yields ``ProviderStreamEvent`` instances in the following order::

            StreamMessageStart          # always first: model name, request ID
            StreamTextDelta*            # zero or more text chunks
            StreamThinkingDelta*        # zero or more thinking chunks (if model supports it)
            StreamToolUseStart          # begins a tool call (id + name)
            StreamToolUseInputDelta*    # partial JSON of tool input
            StreamToolUseStop           # ends tool call with parsed input dict
            ... (more text/tool blocks possible) ...
            StreamMessageDelta          # always near-last: stop_reason + final usage
            StreamMessageStop           # always last: signals end of message

        On error at any point, yields ``StreamError`` and returns.

        Args:
            messages: Conversation history as dicts (format depends on provider).
            system: System prompt sections (joined by provider as needed).
            tools: Tool definitions the model can call.
            model: Override the provider's default model.
            max_tokens: Maximum output tokens for this request.
            thinking: Extended thinking configuration.
            stop_sequences: Custom stop sequences.
        """
        ...  # pragma: no cover

    @abstractmethod
    def get_model_info(self, model: str | None = None) -> ModelInfo:
        """Return information about the model's capabilities."""
        ...  # pragma: no cover

