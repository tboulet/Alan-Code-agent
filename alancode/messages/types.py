"""Message types for Alan Code.

These dataclasses represent all messages flowing through the system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal
from uuid import UUID, uuid4


# ── Content block types ──────────────────────────────────────────────────────


@dataclass
class TextBlock:
    """A text content block."""
    text: str
    type: Literal["text"] = "text"


@dataclass
class ToolUseBlock:
    """A tool call request from the model."""
    id: str
    name: str
    input: dict[str, Any]
    type: Literal["tool_use"] = "tool_use"


@dataclass
class ToolResultBlock:
    """A tool execution result."""
    tool_use_id: str
    content: str | list[TextBlock]
    is_error: bool = False
    type: Literal["tool_result"] = "tool_result"


@dataclass
class ThinkingBlock:
    """Extended thinking content (model's internal reasoning)."""
    thinking: str
    signature: str = ""
    type: Literal["thinking"] = "thinking"


@dataclass
class RedactedThinkingBlock:
    """Redacted thinking content."""
    data: str
    type: Literal["redacted_thinking"] = "redacted_thinking"


@dataclass
class ImageBlock:
    """An image content block."""
    source: dict[str, Any]
    type: Literal["image"] = "image"


# Union of all content block types
ContentBlock = (
    TextBlock
    | ToolUseBlock
    | ToolResultBlock
    | ThinkingBlock
    | RedactedThinkingBlock
    | ImageBlock
)

# Content that can appear in user messages sent to the API
UserContentBlock = TextBlock | ToolResultBlock | ImageBlock

# Content that can appear in assistant messages from the API
AssistantContentBlock = TextBlock | ToolUseBlock | ThinkingBlock | RedactedThinkingBlock


# ── Usage tracking ───────────────────────────────────────────────────────────


@dataclass
class Usage:
    """Token usage for a single API response."""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    @property
    def total_input(self) -> int:
        return (
            self.input_tokens
            + self.cache_creation_input_tokens
            + self.cache_read_input_tokens
        )

    def accumulate(self, other: Usage) -> None:
        """Add another Usage's counts to this one (mutating)."""
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_creation_input_tokens += other.cache_creation_input_tokens
        self.cache_read_input_tokens += other.cache_read_input_tokens


# ── Message origin ───────────────────────────────────────────────────────────


@dataclass
class MessageOrigin:
    """Provenance of a message."""
    kind: str  # 'human', 'tool', 'system', 'compact', 'meta'
    source: str | None = None  # e.g., tool name, hook name


# ── Core message types ───────────────────────────────────────────────────────


@dataclass
class UserMessage:
    """A user-role message (human input, tool results, or system-injected context)."""
    content: str | list[UserContentBlock]
    uuid: UUID = field(default_factory=uuid4)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Metadata flags
    hide_in_ui: bool = False  # If True: not shown in chat UI, but IS sent to LLM
    hide_in_api: bool = False  # If True: shown in UI only, NOT sent to LLM
    is_compact_summary: bool = False  # Output of compaction

    # Tool result linkage
    tool_use_result: Any = None  # Structured tool output
    source_tool_assistant_uuid: UUID | None = None  # Links tool_result to its tool_use

    # Permission mode when sent (for rewind)
    permission_mode: str | None = None
    origin: MessageOrigin | None = None

    # Compaction metadata (only on compact summary messages)
    summarize_metadata: dict[str, Any] | None = None

    type: Literal["user"] = "user"


@dataclass
class AssistantMessage:
    """An LLM response message."""
    content: list[AssistantContentBlock]
    uuid: UUID = field(default_factory=uuid4)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # API metadata
    model: str = ""
    stop_reason: str | None = None
    usage: Usage = field(default_factory=Usage)
    request_id: str | None = None

    # Error tracking
    is_api_error_message: bool = False
    api_error: str | None = None  # 'invalid_request', 'max_output_tokens', etc.
    error_details: str | None = None

    # Display flag
    hide_in_api: bool = False  # If True: shown in UI only, NOT sent to LLM

    type: Literal["assistant"] = "assistant"

    @property
    def text(self) -> str:
        """Extract concatenated text content."""
        parts = []
        for block in self.content:
            if isinstance(block, TextBlock):
                parts.append(block.text)
        return "".join(parts)

    @property
    def tool_use_blocks(self) -> list[ToolUseBlock]:
        """Extract all tool_use blocks."""
        return [b for b in self.content if isinstance(b, ToolUseBlock)]

    @property
    def has_tool_use(self) -> bool:
        return any(isinstance(b, ToolUseBlock) for b in self.content)


# ── System message subtypes ──────────────────────────────────────────────────


class SystemMessageSubtype(str, Enum):
    COMPACT_BOUNDARY = "compact_boundary"
    COMPACT_CLEAR_BOUNDARY = "compact_clear_boundary"
    API_ERROR = "api_error"
    LOCAL_COMMAND = "local_command"
    INFORMATIONAL = "informational"
    MEMORY_SAVED = "memory_saved"
    STOP_HOOK_SUMMARY = "stop_hook_summary"
    TURN_DURATION = "turn_duration"


@dataclass
class CompactMetadata:
    """Metadata for compaction boundary markers."""
    trigger: Literal["manual", "auto"]
    pre_tokens: int
    user_context: str | None = None
    messages_summarized: int | None = None


@dataclass
class CompactClearMetadata:
    """Metadata for Layer B (clear) boundary markers."""
    trigger: Literal["auto"]
    pre_tokens: int
    tokens_saved: int
    compacted_tool_ids: list[str] = field(default_factory=list)
    cleared_attachment_uuids: list[str] = field(default_factory=list)


@dataclass
class SystemMessage:
    """A system-level message (not the API system prompt)."""
    content: str
    subtype: SystemMessageSubtype
    uuid: UUID = field(default_factory=uuid4)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    level: str = "info"  # 'info', 'warning', 'error'
    hide_in_ui: bool = False

    # Subtype-specific metadata
    compact_metadata: CompactMetadata | None = None
    compact_clear_metadata: CompactClearMetadata | None = None

    type: Literal["system"] = "system"


# ── Attachment messages ──────────────────────────────────────────────────────


@dataclass
class Attachment:
    """A contextual attachment (file contents, search results, etc.)."""
    type: str  # 'edited_text_file', 'hook_stopped_continuation', 'max_iterations_per_turn_reached', etc.
    content: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AttachmentMessage:
    """An attachment injected between turns."""
    attachment: Attachment
    uuid: UUID = field(default_factory=uuid4)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    type: Literal["attachment"] = "attachment"


@dataclass
class ProgressMessage:
    """Real-time progress update from tool execution."""
    tool_use_id: str
    data: dict[str, Any]
    parent_tool_use_id: str = ""
    uuid: UUID = field(default_factory=uuid4)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    type: Literal["progress"] = "progress"


# ── Stream events ────────────────────────────────────────────────────────────


@dataclass
class RequestStartEvent:
    """Marks the start of a new API request."""
    type: Literal["stream_request_start"] = "stream_request_start"


# ── Union type ───────────────────────────────────────────────────────────────


Message = (
    UserMessage
    | AssistantMessage
    | SystemMessage
    | AttachmentMessage
    | ProgressMessage
)

StreamEvent = (
    RequestStartEvent
    | AssistantMessage
    | ProgressMessage
)

# All things that can flow through the query loop generator
QueryYield = Message | StreamEvent | RequestStartEvent


# ── Helpers ──────────────────────────────────────────────────────────────────


def is_compact_boundary(message: Message) -> bool:
    """Check if a message is a compaction boundary marker."""
    return (
        isinstance(message, SystemMessage)
        and message.subtype == SystemMessageSubtype.COMPACT_BOUNDARY
    )


def get_messages_after_compact_boundary(messages: list[Message]) -> list[Message]:
    """Return messages from the last compact boundary onward.
    If no boundary exists, returns all messages.
    """
    for i in range(len(messages) - 1, -1, -1):
        if is_compact_boundary(messages[i]):
            return messages[i:]
    return messages


def get_last_assistant_message(messages: list[Message]) -> AssistantMessage | None:
    """Find the last assistant message in a list."""
    for msg in reversed(messages):
        if isinstance(msg, AssistantMessage):
            return msg
    return None
