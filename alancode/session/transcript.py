"""Transcript recording for session persistence.

Each session is stored as a JSONL file under ``.alan/sessions/<session_id>/transcript.jsonl``.
One JSON object per line, one line per message.  This allows efficient
append-only writes and streaming reads.
"""

import json
import logging
import os
from pathlib import Path
from datetime import datetime, timezone
from uuid import UUID

from alancode.messages.types import (
    AssistantMessage,
    Attachment,
    AttachmentMessage,
    CompactMetadata,
    ContentBlock,
    ImageBlock,
    Message,
    CompactClearMetadata,
    RedactedThinkingBlock,
    SystemMessage,
    SystemMessageSubtype,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    Usage,
    UserMessage,
    MessageOrigin,
)
from alancode.session.session import get_session_dir

logger = logging.getLogger(__name__)


# ── Paths ──────────────────────────────────────────────────────────────────


def get_session_transcript_path(cwd: str, session_id: str) -> Path:
    """Return the transcript JSONL file path for a given session.

    New layout: ``.alan/sessions/<session_id>/transcript.jsonl``
    """
    return get_session_dir(cwd, session_id) / "transcript.jsonl"



# ── Write / Read ───────────────────────────────────────────────────────────


async def record_transcript(
    session_id: str,
    messages: list[Message],
    *,
    cwd: str | None = None,
) -> None:
    """Write *messages* to the session transcript (JSONL format).

    Overwrites any previous transcript for the same session.
    The first line is a metadata object containing the session's *cwd*,
    session ID, and creation timestamp so that ``get_last_session_id``
    can filter by working directory.

    When *cwd* is provided, uses the new per-project session layout
    (``.alan/sessions/<id>/transcript.jsonl``).
    """
    if not cwd:
        raise ValueError("cwd is required for record_transcript")
    path = get_session_transcript_path(cwd, session_id)

    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from alancode.utils.atomic_io import atomic_write_text
        metadata = {
            "_metadata": {
                "cwd": cwd or "",
                "session_id": session_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        }
        lines = [json.dumps(metadata, default=str)]
        lines.extend(
            json.dumps(message_to_dict(msg), default=str) for msg in messages
        )
        atomic_write_text(path, "\n".join(lines) + "\n")
    except OSError as exc:
        logger.warning("Failed to write transcript %s: %s", path, exc)


async def load_transcript(session_id: str, *, cwd: str | None = None) -> list[Message] | None:
    """Load messages from a session transcript.

    Returns ``None`` if the file does not exist or cannot be read.
    The first line may be a metadata object (``_metadata`` key) which is
    skipped when reconstructing messages.  Individual malformed lines are
    skipped with a warning.
    """
    if not cwd:
        raise ValueError("cwd is required for load_transcript")
    path = get_session_transcript_path(cwd, session_id)

    if not path.is_file():
        return None

    messages: list[Message] = []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    # Skip metadata line
                    if "_metadata" in d:
                        continue
                    messages.append(dict_to_message(d))
                except (json.JSONDecodeError, KeyError, TypeError) as exc:
                    logger.warning(
                        "Skipping malformed line %d in %s: %s", lineno, path, exc
                    )
    except OSError as exc:
        logger.warning("Failed to read transcript %s: %s", path, exc)
        return None

    return messages if messages else None


# ── Serialization helpers ──────────────────────────────────────────────────


def _uuid_to_str(val: UUID | None) -> str | None:
    """Convert a UUID to its string representation, or pass through None."""
    if val is None:
        return None
    return str(val)


def _content_block_to_dict(block: ContentBlock) -> dict:
    """Serialize a single content block to a dict."""
    if isinstance(block, TextBlock):
        return {"type": "text", "text": block.text}
    if isinstance(block, ToolUseBlock):
        return {
            "type": "tool_use",
            "id": block.id,
            "name": block.name,
            "input": block.input,
        }
    if isinstance(block, ToolResultBlock):
        content = block.content
        if isinstance(content, list):
            content = [_content_block_to_dict(b) for b in content]
        return {
            "type": "tool_result",
            "tool_use_id": block.tool_use_id,
            "content": content,
            "is_error": block.is_error,
        }
    if isinstance(block, ThinkingBlock):
        return {
            "type": "thinking",
            "thinking": block.thinking,
            "signature": block.signature,
        }
    if isinstance(block, RedactedThinkingBlock):
        return {"type": "redacted_thinking", "data": block.data}
    if isinstance(block, ImageBlock):
        return {"type": "image", "source": block.source}
    # Fallback: try to use __dict__
    return {"type": getattr(block, "type", "unknown"), **getattr(block, "__dict__", {})}


def _dict_to_content_block(d: dict) -> ContentBlock:
    """Deserialize a dict into the appropriate content block type."""
    block_type = d.get("type", "")
    if block_type == "text":
        return TextBlock(text=d["text"])
    if block_type == "tool_use":
        return ToolUseBlock(id=d["id"], name=d["name"], input=d.get("input", {}))
    if block_type == "tool_result":
        content = d.get("content", "")
        if isinstance(content, list):
            content = [_dict_to_content_block(b) for b in content]
        return ToolResultBlock(
            tool_use_id=d["tool_use_id"],
            content=content,
            is_error=d.get("is_error", False),
        )
    if block_type == "thinking":
        return ThinkingBlock(
            thinking=d.get("thinking", ""), signature=d.get("signature", "")
        )
    if block_type == "redacted_thinking":
        return RedactedThinkingBlock(data=d.get("data", ""))
    if block_type == "image":
        return ImageBlock(source=d.get("source", {}))
    # Fallback
    return TextBlock(text=str(d))


def message_to_dict(msg: Message) -> dict:
    """Serialize a message to a JSON-compatible dict."""
    if isinstance(msg, UserMessage):
        content = msg.content
        if isinstance(content, list):
            content = [_content_block_to_dict(b) for b in content]
        out = {
            "type": "user",
            "content": content,
            "uuid": str(msg.uuid),
            "timestamp": msg.timestamp,
            "hide_in_ui": msg.hide_in_ui,
            "hide_in_api": msg.hide_in_api,
            "is_compact_summary": msg.is_compact_summary,
            "permission_mode": msg.permission_mode,
        }
        # Preserve the tool_use → tool_result link so pairing survives resume.
        if msg.source_tool_assistant_uuid is not None:
            out["source_tool_assistant_uuid"] = str(msg.source_tool_assistant_uuid)
        if msg.origin is not None:
            out["origin"] = {
                "kind": msg.origin.kind,
                "source": msg.origin.source,
            }
        return out

    if isinstance(msg, AssistantMessage):
        return {
            "type": "assistant",
            "content": [_content_block_to_dict(b) for b in msg.content],
            "uuid": str(msg.uuid),
            "timestamp": msg.timestamp,
            "model": msg.model,
            "stop_reason": msg.stop_reason,
            "usage": {
                "input_tokens": msg.usage.input_tokens,
                "output_tokens": msg.usage.output_tokens,
                "cache_creation_input_tokens": msg.usage.cache_creation_input_tokens,
                "cache_read_input_tokens": msg.usage.cache_read_input_tokens,
            },
            "is_api_error_message": msg.is_api_error_message,
            "api_error": msg.api_error,
            "error_details": msg.error_details,
            "hide_in_api": msg.hide_in_api,
        }

    if isinstance(msg, SystemMessage):
        d: dict = {
            "type": "system",
            "content": msg.content,
            "subtype": msg.subtype.value,
            "uuid": str(msg.uuid),
            "timestamp": msg.timestamp,
            "level": msg.level,
            "hide_in_ui": msg.hide_in_ui,
        }
        if msg.compact_metadata is not None:
            d["compact_metadata"] = {
                "trigger": msg.compact_metadata.trigger,
                "pre_tokens": msg.compact_metadata.pre_tokens,
                "user_context": msg.compact_metadata.user_context,
                "messages_summarized": msg.compact_metadata.messages_summarized,
            }
        if msg.compact_clear_metadata is not None:
            d["compact_clear_metadata"] = {
                "trigger": msg.compact_clear_metadata.trigger,
                "pre_tokens": msg.compact_clear_metadata.pre_tokens,
                "tokens_saved": msg.compact_clear_metadata.tokens_saved,
                "compacted_tool_ids": msg.compact_clear_metadata.compacted_tool_ids,
                "cleared_attachment_uuids": msg.compact_clear_metadata.cleared_attachment_uuids,
            }
        return d

    if isinstance(msg, AttachmentMessage):
        return {
            "type": "attachment",
            "uuid": str(msg.uuid),
            "timestamp": msg.timestamp,
            "attachment": {
                "type": msg.attachment.type,
                "content": msg.attachment.content,
                "metadata": msg.attachment.metadata,
            },
        }

    # ProgressMessage or unknown — best-effort
    return {"type": getattr(msg, "type", "unknown"), **getattr(msg, "__dict__", {})}


def dict_to_message(d: dict) -> Message:
    """Deserialize a message from a dict.

    Reconstructs the correct Message subtype based on the ``type`` field.
    """
    msg_type = d.get("type", "")

    if msg_type == "user":
        content = d.get("content", "")
        if isinstance(content, list):
            content = [_dict_to_content_block(b) for b in content]
        src_uuid_str = d.get("source_tool_assistant_uuid")
        src_uuid = UUID(src_uuid_str) if src_uuid_str else None
        origin_d = d.get("origin")
        origin_obj = None
        if isinstance(origin_d, dict) and "kind" in origin_d:
            from alancode.messages.types import MessageOrigin
            origin_obj = MessageOrigin(
                kind=origin_d["kind"], source=origin_d.get("source"),
            )
        return UserMessage(
            content=content,
            uuid=UUID(d["uuid"]) if "uuid" in d else None,
            timestamp=d.get("timestamp", ""),
            hide_in_ui=d.get("hide_in_ui", False),
            hide_in_api=d.get("hide_in_api", False),
            is_compact_summary=d.get("is_compact_summary", False),
            permission_mode=d.get("permission_mode"),
            source_tool_assistant_uuid=src_uuid,
            origin=origin_obj,
        )

    if msg_type == "assistant":
        content_blocks = [
            _dict_to_content_block(b) for b in d.get("content", [])
        ]
        usage_d = d.get("usage", {})
        return AssistantMessage(
            content=content_blocks,
            uuid=UUID(d["uuid"]) if "uuid" in d else None,
            timestamp=d.get("timestamp", ""),
            model=d.get("model", ""),
            stop_reason=d.get("stop_reason"),
            usage=Usage(
                input_tokens=usage_d.get("input_tokens", 0),
                output_tokens=usage_d.get("output_tokens", 0),
                cache_creation_input_tokens=usage_d.get("cache_creation_input_tokens", 0),
                cache_read_input_tokens=usage_d.get("cache_read_input_tokens", 0),
            ),
            is_api_error_message=d.get("is_api_error_message", False),
            api_error=d.get("api_error"),
            error_details=d.get("error_details"),
            hide_in_api=d.get("hide_in_api", False),
        )

    if msg_type == "system":
        compact_meta = None
        if "compact_metadata" in d and d["compact_metadata"] is not None:
            cm = d["compact_metadata"]
            compact_meta = CompactMetadata(
                trigger=cm["trigger"],
                pre_tokens=cm["pre_tokens"],
                user_context=cm.get("user_context"),
                messages_summarized=cm.get("messages_summarized"),
            )
        compact_clear_meta = None
        if "compact_clear_metadata" in d and d["compact_clear_metadata"] is not None:
            mm = d["compact_clear_metadata"]
            compact_clear_meta = CompactClearMetadata(
                trigger=mm["trigger"],
                pre_tokens=mm["pre_tokens"],
                tokens_saved=mm["tokens_saved"],
                compacted_tool_ids=mm.get("compacted_tool_ids", []),
                cleared_attachment_uuids=mm.get("cleared_attachment_uuids", []),
            )
        return SystemMessage(
            content=d.get("content", ""),
            subtype=SystemMessageSubtype(d["subtype"]),
            uuid=UUID(d["uuid"]) if "uuid" in d else None,
            timestamp=d.get("timestamp", ""),
            level=d.get("level", "info"),
            hide_in_ui=d.get("hide_in_ui", False),
            compact_metadata=compact_meta,
            compact_clear_metadata=compact_clear_meta,
        )

    if msg_type == "attachment":
        att_d = d.get("attachment", {})
        return AttachmentMessage(
            attachment=Attachment(
                type=att_d.get("type", ""),
                content=att_d.get("content", ""),
                metadata=att_d.get("metadata", {}),
            ),
            uuid=UUID(d["uuid"]) if "uuid" in d else None,
            timestamp=d.get("timestamp", ""),
        )

    # Fallback: return as a UserMessage with raw content
    logger.warning("Unknown message type %r, falling back to UserMessage", msg_type)
    return UserMessage(content=str(d))
