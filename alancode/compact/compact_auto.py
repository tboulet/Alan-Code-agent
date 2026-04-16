"""Auto-compaction — summarize conversation when context grows too large.

Layer C of the compaction hierarchy. Forks a tool-less LLM call that
produces an ``<analysis>`` + ``<summary>`` response; the summary
replaces the pre-boundary history.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from alancode.messages.types import (
    Message,
    UserMessage,
    AssistantMessage,
    SystemMessage,
    TextBlock,
    ThinkingBlock,
    RedactedThinkingBlock,
    ToolUseBlock,
    ToolResultBlock,
    ImageBlock,
    get_messages_after_compact_boundary,
)
from alancode.providers.base import StreamTextDelta, StreamError
from alancode.messages.factory import (
    create_compact_boundary_message,
    create_user_message,
)
from alancode.messages.normalization import normalize_messages_for_api
from alancode.messages.serialization import messages_to_openai_dicts
from alancode.compact.prompt import (
    get_compact_prompt,
    format_compact_summary,
    get_post_compact_message,
    get_post_compact_notification,
)
from alancode.utils.tokens import (
    estimate_message_tokens,
    get_auto_compact_threshold,
    rough_token_count,
)

logger = logging.getLogger(__name__)


@dataclass
class CompactionResult:
    summary_messages: list[UserMessage]
    boundary_message: SystemMessage
    pre_compact_token_count: int
    post_compact_token_count: int




# ---------------------------------------------------------------------------
# PTL (Prompt Too Long) retry support
# ---------------------------------------------------------------------------


def truncate_middle_for_ptl(
    messages: list[UserMessage | AssistantMessage],
) -> list[UserMessage | AssistantMessage] | None:
    """Cut ~20% from the middle, keeping start + end.

    System prompt is a separate parameter and is never touched.
    The first few messages (original context) and the last messages
    (recent work) are preserved. The middle is cut.

    Returns None if too few messages to cut.
    """
    if len(messages) <= 4:
        return None
    n = len(messages)
    cut_size = max(1, n // 5)  # ~20% of messages
    cut_start = (n - cut_size) // 2  # center the cut
    cut_end = cut_start + cut_size
    return messages[:cut_start] + messages[cut_end:]


# ---------------------------------------------------------------------------
# Main compaction entry point
# ---------------------------------------------------------------------------


async def compaction_auto(
    messages: list[Message],
    provider: Any,  # LLMProvider
    *,
    model: str | None = None,
    custom_instructions: str | None = None,
    session_id: str | None = None,
    memory_mode: str = "on",
    settings: dict | None = None,
) -> CompactionResult | None:
    """Compact the conversation by summarizing it via LLM (Layer C).

    1. Build compact system prompt (replacement, not the main prompt)
    2. Build compact user message with 9-section template
    3. Normalize messages for API
    4. PTL retry loop: call provider, retry with truncation if prompt too long
    5. Extract summary via format_compact_summary
    6. Build CompactionResult with boundary + summary + notification

    Returns None if compaction fails after all retries.
    """

    # Get messages from last compact boundary onward
    relevant_messages = get_messages_after_compact_boundary(messages)

    pre_compact_token_count = estimate_message_tokens(relevant_messages)
    logger.info(
        "Starting compaction: %d messages, ~%d tokens",
        len(relevant_messages),
        pre_compact_token_count,
    )

    # 1. Pre-truncate oversized tool results before compaction
    # Without this, a single huge tool result (e.g., 216K chars) would be
    # included in the compaction request, exceeding the LLM's context window.
    from alancode.compact.compact_truncate import compaction_truncate_tool_results
    truncated_messages = compaction_truncate_tool_results(
        relevant_messages, settings=settings,
    )

    # 2. Build compact system prompt (REPLACEMENT, not appended)
    compact_system = ["You are a helpful AI assistant tasked with summarizing conversations."]

    # 3. Build compact user message
    compact_prompt = get_compact_prompt(custom_instructions)

    # 4. Normalize messages for API
    api_messages = normalize_messages_for_api(truncated_messages)
    api_messages_dicts = messages_to_openai_dicts(api_messages)

    # Add compact prompt as the final user message
    api_messages_dicts.append({"role": "user", "content": compact_prompt})

    # 4. PTL retry loop
    s = settings or {}
    max_ptl_retries = s.get("max_compact_ptl_retries", 3)
    compact_max_output_tokens = s.get("compact_max_output_tokens", 20_000)

    response_text = ""
    kwargs: dict[str, Any] = {}
    if model is not None:
        kwargs["model"] = model

    for attempt in range(max_ptl_retries + 1):
        response_text = ""
        try:
            async for event in provider.stream(
                api_messages_dicts,
                compact_system,
                tools=[],
                max_tokens=compact_max_output_tokens,
                **kwargs,
            ):
                if isinstance(event, StreamTextDelta):
                    response_text += event.text
                elif isinstance(event, StreamError):
                    error_msg = event.error.lower() if event.error else ""
                    if "prompt" in error_msg and "too long" in error_msg:
                        raise _PromptTooLongError(event.error)
                    # Other errors: log and fail
                    logger.warning("Compaction stream error: %s", event.error)
                    response_text = ""
                    break

            if response_text.strip():
                break  # Success

        except _PromptTooLongError:
            if attempt >= max_ptl_retries:
                logger.error(
                    "Compaction failed: prompt too long after %d retries", max_ptl_retries
                )
                return None
            # Truncate and retry
            # Remove the compact prompt (last element), truncate, re-add
            # Re-normalize as UserMessage/AssistantMessage for truncation
            truncated = truncate_middle_for_ptl(api_messages)
            if truncated is None:
                logger.error("Too few messages to truncate for PTL retry")
                return None
            api_messages = truncated
            api_messages_dicts = messages_to_openai_dicts(api_messages)
            api_messages_dicts.append({"role": "user", "content": compact_prompt})
            logger.info(
                "PTL retry %d: truncated to %d messages",
                attempt + 1,
                len(api_messages),
            )
            continue

        except Exception as e:
            logger.error("Compaction failed with exception: %s", e)
            return None

    if not response_text.strip():
        logger.error("LLM returned empty response for compaction")
        return None

    # 5. Extract summary
    summary = format_compact_summary(response_text)
    logger.info(
        "Compaction complete: summary is ~%d tokens",
        rough_token_count(summary),
    )

    # Transcript path is not available here (we don't have cwd).
    # The post-compact notification will omit it.
    transcript_path: str | None = None

    # 6. Build the boundary message
    boundary_message = create_compact_boundary_message(
        trigger="auto",
        pre_tokens=pre_compact_token_count,
        messages_summarized=len(relevant_messages),
    )

    # Build the summary user message (with post-compact wrapper)
    post_compact_text = get_post_compact_message(
        response_text,
        transcript_path=transcript_path,
        memory_mode=memory_mode,
    )
    summary_message = create_user_message(
        post_compact_text,
        is_compact_summary=True,
    )

    # Build notification message (system reminder for the model)
    notification_text = get_post_compact_notification(memory_mode=memory_mode)
    notification_message = create_user_message(
        notification_text,
        hide_in_ui=True,
    )

    # Calculate post-compact token count
    post_compact_messages = [boundary_message, summary_message, notification_message]
    post_compact_token_count = estimate_message_tokens(post_compact_messages)

    return CompactionResult(
        summary_messages=[summary_message, notification_message],
        boundary_message=boundary_message,
        pre_compact_token_count=pre_compact_token_count,
        post_compact_token_count=post_compact_token_count,
    )


class _PromptTooLongError(Exception):
    """Internal sentinel for prompt-too-long errors during compaction."""
    pass
