"""The agentic query loop.

The heart of Alan Code — a while-true async generator that:
1. Prepares messages (compaction pipeline)
2. Calls the LLM (streaming)
3. Processes the response
4. Executes tools if requested
5. Loops back

See docs/architecture/query-loop.md for the phase-by-phase walkthrough.
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, AsyncGenerator

from alancode.messages.types import (
    AssistantContentBlock,
    AssistantMessage,
    AttachmentMessage,
    Message,
    QueryYield,
    RequestStartEvent,
    SystemMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    Usage,
    UserMessage,
    get_messages_after_compact_boundary,
)
from alancode.messages.factory import (
    create_assistant_error_message,
    create_attachment_message,
    create_compact_boundary_message,
    create_user_interruption_message,
    create_user_message,
)
from alancode.messages.normalization import normalize_messages_for_api
from alancode.messages.serialization import messages_to_openai_dicts
from alancode.providers.base import (
    LLMProvider,
    StreamError,
    StreamMessageDelta,
    StreamMessageStart,
    StreamTextDelta,
    StreamThinkingDelta,
    StreamToolUseInputDelta,
    StreamToolUseStart,
    StreamToolUseStop,
    ToolSchema,
)
from alancode.api.errors import is_prompt_too_long
from alancode.api.retry import stream_with_retry
from alancode.api.cost_tracker import CostTracker
from alancode.budget import ConfigError, clamp_output_budget, resolve_context_budget
from alancode.tools.base import Tool, ToolUseContext
from alancode.tools.registry import tools_to_schemas
from alancode.tools.orchestration import run_tools
from alancode.compact.compact_truncate import compaction_truncate_tool_results
from alancode.compact.compact_clear import compaction_clear_tool_results
from alancode.compact.compact_auto import compaction_auto
from alancode.tools.text_tool_parser import extract_tool_calls_from_text, MAX_TEXT_TOOL_RETRIES
from alancode.query.state import LoopState
from alancode.settings import SETTINGS_DEFAULTS
from alancode.skills.tool_filter import filter_tools_for_skill
from alancode.utils.tokens import estimate_message_tokens, predicted_next_call_tokens

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# System reminders — injected between iterations as <system-reminder> messages
# ---------------------------------------------------------------------------


def _build_turn_reminders(context: ToolUseContext) -> list[UserMessage]:
    """Build system reminders injected once at the start of each turn.

    Contains: current date + time (to the minute).
    These complement the system prompt's date (which is fixed for the session).
    Marked hide_in_ui=True. Can be safely dropped during compaction.
    """
    now = datetime.now(timezone.utc).astimezone()
    date_str = now.strftime("%Y-%m-%d %H:%M")

    reminder_text = (
        "<system-reminder>\n"
        f"# currentDateTime\nCurrent date and time: {date_str}\n"
        "</system-reminder>"
    )
    return [create_user_message(reminder_text, hide_in_ui=True)]




def _is_plain_user_text(msg) -> bool:
    """A message that can safely start (or restart) an API conversation:
    plain user text, not an orphan tool result whose tool_use was cut."""
    if not isinstance(msg, UserMessage):
        return False
    if isinstance(msg.content, str):
        return True
    return not any(isinstance(b, ToolResultBlock) for b in msg.content)


def _hard_truncate_fallback(
    messages: list[Message], target_tokens: int,
) -> tuple[list[Message], int]:
    """Deterministic, LLM-free last resort when summarization keeps failing.

    Middle-out, by information value: the HEAD (a prior compact boundary +
    its summary when present - the compressed representation of the whole
    earlier conversation - otherwise the opening plain-user message, which
    anchors the task) and the TAIL (recent work) are the most valuable parts
    of the history; the middle is the most expendable. So the head is
    preserved, then messages are dropped oldest-first from just after it
    until the estimate fits under *target_tokens*, and the seam is advanced
    until the remainder starts on a plain user message (an orphan tool
    result at the seam would be rejected by the API).

    Escape hatch: if the head alone exceeds the target (e.g. a giant first
    paste), survival wins and the head is dropped too.

    Returns (kept_messages, dropped_count). Information is lost, but the
    session lives - strictly better than the dead-end the circuit breaker
    used to produce (invariant I6).
    """
    if not messages:
        return [], 0

    # Head block: leading compact boundary + its summary message(s), when
    # present; otherwise the opening message if it is plain user text.
    head_end = 0
    if isinstance(messages[0], SystemMessage):
        head_end = 1
        while head_end < len(messages) and getattr(
            messages[head_end], "is_compact_summary", False
        ):
            head_end += 1
    elif _is_plain_user_text(messages[0]):
        head_end = 1
    head = list(messages[:head_end])
    tail = list(messages[head_end:])
    dropped = 0

    # Drop a too-big head up front (escape hatch): otherwise every head+tail
    # estimate stays over target and the tail is emptied to compensate for the
    # head's weight, sacrificing recent work to save an unkeepable head.
    if head and estimate_message_tokens(head) > target_tokens:
        dropped += len(head)
        head = []

    while tail and estimate_message_tokens(head + tail) > target_tokens:
        tail.pop(0)
        dropped += 1
    # Clean seam: the first post-head message must be able to start the
    # (rest of the) conversation.
    while tail and not _is_plain_user_text(tail[0]):
        tail.pop(0)
        dropped += 1

    return head + tail, dropped


def _drain_message_queue(msg_queue) -> list[UserMessage]:
    """Drain queued messages from inject_message() into user messages.

    Accepts either a ``queue.SimpleQueue`` or a plain list.
    Messages are consumed and wrapped as user messages.
    """
    import queue as _queue_mod

    if msg_queue is None:
        return []

    messages: list[UserMessage] = []
    if isinstance(msg_queue, _queue_mod.SimpleQueue):
        while not msg_queue.empty():
            try:
                text = msg_queue.get_nowait()
                messages.append(create_user_message(text))
            except _queue_mod.Empty:
                break
    elif isinstance(msg_queue, list):
        while msg_queue:
            text = msg_queue.pop(0)
            messages.append(create_user_message(text))

    return messages


# ---------------------------------------------------------------------------
# Query parameters
# ---------------------------------------------------------------------------


@dataclass
class QueryParams:
    """Parameters for the query loop."""
    messages: list[Message]
    system_prompt: list[str]
    provider: LLMProvider
    tools: list[Tool]
    context: ToolUseContext
    cost_tracker: CostTracker
    model: str | None = None
    system_static_boundary: int = 0
    max_iterations_per_turn: int | None = None
    max_output_tokens: int | None = None
    # Memory mode
    memory_mode: str = "on"  # "on", "off", "intensive"
    # Permission callback
    permission_callback: Any = None  # async fn(tool, input, context) -> PermissionResult
    # Abort signal
    abort_event: asyncio.Event | None = None
    # Queued messages from ask_while_running / inject_message (shared list reference)
    message_queue: list[str] | None = None
    # Full settings dict (all parameters, flat)
    settings: dict = None  # type: ignore[assignment]
    # Seed values for the pre-call token estimate: the last API call's
    # reported usage (persisted across resume). Zero on a fresh agent.
    last_input_tokens_seed: int = 0
    last_output_tokens_seed: int = 0
    # LLM perspective callback (called with api_messages_dicts before each API call)
    llm_perspective_callback: Any = None  # Callable[[list[dict]], None] | None

    def __post_init__(self):
        if self.settings is None:
            self.settings = dict(SETTINGS_DEFAULTS)



# ---------------------------------------------------------------------------
# The agentic loop
# ---------------------------------------------------------------------------


async def query_loop(params: QueryParams) -> AsyncGenerator[QueryYield, None]:
    """The main agentic loop. Yields stream events and messages.

    The caller iterates this generator to completion. Each iteration of the
    inner while-loop corresponds to one LLM round-trip (possibly followed by
    tool execution).
    """
    state = LoopState(
        messages=list(params.messages),
        last_input_tokens=params.last_input_tokens_seed,
        last_output_tokens=params.last_output_tokens_seed,
        messages_len_at_last_call=(
            len(params.messages) if params.last_input_tokens_seed > 0 else 0
        ),
    )

    # -- Per-turn budget resolution ------------------------------------
    # Model geometry and settings are fixed for the whole turn, so every
    # derived limit (threshold, clear target, blocking, caps) is resolved
    # once here. See alancode/budget.py for the DAG.
    model_info = params.provider.get_model_info(params.model)
    try:
        budget = resolve_context_budget(model_info, params.settings)
    except ConfigError as e:
        yield create_assistant_error_message(
            "Invalid context/budget configuration.", error_details=str(e),
        )
        return
    threshold_tokens = budget.threshold_compaction
    blocking_limit = budget.max_context_size_allowed

    # -- Floor check (once per turn: inputs are fixed) -----------------
    # System prompt + tool schemas are sent on every call and cannot be
    # compacted. If they alone exceed the blocking limit, the turn is
    # unwinnable regardless of the conversation - fail fast with a
    # diagnosis instead of letting Phase 3 give misleading /compact advice.
    tool_schemas_for_floor = [
        t.to_schema() if hasattr(t, "to_schema") else t for t in params.tools
    ]
    floor_tokens = predicted_next_call_tokens(
        params.model,
        [],
        system=params.system_prompt,
        tools=tool_schemas_for_floor,
        last_input_tokens=0,
        last_output_tokens=0,
        new_messages_since_last_call=None,
    )
    if floor_tokens >= blocking_limit:
        yield create_assistant_error_message(
            f"Context window {budget.context_window} is too small "
            f"for this configuration.",
            error_details=(
                f"System prompt + tool schemas alone need "
                f"{floor_tokens} tokens, but the blocking limit is "
                f"{blocking_limit} ({budget.context_window} CW - "
                f"{budget.max_output_tokens} output budget - "
                f"{budget.safety_margin} margin). "
                f"Use a model with a larger context window."
            ),
        )
        return

    iteration = 0

    while True:
        # -- Phase 1: Check abort ----------------------------------------
        if params.abort_event and params.abort_event.is_set():
            yield create_user_interruption_message(tool_use=False)
            return

        yield RequestStartEvent()

        # -- Phase 1.5: Inject system reminders --------------------------
        injected: list[UserMessage] = []

        # Turn reminders (date+time): only on the first iteration of the turn
        if iteration == 0:
            for reminder in _build_turn_reminders(params.context):
                injected.append(reminder)
                yield reminder

        # Queued messages from inject_message(): every iteration
        for queued_msg in _drain_message_queue(params.message_queue):
            injected.append(queued_msg)
            yield queued_msg

        if injected:
            state.messages = state.messages + injected

        # -- Phase 2: Message preparation (compaction pipeline) ----------
        messages_for_query = get_messages_after_compact_boundary(state.messages)

        # Layer A: per-result size cap (always on, middle-out truncation)
        a_truncated = 0
        if params.settings.get("compaction_truncate_enabled", True):
            messages_for_query, a_truncated = compaction_truncate_tool_results(
                messages_for_query, max_chars=budget.tool_result_cap_chars,
            )

        # Layer B: damage control - clear old tool results down to the
        # clear target G (strictly above the compaction threshold T, so B
        # can never pre-empt Layer C, the information-preserving path).
        # Inactive in normal operation; fires only when C failed or could
        # not keep up.
        b_tokens_saved = 0
        if params.settings.get("compaction_clear_enabled", True):
            messages_for_query, b_tokens_saved = compaction_clear_tool_results(
                messages_for_query,
                clear_target_tokens=budget.tool_result_clear_target,
            )

        # Layer C: compaction_auto (summarize if at/over the threshold).
        # Pre-call token estimate: max(usage_based, full_estimate), where
        # usage_based = last call's exact usage + estimate of the delta.
        # When Layer A or B just removed content, the last call's usage
        # still counts it - the floor is stale - so we trust only the
        # direct estimate of the actual post-A/B payload.
        layers_modified = a_truncated > 0 or b_tokens_saved > 0
        seed_input, seed_output = (
            (0, 0)
            if layers_modified
            else (state.last_input_tokens, state.last_output_tokens)
        )
        new_since_last = (
            state.messages[state.messages_len_at_last_call :]
            if seed_input > 0
            else None
        )
        current_tokens = predicted_next_call_tokens(
            params.model,
            messages_for_query,
            system=params.system_prompt,
            tools=[t.to_schema() if hasattr(t, "to_schema") else t for t in params.tools],
            last_input_tokens=seed_input,
            last_output_tokens=seed_output,
            new_messages_since_last_call=new_since_last,
        )
        if params.settings.get("compaction_auto_enabled", True) and current_tokens >= threshold_tokens:
            # Check circuit breaker
            failures = (state.auto_compact_tracking or {}).get("consecutive_failures", 0)
            max_failures = params.settings.get("max_consecutive_compact_failures", 3)
            if failures >= max_failures:
                # Liveness fallback (invariant I6): summarization keeps
                # failing, but the session must not die. Hard-truncate the
                # oldest history deterministically (no LLM involved), tell
                # the user and the model, and continue the turn. Target has
                # headroom below T so Layer C is not immediately re-triggered;
                # the failure counter resets so C gets fresh chances if the
                # conversation grows back over the threshold.
                fallback_target = int(threshold_tokens * 0.8)
                pre_fallback_tokens = estimate_message_tokens(messages_for_query)
                fallback_messages, dropped = _hard_truncate_fallback(
                    messages_for_query, fallback_target,
                )
                # A boundary marker makes the truncation DURABLE: future
                # turns cut here (get_messages_after_compact_boundary),
                # exactly like a successful Layer C - without it, the next
                # turn would rebuild the full oversized history from the
                # agent's permanent record and hit the same wall again.
                fallback_boundary = create_compact_boundary_message(
                    trigger="auto",
                    pre_tokens=pre_fallback_tokens,
                    messages_summarized=dropped,
                )
                notice = create_user_message(
                    f"Summarization failed {failures} times consecutively. "
                    f"{dropped} older message(s) were hard-truncated from the "
                    "context to keep the session alive. Earlier conversation "
                    "details are gone - re-read files if something is missing.",
                    hide_in_ui=False,
                )
                yield fallback_boundary
                yield notice
                messages_for_query = [fallback_boundary, notice] + fallback_messages
                layers_modified = True
                state.auto_compact_tracking = {
                    "compacted": False,
                    "turn_counter": 0,
                    "consecutive_failures": 0,
                }
                logger.warning(
                    "Compaction circuit breaker tripped: hard-truncated %d "
                    "message(s) as liveness fallback", dropped,
                )
            else:
                logger.info("Auto-compaction triggered")
                try:
                    result = await compaction_auto(
                        messages_for_query,
                        params.provider,
                        model=params.model,
                        memory_mode=params.memory_mode,
                        settings=params.settings,
                        budget=budget,
                    )
                    if result:
                        # Yield compaction artefacts so the caller can display/store them
                        yield result.boundary_message
                        for msg in result.summary_messages:
                            yield msg
                        messages_for_query = [result.boundary_message] + result.summary_messages
                        # The payload was just replaced wholesale - the
                        # usage floor is stale for Phase 3 too.
                        layers_modified = True
                        # Update tracking
                        state.auto_compact_tracking = {
                            "compacted": True,
                            "turn_counter": 0,
                            "consecutive_failures": 0,
                        }
                    else:
                        state.auto_compact_tracking = {
                            "compacted": False,
                            "turn_counter": 0,
                            "consecutive_failures": failures + 1,
                        }
                except Exception as e:
                    logger.warning("Auto-compact failed: %s", e)
                    state.auto_compact_tracking = {
                        "compacted": False,
                        "turn_counter": 0,
                        "consecutive_failures": failures + 1,
                    }

        # -- Phase 3: Blocking limit check -------------------------------
        # Same staleness rule as above: after any A/B/C modification this
        # iteration, the last call's usage no longer describes the payload.
        seed_input, seed_output = (
            (0, 0)
            if layers_modified
            else (state.last_input_tokens, state.last_output_tokens)
        )
        current_tokens = predicted_next_call_tokens(
            params.model,
            messages_for_query,
            system=params.system_prompt,
            tools=[t.to_schema() if hasattr(t, "to_schema") else t for t in params.tools],
            last_input_tokens=seed_input,
            last_output_tokens=seed_output,
            new_messages_since_last_call=None,
        )
        if current_tokens >= blocking_limit:
            yield create_assistant_error_message(
                "Conversation too long. Please run /compact or start a new session."
            )
            return

        # -- Phase 4: API call (streaming) -------------------------------
        api_messages = normalize_messages_for_api(messages_for_query)
        api_messages_dicts = messages_to_openai_dicts(api_messages)

        # Notify LLM perspective observers (GUI)
        if params.llm_perspective_callback:
            params.llm_perspective_callback(api_messages_dicts, params.system_prompt)

        # Model-invoked skills may restrict the tool set for the rest of
        # the turn (see ToolUseContext.active_skill_filter).
        effective_tools = params.tools
        if params.context.active_skill_filter:
            effective_tools = filter_tools_for_skill(
                params.tools, params.context.active_skill_filter
            )

        # Don't pass tool schemas to the provider when using text-based
        # tool calling — tools are communicated via the system prompt instead.
        if params.settings.get("tool_call_format"):
            tool_schemas = []
        else:
            tool_schemas = [
                ToolSchema(**s) for s in tools_to_schemas(effective_tools)
            ]

        requested_max_tokens = (
            state.max_output_tokens_override or budget.max_output_tokens
        )
        max_tokens = clamp_output_budget(
            budget, current_tokens, requested_max_tokens,
        )

        # Accumulators for the streamed response
        assistant_content: list[AssistantContentBlock] = []
        tool_use_blocks: list[ToolUseBlock] = []
        current_usage = Usage()
        current_model = params.model
        stop_reason: str | None = None
        request_id: str | None = None

        try:
            async for event in stream_with_retry(
                params.provider,
                api_messages_dicts,
                params.system_prompt,
                tool_schemas,
                model=params.model,
                max_tokens=max_tokens,
                system_static_boundary=params.system_static_boundary,
            ):
                # --- StreamMessageStart ---
                if isinstance(event, StreamMessageStart):
                    current_model = event.model
                    request_id = event.request_id
                    if event.usage:
                        current_usage = Usage(
                            **{
                                k: v
                                for k, v in event.usage.items()
                                if k in Usage.__dataclass_fields__
                            }
                        )

                # --- Text delta ---
                elif isinstance(event, StreamTextDelta):
                    if assistant_content and isinstance(assistant_content[-1], TextBlock):
                        assistant_content[-1].text += event.text
                    else:
                        assistant_content.append(TextBlock(text=event.text))
                    # Yield a virtual message for real-time display
                    yield AssistantMessage(
                        content=[TextBlock(text=event.text)],
                        model=current_model,
                        hide_in_api=True,
                    )

                # --- Tool use lifecycle ---
                elif isinstance(event, StreamToolUseStart):
                    pass  # Start tracked via StreamToolUseStop

                elif isinstance(event, StreamToolUseInputDelta):
                    pass  # Partial JSON tracked via StreamToolUseStop

                elif isinstance(event, StreamToolUseStop):
                    block = ToolUseBlock(
                        id=event.id, name=event.name, input=event.input
                    )
                    assistant_content.append(block)
                    tool_use_blocks.append(block)

                # --- Thinking delta ---
                elif isinstance(event, StreamThinkingDelta):
                    if assistant_content and isinstance(assistant_content[-1], ThinkingBlock):
                        assistant_content[-1].thinking += event.thinking
                    else:
                        assistant_content.append(ThinkingBlock(thinking=event.thinking))
                    # Yield a virtual message for real-time thinking display
                    yield AssistantMessage(
                        content=[ThinkingBlock(thinking=event.thinking)],
                        model=current_model,
                        hide_in_api=True,
                    )

                # --- Message-level metadata ---
                elif isinstance(event, StreamMessageDelta):
                    stop_reason = event.stop_reason
                    if event.usage:
                        for k, v in event.usage.items():
                            if hasattr(current_usage, k):
                                setattr(
                                    current_usage,
                                    k,
                                    getattr(current_usage, k) + v,
                                )

                # --- Stream error ---
                elif isinstance(event, StreamError):
                    yield create_assistant_error_message(
                        event.error, api_error=event.error_type
                    )
                    return

        except Exception as e:
            if (
                is_prompt_too_long(str(e))
                and not state.has_attempted_emergency_compact
            ):
                logger.info("Emergency compaction triggered (prompt too long)")
                state.has_attempted_emergency_compact = True
                try:
                    emergency_result = await compaction_auto(
                        messages_for_query,
                        params.provider,
                        model=params.model,
                        memory_mode=params.memory_mode,
                        settings=params.settings,
                        budget=budget,
                    )
                    if emergency_result:
                        yield emergency_result.boundary_message
                        for msg in emergency_result.summary_messages:
                            yield msg
                        state.messages = (
                            [emergency_result.boundary_message]
                            + emergency_result.summary_messages
                        )
                        state.transition = "emergency_compact_retry"
                        continue
                except Exception as compact_error:
                    logger.warning(
                        "Emergency compaction failed: %s", compact_error,
                    )

            logger.error("Query error: %s", e)
            yield create_assistant_error_message(
                str(e),
                api_error=(
                    "prompt_too_long"
                    if is_prompt_too_long(str(e))
                    else None
                ),
            )
            return

        # -- Phase 5: Build final assistant message ----------------------
        # Fix for thinking models: if all content is in
        # ThinkingBlocks and no TextBlock exists, the model's answer is
        # embedded at the end of the thinking. Mark thinking as the response.
        has_text = any(isinstance(b, TextBlock) and b.text.strip() for b in assistant_content)
        has_thinking = any(isinstance(b, ThinkingBlock) for b in assistant_content)
        if has_thinking and not has_text and not tool_use_blocks:
            # The thinking IS the response — add a note as text
            logger.info("Thinking model returned empty content — using thinking as response")

        assistant_msg = AssistantMessage(
            content=assistant_content,
            model=current_model,
            stop_reason=stop_reason,
            usage=current_usage,
            request_id=request_id,
        )

        # -- Phase 5.25: Extract thinking from text -------------------------
        # Some models (e.g. Qwen3 thinking variants via Ollama/LiteLLM) embed
        # <think>...</think> in the text content instead of using separate
        # thinking events. Extract it into a ThinkingBlock.
        if not any(isinstance(b, ThinkingBlock) for b in assistant_content):
            full_text_for_thinking = "".join(
                b.text for b in assistant_content if isinstance(b, TextBlock)
            )
            if "<think>" in full_text_for_thinking or "</think>" in full_text_for_thinking:
                from alancode.tools.text_tool_parser import _extract_thinking
                thinking_text, remaining_text = _extract_thinking(full_text_for_thinking)
                if thinking_text:
                    new_blocks: list[AssistantContentBlock] = [
                        ThinkingBlock(thinking=thinking_text),
                    ]
                    if remaining_text:
                        new_blocks.append(TextBlock(text=remaining_text))
                    # Preserve non-text blocks (tool_use, etc.)
                    for b in assistant_content:
                        if not isinstance(b, TextBlock):
                            new_blocks.append(b)
                    assistant_content = new_blocks
                    assistant_msg = AssistantMessage(
                        content=assistant_content,
                        model=current_model,
                        stop_reason=stop_reason,
                        usage=current_usage,
                        request_id=request_id,
                    )

        # -- Phase 5.5: Text-based tool call extraction --------------------
        # If the model doesn't support native tool calling, extract tool
        # calls from the text output using the configured format parser.
        # On malformed tool calls, feed back an error and let the model retry.
        tool_call_format = params.settings.get("tool_call_format")
        if (
            tool_call_format
            and not tool_use_blocks
        ):
            full_text = "".join(
                b.text for b in assistant_content if isinstance(b, TextBlock)
            )
            if full_text:
                parse_result = extract_tool_calls_from_text(
                    full_text, format=tool_call_format,
                )

                if parse_result.tool_calls:
                    logger.info(
                        "Extracted %d tool call(s) from text (format=%s)",
                        len(parse_result.tool_calls), tool_call_format,
                    )
                    new_content: list[AssistantContentBlock] = []
                    if parse_result.thinking:
                        new_content.append(ThinkingBlock(thinking=parse_result.thinking))
                    if parse_result.cleaned_text:
                        new_content.append(TextBlock(text=parse_result.cleaned_text))
                    for pc in parse_result.tool_calls:
                        call_id = f"text_{uuid.uuid4().hex[:8]}"
                        block = ToolUseBlock(
                            id=call_id,
                            name=pc.name,
                            input=pc.input,
                        )
                        new_content.append(block)
                        tool_use_blocks.append(block)

                    assistant_msg = AssistantMessage(
                        content=new_content,
                        model=current_model,
                        stop_reason=stop_reason,
                        usage=current_usage,
                        request_id=request_id,
                    )

                elif parse_result.error:
                    # Model attempted a tool call but used wrong format.
                    # Feed back the error and retry (up to MAX_TEXT_TOOL_RETRIES).
                    retry_count = getattr(state, "_text_tool_retries", 0)
                    if retry_count < MAX_TEXT_TOOL_RETRIES:
                        state._text_tool_retries = retry_count + 1  # type: ignore[attr-defined]
                        logger.warning(
                            "Malformed text tool call (retry %d/%d): %s",
                            retry_count + 1, MAX_TEXT_TOOL_RETRIES,
                            parse_result.error[:100],
                        )
                        # Yield the malformed assistant message + error feedback
                        yield assistant_msg
                        error_msg = create_user_message(
                            parse_result.error,
                            hide_in_ui=False,
                        )
                        yield error_msg
                        state.messages = list(messages_for_query) + [assistant_msg, error_msg]
                        state.transition = "text_tool_retry"
                        continue
                    else:
                        logger.error("Text tool call retries exhausted (%d)", MAX_TEXT_TOOL_RETRIES)

                elif parse_result.thinking or parse_result.cleaned_text != full_text:
                    # No tool calls but thinking was extracted or text changed — rebuild
                    rebuilt_content: list[AssistantContentBlock] = []
                    if parse_result.thinking:
                        rebuilt_content.append(ThinkingBlock(thinking=parse_result.thinking))
                    if parse_result.cleaned_text:
                        rebuilt_content.append(TextBlock(text=parse_result.cleaned_text))
                    assistant_msg = AssistantMessage(
                        content=rebuilt_content,
                        model=current_model,
                        stop_reason=stop_reason,
                        usage=current_usage,
                        request_id=request_id,
                    )

        # Yield the (possibly rebuilt) assistant message
        yield assistant_msg

        # Track cost and remember last-call usage for next iteration's
        # pre-call estimate (see predicted_next_call_tokens).
        params.cost_tracker.add_usage(current_usage, current_model)
        if current_usage.input_tokens > 0:
            state.last_input_tokens = current_usage.input_tokens
            state.last_output_tokens = current_usage.output_tokens
            state.messages_len_at_last_call = len(state.messages)

        # -- Phase 6: Check abort after streaming ------------------------
        if params.abort_event and params.abort_event.is_set():
            yield create_user_interruption_message(tool_use=False)
            return

        # -- Phase 7: Handle no tool use (completion or recovery) --------
        if not tool_use_blocks:
            # Max-output-tokens recovery
            if stop_reason == "max_tokens" or assistant_msg.api_error == "max_output_tokens":
                # Try escalation first (bump to 64K)
                if (
                    state.max_output_tokens_override is None
                    and not params.max_output_tokens
                ):
                    escalated = params.settings.get("escalated_max_tokens", 64000)
                    logger.info("Escalating max_tokens to %d", escalated)
                    state.max_output_tokens_override = escalated
                    state.messages = list(messages_for_query)
                    state.transition = "max_output_tokens_escalate"
                    continue

                # Multi-turn recovery
                if state.max_output_tokens_recovery_count < params.settings.get("max_output_tokens_recovery_limit", 3):
                    state.max_output_tokens_recovery_count += 1
                    recovery_msg = create_user_message(
                        "Output token limit hit. Resume directly -- no apology, no recap. "
                        "Pick up mid-thought. Break remaining work into smaller pieces.",
                        hide_in_ui=True,
                    )
                    state.messages = list(messages_for_query) + [
                        assistant_msg,
                        recovery_msg,
                    ]
                    state.max_output_tokens_override = None
                    state.transition = "max_output_tokens_recovery"
                    continue

            # Normal completion
            return

        # -- Phase 8: Tool execution -------------------------------------
        tool_results: list[UserMessage] = []

        async for update in run_tools(
            tool_use_blocks, effective_tools, params.context,
            max_concurrency=params.settings.get("max_tool_concurrency", 10),
            permission_callback=params.permission_callback,
        ):
            if update.message:
                yield update.message
                tool_results.append(update.message)

        # Check abort after tools
        if params.abort_event and params.abort_event.is_set():
            yield create_user_interruption_message(tool_use=True)
            return

        # -- Phase 8.5: Memory reminder (intensive mode) -----------------
        state.turns_since_memory_update += 1
        if (
            params.memory_mode == "intensive"
            and state.turns_since_memory_update >= params.settings.get("memory_reminder_threshold", 10)
        ):
            memory_reminder = create_user_message(
                "<system-reminder>\n"
                "Several turns have passed since the last memory update. "
                "Consider whether any recent corrections, decisions, or preferences "
                "are worth saving to memory.\n"
                "</system-reminder>",
                hide_in_ui=True,
            )
            tool_results.append(memory_reminder)
            yield memory_reminder
            state.turns_since_memory_update = 0

        # -- Phase 9: Check max turns ------------------------------------
        state.iteration_count += 1
        if params.max_iterations_per_turn and state.iteration_count >= params.max_iterations_per_turn:
            yield create_attachment_message(
                "max_iterations_per_turn_reached",
                metadata={
                    "max_iterations_per_turn": params.max_iterations_per_turn,
                    "iteration_count": state.iteration_count,
                },
            )
            return

        # -- Phase 10: Assemble next iteration ---------------------------
        state.messages = list(messages_for_query) + [assistant_msg] + tool_results
        state.max_output_tokens_recovery_count = 0
        state.max_output_tokens_override = None
        state.transition = "next_turn"
        iteration += 1
    # end while True
