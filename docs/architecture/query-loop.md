# The query loop, phase by phase

`alancode/query/loop.py::query_loop` is an async generator that drives every agent turn. This page walks through what it does phase by phase, with file:line anchors.

Prerequisites: [concepts/agent-loop.md](../concepts/agent-loop.md) for the vocabulary (iteration vs turn vs session).

## Entry

```python
async def query_loop(params: QueryParams) -> AsyncGenerator[QueryYield, None]:
    state = LoopState(
        messages=list(params.messages),
        last_input_tokens=params.last_input_tokens_seed,
        last_output_tokens=params.last_output_tokens_seed,
        messages_len_at_last_call=(
            len(params.messages) if params.last_input_tokens_seed > 0 else 0
        ),
    )
    iteration = 0
    while True:
        # phases 1–10 ...
```

`QueryParams` carries everything the loop needs (messages, backend, tools, permission callback, abort event, etc.). `LoopState` is mutable per-turn state.

## Phase 1 — Abort check

```python
if params.abort_event and params.abort_event.is_set():
    yield create_user_interruption_message(tool_use=False)
    return
```

Checked at the top of every iteration. Ctrl+C during the prompt or tool execution sets `abort_event`; we see it here and bail. A user-interruption message is yielded so the caller can display "Turn interrupted."

## Phase 1.5 — System-reminder injection

```python
yield RequestStartEvent()

injected: list[UserMessage] = []
if iteration == 0:
    for reminder in _build_turn_reminders(params.context):
        injected.append(reminder)
        yield reminder

for queued_msg in _drain_message_queue(params.message_queue):
    injected.append(queued_msg)
    yield queued_msg

if injected:
    state.messages = state.messages + injected
```

- `RequestStartEvent` signals to the UI that a new API call is starting (used to trigger "Thinking..." indicators).
- On iteration 0 of the turn, inject a date/time `<system-reminder>` — keeps the model aware of wall-clock time across long sessions.
- Drain any messages injected via `agent.inject_message(...)` from another task.

All injected messages are `hide_in_ui=True` — they go to the API but not to the user's chat panel.

## Phase 2 — Compaction pipeline

The pre-call gatekeeper. See [concepts/context-and-compaction.md](../concepts/context-and-compaction.md) for the full layer walkthrough.

```python
messages_for_query = get_messages_after_compact_boundary(state.messages)

model_info = params.backend.get_model_info(params.model)
budget = resolve_context_budget(model_info, params.settings)

# Layer A
messages_for_query, truncated = compaction_truncate_tool_results(
    messages_for_query, max_chars=budget.tool_result_cap_chars
)

# Layer B
messages_for_query, tokens_saved = compaction_clear_tool_results(
    messages_for_query, clear_target_tokens=budget.tool_result_clear_target
)

# Layer C (pre-call estimate via `predicted_next_call_tokens`)
current_tokens = predicted_next_call_tokens(
    params.model, messages_for_query,
    system=params.system_prompt,
    tools=[t.to_schema() ... for t in params.tools],
    last_input_tokens=state.last_input_tokens,
    last_output_tokens=state.last_output_tokens,
    new_messages_since_last_call=(
        state.messages[state.messages_len_at_last_call:]
        if state.last_input_tokens > 0 else None
    ),
)
if current_tokens >= budget.threshold_compaction:
    # fire Layer C
    result = await compaction_auto(..., budget=budget)
    if result:
        messages_for_query = [result.boundary_message, *result.summary_messages]
```

Layers run in order. Layer A always enforces its per-result cap. Layer B only clears down to a target above the summarization threshold, so it cannot replace Layer C's information-preserving role.

`predicted_next_call_tokens` is `max(usage_based, full_estimate)` where:
- `usage_based = last_input_tokens + last_output_tokens + tokens(new_messages_since_last_call)`
- `full_estimate = litellm.token_counter(...)` or chars/3 fallback

Taking the max protects against under-budgeting.

## Phase 3 — Blocking-limit check

```python
blocking_limit = budget.max_context_size_allowed  # CW - max_output_tokens - margin
if compaction_failed and current_tokens >= blocking_limit:
    messages_for_query = hard_truncate_messages(...)

if current_tokens >= blocking_limit:
    yield create_assistant_error_message(
        "Conversation too long. Please run /compact or start a new session."
    )
    return
```

The limit is the point where the reserved response no longer fits (`alancode/budget.py` derives it once per turn). If LLM summarization failed, a deterministic fallback first keeps a useful opening and a structurally valid recent tail. Refusal remains for an irreducible payload, such as a system prompt and tool schemas that cannot fit by themselves.

## Phase 4 — API call (streaming)

```python
api_messages = normalize_messages_for_api(messages_for_query)
api_messages_dicts = messages_to_openai_dicts(api_messages)

if params.llm_perspective_callback is not None:
    params.llm_perspective_callback(api_messages_dicts, params.system_prompt)

stream = stream_with_retry(
    params.backend,
    api_messages_dicts,
    system=params.system_prompt,
    tools=tool_schemas,
    model=params.model,
    max_tokens=max_tokens,
)

async for event in stream:
    # dispatch into current_usage / text / tool_use accumulation
```

- `normalize_messages_for_api` strips hidden messages, merges same-role neighbours, drops orphan tool_results. See [architecture/messages-and-api.md](messages-and-api.md).
- `stream_with_retry` handles retryable rate limits, overloads, timeouts, connection failures, and HTTP 5xx errors with exponential backoff and jitter. It retries only before response content is emitted; a partial stream is never replayed. After persistent pre-content HTTP 5xx failures, the loop makes one emergency-compaction attempt because some local servers report context overflow as an opaque 500. The partial-stream marker also prevents this higher-level recovery from replaying content.
- Malformed native tool arguments are fed back to the model for a corrected call, with a bounded retry count. They do not end the turn immediately.
- Events are dispatched into `current_usage` (from `message_delta`), `TextBlock` / `ThinkingBlock` / `ToolUseBlock` accumulators.

## Phase 5 — Response assembly

```python
assistant_msg = AssistantMessage(
    content=assembled_blocks,
    model=current_model,
    stop_reason=stop_reason,
    usage=current_usage,
    ...
)
```

All the streamed bits become a single `AssistantMessage`. Text blocks have been streamed with `hide_in_api=True`; the final message is yielded with `hide_in_api=False` — the final view the caller stores.

Inline `<think>` content is separated into `ThinkingBlock`. When a text-tool protocol is active, parsing checks visible text first and then thinking/reasoning content, so a call buried by a reasoning model is still executed without exposing private reasoning. A response with no visible text and no tool call becomes an `empty_response` assistant error rather than a successful blank turn.

## Phase 6 — Yield + calibration

```python
yield assistant_msg

params.cost_tracker.add_usage(current_usage, current_model)
if current_usage.input_tokens > 0:
    state.last_input_tokens = current_usage.input_tokens
    state.last_output_tokens = current_usage.output_tokens
    state.messages_len_at_last_call = len(state.messages)
```

Store the reported usage for next iteration's pre-call estimate.

## Phase 7 — Abort & recovery (no tool use path)

```python
if params.abort_event and params.abort_event.is_set():
    # Close every unexecuted tool call with a synthetic error result.
    yield interruption_results
    yield create_user_interruption_message(tool_use=bool(tool_use_blocks))
    return

if not tool_use_blocks:
    # Possibly recover from max_output_tokens mid-thought
    if stop_reason == "max_tokens":
        # Escalate the automatic budget from 8k -> 64k
        if not state.max_output_tokens_override and not params.max_output_tokens:
            state.max_output_tokens_override = escalated_max_tokens
            state.transition = "max_output_tokens_escalation"
            continue
        # Multi-turn "Resume directly" recovery
        if state.max_output_tokens_recovery_count < limit:
            state.max_output_tokens_recovery_count += 1
            yield recovery_msg   # "Resume directly..."
            continue

    # Normal completion
    return
```

If the automatically sized output is cut off, Alan first retries the original request with `escalated_max_tokens` (64,000 by default, clamped to legal room). An explicit `max_output_tokens` is never escalated. If the larger/explicit call is still cut off, Alan stores the partial assistant response and injects a hidden "Resume directly" user turn, up to `max_output_tokens_recovery_limit`. Prompt-too-long errors trigger one emergency-compaction attempt around the backend stream.

## Phase 8 — Tool execution

```python
async for update in run_tools(
    tool_use_blocks, effective_tools, params.context,
    max_concurrency=params.settings.get("max_tool_concurrency", 10),
    permission_callback=params.permission_callback,
):
    if update.message:
        yield update.message
        tool_results.append(update.message)
```

`run_tools` in `alancode/tools/orchestration.py` batches the `ToolUseBlock`s into concurrent (for read-only) or serial (for write/exec) tasks. Each emits a `UserMessage` with a `ToolResultBlock`.

If a model-invoked skill restricts its allowed tools, `effective_tools` contains only that subset. If the turn is interrupted, Alan emits synthetic error results for every tool call that did not complete so the stored backend history remains valid.

For each tool, `run_tool_use` (in `alancode/tools/execution.py`):

1. Validate input via `tool.validate_input(args, ctx)`.
2. Fire pre-tool-use hook.
3. Run permission pipeline (`check_permissions`).
4. Call `tool.call(args, ctx)`.
5. Fire post-tool-use hook.
6. Return the `ToolResult` wrapped as a tool_result message.

## Phase 8.5 — Intensive-mode memory reminder

```python
state.turns_since_memory_update += 1
if (
    params.memory_mode == "intensive"
    and state.turns_since_memory_update >= params.settings.get("memory_reminder_threshold", 10)
):
    memory_reminder = create_user_message("<system-reminder>...</system-reminder>", hide_in_ui=True)
    tool_results.append(memory_reminder)
    yield memory_reminder
    state.turns_since_memory_update = 0
```

Periodic nudge to save memory in intensive mode.

## Phase 9 — Max-iterations check

```python
state.iteration_count += 1
if params.max_iterations_per_turn and state.iteration_count >= params.max_iterations_per_turn:
    yield create_attachment_message(
        "max_iterations_per_turn_reached",
        metadata={"max_iterations_per_turn": params.max_iterations_per_turn, "iteration_count": state.iteration_count},
    )
    return
```

Hard cap to prevent runaway loops.

## Phase 10 — Next iteration

```python
state.messages = list(messages_for_query) + [assistant_msg, *tool_results]
state.transition = None
state.max_output_tokens_override = None
iteration += 1
# loop
```

Assemble next iteration's starting state. Carry over `transition` for logging only.

## Where the loop exits

Terminal conditions — the loop `return`s:
- Phase 1 / 7: abort fired.
- Phase 3: blocking limit hit.
- Phase 7: tool-free response (normal completion).
- Phase 9: max iterations hit.

Or it propagates an exception to the caller if something goes badly wrong.

## Related

- [architecture/overview.md](overview.md) — how the loop fits in the broader system.
- [architecture/messages-and-api.md](messages-and-api.md) — what phase 4's `normalize_messages_for_api` does.
- [concepts/context-and-compaction.md](../concepts/context-and-compaction.md) — phase 2 in depth.
- [concepts/tools-and-permissions.md](../concepts/tools-and-permissions.md) — phase 8 in depth.
