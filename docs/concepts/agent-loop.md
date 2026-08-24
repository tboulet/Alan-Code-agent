# The agent loop

Alan Code's heart is a single async generator that drives every conversation: `query_loop()` in `alancode/query/loop.py`. Understanding how it's structured — and the vocabulary Alan uses around it — makes the rest of the system easier to reason about.

## Vocabulary

Three terms with precise meanings:

| Term | Definition |
|---|---|
| **Iteration** | One pass through `query_loop`'s while-loop — one API call, optionally followed by tool execution. |
| **Turn** | Everything that happens between two user inputs. A turn contains 1+ iterations until the agent stops and waits for input again. |
| **Session** | Full conversation from start until `/clear` or process exit. Persisted on disk; can be resumed with `--resume`. |

So:

- The user says "Fix this bug" → that starts a **turn**.
- Inside the turn, Alan may run multiple **iterations**: call the LLM → get a `tool_use` → run the tool → call the LLM again with the result → ... → final text reply.
- The whole conversation history across turns is the **session**.

This terminology explains the name `max_iterations_per_turn` (formerly `max_turns`), but the exact implementation is narrower: it counts completed model→tool execution cycles, not recovery-only model calls and not user messages in the session.

## The loop structure

Each iteration runs through 10 phases. Simplified pseudo-code:

```
while True:
    # 1. Check abort (Ctrl+C)
    # 2. Inject turn-start reminders (date/time), drain queued messages
    # 3. Compaction pre-check:
    #       - Layer A: truncate oversized tool results
    #       - Layer B: clear old tool results  
    #       - Layer C: auto-compact if still above threshold
    # 4. Blocking-limit check (refuse call if too close to ceiling)
    # 5. API call (streaming)
    # 6. Process response — collect content blocks + tool_use blocks
    # 7. Handle no-tool-use responses (completion or recovery)
    # 8. Execute tools (concurrent for read-only, serial for writes)
    # 9. Check max_iterations_per_turn
    # 10. Loop back
```

Each phase is small and local. See [architecture/query-loop.md](../architecture/query-loop.md) for the full phase-by-phase walkthrough with file:line pointers.

## What ends a turn

A turn ends when:
- The model returns a text-only response with no tool calls.
- The user hits Ctrl+C (clean abort).
- `max_iterations_per_turn` is reached.
- A blocking error is hit (context overflow despite compaction, repeated output-token limits, etc.).

When any of these happen, control returns to the REPL which prints the turn's cost summary and waits for the next user input.

## Streaming

Every API call streams. You see:
- Token-by-token text (Rich's live-print on CLI, WebSocket events in GUI).
- Incremental "thinking" blocks for models that support them.
- Tool call blocks render in a boxed panel the moment they arrive.

The stream is driven by `backend.stream(...)` which yields structured events: `StreamTextDelta`, `StreamToolUseStart`, `StreamToolUseStop`, etc. The loop consumes these, assembles them into messages, and yields the result to the caller.

## Error recovery

Three kinds of errors the loop handles transparently:

1. **Output token limit hit mid-thought**: the assistant gets cut off. If the resolved starting budget is below `escalated_max_tokens` (64k by default), the loop retries at that higher, window-clamped target. If still cut off, it injects "Resume directly, no apology, pick up mid-thought" up to 3 times. Suspect truncated tool calls are answered with synthetic errors and never executed.
2. **Prompt too long (413)**: triggers an emergency compaction and re-runs with the summarized history.
3. **Retryable network errors (rate limits, timeouts, 529)**: handled in `alancode/api/retry.py` with exponential backoff.

Non-retryable errors (400, 401, 403) skip transport retries. The query loop normally turns them into a final assistant error event; unexpected exceptions still propagate to the caller.

## Abort handling

Ctrl+C / `abort()`:
- Sets an `asyncio.Event` the loop checks before a model call, after a complete stream, and after a tool batch. It is cooperative and does not forcibly cancel an in-flight backend request or a tool that does not poll the signal.
- Causes `ask_user_callback` to raise `CancelledError`, which propagates through the tool execution layer.
- The REPL catches it, prints "Turn interrupted.", clears the abort flag, and waits for new input.

The session's `_last_usage` and `turn_count` are still flushed to disk via a best-effort block in the agent's `finally`, so accounting survives the interrupt.

## State management

Between iterations the loop carries a `LoopState` (`alancode/query/state.py`):

- `messages` — the full list.
- `iteration_count` - how many tool-execution cycles this turn has completed (not every recovery-only API call).
- `max_output_tokens_recovery_count` - how many hidden "Resume directly" continuation turns were attempted.
- `has_attempted_emergency_compact` — one-shot per turn.
- `last_input_tokens` / `last_output_tokens` — used by the pre-call compaction estimate.
- `max_output_tokens_override` - temporary escalation target for the next retry.

When a turn ends, `LoopState` is discarded. The durable state is `self._messages` on the agent and `SessionState` on disk.

## Related reading

- [reference/settings.md](../reference/settings.md) — tune compaction thresholds, max iterations, retry budget.
- [concepts/context-and-compaction.md](context-and-compaction.md) — the three compaction layers in detail.
- [concepts/tools-and-permissions.md](tools-and-permissions.md) — how tool execution actually happens inside phase 8.
- [architecture/query-loop.md](../architecture/query-loop.md) — phase-by-phase code walkthrough for contributors.
