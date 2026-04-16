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

This terminology is why `max_iterations_per_turn` (the setting formerly called `max_turns`) is named the way it is — it caps how many API calls a single user message can trigger, not how many user messages a session can have.

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

The stream is driven by `provider.stream(...)` which yields structured events: `StreamTextDelta`, `StreamToolUseStart`, `StreamToolUseStop`, etc. The loop consumes these, assembles them into messages, and yields the result to the caller.

## Error recovery

Three kinds of errors the loop handles transparently:

1. **Output token limit hit mid-thought**: the assistant gets cut off. The loop escalates `max_tokens` from 8k → 64k and retries. If still cut off, injects "Resume directly, no apology, pick up mid-thought" up to 3 times.
2. **Prompt too long (413)**: triggers an emergency compaction and re-runs with the summarized history.
3. **Retryable network errors (rate limits, timeouts, 529)**: handled in `alancode/api/retry.py` with exponential backoff.

Non-retryable errors (400, 401, 403) propagate immediately — they're user-actionable (wrong key, bad request shape).

## Abort handling

Ctrl+C at any point:
- Sets an `asyncio.Event` the loop checks at phase 1 and phase 7.
- Causes `ask_user_callback` to raise `CancelledError`, which propagates through the tool execution layer.
- The REPL catches it, prints "Turn interrupted.", clears the abort flag, and waits for new input.

The session's `_last_usage` and `turn_count` are still flushed to disk via a best-effort block in the agent's `finally`, so accounting survives the interrupt.

## State management

Between iterations the loop carries a `LoopState` (`alancode/query/state.py`):

- `messages` — the full list.
- `iteration_count` — how many API calls this turn has made.
- `max_output_tokens_recovery_count` — for the 8k→64k escalation.
- `has_attempted_emergency_compact` — one-shot per turn.
- `last_input_tokens` / `last_output_tokens` — used by the pre-call compaction estimate.
- `cached_model_info` — avoid re-querying provider for context window each iteration.

When a turn ends, `LoopState` is discarded. The durable state is `self._messages` on the agent and `SessionState` on disk.

## Related reading

- [reference/settings.md](../reference/settings.md) — tune compaction thresholds, max iterations, retry budget.
- [concepts/context-and-compaction.md](context-and-compaction.md) — the three compaction layers in detail.
- [concepts/tools-and-permissions.md](tools-and-permissions.md) — how tool execution actually happens inside phase 8.
- [architecture/query-loop.md](../architecture/query-loop.md) — phase-by-phase code walkthrough for contributors.
