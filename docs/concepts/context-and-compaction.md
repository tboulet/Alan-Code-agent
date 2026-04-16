# Context and compaction

Long sessions run out of context. The model's context window is fixed — Claude Sonnet 4 has 200k tokens, GPT-4o has 128k, Gemini 2.5 has 1M. Without intervention, a dense debugging session fills the window in 15–30 turns.

Alan Code solves this with a three-layer compaction pipeline that runs **before every API call**, progressively freeing space only when needed. You almost never have to think about it — but when you do, here's how it works.

## The one-line summary you see after each turn

```
Session: 8,118 in + 153 out = $0.0082 (estimated) | Conversation: 8,271 / 200,000 (4%)
```

- **Session** — cumulative tokens + $ since the session started.
- **Conversation** — how full the context window is **right now**. When this approaches 80 %, compaction will fire on the next call.

## The four concentric thresholds

Alan uses four concentric token thresholds, each with its own reaction:

| Threshold | Triggers | Effect |
|---|---|---|
| **Warning threshold** (`context_window - 20k`) | Display a "getting full" signal to the user. | Informational. |
| **Auto-compact threshold** (`context_window * 80 %`) | Layer A → B → C run in order until we're back under it. | Compaction happens on the *next* pre-call check. |
| **Compact buffer** (`context_window - 13k`) | Emergency compact inside the loop. | Last-chance summary on a 413 response. |
| **Blocking limit** (`context_window - 3k`) | Refuse the API call outright. | Turn ends with "Conversation too long. Please run /compact or start a new session." |

All four are tunable via settings — see [reference/settings.md](../reference/settings.md).

## The three compaction layers

Each iteration, if the predicted pre-call token count is over the threshold, layers run in order. Any layer that brings us below threshold stops the chain.

### Layer A — Truncate oversized tool results

`alancode/compact/compact_truncate.py`

Rewrites individual `tool_result` blocks whose content exceeds `tool_result_max_chars` (default 20 000 chars). The block is replaced with:

```
[ALAN-TRUNCATED] Tool result truncated — 216000 chars exceeded 20000 limit.
```

The `[ALAN-TRUNCATED]` sentinel lets later compaction passes (and debugging) tell synthetic content from real. The structure of the message is preserved (it stays a `tool_result` with the same `tool_use_id`), so the conversation shape is intact.

**When it helps**: a single bloated tool output (e.g., `cat` on a 500 KB log) dominates the context. This layer chops just that one block without touching surrounding messages.

### Layer B — Clear old tool results

`alancode/compact/compact_clear.py`

Replaces the **content** of older `tool_result` blocks with a short sentinel, keeping only the N most recent (`compact_clear_keep_recent = 10` by default). The model still sees that a tool was called, but the output is reduced to:

```
[cleared to free context space]
```

**When it helps**: the agent has called `Read` 50 times; each result is small but together they dominate. This flattens the long tail.

### Layer C — Auto-compact (forked summarizer)

`alancode/compact/compact_auto.py`

The heavy hitter. If we're still over threshold after A and B:

1. Fork a **separate** LLM call with **no tools** and a specific summarization prompt (the 9-section template in `alancode/compact/prompt.py`).
2. That call produces an `<analysis>…</analysis><summary>…</summary>` response.
3. The summary replaces the pre-compaction history. A `SystemMessage(subtype=COMPACT_BOUNDARY)` marker is inserted so later compactions know where the cutoff is.
4. A post-compact user message is injected: *"This session is being continued from a previous conversation that ran out of context. Continue from where it left off without asking questions."*

**When it helps**: the conversation has substantial back-and-forth that no mechanical truncation can compress. The summary captures the intent, key decisions, pending tasks, and the exact current state.

## The emergency path

If an API call **still** fails with `prompt too long` (the 413 path) despite the pre-call check:

1. The stream error handler catches the PTL signal.
2. Runs Layer C synchronously as an emergency compaction.
3. Retries the call with the summarized history.

This is a belt-and-suspenders measure — rare in practice but essential for reliability.

## Manual compaction

```
> /compact
```

Runs Layer C on demand, whether or not you're near the threshold. Useful before switching models mid-session (smaller context windows), or when you want to proactively condense a rambling exploration before continuing.

```
> /compact focus on the bug we just fixed, not the earlier refactoring
```

Any text after `/compact` is appended as *"Additional Instructions"* to the summarizer prompt, steering what to emphasize.

## The circuit breaker

If Layer C fails three times in a row (`max_consecutive_compact_failures = 3`), the circuit breaker fires and Alan surfaces:

```
Compaction has failed 3 times consecutively. Use /clear to start fresh.
```

Three failures strongly suggests an adversarial state (token-counting off by tens of thousands, summary prompt confusing the model, etc.). Rather than burn money in a loop, Alan bails out.

## Tuning

Settings in `.alan/settings.json` (or `/settings <key> <value>` at runtime):

| Setting | Default | What it does |
|---|---|---|
| `compaction_threshold_percent` | 80 | When auto-compact kicks in, as % of context window. |
| `tool_result_max_chars` | 20 000 | Layer A's per-tool-result size cap. |
| `compact_clear_keep_recent` | 10 | Layer B's "keep recent N" count. |
| `compact_max_output_tokens` | 20 000 | Output budget for the Layer C summary call. |
| `auto_compact_buffer_tokens` | 13 000 | How close to ceiling before emergency compact. |
| `blocking_limit_buffer_tokens` | 3 000 | Hard floor — refuse calls below this. |
| `max_consecutive_compact_failures` | 3 | Circuit breaker threshold. |

## Inspecting what happened

In the GUI, the **LLM Perspective** panel shows you the exact payload sent on each call, including any post-compact summary injected as a user message. This is the best debugging view when you want to understand what Alan remembers and what got compacted away.

From the CLI, `/status` shows the current `Conversation` tokens, and the session transcript on disk (`.alan/sessions/<id>/transcript.jsonl`) records every message including compaction boundaries.

## Related

- [concepts/agent-loop.md](agent-loop.md) — where in the loop compaction runs.
- [reference/settings.md](../reference/settings.md) — all tuning knobs.
- [reference/cost.md](../reference/cost.md) — what the status line numbers mean.
- [architecture/query-loop.md](../architecture/query-loop.md) — phase 2 of the loop is the compaction pipeline.
