# Context and compaction

Every model has a finite context window shared by the system prompt, conversation, tool definitions, and the model's next response. A long session or a burst of tool output can exhaust that space.

Alan Code solves this with a three-layer compaction pipeline that runs **before every API call**, progressively freeing space only when needed. You almost never have to think about it — but when you do, here's how it works.

## The one-line summary you see after each turn

```
Session: 8,118 in + 153 out = $0.0082 (estimated) | Conversation: 8,271 / 200,000 (4%)
```

- **Session** — cumulative tokens + $ since the session started.
- **Conversation** - how full the context window is **right now**. The compaction trigger is lower because Alan reserves room for the next response and a safety margin.

## The budget and its thresholds

Alan first reserves room for the next response and a safety margin. The remainder is the **usable input budget**. Its active thresholds are:

| Threshold | Triggers | Effect |
|---|---|---|
| **Compaction threshold T** (80% of usable input by default) | Layer C summarizes the conversation. | Compaction happens before the provider call. |
| **Clear target G** (between T and the blocking limit) | Layer B clears old tool results down to G. | Damage control when the payload has grown well past T. |
| **Blocking limit** (context window minus response reservation and safety margin) | The request cannot safely be sent unchanged. | Alan uses the deterministic fallback after a failed compaction, or reports that the fixed prompt and tools cannot fit. |

Layer A's per-result cap is also derived from T. See [reference/settings.md](../reference/settings.md) for the available overrides.

## The three compaction layers

Each iteration, if the predicted pre-call token count is over the threshold, layers run in order. Any layer that brings us below threshold stops the chain.

### Layer A — Truncate oversized tool results

`alancode/compact/compact_truncate.py`

Rewrites individual `tool_result` blocks whose content exceeds `tool_result_max_chars`. Its automatic value is the smaller of 10,000 characters and roughly 10% of T converted to characters. The block is replaced with:

```
<first 60% of the cap>
[ALAN-TRUNCATED: middle 91% of output elided (196,000 of 216,000 chars)]
<last 40% of the cap>
```

Truncation is middle-out: the head (structure, first errors) and the tail (conclusions, final state) survive; only the middle is elided. The `[ALAN-TRUNCATED` sentinel lets later compaction passes (and debugging) tell synthetic content from real. The structure of the message is preserved (it stays a `tool_result` with the same `tool_use_id`), so the conversation shape is intact.

**When it helps**: a single bloated tool output (e.g., `cat` on a 500 KB log) dominates the context. This layer chops just that one block without touching surrounding messages.

### Layer B — Clear old tool results

`alancode/compact/compact_clear.py`

Damage control, not economization: inactive until the estimated size exceeds the *clear target* G (halfway between the compaction threshold T and the blocking limit - see `alancode/budget.py`). Above G, it replaces the **content** of `tool_result` blocks oldest-first, stopping as soon as the estimate is back at G - so it can never bring the size below T, which means it never pre-empts Layer C (the information-preserving path). There is no keep-recent floor; recent results are protected by the target itself. The model still sees that a tool was called, but the output is reduced to:

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

## Recovery paths

If an API call **still** fails with `prompt too long` (the 413 path) despite the pre-call check:

1. The stream error handler catches the PTL signal.
2. Runs Layer C synchronously as an emergency compaction. The summarizer can itself shorten its input and retry when its request is too large.
3. Retries the call with the summarized history.

If summarization fails and the normal request is already at the blocking limit, Alan deterministically drops the oldest history while retaining a useful opening and a structurally valid recent tail. It records a compaction boundary and a visible notice before retrying. The same fallback runs after the configured number of consecutive compaction failures.

## Manual compaction

```
> /compact
```

Runs Layer C on demand, whether or not you're near the threshold. Useful before switching models mid-session (smaller context windows), or when you want to proactively condense a rambling exploration before continuing.

```
> /compact focus on the bug we just fixed, not the earlier refactoring
```

Any text after `/compact` is appended as *"Additional Instructions"* to the summarizer prompt, steering what to emphasize.

## Repeated compaction failures

If Layer C reaches `max_consecutive_compact_failures` (default 3), Alan stops spending calls on the summarizer and uses the deterministic fallback described above. The session remains usable, but older detail is lost.

## Tuning

Settings in `.alan/settings.json` (or `/settings <key> <value>` at runtime):

| Setting | Default | What it does |
|---|---|---|
| `context_window` | "auto" | Detected from registry/server/probe; int to override. |
| `compaction_threshold_percent` | "auto" (80) | When auto-compact kicks in, as % of the usable input budget. |
| `tool_result_max_chars` | "auto" | Layer A's per-result cap: min(10k, 10% of threshold). |
| `compact_max_output_tokens` | "auto" | Layer C summary budget, clamped to fit the window. |
| `max_consecutive_compact_failures` | 3 | After N failures: hard-truncate fallback (session lives). |

The blocking limit is no longer a setting: it is derived as `context_window - max_output_tokens - safety_margin` (the point where a full response no longer fits). See `alancode/budget.py` for the whole derivation DAG.

## Inspecting what happened

In the GUI, the **LLM Perspective** panel shows you the exact payload sent on each call, including any post-compact summary injected as a user message. This is the best debugging view when you want to understand what Alan remembers and what got compacted away.

From the CLI, `/status` shows the current `Conversation` tokens, and the session transcript on disk (`.alan/sessions/<id>/transcript.jsonl`) records every message including compaction boundaries.

## Related

- [concepts/agent-loop.md](agent-loop.md) — where in the loop compaction runs.
- [reference/settings.md](../reference/settings.md) — all tuning knobs.
- [reference/cost.md](../reference/cost.md) — what the status line numbers mean.
- [architecture/query-loop.md](../architecture/query-loop.md) — phase 2 of the loop is the compaction pipeline.
