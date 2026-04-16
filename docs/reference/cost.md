# Cost & token tracking

After every agent turn, Alan Code prints a one-line summary beneath the response:

```
  Session: 16,378 in + 53 out = $0.0050 (estimated) | Conversation: 16,356 / 1,048,576 (1%)
```

## What each field means

- **Session (in / out)** — total input and output tokens since the session started (not just the last turn). Persists across turns, cleared on `/clear` or a new session.
- **`= $…` (estimated)** — a best-effort USD estimate from the LiteLLM pricing registry. Shows `unknown` when the model isn't in the registry (common for local models, new releases, and fine-tunes).
- **Conversation** — current conversation size in tokens vs. the model's context window, with percentage. Updated every turn; once it reaches the configured compaction threshold (`compaction_threshold_percent`, default 80%), Alan starts compacting.

## Deeper breakdown — `/status`

The `/status` command shows the full accounting:

| Row | Description |
|---|---|
| `Input tokens` | Non-cached input sent to the model. |
| `Cache creation tokens` | Tokens written to the prompt cache this session. Billed higher than regular input on Anthropic; one-time per cache entry. |
| `Cache read tokens` | Tokens served from the prompt cache. Billed at ~10% of regular input — this is where prompt caching pays off. |
| `Total input` | Sum of the three above. This is what the "in" in the one-liner refers to. |
| `Output tokens` | Model-generated output tokens across the session. |
| `Estimated cost` | USD estimate. `unknown` if the model's pricing isn't registered. |

## Why "estimated"

Alan Code computes cost client-side from token counts × registered per-token prices. It does not (yet) consume the provider's billing APIs. So:

- Numbers are close but not authoritative. Check your provider's dashboard for exact figures.
- Cost is unknown for models LiteLLM doesn't price (local models, recent releases, fine-tunes).
- Cache-pricing math is applied when the provider's cache metadata is returned on the response (Anthropic, some LiteLLM endpoints).

## Tuning the budget

Key settings that affect cost behavior (see [`cli.md`](cli.md)):

- `max_iterations_per_turn` — hard cap on how many API calls a single user message can consume.
- `max_output_tokens` — ceiling on per-call output, with internal escalation up to `escalated_max_tokens` when the model hits the limit and needs to recover.
- `compaction_threshold_percent` — at what fraction of the context window Alan starts compacting to avoid hitting the hard ceiling (where calls would be refused).
- `auto_compact_buffer_tokens` — how much headroom under the context window triggers automatic compaction.

## Programmatic access

From `AlanCodeAgent`:

```python
agent.cost_usd       # float — session cost in USD
agent.cost_unknown   # bool  — True when pricing isn't available
agent.usage          # Usage — dataclass with input/output/cache breakdown
```

See `alancode/messages/types.py → Usage` for the full shape.
