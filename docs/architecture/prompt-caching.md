# Prompt caching

Anthropic's prompt caching feature lets clients mark sections of the prompt as cacheable — subsequent calls with the same cached prefix pay ~10 % of the normal input-token cost for that portion. For long-running agent sessions, this is the difference between a $0.50 conversation and a $5 one.

This page explains how Alan arranges its Anthropic API calls to maximise cache hits.

## The principle

Anthropic caches prompt prefixes at **cache breakpoints**. A breakpoint is a `cache_control: {"type": "ephemeral"}` annotation on a content block. The cache is addressed by the **entire prefix up to and including the breakpoint**.

Three rules:

1. If the prefix is identical to a previous call → cache hit (10 % of normal cost for those tokens).
2. If any byte earlier in the prefix differs → cache miss (full cost).
3. Maximum 4 breakpoints per call.

So the question becomes: **where to place breakpoints to maximise hits across our typical workload**?

## Alan's 4-block strategy

Alan uses (up to) 4 blocks in the `system` list, ordered from most-static to most-dynamic:

```python
system = [
    {"type": "text", "text": intro,                 "cache_control": {"type": "ephemeral"}},
    {"type": "text", "text": static_sections_2_8,   "cache_control": {"type": "ephemeral"}},
    {"type": "text", "text": dynamic_sections_9_14, "cache_control": {"type": "ephemeral"}},  # (optional)
    {"type": "text", "text": attribution_footer},                                           # no breakpoint
]
```

- **Block 1 — Intro**: single paragraph. Bytes-identical across every single call. Anthropic-wide cache scope — even the same intro from a different Alan user will hit.
- **Block 2 — Static sections (2–8)**: system rules, doing tasks, executing actions, using tools, tone, communication, session-specific guidance. ~20k tokens. Identical across sessions using the same tool set. Session-wide cache scope.
- **Block 3 — Dynamic sections (9–14)**: environment, skills, memory, scratchpad, ALAN.md, tool-format. Changes per session, per mode, per project. Session-wide cache scope (if any of these content changes, the cache invalidates).
- **No breakpoint** on the attribution footer / whatever comes last.

## Actual layout in the code

`alancode/providers/anthropic_provider.py` builds the system list from the `system_prompt: list[str]` that `query_loop` passes. Each entry gets wrapped into a text block; the implementation picks cache-control placement based on which blocks are "static" vs "dynamic".

In practice we set breakpoints at indices 0 and 1 (intro + accumulated static sections) to keep within the 4-breakpoint limit when messages also want their own cache breakpoints.

## Messages caching

Beyond the system prompt, Anthropic also allows caching at the **tail of the messages list** — useful when you're sending the same conversation repeatedly (as in agent loops with iterative tool results).

Alan places a cache breakpoint on the most recent assistant message's last block when it's large enough to be worth caching (heuristic: >1000 tokens). This caches the prefix ending at the last assistant message, so the next iteration (which appends a tool_result and continues) gets the whole history cached for 10 % cost.

## Cache invalidation triggers

Any change that alters the prefix's bytes invalidates the cache from that point onward:

- **Beta header changes** — switching `anthropic-beta: prompt-caching-2024-07-31,extended-thinking-2025-01-02` vs just one. Alan sets these statically per session so they don't change mid-session.
- **Model switch** (`/model`) — different model → different cache space entirely.
- **`ALAN.md` edit** — dynamic-block content changes → invalidates block 3 and everything after. The static first two blocks still hit.
- **Skill added/removed** — same.
- **Memory file save** — memory MEMORY.md is in block 3; a save invalidates block 3.
- **Session timestamp in environment section** — set once on session start, doesn't change within a session.

## Reading the cost summary

When caching is working:

```
Session: 12,480 in + 187 out = $0.0128 (estimated) | Conversation: 24,311 / 200,000 (12%)
```

The `12,480 in` breaks down (in `/status`):

```
Input tokens              482     ← Real new input (new user message + prior assistant response)
Cache creation tokens    7,812    ← First-time cache writes (paid at 1.25x normal input)
Cache read tokens       12,017    ← Cache hits (paid at 0.10x normal input)  
Total input             20,311
```

High cache-read ratio = good caching is working. Low cache-read after the first few turns = cache is being invalidated somehow; investigate what changed.

## Debugging cache behaviour

Enable `--verbose` to log Anthropic beta headers and cache-control markers per call.

In the GUI, the LLM Perspective panel shows the exact system prompt list — compare call-to-call to spot what changed when cache reads drop.

If a setting change (e.g. switching memory from `off` to `on`) invalidates a large prefix, you'll see a one-time spike in cache-creation tokens before the new state stabilises.

## LiteLLM and non-Anthropic providers

- **LiteLLM to Anthropic**: LiteLLM forwards cache-control markers unchanged. Same behaviour as direct Anthropic.
- **LiteLLM to OpenAI**: OpenAI introduced prompt caching in late 2024 but without explicit breakpoints — automatic, prefix-based. Alan doesn't annotate anything; OpenAI caches whatever's stable.
- **LiteLLM to Gemini / Bedrock / others**: no caching or different semantics. Our cache-control markers are ignored harmlessly.
- **Local models (vLLM, etc.)**: no caching. Full-cost input every call. For long sessions with local models, compaction matters more.

## Related

- [architecture/system-prompt.md](system-prompt.md) — the 14 sections that feed into the cache blocks.
- [reference/cost.md](../reference/cost.md) — what the status line numbers mean.
- [reference/providers.md](../reference/providers.md) — per-provider caching support.
- [Anthropic's prompt caching docs](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching).
