# Prompt caching

Prompt caching lets providers reuse previously computed prefixes, reducing input cost by up to 90% on cached tokens. For multi-turn agent sessions, this is the single largest cost lever.

## How it works

Caching is provider-specific. Alan Code applies caching markers where possible and lets providers handle the rest:

- **Anthropic** (direct): `cache_control: {"type": "ephemeral"}` markers on content blocks. Prefix up to the marker is cached. Max 4 breakpoints per request. Cache hits cost 10% of regular input; writes cost 1.25x.
- **OpenAI**: Automatic prefix-based caching. No markers needed.
- **OpenRouter → Anthropic**: Passes `cache_control` through to Anthropic's API. Same mechanics.
- **Local models**: No caching.

## Alan's caching strategy

### Anthropic provider (`anthropic_provider.py`)

Places up to 4 `cache_control` breakpoints per request:

1. **Last tool definition** — caches all tool schemas (~5-10K tokens)
2. **Last static system prompt section** — caches tools + stable prompt sections (intro, rules, guidelines)
3. **Last system prompt section** — caches tools + full system prompt including dynamic sections
4. **Last assistant message** — caches the entire conversation prefix

The system prompt is split into static (sections 0-6, byte-identical across calls) and dynamic (sections 7+, stable within a session but can change on memory/skill/ALAN.md updates). This split is communicated via `system_static_boundary` from `get_system_prompt()`.

### LiteLLM provider (`litellm_provider.py`)

Uses the same `cache_control` markers injected into system message content blocks, tool definitions, and assistant messages. LiteLLM passes these through to providers that support them and ignores them for providers that don't.

## Cache invalidation

Changes that invalidate part of the cache:

| Change | Breakpoints invalidated | Still cached |
|---|---|---|
| Memory save (`/save`, intensive mode) | BP3 (dynamic system) | BP1 (tools), BP2 (static system) |
| Skill created/removed | BP3 | BP1, BP2 |
| ALAN.md edited | BP3 | BP1, BP2 |
| New user message (normal turn) | BP4 (conversation) | BP1, BP2, BP3 |
| Model switch (`/model`) | All (different cache space) | None |

## Minimum prefix size

Caching is silently ignored if the prefix is below the model's minimum:

| Model tier | Minimum |
|---|---|
| Sonnet / Opus | 1024 tokens |
| Haiku | 2048 tokens |

Alan's system prompt is ~15-20K tokens, so this minimum is always exceeded.

## Verifying caching works

In `/status`, check `Cache creation tokens` and `Cache read tokens`:

| Turn | Cache creation | Cache read | Interpretation |
|---|---|---|---|
| 1 | ~prefix size | 0 | Cache populated. Expected. |
| 2+ | 0 | ~prefix size | Cache hit. Working. |
| N (after pause) | ~prefix size | 0 | Cache evicted (TTL expired). |

**Note:** LiteLLM's streaming mode may not propagate cache token details from some providers (e.g., OpenRouter). Cache tokens may show as 0 in `/status` even when caching is active. The cost savings still apply on the provider's billing side. The Anthropic direct provider reports cache tokens accurately.

## Related

- [reference/cost.md](../reference/cost.md) — what the status line numbers mean.
- [reference/providers.md](../reference/providers.md) — per-provider caching support.
