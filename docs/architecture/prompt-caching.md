# Prompt caching

Prompt caching lets backends reuse previously computed prefixes, reducing input cost by up to 90% on cached tokens. For multi-turn agent sessions, this is the single largest cost lever.

## How it works

Caching is backend-specific. Alan Code applies caching markers where possible and lets backends handle the rest:

- **Anthropic** (direct): `cache_control: {"type": "ephemeral"}` markers on content blocks. Prefix up to the marker is cached. Max 4 breakpoints per request. Cache hits cost 10% of regular input; writes cost 1.25x.
- **OpenAI**: Automatic prefix-based caching. No markers needed.
- **OpenRouter → Anthropic**: Passes `cache_control` through to Anthropic's API. Same mechanics.
- **Local models**: No caching.

## Alan's caching strategy

### Anthropic backend (`anthropic_backend.py`)

Places up to 4 `cache_control` breakpoints per request:

1. **Last tool definition** — caches all tool schemas (~5-10K tokens)
2. **Last static system prompt section** — caches tools + stable prompt sections (intro, rules, guidelines)
3. **Last system prompt section** — caches tools + full system prompt including dynamic sections
4. **Last assistant message** — caches the entire conversation prefix

The system prompt is split at `system_static_boundary`. The built-in/environment/scratchpad prefix is stable for one agent; skills, memory, project instructions, and explicit appended instructions follow it and may change. `get_system_prompt()` communicates the boundary to each backend.

### LiteLLM backend (`litellm_backend.py`)

Uses the same `cache_control` markers injected into system message content blocks, tool definitions, and assistant messages. LiteLLM passes these through to backends that support them and ignores them for backends that don't.

## Cache invalidation

Changes that invalidate part of the cache:

| Change | Breakpoints invalidated | Still cached |
|---|---|---|
| Memory save (`/save`, intensive mode) | BP3 (dynamic system) | BP1 (tools), BP2 (static system) |
| Skill created/removed | BP3 | BP1, BP2 |
| ALAN.md edited | BP3 | BP1, BP2 |
| New user message (normal turn) | BP4 (conversation) | BP1, BP2, BP3 |
| Model switch (`/model`) | All (different cache space) | None |



## Related

- [reference/cost.md](../reference/cost.md) — what the status line numbers mean.
