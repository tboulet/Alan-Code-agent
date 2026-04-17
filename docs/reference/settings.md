# Settings reference

Every key in `.alan/settings.json` with its default, type, and effect. See [guides/configuration.md](../guides/configuration.md) for how settings resolve through the priority chain.

## Quick reference table

| Key | Type | Default | Area |
|---|---|---|---|
| `provider` | string | `litellm` | Provider |
| `model` | string | `anthropic/claude-sonnet-4-6` | Provider |
| `api_key` | string \| null | `null` (from env) | Provider — ephemeral, not persisted |
| `base_url` | string \| null | `null` | Provider |
| `tool_call_format` | string \| null | `null` | Provider — `hermes`, `glm`, `alan` |
| `permission_mode` | string | `edit` | Session |
| `max_iterations_per_turn` | int \| null | `null` (unlimited) | Session |
| `max_output_tokens` | int \| null | `null` | Session |
| `custom_system_prompt` | string \| null | `null` | System prompt |
| `append_system_prompt` | string \| null | `null` | System prompt |
| `memory` | string | `off` | Memory |
| `verbose` | bool | `false` | Logging |
| `hooks` | object | `{}` | Hooks |
| `compact_max_output_tokens` | int | `20_000` | Compaction |
| `capped_default_max_tokens` | int | `8_000` | Output control |
| `escalated_max_tokens` | int | `64_000` | Output control |
| `auto_compact_buffer_tokens` | int | `13_000` | Compaction |
| `warning_threshold_buffer_tokens` | int | `20_000` | Compaction |
| `blocking_limit_buffer_tokens` | int | `3_000` | Compaction |
| `max_consecutive_compact_failures` | int | `3` | Compaction |
| `compaction_threshold_percent` | int | `80` | Compaction |
| `max_compact_ptl_retries` | int | `3` | Compaction |
| `max_output_tokens_recovery_limit` | int | `3` | Error recovery |
| `max_tool_concurrency` | int | `10` | Tool execution |
| `tool_result_max_chars` | int | `20_000` | Tool execution |
| `compact_clear_keep_recent` | int | `10` | Compaction |
| `thinking_budget_default` | int | `10_000` | Thinking |
| `memory_reminder_threshold` | int | `10` | Memory |
| `max_scratchpad_sessions` | int | `5` | Sessions |
| `compaction_truncate_enabled` | bool | `true` | Compaction layer toggle |
| `compaction_clear_enabled` | bool | `true` | Compaction layer toggle |
| `compaction_auto_enabled` | bool | `true` | Compaction layer toggle |

Source of truth: `alancode/settings.py::SETTINGS_DEFAULTS`.

---

## Provider

### `provider`
Which LLM backend to use.
- `"anthropic"` — Anthropic's API directly (supports prompt caching, extended thinking, native tool use).
- `"litellm"` — Everything else (OpenAI, OpenRouter, Gemini, Vertex, Bedrock, Ollama, vLLM, SGLang, local servers).
- `"scripted"` — Deterministic test provider. See [reference/python-api.md](python-api.md).

### `model`
Model identifier. For LiteLLM, use `provider/model` form (`openai/gpt-4o`, `openrouter/google/gemini-2.5-pro`).

### `api_key`
If `null`, read from the provider's environment variable at init time. **Never persisted to disk** (flagged ephemeral).

### `base_url`
Override the API endpoint. Set for local servers (`http://localhost:8000/v1`).

### `tool_call_format`
Text-based tool-call protocol for models without native function calling. Options: `"hermes"`, `"glm"`, `"alan"`. When set, tool definitions are injected into the system prompt instead of being passed as API tool schemas, and the model's text output is parsed for tool calls. `null` (default) means use the model's native function calling.

---

## Session

### `permission_mode`
- `"yolo"` — allow everything without asking.
- `"edit"` (default) — allow read, ask for write/exec.
- `"safe"` — ask for everything except pure reads.

### `max_iterations_per_turn`
Hard cap on API calls per user message. `null` = unlimited. Prevents runaway loops; ignores reasoning-loops that should stop naturally.

### `max_output_tokens`
Ceiling on output tokens per call. Internally escalated up to `escalated_max_tokens` on recovery.

---

## System prompt

### `custom_system_prompt`
Replaces Alan's built-in system prompt entirely (sections 1–9). Sections 10–14 (skills, memory, scratchpad, ALAN.md, tool format) still append. Use with care — you lose all the tool-use guidance and safety instructions baked in.

### `append_system_prompt`
Appended to Alan's built-in system prompt. Safer way to inject project-specific nudges beyond what `ALAN.md` offers. Not cacheable by Anthropic (changes per session).

---

## Memory

### `memory`
- `"off"` (default) — no read/write of memory files.
- `"on"` — read on start, write only on explicit `/save` or user request.
- `"intensive"` — also proactively save after significant turns.

### `memory_reminder_threshold`
In `intensive` mode, iterations between memory-save reminders. Default 10.

---

## Logging

### `verbose`
If `true`, debug-level logging to stderr. Same effect as `--verbose` flag.

---

## Hooks

### `hooks`
Dict mapping hook-type name to list of hook configs. See [guides/hooks.md](../guides/hooks.md) for the schema and examples.

---

## Compaction

### `compaction_threshold_percent`
When Layer C (auto-compact) kicks in, as a percentage of the context window. Default 80.

### `tool_result_max_chars`
Layer A truncates any single tool result exceeding this. Default 20 000 chars.

### `compact_clear_keep_recent`
Layer B clears old tool results but keeps the most recent N. Default 10.

### `compact_max_output_tokens`
Output budget for the Layer C summarization call. Default 20 000.

### `auto_compact_buffer_tokens`
Emergency compaction trigger — if predicted tokens would land within this margin of the context ceiling. Default 13 000.

### `warning_threshold_buffer_tokens`
User-facing "context is filling up" warning trigger. Default 20 000.

### `blocking_limit_buffer_tokens`
Hard floor — refuses API calls that would land this close to the ceiling. Default 3 000.

### `max_consecutive_compact_failures`
Circuit-breaker threshold. After N failed compactions in a row, Alan surfaces an error and stops trying. Default 3.

### `max_compact_ptl_retries`
Prompt-too-long retries during the compaction summarize step itself. Default 3.

### `compaction_truncate_enabled` / `compaction_clear_enabled` / `compaction_auto_enabled`
Independent toggles for compaction layers A/B/C. All `true` by default.

---

## Output control

### `capped_default_max_tokens`
Default `max_tokens` per API call, even when the model would accept more. A slot-reservation optimization: by keeping this small, more context-window space is available for input. Default 8 000.

### `escalated_max_tokens`
Retry budget after the capped default is hit mid-generation. Default 64 000 — practical ceiling for most current models.

---

## Error recovery

### `max_output_tokens_recovery_limit`
When the model keeps getting cut off at `max_tokens`, how many "Resume directly" injections to try before giving up. Default 3.

---

## Tool execution

### `max_tool_concurrency`
Max parallel read-only tool executions. Write and exec tools always run serially. Default 10.

---

## Thinking

### `thinking_budget_default`
For models supporting extended thinking (Claude Sonnet 4, DeepSeek R1, o1-style): token budget the model can burn on reasoning before the visible response. `0` disables. Default 10 000.

---

## Sessions

### `max_scratchpad_sessions`
How many scratchpad directories to keep. Older ones are GC'd. Default 5.

---

## Validation

Invalid values at load-time (wrong type, out-of-enum string, negative integer where positive required) fall back to the default with a WARNING logged to stderr. Settings are never silently dropped — bad values are visible.

Validators live in `alancode/settings.py::_VALIDATORS`.

## Related

- [guides/configuration.md](../guides/configuration.md) — priority chain, how to change settings at runtime.
- [reference/cli.md](cli.md) — CLI flags map 1-to-1 with the most common settings.
- [reference/slash-commands.md](slash-commands.md) — `/settings` and `/settings-project`.
