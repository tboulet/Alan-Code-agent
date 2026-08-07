# Settings reference

Every key in `.alan/settings.json` with its default, type, and effect. See [guides/configuration.md](../guides/configuration.md) for how settings resolve through the priority chain.

## Quick reference table

| Key | Type | Default | Area |
|---|---|---|---|
| `backend` | string | `anthropic-native` | Backend |
| `model` | string | `claude-sonnet-4-6` | Backend |
| `api_key` | string \| null | `null` (from env) | Backend — ephemeral, not persisted |
| `base_url` | string \| null | `null` | Backend |
| `request_timeout` | int \| `"auto"` | `"auto"` | Backend request timeout |
| `tool_call_format` | string \| null | `null` | Backend text-tool protocol |
| `permission_mode` | string | `edit` | Session |
| `max_iterations_per_turn` | int \| null | `null` (unlimited) | Session |
| `max_output_tokens` | int \| null | `null` | Session |
| `custom_system_prompt` | string \| null | `null` | System prompt |
| `append_system_prompt` | string \| null | `null` | System prompt |
| `memory` | string | `off` | Memory |
| `verbose` | bool | `false` | Logging |
| `hooks` | object | `{}` | Hooks |
| `context_window` | int \| `"auto"` | `"auto"` | Budget |
| `compact_max_output_tokens` | int \| `"auto"` | `"auto"` | Compaction |
| `capped_default_max_tokens` | int | `8_000` | Output control |
| `escalated_max_tokens` | int | `64_000` | Output control |
| `auto_compact_buffer_tokens` | int | `13_000` | Compaction |
| `warning_threshold_buffer_tokens` | int | `20_000` | Compaction |
| `max_consecutive_compact_failures` | int | `3` | Compaction |
| `compaction_threshold_percent` | int \| `"auto"` | `"auto"` (80) | Compaction |
| `max_compact_ptl_retries` | int | `3` | Compaction |
| `max_output_tokens_recovery_limit` | int | `3` | Error recovery |
| `max_tool_concurrency` | int | `10` | Tool execution |
| `tool_result_max_chars` | int \| `"auto"` | `"auto"` | Tool execution |
| `thinking_budget_default` | int | `10_000` | Thinking |
| `memory_reminder_threshold` | int | `10` | Memory |
| `max_scratchpad_sessions` | int | `5` | Sessions |
| `compaction_truncate_enabled` | bool | `true` | Compaction layer toggle |
| `compaction_clear_enabled` | bool | `true` | Compaction layer toggle |
| `compaction_auto_enabled` | bool | `true` | Compaction layer toggle |

Source of truth: `alancode/settings.py::SETTINGS_DEFAULTS`.

---

## Backend

### `backend`
Transport (advanced — inferred from `model` when not set explicitly).
- `"anthropic-native"` — direct Anthropic SDK (`cache_control`, native thinking, native `tool_use`). Default for bare `claude-*` model names.
- `"auto"` — universal LiteLLM transport (OpenAI, OpenRouter, Gemini, Vertex, Bedrock, Ollama, vLLM, SGLang, local servers). Default for everything else.
- `"scripted"` — deterministic test backend. See [reference/python-api.md](python-api.md).

### `model`
Model identifier. Bare names (`claude-sonnet-4-6`, `gpt-4o`) or LiteLLM-style `provider/model` prefixes (`openrouter/google/gemini-2.5-pro`, `ollama/llama3.1`, `anthropic/claude-sonnet-4-6`).

Changing `model` mid-session also re-infers `backend` (bare `claude-*` → `anthropic-native`, anything else → `auto`).

### `api_key`
If `null`, read from the model provider's environment variable at init time. **Never persisted to disk** (flagged ephemeral).

### `base_url`
Override the API endpoint. Set for local servers (`http://localhost:8000/v1`).

### `request_timeout`
Model-request timeout in seconds. `"auto"` uses the SDK default for normal cloud routes and 3,600 seconds for a custom `base_url`, which accommodates slow local inference. An explicit positive integer overrides either behavior.

### `tool_call_format`
Text-based tool-call protocol for models without reliable native function calling. Options: `"hermes"`, `"hermes_xml"`, `"glm"`, `"alan"`, and `"meta_json"`. When set, tool definitions are injected into the system prompt instead of being passed as API tool schemas. Alan parses both visible text and reasoning/thinking output, because some reasoning models place the call there. `null` (default) means use native structured tool calls.

---

## Session

### `permission_mode`
- `"yolo"` — allow everything without asking.
- `"edit"` (default) — allow read, ask for write/exec.
- `"safe"` — ask for everything except pure reads.

### `max_iterations_per_turn`
Hard cap on API calls per user message. `null` = unlimited. Prevents runaway loops; ignores reasoning-loops that should stop naturally.

### `max_output_tokens`
Explicit ceiling on output tokens per call. Alan honors an explicit value; automatic first-retry escalation applies only when this setting is `null` or `"auto"`. Every request is still clamped so input, output, and safety margin fit the context window.

---

## System prompt

### `custom_system_prompt`
Replaces Alan's built-in prompt, skills, memory, scratchpad, and `ALAN.md` context. Required text-tool schemas still append when `tool_call_format` is set. Use with care: you lose Alan's normal tool guidance and safety instructions.

### `append_system_prompt`
Appended after the normal built-in/project prompt. If `custom_system_prompt` is also set, it appends directly after that replacement instead. This is the safer way to add a focused instruction without discarding Alan's defaults.

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
If `true`, debug-level logging to stderr. Same effect as `--verbose`; it changes the level, not the format. CLI records always use:

```text
2026-08-08 14:32:10,123 [Alan Code] alancode.query.loop DEBUG: message
```

---

## Hooks

### `hooks`
Dict mapping hook-type name to list of hook configs. See [guides/hooks.md](../guides/hooks.md) for the schema and examples.

---

## Compaction

### `context_window`
Overrides the model's detected context window. `"auto"` (default) resolves it from the model registry, the serving endpoint's metadata, or a one-time probe (see `alancode/budget.py` and `alancode/backends/cw_probe.py`). Set an integer only when detection is wrong.

### `compaction_threshold_percent`
When Layer C (auto-compact) kicks in, as a percentage of the *usable input budget* (context window minus output reservation and margin). `"auto"` = 80.

### `tool_result_max_chars`
Layer A truncates any single tool result exceeding this (middle-out: head and tail kept, sentinel between). `"auto"` = `min(10 000, 10% of the compaction threshold in chars)`.

### `compact_max_output_tokens`
Output budget for the Layer C summarization call. `"auto"` = `min(20 000, what fits in the window)`; always clamped at call time so the summarizer request is legal.

### `auto_compact_buffer_tokens`
Retained for settings-file compatibility. The 1.2 loop derives its active thresholds from the resolved context budget instead.

### `warning_threshold_buffer_tokens`
Retained for settings-file compatibility; it currently has no effect on loop behavior.

### `max_consecutive_compact_failures`
Circuit-breaker threshold. After N failed compactions in a row, Alan hard-truncates the oldest history (deterministic, no LLM) with a visible notice and keeps the session alive. Default 3.

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
When the model keeps getting cut off at `max_tokens`, how many "Resume directly" continuation turns to try before giving up. With automatic output sizing, Alan first retries the original request with `escalated_max_tokens`; continuation turns begin only if that retry is also cut off. Default 3.

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

Validators live in `alancode/settings.py::SETTING_VALIDATORS`.

## Related

- [guides/configuration.md](../guides/configuration.md) — priority chain, how to change settings at runtime.
- [reference/cli.md](cli.md) — CLI flags map 1-to-1 with the most common settings.
- [reference/slash-commands.md](slash-commands.md) — `/settings` and `/settings-project`.
