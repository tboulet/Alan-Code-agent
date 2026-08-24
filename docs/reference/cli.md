# CLI parameters

Every parameter follows the same priority chain:

> **CLI flag > resumed-session snapshot (with `--resume`/`--continue`) > `.alan/settings.json` (new-session project defaults) > built-in defaults**

So anything you can pass on the command line can also be put in `.alan/settings.json` once per project, and you only need the flag when you want to override that default.

Run `alancode --help` for a quick version of this table.

## Model & backend

| Flag | Description | Default |
|---|---|---|
| `--model` | Model name. Bare names (`claude-sonnet-4-6`, `gpt-4o`) or LiteLLM-style `provider/model` prefixes (`ollama/llama3.1`, `openrouter/google/gemini-2.5-pro`, `gemini/gemini-2.5-flash`, `anthropic/claude-sonnet-4-6`). | `claude-sonnet-4-6` |
| `--backend` | Transport (advanced — inferred from `--model` when not set). One of `auto` (universal LiteLLM transport), `anthropic-native` (direct Anthropic SDK with `cache_control`, native thinking, native `tool_use`), or `scripted` (tests). | inferred |
| `--api-key` | API key. If omitted, read from the backend's usual environment variable (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `OPENROUTER_API_KEY`, …). | (env) |
| `--base-url` | Override the API base URL. Set this for local OpenAI-compatible servers, e.g. `http://localhost:8000/v1`. See [`local-models.md`](local-models.md). | *(backend default)* |
| `--request-timeout SECONDS` | Model-request timeout. Custom endpoints default to 3,600 seconds when omitted. | automatic |
| `--context-window TOKENS`, `--cw TOKENS` | Trust this context-window size instead of automatic registry/server/probe resolution. | automatic |

### Backend inference

When `--backend` isn't set, it's chosen from the model string:

- Bare Claude name (e.g. `claude-sonnet-4-6`, `claude-opus-4-7`) → `anthropic-native`. Unlocks `cache_control`, native thinking, and native `tool_use`.
- Anything else (`gpt-4o`, `ollama/llama3.1`, `openrouter/...`, `gemini/...`, `anthropic/claude-...`) → `auto` (LiteLLM transport).

The `anthropic/...` prefix is the explicit escape hatch for using Claude through LiteLLM (e.g. routing via a LiteLLM Proxy for centralized logging).

### Tool calling format

By default, Alan uses native structured tool calling. If your model/server combination does not support that reliably, use `--tool-call-format` to specify a text-based format (see [`local-models.md`](local-models.md) for details).

| Flag | Description | Default |
|---|---|---|
| `--tool-call-format` | Text-based tool-call format for models without native tool calling: `hermes`, `hermes_xml`, `glm`, `alan`, `meta_json`, `bash_block`, `kimi`, `kimi_k3`, `deepseek`, `minimax`, or `auto`. | *(none - uses native)* |

## Session behavior

| Flag | Description | Default |
|---|---|---|
| `--permission-mode` | `safe` (auto-read, ask for writes + exec), `edit` (auto-read/write, ask for exec), `yolo` (allow everything). | `edit` |
| `--max-iterations-per-turn` | Hard cap on completed model→tool cycles per user message; recovery-only calls are not counted. | unlimited |
| `--max-output-tokens` | Starting output budget per model call. A length-truncated response may retry at `escalated_max_tokens`; set that setting at or below this value for a hard ceiling. | automatic (model default, capped at 25% of the context window) |
| `--escalated-max-tokens` | Retry budget after output truncation; set at/below the starting budget for a hard ceiling. | `64,000` |
| `--empty-response-retries` | Corrective retries for wholly empty or reasoning-only replies with no visible answer/tool. | `2` |
| `--persist-thinking`, `--no-persist-thinking` | Re-inject or omit prior reasoning in later requests. Does not enable provider-side thinking. | disabled |
| `--memory` | Memory mode: `off` (default), `on`, `intensive`. | `off` |
| `--verbose` | Enable debug-level logging. | `false` |

## Session resumption

| Flag | Description |
|---|---|
| `--resume` | Resume the most recent session in the current working directory. |
| `--continue [prefix]` | Without argument: list recent sessions. With a session-ID prefix: resume that specific one. |

## Mode

| Flag | Description |
|---|---|
| *(none — default)* | Interactive CLI mode. |
| `--gui` | Launch the local browser GUI. |
| `--print PROMPT` | Non-interactive: send one prompt, print the final answer, exit. |

## Utilities

| Flag | Description |
|---|---|
| `--version` | Show the installed version and exit. |
| `--help` | Show the built-in argument reference and exit. |

---

## Setting via `.alan/settings.json`

Every flag above maps to a key in the project's `.alan/settings.json`. The file is auto-generated on first run. When a later version adds a setting, its built-in default fills the missing value in memory without overwriting the existing file.

Example:

```json
{
  "backend": "anthropic-native",
  "model": "claude-sonnet-4-6",
  "permission_mode": "edit",
  "memory": "off"
}
```

## Modifying at runtime

Use the `/settings <key>=<value>` slash command (see [`slash-commands.md`](slash-commands.md)) to change a setting mid-session. Backend-related changes (`backend`, `model`, `api_key`, `base_url`, `request_timeout`, `context_window`) create a fresh `LLMBackend` and close the old one; a failed recreation leaves the active settings and backend unchanged. Other settings take effect on the next turn. Changing `model` also re-infers the backend per the rule above.
