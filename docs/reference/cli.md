# CLI parameters

Every parameter follows the same priority chain:

> **CLI flag > `.alan/settings.json` (per-project) > built-in defaults**

So anything you can pass on the command line can also be put in `.alan/settings.json` once per project, and you only need the flag when you want to override that default.

Run `alancode --help` for a quick version of this table.

## Provider & model

| Flag | Description | Default |
|---|---|---|
| `--provider` | LLM provider: `litellm`, `anthropic`, or `scripted` (testing). | `litellm` |
| `--model` | Model name (LiteLLM format: `provider/model`). | `anthropic/claude-sonnet-4-6` |
| `--api-key` | API key. If omitted, read from the provider's usual environment variable (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `OPENROUTER_API_KEY`, …). | (env) |
| `--base-url` | Override the API base URL. Set this for local OpenAI-compatible servers, e.g. `http://localhost:8000/v1`. See [`local-models.md`](local-models.md). | *(provider default)* |

### Tool calling format

By default, Alan uses provider-native tool calling. If your provider/model doesn't support that, use `--tool-call-format` to specify a text-based format (see [`local-models.md`](local-models.md) for details).

| Flag | Description | Default |
|---|---|---|
| `--tool-call-format` | Text-based tool-call format for models without native tool calling: `hermes`, `glm`, or `alan`. | *(none — uses native)* |

## Session behavior

| Flag | Description | Default |
|---|---|---|
| `--permission-mode` | `safe` (ask for every tool), `edit` (ask for writes + exec), `yolo` (allow everything). | `edit` |
| `--max-iterations-per-turn` | Hard cap on model calls per user message. | unlimited |
| `--max-output-tokens` | Per-call output token limit. Subject to internal escalation up to `escalated_max_tokens` on recovery. | *(provider default)* |
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

Every flag above maps to a key in the project's `.alan/settings.json`. The file is auto-generated on first run with sensible defaults and **auto-migrated** when new options are added in future versions — your existing values are preserved.

Example:

```json
{
  "provider": "litellm",
  "model": "openrouter/google/gemini-2.5-pro",
  "permission_mode": "edit",
  "memory": "off",
  "max_turns": 30
}
```

## Modifying at runtime

Use the `/settings <key> <value>` slash command (see [`slash-commands.md`](slash-commands.md)) to change a setting mid-session. Provider-related changes trigger provider recreation; other settings take effect on the next turn.
