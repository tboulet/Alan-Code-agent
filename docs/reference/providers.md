# Backends and providers

Alan Code splits "how it talks to a model" (the **backend**) from "which service runs the model" (the **provider**, encoded as a prefix in the model string).

## At a glance

There are three backends:

| Backend | Setting | Best for | Notes |
|---|---|---|---|
| `auto` *(default for non-Claude models)* | `--backend auto` | Everything except Claude direct: OpenAI, OpenRouter, Gemini, Ollama, vLLM, Bedrock, … | Universal transport — LiteLLM under the hood, supports 50+ providers via the model-string prefix. |
| `anthropic-native` *(default for bare `claude-*`)* | `--backend anthropic-native` | Claude models | Direct Anthropic SDK with `cache_control`, native thinking, native `tool_use`. |
| `scripted` | `--backend scripted` | Tests, CI, demos | No network, no cost, deterministic. |

The provider — OpenAI, Ollama, OpenRouter, etc. — is **not** a backend. It lives inside the model string as a prefix (LiteLLM convention).

You usually don't pass `--backend` at all — it's inferred from `--model`. Pass `--model` only.

---

## Backend inference

When `--backend` isn't set, it's chosen from the model string:

- Bare Claude name (`claude-sonnet-4-6`, `claude-opus-4-7`, …) → `anthropic-native`.
- Anything else → `auto` (LiteLLM transport).

The `anthropic/...` prefix is the explicit escape hatch for using Claude through LiteLLM (e.g. routing via a LiteLLM Proxy for centralized logging).

| Model string | Inferred backend | API key used |
|---|---|---|
| `claude-sonnet-4-6` | `anthropic-native` | `ANTHROPIC_API_KEY` |
| `anthropic/claude-sonnet-4-6` | `auto` (LiteLLM → Anthropic) | `ANTHROPIC_API_KEY` |
| `gpt-4o`, `gpt-4.1` | `auto` (LiteLLM → OpenAI) | `OPENAI_API_KEY` |
| `openrouter/...`, `ollama/...`, `gemini/...`, … | `auto` | provider's env var |

---

## `anthropic-native` backend

**Class**: `alancode.providers.anthropic_provider.AnthropicProvider`

Uses the official `anthropic` SDK. Gets Alan the best of what Anthropic offers:

- **Prompt caching** — system prompt is split into 4 cache blocks, slashing cost on multi-turn conversations.
- **Extended thinking** — Claude Sonnet 4's `thinking` mode is supported (budget controlled by `thinking_budget_default` setting).
- **Native tool use** — structured `tool_use` blocks, clean tool_use → tool_result linking.

### Models

`claude-sonnet-4-6` (default), `claude-opus-4-7`, `claude-haiku-4-5`, older Sonnet/Opus/Haiku versions. Use the exact model string from Anthropic's API docs.

### Configuration

```bash
export ANTHROPIC_API_KEY=sk-ant-...
alancode --model claude-sonnet-4-6      # backend inferred
# or, explicit:
alancode --backend anthropic-native --model claude-sonnet-4-6
```

```python
AlanCodeAgent(model="claude-sonnet-4-6")  # backend inferred
```

### Pricing

Alan has per-model Anthropic pricing hardcoded in `alancode/api/cost_tracker.py::ANTHROPIC_PRICING`. Cost displayed is accurate to the cent (includes cache read/write differentiation).

---

## `auto` backend (LiteLLM transport)

**Class**: `alancode.providers.litellm_provider.LiteLLMProvider`

Wrapper around [LiteLLM](https://docs.litellm.ai/), giving you OpenAI, OpenRouter, Gemini, Vertex, Bedrock, Ollama, vLLM, SGLang, and dozens more from one config.

### Model string convention

LiteLLM expects `provider/model`:

| Provider | Example model string |
|---|---|
| OpenAI | `gpt-4o`, `gpt-4.1`, `openai/gpt-4o` (explicit form) |
| Anthropic (via LiteLLM) | `anthropic/claude-sonnet-4-6` |
| OpenRouter | `openrouter/google/gemini-2.5-pro`, `openrouter/meta-llama/llama-3.3-70b-instruct` |
| Google Gemini (direct) | `gemini/gemini-2.5-pro` |
| Vertex AI | `vertex_ai/gemini-pro` |
| Bedrock | `bedrock/anthropic.claude-3-sonnet-20240229-v1:0` |
| Ollama | `ollama/qwen2.5-coder:7b` (no API key) |
| vLLM / SGLang / any OpenAI-compatible | `openai/<your-model>` + `--base-url http://localhost:8000/v1` |

### Configuration

```bash
# OpenRouter
export OPENROUTER_API_KEY=sk-or-...
alancode --model openrouter/google/gemini-2.5-pro

# OpenAI
export OPENAI_API_KEY=sk-...
alancode --model gpt-4o

# Local
alancode --model openai/my-vllm-model --base-url http://localhost:8000/v1
```

### Which env var for which provider

LiteLLM reads the standard env var for each upstream provider:

| Provider | Env var |
|---|---|
| OpenAI | `OPENAI_API_KEY` |
| Anthropic (via LiteLLM) | `ANTHROPIC_API_KEY` |
| OpenRouter | `OPENROUTER_API_KEY` |
| Google Gemini | `GEMINI_API_KEY` or `GOOGLE_API_KEY` |
| Vertex AI | `GOOGLE_APPLICATION_CREDENTIALS` (service account JSON) |
| Bedrock | `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY` + `AWS_REGION_NAME` |
| Local (Ollama, vLLM, SGLang) | none — `--base-url` is enough |

### Pricing

LiteLLM ships its own pricing registry. Cost display reads from `litellm.model_cost`. Unknown or unpriced models show `(estimated)` or nothing — see [reference/cost.md](cost.md).

### Tool calling

- **Native tool use**: most modern models (Claude, GPT-4o, Gemini 2.5, Llama 3.3 with function calling) are auto-detected and pass `tools=[...]` natively.
- **Text-based fallback**: for models without native function calling, set `--tool-call-format hermes|glm|alan`. The schema is rendered as text in the system prompt; output is parsed with regex. See [reference/cli.md](cli.md) for details.

### Text-based tool calling

For models without native tool calling support, set `--tool-call-format` to enable text-based tool calling. Alan injects tool schemas into the system prompt and parses tool calls from the model's text output:

```bash
--tool-call-format hermes   # Hermes <tool_call> format
--tool-call-format glm      # GLM XML format
--tool-call-format alan     # Alan's own format (most portable)
```

When `--tool-call-format` is not set (default), Alan uses native function calling.

---

## `scripted` backend

**Class**: `alancode.providers.scripted_provider.ScriptedProvider`

A testing-oriented backend that returns pre-canned responses. No network, no cost, fully deterministic. Used in Alan's own test suite and the `--scripted` mode of the auto-fix-loop example.

### Usage

```python
from alancode.providers.scripted_provider import (
    ScriptedProvider, text, tool_call, multi_tool_call,
)

provider = ScriptedProvider.from_responses([
    text("Hello!"),
    tool_call("Bash", {"command": "ls"}),
    text("Done."),
])

agent = AlanCodeAgent(backend=provider, permission_mode="yolo")
```

Each list entry is the response on the Nth iteration. `text(...)` returns a text-only response; `tool_call(...)` emits a tool_use; `multi_tool_call(...)` emits several tool_use blocks in one response.

See `alancode/providers/scripted_provider.py` for the full helper API (rules, turn-indexed responses, etc.).

---

## Adding a custom backend

All backends implement the `LLMProvider` ABC in `alancode/providers/base.py`:

```python
from alancode.providers.base import LLMProvider, StreamEvent

class MyBackend(LLMProvider):
    async def stream(self, messages, system, tools, *, model, max_tokens, thinking, **kwargs) -> AsyncGenerator[StreamEvent]:
        ...
    def get_model_info(self, model) -> ModelInfo:
        ...
```

Then inject it into the agent directly:

```python
agent = AlanCodeAgent(backend=MyBackend(...))
```

The `--backend` CLI flag only knows the three built-ins (`auto`, `anthropic-native`, `scripted`), but the constructor accepts any `LLMProvider` instance.

---

## Migration from older versions

Older versions exposed `--provider {litellm,anthropic,scripted}`. The flag, the `provider` settings key, and the `/provider` slash command are all kept as deprecated aliases for one release:

| Old | New |
|---|---|
| `--provider litellm` | drop the flag (or `--backend auto`) |
| `--provider anthropic` | `--backend anthropic-native` (or just `--model claude-sonnet-4-6`) |
| `--provider scripted` | `--backend scripted` |
| `--provider <other>` | error suggesting `--model <other>/<name>` (the prefix form) |

Old `.alan/settings.json` files with `"provider": "..."` are auto-migrated on first read.

---

## Related

- [reference/cli.md](cli.md) — backend-related CLI flags.
- [reference/settings.md](settings.md) — persistent backend configuration.
- [reference/local-models.md](local-models.md) — detailed local-model setup.
- [reference/python-api.md](python-api.md) — using backends programmatically.
