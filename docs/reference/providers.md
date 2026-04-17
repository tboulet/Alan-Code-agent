# Providers

Alan Code supports three providers. Pick one based on which model you want to use.

## At a glance

| Provider | Setting | Best for | Notes |
|---|---|---|---|
| Anthropic | `--provider anthropic` | Claude models | Direct SDK, native prompt caching, extended thinking. |
| LiteLLM | `--provider litellm` | Everything else | One abstraction over 50+ backends. |
| Scripted | `--provider scripted` | Tests, CI, demos | No network, no cost, deterministic. |

---

## Anthropic

**Class**: `alancode.providers.anthropic_provider.AnthropicProvider`

Uses the official `anthropic` SDK. Gets Alan the best of what Anthropic offers:
- **Prompt caching** — system prompt is split into 4 cache blocks, slashing cost on multi-turn conversations.
- **Extended thinking** — Claude Sonnet 4's `thinking` mode is supported (budget controlled by `thinking_budget_default` setting).
- **Native tool use** — structured `tool_use` blocks, clean tool_use → tool_result linking.

### Models

`claude-sonnet-4-6` (default), `claude-opus-4-6`, `claude-haiku-4-5`, older Sonnet/Opus/Haiku versions. Use the exact `model` string from Anthropic's API docs.

### Configuration

```bash
export ANTHROPIC_API_KEY=sk-ant-...
alancode --model claude-sonnet-4-6
```

Or in Python:

```python
AlanCodeAgent(provider="anthropic", model="claude-sonnet-4-6")
```

### Pricing

Alan has per-model Anthropic pricing hardcoded in `alancode/api/cost_tracker.py::ANTHROPIC_PRICING`. Cost displayed is accurate to the cent (includes cache read/write differentiation).

---

## LiteLLM

**Class**: `alancode.providers.litellm_provider.LiteLLMProvider`

Wrapper around [LiteLLM](https://docs.litellm.ai/), giving you OpenAI, OpenRouter, Gemini, Vertex, Bedrock, Ollama, vLLM, SGLang, and dozens more from one config.

### Model string convention

LiteLLM expects `provider/model`:

| Provider | Example model string |
|---|---|
| OpenAI | `openai/gpt-4o`, `openai/gpt-4o-mini`, `openai/o1-preview` |
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
alancode --provider litellm --model openrouter/google/gemini-2.5-pro

# OpenAI
export OPENAI_API_KEY=sk-...
alancode --provider litellm --model openai/gpt-4o

# Local
alancode --provider litellm --model openai/my-vllm-model --base-url http://localhost:8000/v1
```

### Which env var for which provider

LiteLLM reads the standard env var for each backend:

| Backend | Env var |
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
--tool-call-format alan      # Alan's own format (most portable)
```

When `--tool-call-format` is not set (default), Alan uses native function calling.

---

## Scripted

**Class**: `alancode.providers.scripted_provider.ScriptedProvider`

A testing-oriented provider that returns pre-canned responses. No network, no cost, fully deterministic. Used in Alan's own test suite and the `--scripted` mode of the auto-fix-loop example.

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

agent = AlanCodeAgent(provider=provider, permission_mode="yolo")
```

Each list entry is the response on the Nth iteration. `text(...)` returns a text-only response; `tool_call(...)` emits a tool_use; `multi_tool_call(...)` emits several tool_use blocks in one response.

See `alancode/providers/scripted_provider.py` for the full helper API (rules, turn-indexed responses, etc.).

---

## Adding a custom provider

All providers implement the `LLMProvider` ABC in `alancode/providers/base.py`:

```python
from alancode.providers.base import LLMProvider, StreamEvent

class MyProvider(LLMProvider):
    async def stream(self, messages, system, tools, *, model, max_tokens, thinking, **kwargs) -> AsyncGenerator[StreamEvent]:
        ...
    def get_model_info(self, model) -> ModelInfo:
        ...
```

Then inject it into the agent directly:

```python
agent = AlanCodeAgent(provider=MyProvider(...))
```

The `--provider` CLI flag only knows the three built-ins, but the constructor accepts any `LLMProvider` instance.

## Local models

Setting up vLLM / SGLang / Ollama / llama.cpp against Alan deserves a page of its own: [reference/local-models.md](local-models.md).

## Related

- [reference/cli.md](cli.md) — provider-related CLI flags.
- [reference/settings.md](settings.md) — persistent provider configuration.
- [reference/local-models.md](local-models.md) — detailed local-model setup.
- [reference/python-api.md](python-api.md) — using providers programmatically.
