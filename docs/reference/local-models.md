# Local Models

Alan Code works with any LLM served via an OpenAI-compatible API. Use `--base-url` to point at your local server.

## Supported servers

| Server | Alan command |
|---|---|
| vLLM | `alancode --model openai/<model> --base-url http://localhost:8000/v1` |
| Ollama | `alancode --model ollama/<model>` |
| SGLang | `alancode --model openai/<model> --base-url http://localhost:8000/v1` |

Ollama uses the `ollama/` prefix — LiteLLM auto-detects `localhost:11434`, no `--base-url` needed.

## Tool calling

By default, Alan uses **native tool calling** (the model returns structured `tool_calls`). This works with servers that support it (e.g., vLLM with `--tool-call-parser hermes`, Ollama with tool-capable models).

For models without native tool support, use **text-based tool calling** — Alan injects tool schemas into the system prompt and parses tool calls from the model's text output:

```bash
alancode --model openai/<model> --base-url http://localhost:8000/v1 --tool-call-format hermes
```

Available formats: `hermes`, `glm`, `alan`.

## Model name format

LiteLLM uses the model name prefix to determine the API protocol:

| Prefix | Protocol |
|---|---|
| `openai/<name>` | OpenAI-compatible (vLLM, SGLang, any local server) |
| `ollama/<name>` | Ollama (auto-detects localhost:11434) |
| `anthropic/<name>` | Anthropic API |
| `openrouter/<provider>/<name>` | OpenRouter |

For local servers, use `openai/<model>` + `--base-url`.

## Troubleshooting

| Problem | Fix |
|---|---|
| Tool calling errors | Use `--tool-call-format hermes` for text-based tool calling |
| Context window exceeded | Reduce `--max-output-tokens` or increase `--max-model-len` on the server |
| CUDA out of memory | Reduce `--max-model-len` or use a smaller model |
| Cost shows $0.0000 | Expected for self-hosted models (not in pricing registry) |
