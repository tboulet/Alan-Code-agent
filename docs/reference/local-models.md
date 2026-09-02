# Local Models

Alan Code works with any LLM served via an OpenAI-compatible API. Use `--base-url` to point at your local server.

## Supported servers

| Server | Alan command |
|---|---|
| vLLM | `alancode --model openai/<model> --base-url http://localhost:8000/v1` |
| Ollama | `alancode --model ollama/<model>` |
| SGLang | `alancode --model openai/<model> --base-url http://localhost:8000/v1` |
| SimpleLM | `alancode --model openai/<model> --base-url http://localhost:9876/v1` |

Ollama uses the `ollama/` prefix — LiteLLM auto-detects `localhost:11434`, no `--base-url` needed.

## Tool calling

By default, Alan uses **native tool calling**: schemas are sent in the request and streamed OpenAI-compatible `tool_calls` are assembled by ID/index before execution. This works with servers that support it, including SimpleLM with `--tool-parser universal`, vLLM with a matching `--tool-call-parser`, and tool-capable Ollama models.

For models without native tool support, use **text-based tool calling** — Alan injects tool schemas into the system prompt and parses tool calls from the model's text output:

```bash
alancode --model openai/<model> --base-url http://localhost:8000/v1 --tool-call-format hermes
```

Available formats: `hermes`, `hermes_xml`, `glm`, `alan`, `meta_json`, `bash_block`, `kimi`, `kimi_k3`, `deepseek`, `minimax`, and `auto`.

Text-tool parsing examines visible content and, for formats that permit it, separate reasoning content. This matters for Qwen and other reasoning models that place structured markup inside `<think>` or `reasoning_content`. Thinking stays private while the structured call executes. `bash_block` never parses reasoning, so shell drafts cannot become actions. A wholly empty or reasoning-only response gets up to `empty_response_retries` corrective nudges (default 2); if still empty, it surfaces with `api_error="empty_response"` instead of looking successful.

As practical starting points, use `hermes` for Qwen3-family JSON-in-tag output, `hermes_xml` for function-tag variants, `glm` for GLM output, `meta_json` for Llama JSON output, or `bash_block` for models reliably trained to emit one shell block. `auto` teaches `bash_block` but strict-parses every registered format, which is useful when a model ignores the requested convention. The exact format remains model/template dependent.

## Model name format

LiteLLM uses the model name prefix to determine the API protocol:

| Prefix | Protocol |
|---|---|
| `openai/<name>` | OpenAI-compatible (vLLM, SGLang, any local server) |
| `ollama/<name>` | Ollama (auto-detects localhost:11434) |
| `anthropic/<name>` | Anthropic API |
| `openrouter/<provider>/<name>` | OpenRouter |

For local servers, use `openai/<model>` + `--base-url`.

## Worked example: a 27B GGUF on a 12 GB laptop GPU

Serving `Qwen3.8-27B` (Q4_K_M, 16 GB of weights) on a 12 GB card, with the parts that are easy to get wrong:

```bash
llama-server -m ~/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf \
  --n-gpu-layers 36 --ctx-size 16384 --host 127.0.0.1 --port 8080 --jinja

alancode --model openai/qwen3.8-27b --base-url http://127.0.0.1:8080/v1 \
  --cw 16384 --tool-call-format auto
```

- **`--tool-call-format auto` is doing real work here.** Qwen-family models served locally often emit a tool call as plain text that the server never converts into OpenAI `tool_calls`, so native tool calling silently produces no action. `auto` strict-parses every registered format and executes whichever one parses. A single named format is a worse bet: the same model may emit a fenced bash block on one turn and function-tag markup on the next.
- **`--cw` is not optional.** Alan probes llama.cpp's `/props` for the served window, but a value you set on the server is authoritative; passing it avoids a budget resolved against the model's theoretical 262K maximum.
- **Weights larger than VRAM are fine**, at a price. Layers that do not fit run on the CPU, so throughput falls to a few tokens per second and an agent turn takes tens of seconds. Raise `--n-gpu-layers` until the server reports an allocation failure, then step back one.
- **A GGUF may carry layers your runtime rejects.** Multi-token-prediction builds ship an extra speculative block (`nextn_predict_layers` in the header, an extra index in `block_count`). llama.cpp logs `unused tensor blk.N.* -- ignoring` and loads; other runtimes refuse the file outright with a missing-projection error. Check the header before assuming the download is corrupt.

## Context-window detection

Alan needs the model's real context window to reserve output space and compact at a safe point. With the default `context_window: "auto"`, it tries model-registry data, server metadata, and a one-time backend probe, then caches a successful probe in `~/.alan/context_windows.json`.

For OpenAI-compatible servers Alan queries `/v1/models` and recognizes fields such as `max_model_len`, `max_context_length`, and `context_window`. Ollama is queried through `/api/show`. If metadata and registry lookup fail, a one-time probe is attempted and successful values are cached.

If detection is unavailable, Alan uses a conservative fallback and prints a warning. If you know the server's configured value, set `context_window` in `.alan/settings.json` or pass `--cw TOKENS`; the served window may be smaller than the model's theoretical maximum.

## Slow and offline endpoints

Importing LiteLLM defaults to its packaged model-cost map, so startup does not require a registry download. Pricing lookup is best-effort and never blocks a model request.

For a custom `--base-url`, `request_timeout: "auto"` means 3,600 seconds on both the LiteLLM and Anthropic-native backends. Override it with `--request-timeout SECONDS` or an explicit setting. Timeouts, connection failures, rate limits, overloads, and HTTP 5xx responses are retried with bounded exponential backoff only when no response content has been emitted; Alan never replays an already-partial stream. If a local server keeps returning an opaque pre-content 5xx, Alan makes one emergency-compaction attempt because some servers hide context overflow behind a generic internal error.

Example for a slow SimpleLM server:

```bash
export OPENAI_API_KEY=local
alancode \
  --model openai/Qwen3-32B \
  --base-url http://127.0.0.1:9876/v1 \
  --request-timeout 3600 \
  --tool-call-format hermes
```
