# Running Alan Code with Local Models

Alan Code works with any LLM served via an OpenAI-compatible API. This guide covers vLLM, Ollama, SGLang, and the JZ bridge for remote HPC models.

## Quick Reference

| Server | Install | Serve command | Alan command |
| --- | --- | --- | --- |
| vLLM | `pip install vllm` (separate venv) | `vllm serve <model> --port 8000 ...` | `alancode --provider litellm --model openai/<model> --base-url http://localhost:8000/v1` |
| Ollama | `curl -fsSL https://ollama.com/install.sh \| sh` | `ollama pull <model>` | `alancode --provider litellm --model ollama/<model>` |
| SGLang | `pip install "sglang[all]"` (separate venv) | `python -m sglang.launch_server ...` | `alancode --provider litellm --model openai/<model> --base-url http://localhost:8000/v1` |
| JZ Bridge | See below | `python3 external/jz_bridge/local_server.py` | `alancode --provider litellm --model openai/<model> --base-url http://localhost:9999/v1` |

## Tested Configurations

| Model | Server | Tool mode | GPU VRAM | Status |
| --- | --- | --- | --- | --- |
| Qwen3-4B-Instruct-FP8 | vLLM | Native (vLLM `--tool-call-parser hermes`) | 12 GiB | Working |
| Qwen3-4B-Instruct-FP8 | SGLang | Text-based (`--tool-call-format hermes`) | 12 GiB | Working |
| qwen3:4b (thinking) | Ollama | Native | 12 GiB | LiteLLM streaming bug with thinking models |
| qwen2.5-coder:3b | Ollama | Text-based | 12 GiB | Model too small for reliable tool calling |
| GLM-4.7-FP8 (358B) | JZ Bridge + SGLang | Text-based (`--tool-call-format glm`) | 8x H100 | Working |
| Gemini 2.5 Flash | OpenRouter (cloud) | Native | N/A | Working |

## vLLM (Recommended for local)

### Setup

vLLM needs its own venv to avoid flash-attn conflicts:

```bash
python3 -m venv ~/venvs/venv_vllm
source ~/venvs/venv_vllm/bin/activate
pip install vllm
```

### Serve a model

```bash
source ~/venvs/venv_vllm/bin/activate

vllm serve ~/models/Qwen3-4B-Instruct-2507-FP8 \
  --port 8000 \
  --tool-call-parser hermes \
  --enable-auto-tool-choice \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.95 \
  --enforce-eager
```

Key flags:

- `--tool-call-parser hermes` — native tool calling (vLLM parses model's text output into OpenAI tool_calls format)
- `--enable-auto-tool-choice` — model decides when to use tools
- `--enforce-eager` — disables CUDA graphs (saves VRAM, needed for smaller GPUs)
- `--max-model-len` — context window (reduce if OOM)
- `--max-num-seqs` — max concurrent requests (reduce if OOM)

### Run Alan

```bash
alancode --provider litellm \
  --model openai//home/tboulet/models/Qwen3-4B-Instruct-2507-FP8 \
  --base-url http://localhost:8000/v1 \
  --max-output-tokens 4096
```

Since vLLM's `--tool-call-parser` handles the tool format translation, Alan receives structured `tool_calls` in the response — no text parsing needed. Native tool calling is the default.

### VRAM guidelines

| Model size | FP8 VRAM | Recommended GPU | Context possible |
| --- | --- | --- | --- |
| 4B | ~5 GiB | 12 GiB (RTX 4000 Ada) | 16K with --enforce-eager |
| 8B | ~9 GiB | 16+ GiB (RTX 4090) | 16K+ |
| 14B | ~15 GiB | 24 GiB (RTX 4090) | 8K+ |

## SGLang

### Setup

SGLang also needs its own venv (conflicts with vLLM's flash-attn):

```bash
python3 -m venv ~/venvs/venv_sglang
source ~/venvs/venv_sglang/bin/activate
pip install "sglang[all]"
```

### Serve a model

```bash
source ~/venvs/venv_sglang/bin/activate

python3 -m sglang.launch_server \
  --model-path ~/models/Qwen3-4B-Instruct-2507-FP8 \
  --port 8000 \
  --host 0.0.0.0 \
  --max-running-requests 4 \
  --trust-remote-code \
  --skip-server-warmup
```

### Run Alan

SGLang serves an OpenAI-compatible API but does NOT translate tool calls natively. Use text-based tool calling:

```bash
alancode --provider litellm \
  --model openai//home/tboulet/models/Qwen3-4B-Instruct-2507-FP8 \
  --base-url http://localhost:8000/v1 \
  --tool-call-format hermes \
  --max-output-tokens 4096
```

Alan's text tool parser extracts tool calls from the model's Hermes-format text output.

## Ollama

### Setup

Ollama is a standalone binary — no Python, no venv needed:

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen3:4b
```

Ollama manages its own model storage and serves automatically as a system service on `http://localhost:11434`.

### Run Alan

```bash
alancode --provider litellm \
  --model ollama/qwen3:4b \
  --max-output-tokens 4096
```

Ollama uses the `ollama/` prefix in LiteLLM — no `--base-url` needed (LiteLLM auto-detects localhost:11434).

### Known limitations

- **Thinking models** (qwen3:4b): LiteLLM's streaming mode puts all content into `reasoning_content`, leaving `content` empty. Alan extracts and displays the thinking, but the response quality varies.
- **Small models** (3B and below): Too small for reliable tool calling — they dump raw JSON instead of following tool format instructions.
- **Recommended**: Use 4B+ instruct (non-thinking) variants for best results. If only thinking variants are available, they work but with more verbose output.

## JZ Bridge (Remote HPC models)

For models hosted on Jean Zay or other HPC clusters without direct network access.

### Architecture

```
Your machine                          JZ compute node
─────────────                         ────────────────
local_server.py (FastAPI)             server.py + SGLang
  localhost:9999/v1                     polls input/, POSTs to SGLang
  ↕ SCP pickle files                   writes response to output/
```

### Setup

**1. Start SGLang + bridge server on JZ** (via SLURM):

```bash
cd /home/tboulet/projects/swegrid
deploy_jz python3 scripts/run_jz.py \
  'bash scripts_sglang/sglang_server_GLM-4.7_with_bridge.sh' \
  --h100 --n_gpu=4 --nodes=2 --hour 15
```

**2. Start local proxy server**:

```bash
python3 external/jz_bridge/local_server.py --port 9999
```

**3. Run Alan**:

```bash
alancode --provider litellm \
  --model openai/GLM-4.7-FP8 \
  --base-url http://localhost:9999/v1 \
  --tool-call-format glm
```

### Latency

Each LLM call adds ~5-10s SCP overhead on top of model inference time. SSH connection multiplexing (`ControlMaster` in `~/.ssh/config`) reduces this.

## Tool Calling Modes

| Scenario | Flag | How it works |
| --- | --- | --- |
| vLLM with `--tool-call-parser` | *(default — native)* | Native: vLLM translates model text → OpenAI tool_calls |
| SGLang (no tool parser) | `--tool-call-format hermes` | Text-based: Alan parses Hermes `<tool_call>` tags from text |
| GLM models via SGLang | `--tool-call-format glm` | Text-based: Alan parses GLM's XML `<arg_key>/<arg_value>` tags |
| Unknown model | `--tool-call-format alan` | Text-based: Alan instructs model via system prompt |
| Ollama with tool support | *(default — native)* | Native: Ollama handles tool format |

## Model Name Format

LiteLLM uses the model name prefix to determine the API protocol:

- `openai/<name>` — OpenAI-compatible protocol (vLLM, SGLang, any local server)
- `ollama/<name>` — Ollama protocol (auto-detects localhost:11434)
- `anthropic/<name>` — Anthropic API
- `openrouter/<provider>/<name>` — OpenRouter

For local servers (vLLM, SGLang), use `openai/<model_path_or_name>` + `--base-url`.

## Troubleshooting

**Tool calling errors**: If the model doesn't support native tool calling, use `--tool-call-format hermes` (or `glm` / `alan`) to enable text-based tool calling.

**Context window exceeded**: Reduce `--max-output-tokens` (e.g., `4096`) or increase `--max-model-len` on the server.

**CUDA out of memory**: Add `--enforce-eager`, reduce `--max-model-len`, reduce `--max-num-seqs`, or use a smaller model.

**Empty responses with thinking models**: The model spends all tokens on `<think>` reasoning. Alan displays thinking content in dim italic. Use a non-thinking (instruct) variant for cleaner output.

**Internal Server Error from JZ bridge**: Check that the bridge server is running on JZ and the SLURM job is active.

**flash-attn conflicts**: vLLM and SGLang require different flash-attn versions. Use separate venvs (`~/venvs/venv_vllm`, `~/venvs/venv_sglang`).

**Cost shows $0.0000**: Expected for self-hosted models (not in pricing registry).
