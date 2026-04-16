# Getting started

This walkthrough takes you from `pip install` to your first successful agent turn — about 10 minutes, no prior setup assumed.

## 1. Install

```bash
pip install alancode
```

Requires Python 3.11+. One install gives you the CLI, the browser GUI, the Python library, the Anthropic provider, and LiteLLM support for every other model provider.

## 2. Provide an API key

Pick one — whichever matches the provider you want to use.

```bash
# Anthropic (default)
export ANTHROPIC_API_KEY=sk-ant-...

# OpenRouter — access to OpenAI, Google, Mistral, Meta, etc. via one key
export OPENROUTER_API_KEY=sk-or-...

# OpenAI directly
export OPENAI_API_KEY=sk-...
```

For local models (vLLM, Ollama, SGLang, llama.cpp) no key is needed — see [reference/local-models.md](reference/local-models.md).

## 3. Start a session

From inside the project you want to work on:

```bash
alancode
```

You'll see:

```
╭──────────────────────────────────────────────────╮
│ Alan Code -- Open-source coding agent            │
│ Session: a1b2c3d4... | Model: claude-sonnet-4-6  │
│ Type /help for commands, Ctrl+C to interrupt     │
│ Tip: create ALAN.md (or use /init) to give Alan project context │
╰──────────────────────────────────────────────────╯

>
```

## 4. Ask a question

Try a read-only question — nothing will need approval:

```
> What does this project do?
```

Alan will use the `Read`, `Glob`, and `Grep` tools to inspect the codebase, then answer. You'll see live streaming text, then the tool calls render with green-bordered panels, and finally a summary line like:

```
Session: 8,118 in + 153 out = $0.0082 (estimated) | Conversation: 8,271 / 200,000 (4%)
```

- `Session` = cumulative tokens + USD across the turn.
- `Conversation` = how full the context window is.

## 5. Ask for a change

```
> Add a docstring to the public functions in alancode/agent.py
```

Now Alan will want to use `Edit` — a write tool. By default Alan runs in `edit` permission mode, which asks for approval before write/exec operations:

```
? Allow Edit?
Tool 'Edit' wants to execute with input: {'file_path': '/proj/alancode/agent.py', ...}
  1) Allow
  2) Deny
Your choice: 1
```

After the edit runs, you'll see a green/red diff with line numbers showing exactly what changed. Review and continue.

Use **option 3** if you want to allow the same pattern in future (e.g. "Allow always `git *` commands" for Bash). That rule persists to `.alan/allow_rules.json` in this project.

## 6. Use a slash command

Slash commands are typed into the prompt and handled locally (they don't go through the model). Try:

```
> /status
```

You'll see a table with the model, tokens used, cost, and more. Some useful ones:

- `/help` — list all commands
- `/diff` — show git diff of uncommitted changes
- `/commit` — have Alan draft and create a commit
- `/compact` — manually compact the conversation
- `/exit` — leave the session

Full list in [reference/slash-commands.md](reference/slash-commands.md).

## 7. Resume a session

```bash
alancode --resume
```

Picks up the last session in this directory. The conversation (last 100 messages) is replayed automatically.

To list recent sessions and pick one:

```bash
alancode --continue
```

## 8. Try a different model

```bash
# Google Gemini via OpenRouter
alancode --provider litellm --model openrouter/google/gemini-2.5-pro

# OpenAI directly
alancode --provider litellm --model openai/gpt-4o

# Local Ollama
alancode --provider litellm --model ollama/qwen2.5-coder:7b --base-url http://localhost:11434
```

## 9. Launch the GUI (optional)

```bash
alancode --gui
```

Opens `http://localhost:8420/`. Three panels:
- **Chat** — same as the CLI, but with in-place diff rendering.
- **LLM Perspective** — the exact payload sent to the model each turn (for debugging).
- **Git Tree** — visualise commits + the agent's position + revert/move controls.

## 10. Give Alan project-specific context

Create an `ALAN.md` in your project root:

```bash
alancode
> /init
```

This creates a starter template. Fill it with your project's conventions:

```markdown
# Alan's instructions for this project

- Use `pathlib` instead of `os.path`.
- Tests live under `tests/`. Run with `pytest -x`.
- The CLI entry point is `alancode.cli.main:main`.
- Don't auto-format — we handle formatting manually with ruff.
```

Every session Alan starts in this project loads `ALAN.md` into its system prompt. You can also create an `~/.alan/ALAN.md` file for global instructions across all projects.

## Where to go next

- Understand the core concepts → [concepts/agent-loop.md](concepts/agent-loop.md)
- Use Alan as a Python library → [guides/building-agents.md](guides/building-agents.md)
- Set up a local model → [reference/local-models.md](reference/local-models.md)
- Browse the full CLI and command references → [reference/cli.md](reference/cli.md), [reference/slash-commands.md](reference/slash-commands.md)
