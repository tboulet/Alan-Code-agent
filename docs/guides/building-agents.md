# Building agents as a library

Everything Alan Code does on the command line is driven by the `AlanCodeAgent` class. You can use it directly from Python to build your own coding agents, orchestrators, auto-fix loops, and custom interfaces.

## The minimal case

Three lines for a functional agent:

```python
from alancode import AlanCodeAgent

agent = AlanCodeAgent()
print(agent.query("What does this project do?"))
```

That works identically to `alancode` on the CLI — same tools, same permission pipeline (default `edit` mode, so writes need stdin approval), same compaction.

## The four query APIs

`AlanCodeAgent` exposes a 2×2 matrix. Pick the one that matches your caller:

|  | Sync | Async |
|---|---|---|
| **Final text only** | `query(prompt) -> str` | `query_async(prompt) -> str` (awaitable) |
| **All events (streaming)** | `query_events(prompt) -> list[Event]` | `query_events_async(prompt) -> AsyncGenerator[Event]` |

Only `query_events_async` does real work — the others are thin adapters around it.

### Sync, final text

```python
answer = agent.query("Fix the bug in calc.py")
```

Blocks until the turn completes. Returns the assistant's final text response. Under the hood: runs `asyncio.run` in a worker thread if an event loop is already running (Jupyter-safe).

### Async, final text

```python
import asyncio

async def main():
    agent = AlanCodeAgent()
    answer = await agent.query_async("Fix the bug in calc.py")

asyncio.run(main())
```

Use in async contexts (FastAPI endpoints, async workers, etc.).

### Streaming events

For live rendering, progress bars, custom UIs:

```python
import asyncio
from alancode import AlanCodeAgent
from alancode.messages.types import AssistantMessage, TextBlock, ToolUseBlock

async def main():
    agent = AlanCodeAgent(permission_mode="yolo")
    async for event in agent.query_events_async("List files and summarise."):
        if not isinstance(event, AssistantMessage):
            continue
        for block in event.content:
            if event.hide_in_api and isinstance(block, TextBlock):
                # Streaming delta — print as it arrives
                print(block.text, end="", flush=True)
            elif not event.hide_in_api and isinstance(block, ToolUseBlock):
                # Final message — tool call block
                print(f"\n[tool: {block.name}({block.input})]")

asyncio.run(main())
```

The key idea: `AssistantMessage` arrives **twice** per iteration — once as streaming deltas (`hide_in_api=True`, text chunks) and once as the final assembled message (`hide_in_api=False`, with tool calls). Filter by `hide_in_api` to avoid duplicate output.

## Worked examples in the repo

The examples directory has three ready-to-run scripts:

- [`examples/example_1_cli_agent.py`](../../examples/example_1_cli_agent.py) — 10-line interactive CLI loop.
- [`examples/example_2_auto_fix_loop/`](../../examples/example_2_auto_fix_loop/) — iterate `agent.query()` + `pytest` until tests pass.
- [`examples/example_3_streaming_agent.py`](../../examples/example_3_streaming_agent.py) — async streaming for custom UIs.

Each runs against real LLMs or against the `ScriptedProvider` (no API needed) for deterministic testing.

## Configuration

Pass anything you'd set in `settings.json` as a constructor kwarg:

```python
agent = AlanCodeAgent(
    provider="litellm",
    model="openrouter/google/gemini-2.5-flash",
    permission_mode="yolo",
    max_iterations_per_turn=15,
    max_output_tokens=16_000,
    memory="off",
    cwd="/path/to/project",
    session_id=None,     # None = new session
    api_key=None,        # None = from env
    verbose=False,
    ask_callback=None,   # Custom permission prompt; see below
)
```

Omitted kwargs fall through the priority chain (session → project settings.json → defaults). See [guides/configuration.md](configuration.md).

## Custom permission prompts

By default, `AlanCodeAgent` has no `ask_callback` — so tools that would need approval just get denied. Provide your own to integrate permission prompts into your UI:

```python
async def ask(question: str, options: list[str]) -> str:
    """Return the chosen option (or 'Other' + free text)."""
    # Your custom dialog / API call / whatever.
    print(f"\n{question}")
    for i, opt in enumerate(options, 1):
        print(f"  {i}) {opt}")
    choice = input("> ").strip()
    try:
        return options[int(choice) - 1]
    except (ValueError, IndexError):
        return choice  # treat as free-text answer

agent = AlanCodeAgent(ask_callback=ask, permission_mode="edit")
```

The callback signature is `async def ask(question: str, options: list[str]) -> str`. Return the selected option text, or any other string to feed back as the "tool result" (users can deny with their own reason this way).

## Session persistence

Every turn, messages are persisted to `<cwd>/.alan/sessions/<session_id>/transcript.jsonl`. To resume:

```python
agent = AlanCodeAgent(session_id="a1b2c3...")
```

Session state (cost totals, allow rules, agent position, last usage) comes with it.

## Costs and tokens

```python
agent.usage.input_tokens       # cumulative
agent.usage.output_tokens
agent.cost_usd                 # estimated $ total
agent.cost_unknown             # True if pricing isn't available

agent.last_usage.input_tokens  # most recent call only
```

Both `usage` (cumulative) and `last_usage` (most recent) are `Usage` dataclasses with the full breakdown: input, output, cache-creation, cache-read.

## Scripted provider — deterministic testing

For tests and CI where real API calls aren't desired:

```python
from alancode.providers.scripted_provider import (
    ScriptedProvider, text, tool_call, multi_tool_call,
)

provider = ScriptedProvider.from_responses([
    multi_tool_call(
        ("Bash", {"command": "ls"}),
        ("Read", {"file_path": "/etc/hostname"}),
    ),
    text("Done. The system is ..."),
])

agent = AlanCodeAgent(provider=provider, permission_mode="yolo")
answer = agent.query("check the system")
```

Each entry in the list is what the "provider" returns on the Nth iteration. Zero network, zero cost, fully deterministic.

## Inject messages mid-run

Rarely needed but occasionally useful — send a message to the agent while it's thinking:

```python
agent.inject_message("Actually, focus on calc.py only.")
```

The message gets queued and delivered at the start of the next iteration. Handy for orchestration frameworks that need to steer mid-turn.

## Aborting

```python
agent.abort()
```

Sets the abort event. The next `await` checkpoint catches it and unwinds the turn cleanly. Used by GUI's "Stop" button and Ctrl+C in the CLI.

## Lifecycle

```python
agent = AlanCodeAgent(...)           # sync init; loads session state if session_id given
try:
    agent.query("...")
    agent.query("...")
finally:
    agent.close()                    # async: fires session-end hooks
```

`agent.close()` is async. The CLI calls it for you on `/exit`. Library users should either call it themselves or use a context manager (not yet provided; TODO).

## Related

- [reference/python-api.md](../reference/python-api.md) — full class + method signatures.
- [examples/](../../examples/) — the three worked examples.
- [architecture/query-loop.md](../architecture/query-loop.md) — how the loop actually drives the events.
