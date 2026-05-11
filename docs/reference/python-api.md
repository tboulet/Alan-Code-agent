# Python API reference

The main public class is `alancode.AlanCodeAgent`. This page documents the methods and properties you're likely to use.

For a tutorial-style introduction, see [guides/building-agents.md](../guides/building-agents.md).

## Constructor

```python
from alancode import AlanCodeAgent

AlanCodeAgent(
    *,
    cwd: str | None = None,
    provider: str | LLMProvider = "litellm",  # or "anthropic"
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    permission_mode: str | None = None,
    max_iterations_per_turn: int | None = None,
    max_output_tokens: int | None = None,
    memory: str | None = None,
    tool_call_format: str | None = None,
    session_id: str | None = None,
    ask_callback: Callable | None = None,
    verbose: bool = False,
    extra_tools: list[Tool] | None = None,
    custom_system_prompt: str | None = None,
    gui_label: str | None = None,
    programmatic: bool = False,
    tools: list[Tool] | None = None,
    disabled_tools: list[str] | None = None,
    **provider_kwargs: Any,
)
```

All settings omitted (`None`) fall through to `.alan/settings.json` → built-in defaults. See [guides/configuration.md](../guides/configuration.md).

Key arguments:

- **`cwd`** — working directory the agent operates in. Defaults to `os.getcwd()`.
- **`provider`** — either a string (`"anthropic"`, `"litellm"`, `"scripted"`) or a concrete `LLMProvider` instance (lets you inject a custom provider).
- **`session_id`** — if set, resume an existing session; otherwise a new session ID is generated.
- **`ask_callback`** — `async def callback(question: str, options: list[str]) -> str`. Called when a tool needs user approval. Return the chosen option text (or any string to use as a free-text answer).
- **`extra_tools`** — additional tools appended to the agent's tool list. See [guides/building-agents.md](../guides/building-agents.md) for embedding patterns.
- **`custom_system_prompt`** — when set, replaces Alan's default system prompt sections entirely.
- **`gui_label`** — URL path segment for the GUI bridge. Defaults to the cwd basename.
- **`programmatic`** — when `True`, runs Alan as a library component rather than a developer assistant. See [Programmatic mode](#programmatic-mode) below.
- **`tools`** — explicit base tool list, replacing the default builtins. Composes with `disabled_tools` and `extra_tools`. See [Tool selection](#tool-selection) below.
- **`disabled_tools`** — list of tool names to remove from the base set (e.g. `["WebFetch", "GitCommit"]`).

## Query methods

The 2×2 matrix:

|  | Sync | Async |
|---|---|---|
| **Final text only** | `query(prompt) -> str` | `query_async(prompt) -> str` |
| **Streaming events** | `query_events(prompt) -> list[Event]` | `query_events_async(prompt) -> AsyncGenerator[Event]` |

### `query(prompt: str) -> str`

Run a turn synchronously. Returns the assistant's final text response.

```python
answer = agent.query("Explain the compaction system")
```

Internally runs `asyncio.run`, or dispatches to a worker thread if an event loop is already running (Jupyter-safe).

### `async query_async(prompt: str) -> str`

Same as `query` but awaitable.

```python
answer = await agent.query_async("Explain the compaction system")
```

### `query_events(prompt: str) -> list[Event]`

Synchronous; returns a full list of events after the turn completes. Useful for post-hoc inspection.

### `async query_events_async(prompt: str) -> AsyncGenerator[Event, None]`

The real primitive. Yields events as they're produced:

```python
async for event in agent.query_events_async("Summarize README.md"):
    # handle each event
    pass
```

Events are message dataclasses from `alancode.messages.types`:

| Event | When |
|---|---|
| `RequestStartEvent` | Each API call begins (useful for "Thinking..." indicators). |
| `AssistantMessage` with `hide_in_api=True` | Streaming delta — text chunks, thinking chunks. |
| `AssistantMessage` with `hide_in_api=False` | Final assembled message after the stream completes. Has tool calls. |
| `UserMessage` | Injected (system reminders, tool results). |
| `SystemMessage` | Informational (compaction markers, etc.). |
| `AttachmentMessage` | Structured metadata (e.g., `max_iterations_per_turn_reached`). |
| `ProgressMessage` | Long-running operation updates. |

Filter on `hide_in_api` to distinguish streaming deltas from final messages — see the streaming example in [guides/building-agents.md](../guides/building-agents.md).

## State inspection

| Property | Type | Description |
|---|---|---|
| `agent.session_id` | `str` | Current session ID (auto-generated or passed in). |
| `agent.messages` | `list[Message]` | Copy of the current conversation (safe to mutate the returned list). |
| `agent.usage` | `Usage` | Cumulative tokens across the session. |
| `agent.last_usage` | `Usage` | Usage from the most recent successful API call. |
| `agent.cost_usd` | `float` | Cumulative estimated cost. |
| `agent.cost_unknown` | `bool` | `True` if the model's pricing isn't known. |
| `agent.cwd` | `str` | Working directory. |
| `agent.turn_count` | `int` | Number of user messages processed this session. |

`Usage` has: `input_tokens`, `output_tokens`, `cache_read_input_tokens`, `cache_creation_input_tokens`, plus a `total_input` property summing the three input types.

## Runtime control

### `abort()`

```python
agent.abort()
```

Sets the abort event. The running turn's next `await` checkpoint catches it and unwinds cleanly.

### `inject_message(text: str)`

```python
agent.inject_message("Actually, focus on calc.py only.")
```

Queues a user message to be delivered at the start of the next iteration. Useful for orchestration frameworks that steer mid-turn.

### `update_session_setting(key: str, value: Any) -> str | None`

```python
error = agent.update_session_setting("permission_mode", "yolo")
if error:
    print("Invalid:", error)
```

Validates and updates a setting in-memory + on disk. Returns an error message string on validation failure, or `None` on success. Provider-related settings trigger provider recreation.

## Lifecycle

### `async close()`

```python
await agent.close()
```

Fires `session_end` hooks. Call once when done. The CLI does this on `/exit`.

## Programmatic mode

Use `programmatic=True` when Alan is being driven by another program (a benchmark harness, a parent agent, an automated pipeline) rather than a developer at a terminal. It detaches Alan from project- and host-level state that's normally helpful for an interactive assistant but contaminates a controlled run.

```python
agent = AlanCodeAgent(
    model="claude-sonnet-4-6",
    cwd="/path/to/experiment",
    permission_mode="yolo",
    programmatic=True,
)
```

When `programmatic=True`:

- `~/.alan/ALAN.md` (global instructions) is **not** loaded.
- `<cwd>/ALAN.md` (project instructions) is **not** loaded.
- `~/.alan/memory/MEMORY.md` (global memory index) is **not** loaded.
- AGT (Agentic Git Tree) bootstrap is skipped — no HEAD snapshot, no `.gitignore` mutation.
- The default tool set excludes `WebFetch`, `GitCommit`, and `AskUserQuestion`. `SkillTool` is also not appended.

Project-scoped state in `<cwd>/.alan/sessions/<id>/` (transcript, state, scratchpad) is unchanged — that's the agent's own working memory and is needed for resume.

You can override the curated tool set with `tools=` or refine it with `disabled_tools=` (see below).

## Tool selection

Three knobs control the agent's tool list, applied in order:

1. **Base set.** Resolved from the first of:
   - `tools=[...]` if passed (explicit replacement),
   - the curated programmatic set if `programmatic=True`,
   - all enabled built-in tools otherwise (the `SkillTool` is appended in this case).
2. **Subtract** any names listed in `disabled_tools`.
3. **Append** anything in `extra_tools`.

```python
# Read-only assistant: drop write/exec tools entirely
agent = AlanCodeAgent(disabled_tools=["Bash", "Edit", "Write", "GitCommit"])

# Custom tool list (e.g. for a domain-specific agent)
agent = AlanCodeAgent(tools=[MyDomainTool(), MyOtherTool()])

# Programmatic mode plus an extra custom tool
agent = AlanCodeAgent(programmatic=True, extra_tools=[MyTool()])
```

## Session locking

`SessionState` takes an exclusive `flock` on `<cwd>/.alan/sessions/<session_id>/session.lock` at construction. A second process attempting to open the same session raises `alancode.session.SessionLockedError`. The lock is released by `agent.close()` and on process exit.

## Custom permission callbacks

```python
async def my_ask(question: str, options: list[str]) -> str:
    print(f"\n{question}")
    for i, opt in enumerate(options, 1):
        print(f"  {i}) {opt}")
    choice = input("> ").strip()
    if choice.isdigit() and 1 <= int(choice) <= len(options):
        return options[int(choice) - 1]
    return choice  # free-text answer

agent = AlanCodeAgent(ask_callback=my_ask, permission_mode="edit")
```

The callback is awaited when a tool needs approval. Return one of the option strings to accept the corresponding action (Allow, Deny, Allow always), or return any other string — that string becomes the "tool result" sent back to the model (so the user can deny with a reason in one step).

Ctrl+C from within the callback should raise `KeyboardInterrupt` → Alan converts it to `asyncio.CancelledError` → the turn aborts cleanly.

## Example: a minimal synchronous script

```python
from alancode import AlanCodeAgent

agent = AlanCodeAgent(
    model="openrouter/google/gemini-2.5-flash",
    permission_mode="yolo",  # auto-approve for automation
)

answer = agent.query("What's 2+2?")
print(answer)

print(f"Cost: ${agent.cost_usd:.4f}")
print(f"Tokens: {agent.usage.total_input} in, {agent.usage.output_tokens} out")
```

## Example: async streaming

```python
import asyncio
from alancode import AlanCodeAgent
from alancode.messages.types import AssistantMessage, TextBlock, ToolUseBlock

async def main():
    agent = AlanCodeAgent(permission_mode="yolo")
    async for event in agent.query_events_async("List files and summarize."):
        if not isinstance(event, AssistantMessage):
            continue
        for block in event.content:
            if event.hide_in_api and isinstance(block, TextBlock):
                print(block.text, end="", flush=True)
            elif not event.hide_in_api and isinstance(block, ToolUseBlock):
                print(f"\n[tool: {block.name}({block.input})]")

asyncio.run(main())
```

## Example: injecting a custom backend

```python
from alancode import AlanCodeAgent
from alancode.providers.base import LLMProvider

class MyBackend(LLMProvider):
    async def stream(self, messages, system, tools, *, model, max_tokens, thinking, **kwargs):
        # yield StreamEvent objects
        ...
    def get_model_info(self, model):
        ...

agent = AlanCodeAgent(backend=MyBackend(...))
```

## Related

- [guides/building-agents.md](../guides/building-agents.md) — tutorial-style introduction.
- [reference/tools.md](tools.md) — what tools the agent has access to.
- [reference/settings.md](settings.md) — what kwargs are valid and what defaults apply.
- [architecture/query-loop.md](../architecture/query-loop.md) — what happens inside each `query_events_async` call.
