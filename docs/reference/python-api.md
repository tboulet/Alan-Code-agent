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
    **provider_kwargs: Any,
)
```

All settings omitted (`None`) fall through to `.alan/settings.json` → built-in defaults. See [guides/configuration.md](../guides/configuration.md).

Key arguments:

- **`cwd`** — working directory the agent operates in. Defaults to `os.getcwd()`.
- **`provider`** — either a string (`"anthropic"`, `"litellm"`, `"scripted"`) or a concrete `LLMProvider` instance (lets you inject a custom provider).
- **`session_id`** — if set, resume an existing session; otherwise a new session ID is generated.
- **`ask_callback`** — `async def callback(question: str, options: list[str]) -> str`. Called when a tool needs user approval. Return the chosen option text (or any string to use as a free-text answer).

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
    provider="litellm",
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

## Example: injecting a custom provider

```python
from alancode import AlanCodeAgent
from alancode.providers.base import LLMProvider

class MyProvider(LLMProvider):
    async def stream(self, messages, system, tools, *, model, max_tokens, thinking, **kwargs):
        # yield StreamEvent objects
        ...
    def get_model_info(self, model):
        ...

agent = AlanCodeAgent(provider=MyProvider(...))
```

## Related

- [guides/building-agents.md](../guides/building-agents.md) — tutorial-style introduction.
- [reference/tools.md](tools.md) — what tools the agent has access to.
- [reference/settings.md](settings.md) — what kwargs are valid and what defaults apply.
- [architecture/query-loop.md](../architecture/query-loop.md) — what happens inside each `query_events_async` call.
