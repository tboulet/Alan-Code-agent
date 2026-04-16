# Architecture overview

A 10-000-ft view of how Alan Code is put together. Read this first if you're about to dig into the code.

## The layers

```
┌─────────────────────────────────────────────────────────────────┐
│  UI layer           CLIUI      GUIUI      ScriptedUI            │
│                        │          │             │               │
│                        └──────┬───┴─────────────┘               │
│                               │ SessionUI interface             │
├───────────────────────────────┼─────────────────────────────────┤
│  Session layer        run_session (alancode/cli/repl.py)        │
│                               │                                 │
├───────────────────────────────┼─────────────────────────────────┤
│  Agent layer          AlanCodeAgent (alancode/agent.py)         │
│                               │ .query_events_async             │
├───────────────────────────────┼─────────────────────────────────┤
│  Loop layer           query_loop (alancode/query/loop.py)       │
│                               │ phases 1–10 per iteration       │
├───────────────────────────────┼─────────────────────────────────┤
│  Support          providers   tools   compact   hooks           │
│  subsystems       messages    permissions   memory   skills     │
│                   session     git_tree                          │
└─────────────────────────────────────────────────────────────────┘
```

Each layer has one job:

- **UI layer** presents events to the user (terminal, browser, test harness). All three implement `SessionUI` (`alancode/gui/base.py`).
- **Session layer** is the REPL driver. Handles slash commands, displays events to the UI, runs `run_session` in a loop.
- **Agent layer** is the public API. `AlanCodeAgent` owns the messages list, session state, and provider; exposes `query`/`query_async`/`query_events`/`query_events_async`.
- **Loop layer** is the inner engine. `query_loop` is an async generator that runs one "turn" — repeatedly calling the provider and executing tools.
- **Support subsystems** are everything else, grouped by concern.

## Data flow for one turn

```
user types "fix this bug"
        │
        ▼
  CLIUI.get_input returns the string
        │
        ▼
  run_session sees it, not a slash-command, so calls:
        agent.query_events_async("fix this bug")
        │
        ▼
  AlanCodeAgent.query_events_async:
      - appends UserMessage to self._messages
      - builds QueryParams with the provider, tools, settings, abort event
      - calls query_loop(params)
        │
        ▼
  query_loop (while True):
     phase 1: abort check
     phase 1.5: inject date/time system-reminder
     phase 2: compaction pre-check (truncate → clear → auto)
     phase 3: blocking limit check
     phase 4: provider.stream() — streams response
     phase 5: collect content blocks into AssistantMessage
     phase 6: yield AssistantMessage to caller
     phase 7: abort check
     phase 8: execute tools (orchestration.py runs them concurrent/serial)
             for each tool:
                 validate → pre-hook → permission pipeline → tool.call → post-hook
     phase 8.5: memory reminder (intensive mode)
     phase 9: check max_iterations_per_turn
     phase 10: loop back
        │
  (or exit on "no tool use" terminal condition)
        │
        ▼
  Events yielded back up to agent.query_events_async, which:
      - appends them to self._messages (filtered)
      - yields them to the caller (run_session)
        │
        ▼
  run_session receives each event:
      - ui.on_agent_event(event) → displayed
      - after loop: ui.on_cost(...) → displays cost summary
```

## Key packages

### `alancode.agent`
`AlanCodeAgent` — the public API.

### `alancode.query`
- `loop.py` — `query_loop` async generator, the beating heart.
- `state.py` — `LoopState` dataclass, mutable state between iterations.

### `alancode.providers`
- `base.py` — `LLMProvider` ABC, `StreamEvent` types.
- `anthropic_provider.py` — direct Anthropic SDK wrapper.
- `litellm_provider.py` — LiteLLM adapter.
- `scripted_provider.py` — deterministic test provider.

### `alancode.tools`
- `base.py` — `Tool` ABC and `ToolUseContext`.
- `registry.py` — enumerate built-ins, convert to API schemas.
- `execution.py` — `run_tool_use` — validate + permission + call + hooks.
- `orchestration.py` — batch tool calls (concurrent for reads, serial for writes).
- `builtin/*.py` — the 10 built-in tools.
- `text_tool_parser.py` — hermes/glm/alan formats for non-native models.

### `alancode.messages`
- `types.py` — all message dataclasses (UserMessage, AssistantMessage, blocks).
- `factory.py` — constructors for common messages.
- `normalization.py` — convert internal messages → API-ready form.
- `serialization.py` — to OpenAI-compatible dicts.

### `alancode.session`
- `session.py` — session listing, load/save settings snapshot.
- `state.py` — `SessionState` disk-attached properties (turn_count, cost, allow_rules, etc.).
- `transcript.py` — JSONL transcript serialize/deserialize.

### `alancode.permissions`
- `context.py` — `PermissionMode`, `PermissionBehavior`, `PermissionRule`, `ToolPermissionContext`.
- `pipeline.py` — `check_permissions` decides allow/deny/ask.
- `project_rules.py` — project-level `.alan/allow_rules.json` persistence.

### `alancode.compact`
- `compact_truncate.py` — Layer A (per-tool-result truncation).
- `compact_clear.py` — Layer B (old tool result clearing).
- `compact_auto.py` — Layer C (forked-agent summarization).
- `prompt.py` — the 9-section summarization template.

### `alancode.hooks`
- `registry.py` — load/execute pre/post tool-use hooks.
- `handlers.py` — session-start / session-end hook entry points.

### `alancode.memory`
- `memdir.py` — directory structure, memory index loading.
- `prompt.py` — memory section of the system prompt (off/on/intensive variants).

### `alancode.skills`
- `registry.py` — discover skills from `.alan/skills/`.
- `parser.py` — YAML-frontmatter parser.
- `tool_filter.py` — scope tool access when a skill is active.

### `alancode.git_tree`
- `parser.py` — parse git log into the AGT model.
- `layout.py` — assign (x, y) coords for the GUI tree.
- `operations.py` — `agt_move`, `agt_revert`, etc.
- `memory_snapshots.py` — save/restore `.alan/memory/` across moves.

### `alancode.cli`
- `main.py` — argparse entry point.
- `repl.py` — `run_session` + slash command handlers.
- `display.py` — Rich-based rendering (welcome panel, diffs, etc.).
- `user_input.py` — `ask_user_cli` for permission prompts (prompt-toolkit).

### `alancode.gui`
- `base.py` — `SessionUI` interface.
- `cli_ui.py` — terminal implementation using Rich + prompt-toolkit.
- `gui_ui.py` — FastAPI + WebSocket implementation.
- `server.py` — FastAPI app factory.
- `static/` — HTML / JS / CSS for the browser UI.
- `scripted_ui.py` — deterministic test UI.
- `serialization.py` — convert agent events to wire-format dicts.

### `alancode.api`
- `retry.py` — `with_retry` wrapper around provider streams.
- `cost_tracker.py` — per-session cost accounting + Anthropic pricing.

### `alancode.utils`
- `tokens.py` — tokenizer-backed count for compaction pre-check.
- `atomic_io.py` — `atomic_write_json` / `atomic_write_text` (tmp + rename).
- `env.py` — `get_cwd`, `get_git_status`, `is_git_repo`, etc.

## The public API surface

Users import from `alancode`:

```python
from alancode import AlanCodeAgent
```

Nothing else is currently a stable public export. Internal modules can and do change between versions.

## Where to start reading code

If you want to understand how Alan works:

1. **[`alancode/query/loop.py`](https://github.com/example/alan-code/blob/main/alancode/query/loop.py)** — the whole loop fits in one file. Start at `query_loop` and read the phases.
2. **[`alancode/agent.py`](https://github.com/example/alan-code/blob/main/alancode/agent.py)** — see how `query_events_async` wires up `query_loop`.
3. **[`alancode/prompt/system_prompt.py`](https://github.com/example/alan-code/blob/main/alancode/prompt/system_prompt.py)** — the system prompt sections.
4. **[`alancode/messages/normalization.py`](https://github.com/example/alan-code/blob/main/alancode/messages/normalization.py)** — how internal messages become the API payload.
5. **[`alancode/tools/execution.py`](https://github.com/example/alan-code/blob/main/alancode/tools/execution.py)** — the per-tool execution path.

## Related

- [architecture/query-loop.md](query-loop.md) — phase-by-phase walkthrough.
- [architecture/system-prompt.md](system-prompt.md) — how the system prompt is assembled.
- [architecture/messages-and-api.md](messages-and-api.md) — message normalization pipeline.
- [architecture/prompt-caching.md](prompt-caching.md) — Anthropic cache block strategy.
