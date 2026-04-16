# Development

Setting up Alan Code for contributing.

## Prerequisites

- Python 3.11 or newer.
- Git.
- Optional: [ripgrep](https://github.com/BurntSushi/ripgrep) (`rg`) on your PATH for faster Grep tool performance during dev.

## Clone + install

```bash
git clone https://github.com/<your-fork>/alan-code.git
cd alan-code
python -m venv venv
source venv/bin/activate
pip install -e '.[dev]'
```

`-e` installs in editable mode — changes to `alancode/` take effect immediately without reinstalling. `[dev]` pulls the test tooling (pytest, pytest-asyncio, pytest-cov, ruff).

## Running Alan from the source tree

After editable install:

```bash
alancode --version    # reads from alancode/__version__.py
alancode              # runs from your working copy
```

Edit a file under `alancode/`, save, rerun — changes apply.

## Running tests

```bash
pytest -x -q
```

- `-x` stops at first failure. Useful when iterating.
- `-q` suppresses per-test output. Drop for verbose per-test lines.

Run a specific file:

```bash
pytest tests/unit/test_compaction.py
```

Run tests matching a keyword:

```bash
pytest -k "hook or permission"
```

With coverage:

```bash
pytest --cov=alancode --cov-report=term-missing
```

See [contributing/testing.md](testing.md) for the test organisation.

## Linting

```bash
ruff check .
```

With auto-fix:

```bash
ruff check --fix .
```

No separate formatter — `ruff format` is available but we don't enforce it. Match the surrounding style; PRs that reformat unrelated code will likely be pushed back on.

## Project layout

```
alan-code/
├── alancode/              # the package
│   ├── agent.py           # AlanCodeAgent class
│   ├── query/             # query_loop + state
│   ├── providers/         # Anthropic, LiteLLM, Scripted
│   ├── tools/             # built-in tools + orchestration
│   ├── messages/          # message dataclasses + normalization
│   ├── session/           # session persistence, state, transcripts
│   ├── permissions/       # permission pipeline + rules
│   ├── compact/           # 3-layer compaction
│   ├── hooks/             # pre/post tool-use hooks
│   ├── memory/            # memory system
│   ├── skills/            # skill registry + parser
│   ├── git_tree/          # AGT operations
│   ├── cli/               # CLI entry point + REPL + display
│   ├── gui/               # browser GUI (FastAPI + WebSocket + static/)
│   ├── prompt/            # system prompt assembly
│   ├── api/               # retry, cost tracking
│   └── utils/             # atomic I/O, token counting, env helpers
├── tests/
│   ├── unit/              # fast, no-network, no-disk tests
│   ├── integration/       # full agent runs with scripted provider
│   ├── dummy/             # helpers used across tests
│   └── conftest.py
├── examples/              # runnable example scripts
├── docs/                  # these files
├── pyproject.toml
├── LICENSE
└── README.md
```

Start reading in `alancode/query/loop.py::query_loop`. See [architecture/overview.md](../architecture/overview.md) for the full subsystem map.

## Working with the scripted provider

For code that touches the agent loop or message handling, use the `ScriptedProvider` to write deterministic tests without hitting real APIs:

```python
from alancode.providers.scripted_provider import ScriptedProvider, text, tool_call

provider = ScriptedProvider.from_responses([
    tool_call("Bash", {"command": "ls"}),
    text("Found 3 files"),
])

agent = AlanCodeAgent(provider=provider, permission_mode="yolo")
```

See `tests/integration/test_agent_loop.py` for examples.

## Running the GUI during dev

```bash
alancode --gui
```

Static assets at `alancode/gui/static/` (HTML, JS, CSS). Browser caches them aggressively — **hard-refresh** (Ctrl+Shift+R / Cmd+Shift+R) after JS/CSS edits. A regular reload won't pick them up.

Edits to `alancode/gui/server.py` or `gui_ui.py` require restarting `alancode` (the Python side isn't hot-reloaded).

## Making a change — typical flow

1. Create a feature branch: `git checkout -b feat/my-feature`.
2. Make the change. Keep commits small and focused.
3. Run tests: `pytest -x -q`. Fix regressions.
4. Run lint: `ruff check .`.
5. If you changed the agent behaviour: run a real-model smoke test with `alancode --provider litellm --model openrouter/google/gemini-2.5-flash --permission-mode yolo` in a scratch directory.
6. If you changed prompt behaviour: use `--gui` and inspect the LLM Perspective panel to verify the model sees what you expect.
7. Push and open a PR.

## Commit message style

Look at `git log --oneline -20` for current style. Generally:
- Short imperative mood ("add X", "fix Y", "rename Z").
- No scope prefixes (this isn't Conventional Commits).
- First line under 72 chars; expand in the body only if needed.

## Common tasks

### Add a new tool

1. Create `alancode/tools/builtin/my_tool.py`, subclass `Tool`, implement `name`, `description`, `input_schema`, `permission_level`, `call`.
2. Register in `alancode/tools/builtin/__init__.py` (import + add to `ALL_BUILTIN_TOOLS`).
3. Add schema tests in `tests/unit/test_tools.py`.
4. Document in `docs/reference/tools.md`.

### Add a new setting

1. Add to `alancode/settings.py::SETTINGS_DEFAULTS`.
2. Add a validator in `_VALIDATORS` if the type needs checking.
3. Propagate to wherever it's consumed (usually `QueryParams` → `query_loop`).
4. Add a CLI flag in `alancode/cli/main.py` if appropriate.
5. Document in `docs/reference/settings.md`.

### Add a new slash command

1. Add the entry to `SLASH_COMMANDS` dict in `alancode/cli/repl.py`.
2. Add a dispatch case in `_handle_slash_command`.
3. Write `_handle_<command>` function.
4. Add tests in `tests/unit/test_repl.py` (or integration).
5. Document in `docs/reference/slash-commands.md`.

## Related

- [contributing/testing.md](testing.md) — test organisation + conventions.
- [contributing/release.md](release.md) — PyPI release process.
- [architecture/overview.md](../architecture/overview.md) — the 10,000-ft system view.
