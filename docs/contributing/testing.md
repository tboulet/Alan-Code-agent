# Testing

Alan Code has ~700 tests organised into three tiers.

## Running

```bash
# All tests (unit + integration)
pytest

# Stop at first failure, quieter output
pytest -x -q

# Just one file
pytest tests/unit/test_compaction.py

# Match a name keyword
pytest -k "hook or permission"

# Coverage
pytest --cov=alancode --cov-report=term-missing
```

## Organisation

```
tests/
├── conftest.py         # shared fixtures (tmp session dirs, scripted backends, etc.)
├── unit/               # fast, local, no-network
├── integration/        # full agent turns against the scripted backend
└── testing_*/          # adversarial and regression scenario suites
```

### `unit/`

One file per subsystem. No network, no disk except `tmp_path`. Each test should run in under ~100 ms.

Key files:
- `test_compaction.py` — layers A, B, C logic.
- `test_permissions_extended.py` — permission pipeline, allow rules, modes.
- `test_messages.py` — message types, serialization, normalization.
- `test_session.py` — `SessionState`, transcript roundtrip, locking, and session lookup.
- `test_settings.py` — defaults, validators, save/load.
- `test_tools.py` — tool input validation, schemas.
- `test_text_tool_parser.py` - registered text tool-call format parsers.
- `test_hooks.py` — pre/post hook execution, timeout, action fallback.
- `test_skills.py` — frontmatter parser, registry, validation.
- `test_compaction_upgrade.py` — format_compact_summary, the 9-section prompt.
- `test_thinking_extraction.py` — ThinkingBlock extraction in text-based parsers.
- `test_litellm_backend.py`, `test_retry.py` — stream translation and retry safety.
- `test_backend_lifecycle.py`, `test_concurrent_shared_state.py` — cleanup and multi-session shared-state regressions.

### `integration/`

Tests that exercise the full agent loop through `AlanCodeAgent.query_events_async`, backed by the `ScriptedBackend` to stay deterministic.

- `test_agent_loop.py` — happy path + max_iterations_per_turn + early exit.
- `test_reactive_scenarios.py` — error recovery, multi-tool scenarios.
- `test_query_api.py` — the 2×2 matrix (sync/async × text/events).
- `test_scripted_ui.py` — the scripted UI fixture.
- `test_budget_regression.py` - full-loop context-budget and compaction regressions across small and large windows.
- `test_reasoning_tool_calls.py`, `test_thinking_only_response.py` - reasoning-stream tool parsing and empty-response handling.

The `testing_*` directories contain adversarial scenario suites used to challenge invariants such as legal request sizes, conversation liveness, and tool-history validity. They use deterministic backends and are safe to run with the rest of the suite. Accepted limitations remain as strict expected failures, so an unexpected pass prompts a review.

## Writing new tests

### Unit test template

```python
import pytest
from alancode.compact.compact_truncate import compaction_truncate_tool_results
from alancode.messages.types import UserMessage, ToolResultBlock


def test_truncates_oversized_result():
    big = "X" * 50_000
    messages = [
        UserMessage(content=[ToolResultBlock(tool_use_id="t1", content=big)]),
    ]
    result = compaction_truncate_tool_results(
        messages, max_chars=10_000,
    )
    assert "[ALAN-TRUNCATED]" in str(result[0].content)
```

### Integration test template

```python
import pytest
from alancode import AlanCodeAgent
from alancode.backends.scripted_backend import ScriptedBackend, text


@pytest.mark.asyncio
async def test_simple_turn():
    backend = ScriptedBackend.from_responses([text("Hello!")])
    agent = AlanCodeAgent(backend=backend, permission_mode="yolo")
    try:
        answer = await agent.query_async("ping")
        assert answer == "Hello!"
    finally:
        await agent.close()
```

`asyncio_mode = "auto"` is set in `pyproject.toml`, so async tests just need `@pytest.mark.asyncio`.

## Fixtures worth knowing

In `tests/conftest.py`:

- `tmp_cwd` — temporary directory for session state.
- `scripted_agent` — pre-built `AlanCodeAgent` with a `ScriptedBackend`.

Check `conftest.py` for the current list.

## What NOT to test

- Real API calls. Use `ScriptedBackend`. If you genuinely need to verify behaviour against a real model, do it manually before pushing — don't add it to CI.
- Display formatting beyond a smoke test. Rich's output is implementation detail; over-specifying it creates brittle tests.
- Private method internals when the public behaviour covers it. Prefer black-box tests.

## When to add tests

Always:
- Bug fix → regression test covering the original failing input.
- New tool / new slash command → at minimum a smoke test that it runs and validates input.
- New setting → test that its validator works and that the loop respects it.
- New compaction behaviour → test the specific scenario it fixes.

Skip:
- Trivial display / refactor-only changes.
- Docstring updates.

## CI

Currently the repo runs `pytest -x -q` locally. CI integration (GitHub Actions) is planned but not yet in place — contributors are expected to run tests before pushing.

## Debugging failing tests

```bash
pytest tests/path/to/test.py::TestClass::test_name -v
```

`-v` shows per-assert lines. Add `-s` to see `print()` output (pytest captures it by default).

For async tests hanging:

```bash
pytest tests/... --timeout=10
```

(Requires `pytest-timeout`; not in our dev deps but easy to add locally.)

## Related

- [contributing/development.md](development.md) — setup and dev workflow.
- [architecture/overview.md](../architecture/overview.md) — what each subsystem does (guides where to add tests for changes).
