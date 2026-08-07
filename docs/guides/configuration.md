# Configuration

Alan Code has many knobs — backend, model, permission mode, compaction thresholds, memory behaviour, and more. This guide explains **where settings live** and **how they resolve** so you can predict what's in effect at any moment.

## The priority chain

Every setting resolves through this chain, highest priority first:

1. **Constructor arguments / CLI flags** — `AlanCodeAgent(model="...")` in code, or `alancode --model ...` on the command line. Always win.
2. **Session settings** — `.alan/sessions/<id>/settings.json`. Snapshot of effective settings at session start, used on `--resume` so a resumed session keeps the same config.
3. **Project settings** — `<cwd>/.alan/settings.json`. Auto-generated on first run, checked into git (optionally).
4. **Built-in defaults** — hard-coded in `alancode/settings.py::SETTINGS_DEFAULTS`.

A setting set at level 1 overrides everything below. A setting absent at level 1 falls through to level 2, then 3, then 4.

## The three files

### `.alan/settings.json` (project-level)

```json
{
  "backend": "anthropic-native",
  "model": "claude-sonnet-4-6",
  "permission_mode": "edit",
  "memory": "off",
  "compaction_threshold_percent": 75
}
```

Auto-created on first run with sensible defaults. Commit it if you want teammates to pick up the same config, gitignore it if you don't.

### `.alan/sessions/<id>/settings.json` (per-session snapshot)

Created automatically when a session starts. Locks in the effective config so that resuming the session uses the same settings even if you've since changed `.alan/settings.json`.

You don't edit these manually — they're managed by the session system.

### CLI flags and constructor args

```bash
alancode --model gpt-4o --permission-mode yolo
```

Or in Python:

```python
AlanCodeAgent(
    model="gpt-4o",
    permission_mode="yolo",
    max_iterations_per_turn=15,
)
```

Pass only what you want to override — omitted args fall through to the chain. The transport backend is inferred from `model`; pass `backend=` explicitly only if you need to override the inference.

## Changing a setting mid-session

Three ways:

**Slash command** (recommended for interactive use):
```
> /settings permission_mode=yolo
```

Updates the session's effective setting and persists it to the session snapshot. Takes effect immediately. Backend-related changes (`backend`, `model`, `api_key`, `base_url`, `request_timeout`) recreate the underlying `LLMBackend`; creation is transactional, so a failure keeps the old backend and settings. Changing `model` also re-infers the backend (bare `claude-*` → `anthropic-native`, anything else → `auto`).

**Edit the project file**:
```
> /settings-project permission_mode=yolo
```

Writes to `.alan/settings.json`. Does NOT affect the current session — only future sessions pick this up. Use when you want to change the default for this project.

**Direct file edit**: open `.alan/settings.json` in an editor. Same effect as `/settings-project`.

## Every setting key

Full reference: [reference/settings.md](../reference/settings.md).

Highlights:

| Key | Default | What it does |
|---|---|---|
| `backend` | `anthropic-native` | `auto`, `anthropic-native`, or `scripted` |
| `model` | `claude-sonnet-4-6` | Model identifier; non-Claude routes use LiteLLM provider prefixes |
| `request_timeout` | `"auto"` | SDK default, or 3,600 seconds for a custom endpoint |
| `permission_mode` | `edit` | `yolo`, `edit`, `safe` |
| `memory` | `off` | `off`, `on`, `intensive` |
| `max_iterations_per_turn` | `None` | Cap API calls per user message |
| `context_window` | `"auto"` | Resolve from model/server/probe; set an integer to override |
| `compaction_threshold_percent` | `"auto"` (80) | Layer C threshold as a percentage of usable input space |
| `tool_result_max_chars` | `"auto"` | Context-scaled per-result cap before Layer A truncation |
| `hooks` | `{}` | Pre/post tool-use hooks |

## Where API keys go

**Not in `settings.json`.** The `api_key` field is flagged ephemeral (`_EPHEMERAL_FIELDS` in `alancode/settings.py`) — it never persists to disk. It's only read from:

1. CLI: `--api-key sk-...` (one-shot, not saved).
2. Environment: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `OPENROUTER_API_KEY`, etc.

Put keys in your shell profile or a `.envrc` that `direnv` manages — standard dev-env hygiene.

## First-run setup

On the very first `alancode` invocation in a new project (no `.alan/` yet), a short interactive setup detects available API keys from your environment and writes an initial `.alan/settings.json`.

If you've been using Alan for a while, first-run has already happened — the file exists, setup is skipped on subsequent runs.

## Migrating settings forward

New settings added in future Alan releases are filled from built-in defaults in memory when absent from an older file. Existing values remain unchanged. The project file itself is not rewritten merely because a new default exists.

## Inspecting current settings

```
> /settings
```

With no arguments, prints the full effective settings dict as JSON.

```
> /settings-project
```

Prints the `.alan/settings.json` file specifically.

## The difference: session vs project

Both files overlap 95 %. The difference is their role:

- **Project settings** are the declared baseline for this project.
- **Session settings** are the snapshot that this specific session is using (even after you edit the project file).

Example: you start a session with `permission_mode=edit`, then edit `.alan/settings.json` to `yolo`. The current session and a later `--resume` of it keep the session snapshot. A brand-new session picks up `yolo`; use `/settings permission_mode=yolo` to change the current one.

Most of the time you won't notice — but it explains why editing the project file mid-session seems not to take effect.

## Related

- [reference/settings.md](../reference/settings.md) — every key with its default.
- [reference/cli.md](../reference/cli.md) — every CLI flag.
- [reference/slash-commands.md](../reference/slash-commands.md) — `/settings`, `/settings-project`.
- `alancode/settings.py` — validators, defaults, load/save.
