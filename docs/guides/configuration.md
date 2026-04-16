# Configuration

Alan Code has many knobs — provider, model, permission mode, compaction thresholds, memory behaviour, and more. This guide explains **where settings live** and **how they resolve** so you can predict what's in effect at any moment.

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
  "provider": "litellm",
  "model": "openrouter/google/gemini-2.5-pro",
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
alancode --provider litellm --model openai/gpt-4o --permission-mode yolo
```

Or in Python:

```python
AlanCodeAgent(
    provider="litellm",
    model="openai/gpt-4o",
    permission_mode="yolo",
    max_iterations_per_turn=15,
)
```

Pass only what you want to override — omitted args fall through to the chain.

## Changing a setting mid-session

Three ways:

**Slash command** (recommended for interactive use):
```
> /settings permission_mode=yolo
```

Updates the session's effective setting AND persists to the session snapshot. Takes effect immediately. Provider-related changes (`provider`, `model`, `api_key`, `base_url`, `force_supports_*`) trigger provider recreation.

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
| `provider` | `anthropic` | `anthropic`, `litellm`, or `scripted` |
| `model` | `claude-sonnet-4-6` | Model identifier |
| `permission_mode` | `edit` | `yolo`, `edit`, `safe` |
| `memory` | `off` | `off`, `on`, `intensive` |
| `max_iterations_per_turn` | `None` | Cap API calls per user message |
| `compaction_threshold_percent` | `80` | When auto-compact fires |
| `tool_result_max_chars` | `20_000` | Per-tool-result size before Layer A truncation |
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

New settings added in future Alan releases are auto-merged into your existing `.alan/settings.json` on next load: missing keys get the new default, existing keys preserve your customisations. You never have to re-run `/init` or delete the file to pick up new knobs.

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

Example: you start a session with `permission_mode=edit`, then edit `.alan/settings.json` to `yolo`. This session stays on `edit` until `/clear` or restart. A new session (or `--resume` on this one) picks up `yolo`.

Most of the time you won't notice — but it explains why editing the project file mid-session seems not to take effect.

## Related

- [reference/settings.md](../reference/settings.md) — every key with its default.
- [reference/cli.md](../reference/cli.md) — every CLI flag.
- [reference/slash-commands.md](../reference/slash-commands.md) — `/settings`, `/settings-project`.
- `alancode/settings.py` — validators, defaults, load/save.
