# Slash commands

Slash commands are typed at the prompt and handled locally by Alan Code — they do **not** go through the model. They start with `/` and work the same in CLI and GUI modes.

Typing `/help` in a session prints the currently-registered list.

## Conversation control

| Command | Description |
|---|---|
| `/clear` | Clear the in-memory conversation and reset the latest-usage counters. The old transcript remains recoverable only until another turn rewrites this session's transcript. |
| `/compact [instructions]` | Manually trigger conversation compaction. Optional instructions steer the summary (e.g. `/compact focus on the bug we fixed`). |
| `/exit` | Leave the session cleanly. |

## Session info

| Command | Description |
|---|---|
| `/help` | List all available commands. |
| `/status` | Full session summary: backend, model, session ID, turns, messages, detailed token breakdown (regular / cache-creation / cache-read / output), estimated $ cost, `cwd`, presence of `ALAN.md` and `.alan/settings.json`. |
| `/name <text>` | Set a human-readable name for this session (shown in listings and the GUI). |

## Model & backend

| Command | Description |
|---|---|
| `/model` | Show the current model. |
| `/model <name>` | Switch the active model mid-session. A reminder is injected so the agent knows a switch happened. Changing the model also re-infers the backend (bare `claude-*` → `anthropic-native`; anything else → `auto`). |
| `/backend` | Show the current transport backend. |
| `/backend <name>` | Switch the backend (`auto`, `anthropic-native`, `scripted`). Rarely needed — the backend is inferred from the model string. |

## Settings

| Command | Description |
|---|---|
| `/settings` | Show current session settings. |
| `/settings <key>=<value>` | Update a session setting (e.g. `/settings permission_mode=yolo`). Takes effect immediately; backend-related changes (`backend`, `model`, `api_key`, `base_url`, `request_timeout`, `context_window`) recreate the underlying `LLMBackend`. |
| `/settings-project` | Show project settings from `.alan/settings.json`. |
| `/settings-project <key>=<value>` | Update a project-level default. |

## Memory

| Command | Description |
|---|---|
| `/memory` | Show the current memory mode. |
| `/memory <mode>` | Set mode: `off` (default), `on` (read on start, write on `/save`), `intensive` (also auto-write after significant responses). |
| `/save [note]` | Ask the agent to persist noteworthy info from the conversation into `.alan/memory/`. Optional note becomes the focus of what to save. |

## Git integration

| Command | Description |
|---|---|
| `/diff` | Show the git diff of all uncommitted changes (staged + unstaged), with syntax highlighting. |
| `/commit [guidance]` | Ask the agent to inspect the diff, draft a message, and call `GitCommit`. Optional text guides the generated message. |

## Skills

| Command | Description |
|---|---|
| `/skill list` | List available skills (built-in + user-defined). |
| `/skill <name> [args]` | Invoke a skill. The agent runs with the skill's prompt and (optional) tool filter. |
| `/skill create` | Bootstrap a new skill file interactively. |

## Project context

| Command | Description |
|---|---|
| `/init` | Create a starter `ALAN.md` in the project root. `ALAN.md` is auto-loaded into the system prompt at session start. |

---

**Commands auto-complete** on `/`-prefix input in the CLI (via prompt_toolkit) — start typing a slash and press Tab to cycle.
