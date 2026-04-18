# Slash commands

Slash commands are typed at the prompt and handled locally by Alan Code — they do **not** go through the model. They start with `/` and work the same in CLI and GUI modes.

Typing `/help` in a session prints the currently-registered list.

## Conversation control

| Command | Description |
|---|---|
| `/clear` | Clear the conversation and start a fresh turn. Keeps the session file (use `--resume` later) but drops in-memory messages and compaction state. |
| `/compact [instructions]` | Manually trigger conversation compaction. Optional instructions steer the summary (e.g. `/compact focus on the bug we fixed`). |
| `/exit` | Leave the session cleanly. |

## Session info

| Command | Description |
|---|---|
| `/help` | List all available commands. |
| `/status` | Full session summary: provider, model, session ID, turns, messages, detailed token breakdown (regular / cache-creation / cache-read / output), estimated $ cost, `cwd`, presence of `ALAN.md` and `.alan/settings.json`. |
| `/name <text>` | Set a human-readable name for this session (shown in listings and the GUI). |

## Model & provider

| Command | Description |
|---|---|
| `/model` | Show the current model. |
| `/model <name>` | Switch the active model mid-session. A reminder is injected so the agent knows a switch happened. |
| `/provider` | Show the current provider. |
| `/provider <name>` | Switch provider (`anthropic`, `litellm`). Both commands remind you to update the other if needed. |

## Settings

| Command | Description |
|---|---|
| `/settings` | Show current session settings. |
| `/settings <key> <value>` | Update a session setting (e.g. `/settings permission_mode yolo`). Takes effect immediately; provider-related changes recreate the provider. |
| `/settings-project` | Show project settings from `.alan/settings.json`. |
| `/settings-project <key> <value>` | Update a project-level default. |

## Memory

| Command | Description |
|---|---|
| `/memory` | Show the current memory mode. |
| `/memory <mode>` | Set mode: `off` (default), `on` (read on start, write on `/save`), `intensive` (also auto-write after significant responses). |
| `/save [note]` | Ask the agent to persist noteworthy info from the conversation into `.alan/memory/`. Optional note becomes the focus of what to save. |
| `/memodiff` | Show memory diff vs. the last commit. |

## Git integration

| Command | Description |
|---|---|
| `/diff` | Show the git diff of all uncommitted changes (staged + unstaged), with syntax highlighting. |
| `/commit [message]` | Stage all changes and create a commit. Without an argument, an AI-generated message is used. |

## Agentic Git Tree (AGT)

The Git Tree panel in the GUI corresponds to these commands; they also work from the CLI.

| Command | Description |
|---|---|
| `/move <sha-or-branch>` | Move the agent to a commit or branch. Performs a git checkout and injects a reminder so the agent re-reads files. |
| `/revert [N]` | Revert `N` commits back (default 1). Discards uncommitted changes; the conversation remains. |
| `/convrevert [N]` | Revert `N` steps in the conversation only (the agent "forgets" recent messages). The repo is unchanged. |
| `/allrevert [N]` | Revert both the repo and the conversation by `N` steps, together. |

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
