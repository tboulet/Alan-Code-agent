# Memory

Memory is Alan Code's mechanism for **persisting information across sessions**. Unlike the conversation history (which lives per-session and gets compacted), memories live on disk as markdown files and are loaded at the start of every session where memory is enabled.

## What "memory" is (and isn't)

Memory is **not**:
- A substitute for reading code — facts about your codebase should be derived by actually looking at the files.
- A log of what happened — `git log` and the session transcripts already do that.
- A place for ephemeral conversation state.

Memory **is**:
- User preferences ("I prefer concise responses, don't summarize at the end of every message").
- Non-obvious project context ("the auth middleware rewrite is driven by legal, not tech debt").
- Workflow procedures that aren't in the code ("to run tests, start the DB first with `docker-compose up -d db`").
- Pointers to external systems ("pipeline bugs are tracked in the Linear project INGEST").

The distinction matters because memory is scarce (it costs tokens every session) and stale memory is actively harmful — the agent will make decisions based on facts that may no longer be true.

## Memory modes

Set via `--memory <mode>` or `/memory <mode>`:

| Mode | Read on start | Write |
|---|---|---|
| **`off`** *(default)* | no | no |
| **`on`** | yes | on user request or `/save` |
| **`intensive`** | yes | proactively after significant turns + on `/save` |

### `off` (default)

Memory is completely disabled. The agent is told: *"Memory is currently disabled for this session. Do not attempt to read or write memory files. If the user asks to save something, tell them they can enable memory with `/memory on` or `/memory intensive`."*

This is the default for now because **memory is a feature with a cost**: every memory loaded into the system prompt consumes tokens every turn. A half-maintained memory system can be a net negative.

### `on`

Memory is loaded on session start, and only written when you explicitly say so (via natural language — "remember that I prefer X") or run `/save`. This is the right mode for most users who want persistence without surprise writes.

### `intensive`

Memory is still loaded on start, but the agent also proactively saves after significant turns. The system prompt tells it to watch for:
- Corrections to your approach ("stop doing X", "don't do Y").
- Decisions about project direction, architecture, or workflow.
- Information about the user's role, preferences, or expertise.
- External system references.
- Build/test/deploy procedures it learns.

Every 10 iterations (`memory_reminder_threshold = 10`) Alan injects a reminder: *"Several turns have passed since the last memory update. Consider whether any recent corrections, decisions, or preferences are worth saving."*

This mode is best for long-term collaborators who want Alan to build up a model of how they work over time.

## Where memories live

Two scopes:

- **Project memory** at `<cwd>/.alan/memory/` — specific to the current project.
- **Global memory** at `~/.alan/memory/` — shared across all projects on this machine.

Both directories have the same structure:

```
memory/
├── MEMORY.md              # Index — loaded at session start
├── user/                  # Who the user is (global scope mostly)
├── feedback/              # Corrections and validated approaches (global scope mostly)
├── project/               # This project's decisions and ongoing work (project scope)
├── reference/             # External system pointers (project scope)
└── workflow/              # Build/test/deploy procedures (project scope)
```

`MEMORY.md` is an index (one line per memory with a short hook):

```markdown
- [User prefers concise](user/user-prefers-concise.md) — Terse replies, no trailing summaries
- [Migration uses temp table](project/migration-temp-table.md) — Why the 0042 migration uses a staging table
```

Individual memory files start with YAML frontmatter:

```markdown
---
name: User prefers concise responses
description: Terse replies, no trailing "I did X" summaries
type: feedback
---

Keep responses short and direct. Don't add "Let me know if you want me to explain further" or "Hope this helps!" at the end.

**Why:** User said they can read the diff and find followups boring.
**How to apply:** Every response. Applies to code and prose alike.
```

## The five memory types

| Type | What it captures | Scope tendency |
|---|---|---|
| **`user`** | User's role, goals, expertise, preferences. | global |
| **`feedback`** | Rules the user has given — both corrections and validations. | global |
| **`project`** | Ongoing work, decisions, incidents, motivations. | project |
| **`reference`** | External system pointers (Linear projects, Slack channels, dashboards). | project |
| **`workflow`** | Build / test / deploy / dev procedures. | project |

The types are soft categorisations — a memory's type affects the prompt guidance (how to structure the body, when to save) but not the technical behaviour. All types are loaded uniformly.

## Living documents, not append-only logs

The agent is instructed to **update memories in place, not append**. When a fact changes:
- Existing memory → `Edit` tool to update the relevant lines.
- Superseded entirely → `Bash rm` to delete + `Edit MEMORY.md` to drop the line.
- Genuinely new topic → `Write` a new file + `Edit MEMORY.md` to add the index entry.

This is why Alan defaults to `memory=off`: a poorly-maintained memory that just grows monotonically is worse than no memory. The prompt explicitly tells the agent to prefer updates over new files.

## `/save`

```
> /save
```

Triggers an agent-invoking prompt: *"User requested a memory update. Review the recent conversation for information worth saving or updating. Prefer Edit (or Write for a brand-new file) to modify existing entries in place rather than appending new ones that duplicate or supersede them. Remove or rewrite stale entries instead of leaving them alongside newer facts."*

With an argument: `/save the deploy process changed`. The text is appended to give focus.

## Accessing memories

Alan's system prompt (in `on` / `intensive` modes) includes:

1. The memory instructions describing what, when, and how to save.
2. The contents of **both** `MEMORY.md` files (global first, then project) so the agent knows what's available without having to list the directory.

Individual memory files are **not** loaded — the agent reads them on demand with the `Read` tool when `MEMORY.md`'s description suggests relevance.

On recall, Alan is instructed to **verify before acting**: a memory saying "the `x` function is in `foo.py`" may be stale; the agent should grep to confirm before recommending anything based on it.

## Global vs project

- `~/.alan/memory/` applies to **every** Alan session on this machine. Use for user preferences and universal feedback.
- `<project>/.alan/memory/` applies to this project only. Use for project-specific decisions, external references, and workflows.

Both are loaded simultaneously when memory is enabled. Project memory takes precedence for topic overlap — the agent decides which applies from the descriptions.

## Tuning

| Setting | Default | What it does |
|---|---|---|
| `memory` | `off` | Mode: `off`, `on`, `intensive`. |
| `memory_reminder_threshold` | 10 | Iterations between memory-save reminders (intensive mode only). |

## Related

- [reference/slash-commands.md](../reference/slash-commands.md) — `/memory`, `/save`.
- [concepts/project-context.md](project-context.md) — `ALAN.md` (static per-project instructions, not memory).
- The `/save` flow in [architecture/query-loop.md](../architecture/query-loop.md).
