# Project context — `ALAN.md`

`ALAN.md` is a markdown file in your project root that Alan loads into its system prompt at the start of every session. It's where you encode **project-specific conventions, constraints, and context** that the agent should always follow.

## Creating it

```
> /init
```

Creates a starter template with sections for project overview, conventions, and important files. You then edit it freely.

Or just create the file by hand — it's plain markdown.

## What goes in `ALAN.md`

Good content:

```markdown
# Alan's instructions for this project

## Project overview
This is the Python backend for a contract-management system. Async FastAPI +
PostgreSQL. Key domain concepts: Contract, Party, Obligation.

## Conventions
- Use `pathlib` over `os.path`.
- Async everywhere; no `requests`, use `httpx.AsyncClient`.
- Tests live in `tests/` with pytest-asyncio. Run with `pytest -x`.
- Migrations go through Alembic. Never edit committed migrations — add a new one.

## Important files
- `src/models/` — SQLModel definitions.
- `src/api/` — FastAPI routes.
- `src/services/` — business logic.
- `tests/fixtures/` — shared test fixtures. Prefer adding there over duplicating.

## Things to avoid
- Don't add logging calls to hot paths.
- Don't auto-format — we run ruff manually with a specific config.
- Don't commit anything to `main` — always open a PR.
```

Bad content (these belong in memory or other places):
- ❌ A changelog or "what changed recently" — that's what `git log` is for.
- ❌ Secrets or API keys — `ALAN.md` is committed to git.
- ❌ Information about the user personally — that's memory, not project context.
- ❌ Ephemeral context ("we're in the middle of migrating X") — tell the agent directly; will be irrelevant in a month.

## Global `ALAN.md`

`~/.alan/ALAN.md` is the user-scope equivalent: loaded into every Alan session on this machine, regardless of project. Good for user-wide preferences:

```markdown
## My preferences

- Keep replies under 300 words unless I ask for detail.
- Use UK English spelling.
- When editing code, always run the tests immediately after — don't ask.
```

## Both are loaded

If both exist, both are appended to the system prompt (global first, then project). They complement each other.

## Where it fits in the system prompt

`ALAN.md` content goes in **section 13** of the assembled system prompt (see [architecture/system-prompt.md](../architecture/system-prompt.md)). It comes after the skills and memory sections, as an "append block" — Alan's built-in rules run first, then your project rules. If they conflict, project rules win because they're more specific to the task at hand.

## `ALAN.md` vs memory vs skills

Three places you might put "persistent instructions":

| Mechanism | Scope | Structure | Best for |
|---|---|---|---|
| **`ALAN.md`** | One project (or global user-wide) | Free-form markdown | Project conventions, architecture notes, forbidden patterns |
| **Memory** | Per project or global | Structured markdown files with YAML frontmatter | Ongoing facts, user feedback, stale-prone information |
| **Skills** | Per project or global | Invokable templates with `$ARGUMENTS` | Reusable workflows you trigger on demand |

Rule of thumb:
- "Always do X in this project" → `ALAN.md`.
- "The user told me on Tuesday they prefer Y" → memory (`feedback` type).
- "Execute this 5-step review workflow when I say so" → skill.

## `/init` template

Running `/init` creates:

```markdown
# Project Instructions

<!-- This file is read by Alan Code at the start of every session. -->
<!-- Use it to give Alan context about your project, preferences, and conventions. -->

## Project overview

<!-- Describe your project here. What does it do? What technologies does it use? -->

## Conventions

<!-- List coding conventions, naming patterns, or style preferences. -->
<!-- Example: "Use Google-style docstrings", "Prefer pathlib over os.path" -->

## Important files

<!-- Point Alan to key files or directories it should know about. -->
```

Delete the comments as you fill sections in.

## Related

- [concepts/memory.md](memory.md) — the other persistence mechanism.
- [concepts/skills.md](skills.md) — invokable workflow templates.
- [architecture/system-prompt.md](../architecture/system-prompt.md) — exactly where `ALAN.md` content lands in the prompt.
- [reference/slash-commands.md](../reference/slash-commands.md) — `/init`.
