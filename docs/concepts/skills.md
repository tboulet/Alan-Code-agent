# Skills

Skills are **user-defined prompt templates** the agent can invoke as a first-class tool. Think of them as saved workflows — a single `/skill review-pr` command expands into a detailed multi-step prompt the model then acts on.

## What a skill is, concretely

A markdown file in `.alan/skills/` (project-scoped) or `~/.alan/skills/` (global-scoped) with YAML frontmatter:

```markdown
---
name: review-pr
description: Review a pull request for correctness, style, and unintended changes.
when_to_use: When the user asks for a code review or mentions reviewing a PR.
argument_hint: <pr-number or branch>
allowed-tools: [Bash, Read, Grep]
---

Review the pull request: $ARGUMENTS

Steps:
1. Run `gh pr view $ARGUMENTS` to get the PR metadata.
2. Run `gh pr diff $ARGUMENTS` to see the full diff.
3. Read any test files that changed to understand intent.
4. Check for: correctness bugs, style issues, unintended changes to unrelated files, missing test coverage.
5. Summarize findings — what's good, what needs changes, blocking vs nitpick.
```

When the user or agent invokes `review-pr`, the body (with `$ARGUMENTS` substituted) becomes the next user message.

## Frontmatter fields

| Field | Required | Purpose |
|---|---|---|
| `name` | yes | Short identifier (`kebab-case` recommended). Used in `/skill <name>` and Skill tool calls. |
| `description` | yes | One-line description the model reads to decide relevance. |
| `when_to_use` | no | Trigger hint. Shown as `TRIGGER:` in the system prompt's skills listing. Encourages the model to autonomously invoke the skill when the hint matches. |
| `argument_hint` | no | Displayed in `/skill list` to show expected argument shape, e.g. `<file_path>`. |
| `allowed-tools` | no | Tool filter — when the skill is active, only these tools are available. Either a list or a single string. |

## Two ways to invoke

**1. User runs a slash command:**

```
> /skill review-pr 123
```

Loads the skill body, substitutes `$ARGUMENTS = "123"`, feeds as a user message.

**2. Agent decides to use the Skill tool:**

The model sees in its system prompt (section 10):
```
- **review-pr** <pr-number or branch>: Review a pull request for correctness...
  TRIGGER: When the user asks for a code review or mentions reviewing a PR.
```

If the user says "review PR #123", the model can call `Skill(name="review-pr", arguments="123")` directly — no `/skill` needed from the user.

## Tool filtering

The optional `allowed-tools` field scopes the skill's execution to a subset of tools:

```yaml
allowed-tools: [Read, Grep, Glob]
```

Only these tools are available while the skill is active. Applied during phase 2 of the query loop — if the skill is invoked on iteration N, iteration N+1's tool list filters to only the allowed set. Resets on turn end.

Useful when you want a skill that *must* stay read-only (e.g., a code review skill that shouldn't accidentally modify files).

## Discovery

On session start, Alan walks both skill directories:

- `<cwd>/.alan/skills/*.md` (project skills)
- `~/.alan/skills/*.md` (global skills)

Each is parsed; failures surface in the log with the file path and error. Valid skills are listed in the system prompt's "Available skills" section.

## Listing skills

```
> /skill list
```

Shows every discovered skill with its description, argument hint, and source file.

## Creating a skill

```
> /skill create
```

Bootstraps a new skill file interactively (prompts you for name, description, body).

Or just write the file by hand — skills are plain markdown.

## Examples worth including in your toolkit

**Run tests and fix failures**:
```markdown
---
name: run-tests
description: Run the test suite and fix any failures.
when_to_use: When the user asks to run tests or check if tests pass.
---

Run `pytest -x` and work through any failures. For each failure:
1. Read the test to understand intent.
2. Read the implementation being tested.
3. Propose a fix (don't apply blindly).
4. Apply the fix.
5. Re-run the affected test.

Stop and report if you hit the same failure twice after different fixes.
```

**Draft a commit message**:
```markdown
---
name: commit
description: Review staged changes and draft a commit message.
---

1. Run `git diff --staged`.
2. Draft a 1-2 sentence message following this repo's style (`git log --oneline -10`).
3. Show the message.
4. Commit with GitCommit when approved.
```

**Explain a file**:
```markdown
---
name: explain
description: Explain what a file does at a conceptual level.
argument_hint: <file_path>
allowed-tools: [Read, Grep, Glob]
---

Read $ARGUMENTS and explain:
1. What this file's job is (one paragraph).
2. Key types and functions (with file:line references).
3. How it fits into the broader codebase (use Grep to find callers).
```

## Validation

Bad frontmatter surfaces in the log at session start:

- Missing `name` or `description` → skill skipped with a WARNING.
- Invalid YAML → WARNING with the parse error location.
- `allowed-tools` that isn't a list of strings → WARNING with the bad shape.

No silent failures: if a skill doesn't load, you know.

## Related

- [reference/slash-commands.md](../reference/slash-commands.md) — `/skill list`, `/skill <name>`, `/skill create`.
- [reference/tools.md](../reference/tools.md) — the `Skill` tool.
- `alancode/skills/` — the source code: `parser.py` (frontmatter), `registry.py` (discovery), `tool_filter.py` (scoping).
