# Tools and permissions

Alan Code ships with a handful of built-in tools the model uses to do actual work — read files, run shell commands, edit code, fetch URLs, ask you questions. Whether each tool runs without your approval depends on two things: the tool's **permission level** and the session's **permission mode**.

## The built-in tools

| Tool | What it does | Permission level |
|---|---|---|
| `Bash` | Runs a shell command. | `exec` |
| `Read` | Reads a file. | `read` |
| `Edit` | Exact-string replacement inside a file. | `write` |
| `Write` | Creates or overwrites a file. | `write` |
| `Glob` | Find files by pattern. | `read` |
| `Grep` | Search file contents by regex. | `read` |
| `AskUserQuestion` | Model asks you a multi-choice question. | `read` |
| `WebFetch` | Fetch a URL and strip HTML. | `read` |
| `GitCommit` | Stage + commit with a given message. | `write` |
| `Skill` | Invoke a user-defined skill template. | `read` |

Full schemas and examples: [reference/tools.md](../reference/tools.md).

## Permission levels

Each tool declares its blast radius:

- **`read`** — no side effects on disk, no network writes. Safe to run.
- **`write`** — modifies files in the working directory.
- **`exec`** — runs arbitrary external commands (Bash).

## Permission modes

The session-wide stance on when to ask you before running a tool:

| Mode | read | write | exec |
|---|---|---|---|
| **`safe`** | ✅ auto | 🟡 ask | 🟡 ask |
| **`edit`** *(default)* | ✅ auto | 🟡 ask | 🟡 ask |
| **`yolo`** | ✅ auto | ✅ auto | ✅ auto |

Set per-session with `--permission-mode` or at runtime with `/settings permission_mode=yolo`.

> **Note**: The difference between `safe` and `edit` mainly matters for **hooks** and **allow rules** — both modes ask for write/exec by default, but `safe` is stricter about rules that downgrade to auto-allow. In practice most users pick `edit` for interactive work and `yolo` for trusted autonomous runs like the [auto-fix loop example](../../examples/example_2_auto_fix_loop/).

## The permission prompt

When a tool would need your approval, Alan shows:

```
? Allow Bash?
Tool 'Bash' wants to execute with input: {'command': 'git rev-parse HEAD'}
  1) Allow
  2) Deny
  3) Allow always "git *" commands
  Or type your own answer

Your choice: _
```

- **Allow** — runs this one call.
- **Deny** — the tool is blocked; the model gets a "Permission denied" result and can adapt (e.g. ask you why).
- **Allow always** — only shown for Bash. Extracts the command's first word (`git`, `npm`, `pytest`...) and records a session-scoped rule so future `git *` calls run without asking. Persisted in `.alan/allow_rules.json` so it also applies to future sessions in this project.
- **Type your own answer** — your text is fed back to the model as the tool result. Useful for "deny, and here's why" without a menu option.

Ctrl+C at the prompt cleanly aborts the turn.

## Allow rules

Rules live per-project in `.alan/allow_rules.json`. Example:

```json
[
  {"tool_name": "Bash", "rule_content": "git *", "source": "session"},
  {"tool_name": "Bash", "rule_content": "pytest *", "source": "session"},
  {"tool_name": "Read", "rule_content": null, "source": "project"}
]
```

- `rule_content: null` = blanket rule, matches any input.
- `rule_content: "pattern *"` = matches when the target field (`command` for Bash, `file_path` for Read/Edit/Write, …) starts with `pattern`.

Matching is per-tool-field, not scanning every string in the args — so a rule `Read: "config*"` matches `file_path="config.json"` but not unrelated fields. See [`alancode/permissions/pipeline.py`](https://github.com/example/alan-code/blob/main/alancode/permissions/pipeline.py) for the exact logic.

## Deny rules

There's no CLI to add deny rules today — the concept exists in the code (`alancode/permissions/context.py::PermissionRule` with `behavior=DENY`) but no user-facing command populates them. If you need a blanket block on a tool, configure a [pre-tool-use hook](../guides/hooks.md) instead.

## Hooks — escape hatch for custom policy

Hooks are shell commands (or argv-style commands; see the `shell: false` default) that Alan runs before or after each tool call. A `PreToolUse` hook can inspect the tool name + input, print `{"action": "deny", "message": "..."}` to stdout, and Alan will block the call with that message to the model.

Real use cases:
- Block Bash commands touching production directories.
- Require `rg` queries to be scoped to specific paths.
- Log every edit to an audit file.

See [guides/hooks.md](../guides/hooks.md) for config and worked examples.

## How a tool call actually runs (one iteration)

1. Model emits a `tool_use` block via the stream.
2. Loop collects all `tool_use` blocks for this iteration.
3. For each block, in parallel (for read-only tools) or serially (for writes):
   - `tool.validate_input(args, ctx)` — structural check.
   - Pre-tool-use hook fires (if configured).
   - Permission pipeline: allow-rule? deny-rule? mode-auto? else ask the user.
   - `tool.call(args, ctx)` — the actual work.
   - Post-tool-use hook fires.
4. Each tool's `ToolResult.data` becomes the content of a `tool_result` block in a `UserMessage`.
5. That message gets sent back to the model on the next iteration.

See `alancode/tools/execution.py` for `run_tool_use`, and `alancode/tools/orchestration.py` for the concurrent batching logic (`max_tool_concurrency = 10` by default).

## Related

- [reference/tools.md](../reference/tools.md) — full tool schemas.
- [guides/hooks.md](../guides/hooks.md) — hook configuration.
- [reference/settings.md](../reference/settings.md) — `permission_mode`, `max_tool_concurrency`.
- [concepts/git-tree.md](git-tree.md) — `GitCommit` integration with AGT.
