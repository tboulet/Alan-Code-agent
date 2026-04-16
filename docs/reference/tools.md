# Tools reference

Every built-in tool with its input schema, permission level, and usage notes. For the conceptual overview of how tools interact with permissions and hooks, see [concepts/tools-and-permissions.md](../concepts/tools-and-permissions.md).

## Summary table

| Tool | Permission | Purpose |
|---|---|---|
| [`Bash`](#bash) | `exec` | Run a shell command |
| [`Read`](#read) | `read` | Read a file |
| [`Edit`](#edit) | `write` | Exact-string replacement in a file |
| [`Write`](#write) | `write` | Create or overwrite a file |
| [`Glob`](#glob) | `read` | Find files by pattern |
| [`Grep`](#grep) | `read` | Search file contents by regex |
| [`AskUserQuestion`](#askuserquestion) | `read` | Ask the user a multi-choice question |
| [`WebFetch`](#webfetch) | `read` | Fetch a URL |
| [`GitCommit`](#gitcommit) | `write` | Stage and commit with a message |
| [`Skill`](#skill) | `read` | Invoke a user-defined skill template |

All schemas reject unknown fields (`additionalProperties: false`) — the API surfaces clear "unknown parameter" errors instead of silently dropping.

---

## Bash

**Source**: `alancode/tools/builtin/bash.py`
**Permission level**: `exec`

Runs a shell command. Output is stdout + stderr combined, exit-code-non-zero marks the result as `is_error: true` so the model can tell success from failure.

**Parameters**:

| Field | Type | Required | Description |
|---|---|---|---|
| `command` | string | yes | The shell command. Use `&&` to chain. Quote paths with spaces. |
| `timeout` | integer | no | Milliseconds; default 120 000 (2 min). |
| `purpose` | string | no | One-line summary shown to the user on the approval prompt. |

**System prompt guidance**: "Avoid using this tool to run `cat`, `head`, `tail`, `sed`, `awk`, or `echo` when a dedicated tool (Read/Edit/Write/Glob/Grep) exists."

---

## Read

**Source**: `alancode/tools/builtin/file_read.py`
**Permission level**: `read`

Reads a file with line numbering. Output uses `cat -n` format: `<N>\t<line>`. Large files can be sliced with `offset` + `limit`.

**Parameters**:

| Field | Type | Required | Description |
|---|---|---|---|
| `file_path` | string | yes | Absolute path (recommended) or relative to cwd. |
| `offset` | integer | no | Starting line (1-indexed). |
| `limit` | integer | no | Max lines to read. Default 2000. |

---

## Edit

**Source**: `alancode/tools/builtin/file_edit.py`
**Permission level**: `write`

Exact string replacement. Fails if `old_string` isn't unique (to prevent accidental mass-edits) unless `replace_all=true`.

**Parameters**:

| Field | Type | Required | Description |
|---|---|---|---|
| `file_path` | string | yes | Path to the file. |
| `old_string` | string | yes | Exact match including whitespace. |
| `new_string` | string | yes | Replacement. |
| `replace_all` | boolean | no | Replace every occurrence. Default `false`. |

**Output**: includes a unified diff with an `[ALAN-DIFF]` sentinel, rendered as green/red coloured output in the CLI and as a diff block in the GUI.

The tool requires you to have `Read` the file earlier in the conversation — otherwise it errors. Guards against "edit blind" hallucinations.

---

## Write

**Source**: `alancode/tools/builtin/file_write.py`
**Permission level**: `write`

Creates a new file or overwrites an existing one.

**Parameters**:

| Field | Type | Required | Description |
|---|---|---|---|
| `file_path` | string | yes | Path to write. |
| `content` | string | yes | Full file content. |

**Output**: includes a `[ALAN-DIFF]` unified diff showing what changed (or the full new-file content for a creation).

**Guidance in the schema**: "Prefer the Edit tool for modifying existing files — it only sends the diff. Only use this tool to create new files or for complete rewrites."

---

## Glob

**Source**: `alancode/tools/builtin/glob_tool.py`
**Permission level**: `read`

Find files by pattern. Uses `pathlib.Path.glob` semantics (supports `**`).

**Parameters**:

| Field | Type | Required | Description |
|---|---|---|---|
| `pattern` | string | yes | `**/*.py`, `src/*.ts`, etc. |
| `path` | string | no | Search root. Defaults to cwd. |

**Result**: newest-first list of matching paths. Truncates at 1000 matches; if more exist, output clearly says "first 1000 of 2000+ matches, narrow your pattern".

---

## Grep

**Source**: `alancode/tools/builtin/grep_tool.py`
**Permission level**: `read`

Regex search over file contents. Prefers `rg` (ripgrep), falls back to GNU `grep`, then pure Python. Each fallback has a 30-second wall-clock cap.

**Parameters**:

| Field | Type | Required | Description |
|---|---|---|---|
| `pattern` | string | yes | Regex. |
| `path` | string | no | Directory or file. Default cwd. |
| `glob` | string | no | Filename filter, e.g. `*.py`. |
| `output_mode` | string | no | `files_with_matches` (default), `content`, `count`. |
| `context` | integer | no | Lines before/after each match (content mode). |

---

## AskUserQuestion

**Source**: `alancode/tools/builtin/ask_user.py`
**Permission level**: `read` (no filesystem side effects)

Lets the model ask the user a multi-choice question.

**Parameters**:

| Field | Type | Required | Description |
|---|---|---|---|
| `question` | string | yes | The question. |
| `options` | list[string] | yes | At least 1. User can pick one or type their own. |

**Usage note in the schema**: "Use sparingly — only when you truly need user input to proceed. Frame questions clearly with actionable options. Prefer making reasonable assumptions over asking when the choice is low-risk."

Ctrl+C at the prompt aborts the whole turn (not just the tool).

---

## WebFetch

**Source**: `alancode/tools/builtin/web_fetch.py`
**Permission level**: `read`

Fetches a URL (HTTP/HTTPS), strips HTML, returns text content.

**Parameters**:

| Field | Type | Required | Description |
|---|---|---|---|
| `url` | string | yes | Full URL with scheme. |
| `max_length` | integer | no | Truncate at N chars. |

**Schema guidance**: "For GitHub URLs, prefer using the `gh` CLI via Bash instead (e.g. `gh pr view`, `gh issue view`)."

---

## GitCommit

**Source**: `alancode/tools/builtin/git_commit.py`
**Permission level**: `write`

Stages and commits with a given message. Adds `Co-Authored-By: Alan Code` trailer. The commit SHA is tracked in session state so the GUI's Git Tree panel can colour agent commits blue.

**Parameters**:

| Field | Type | Required | Description |
|---|---|---|---|
| `message` | string | yes | Commit message. |
| `files` | list[string] | no | Specific files to stage. Omit to stage all changes (`git add -A`). |

---

## Skill

**Source**: `alancode/tools/builtin/skill_tool.py`
**Permission level**: `read` (loading a prompt template is read-only)

Invokes a user-defined skill from `.alan/skills/` or `~/.alan/skills/`. The skill's body (with `$ARGUMENTS` substituted) becomes the next user message; the model responds to that.

**Parameters**:

| Field | Type | Required | Description |
|---|---|---|---|
| `name` | string | yes | Skill name from the frontmatter. |
| `arguments` | string | no | Replaces `$ARGUMENTS` in the template. |

See [concepts/skills.md](../concepts/skills.md) for how to write skills.

---

## Schema details

All tool schemas are OpenAI-compatible JSON Schema. The full schema for a tool is available via:

```python
from alancode.tools.registry import tools_to_schemas
from alancode.tools.builtin import ALL_BUILTIN_TOOLS

schemas = tools_to_schemas(ALL_BUILTIN_TOOLS)
```

## How tools are exposed to the model

- **Native tool-use models** (Anthropic, OpenAI, Gemini, most mainstream): schemas passed as the `tools=[...]` API parameter.
- **Text-based tool-use models** (GLM, Hermes fine-tunes): the schema list is rendered into the system prompt. Format depends on `tool_call_format` setting (`hermes`, `glm`, `alan`). See `alancode/tools/text_tool_parser.py`.

## Related

- [concepts/tools-and-permissions.md](../concepts/tools-and-permissions.md) — mental model.
- [reference/slash-commands.md](../reference/slash-commands.md) — user-facing commands (not tools).
- [guides/hooks.md](../guides/hooks.md) — inject policy before/after tool execution.
