# Hooks

Hooks are **shell commands Alan runs at specific lifecycle events**. They let you inject custom policy without modifying the agent code: audit every tool call, block certain operations, preprocess inputs, send notifications — anything a shell script or small program can do.

## Event types

| Event | When it fires | Can block? |
|---|---|---|
| `pre_tool_use` | Before a tool runs, after permission approval. | ✅ Yes — exit with `{"action": "deny"}`. |
| `post_tool_use` | After a tool runs. | No. Informational only. |
| `session_start` | When a session begins. | No. |
| `session_end` | When a session ends. | No. |

## Configuration

In `.alan/settings.json`:

```json
{
  "hooks": {
    "pre_tool_use": [
      {
        "command": "python3 /path/to/my_policy_check.py",
        "tools": ["Bash", "Edit", "Write"],
        "timeout": 5
      }
    ],
    "post_tool_use": [
      {
        "command": "/usr/local/bin/audit-log",
        "timeout": 2
      }
    ],
    "session_start": [
      {
        "command": "echo 'Alan session started'",
        "shell": true
      }
    ]
  }
}
```

Each hook entry supports:

| Field | Required | Default | Purpose |
|---|---|---|---|
| `command` | yes | — | Command to run. Tokenised via `shlex.split` by default (argv-style, no shell interpretation). |
| `tools` | no | `null` (all) | List of tool names this hook applies to. |
| `timeout` | no | `5` | Seconds before the hook is killed. |
| `shell` | no | `false` | Opt-in shell interpretation (pipes, redirects, globs). **Documented as the risky path** — use only when necessary. |

## How a hook sees the event

Alan sends a JSON payload on **stdin** to the hook command:

```json
{
  "hook_type": "pre_tool_use",
  "tool_name": "Bash",
  "tool_input": {"command": "rm -rf /tmp/scratch"},
  "session_id": "a1b2c3d4..."
}
```

Fields depend on the event type. Always include `hook_type`, `session_id`; tool-related events also include `tool_name` and `tool_input`.

## How a hook responds

### `pre_tool_use` — control whether the tool runs

The hook's stdout is parsed as JSON:

```json
{"action": "allow"}
```
```json
{"action": "deny", "message": "Blocked: /tmp/scratch is protected"}
```
```json
{"action": "ask", "message": "This looks destructive — are you sure?"}
```

- `allow` — tool runs normally.
- `deny` — tool is blocked; the `message` is fed back to the model so it can adapt.
- `ask` — force a user permission prompt, even if mode is `yolo` or a rule would otherwise auto-allow. Useful as a "confirm-for-this-pattern" guard.

If the hook prints non-JSON, or doesn't print anything, the action defaults to `allow` **unless** the hook exited non-zero — in which case Alan denies with the exit-code message.

### Timeouts and errors

If the hook times out (default 5 s) or crashes, Alan falls back to **`ask`** for `pre_tool_use` — a broken safety-critical hook must not silently allow. For `post_tool_use` / session events (informational), timeout falls back to `allow` since there's nothing to block.

This fallback behaviour was tightened in response to an audit: originally timeouts defaulted to `allow` everywhere, which made broken security hooks invisible.

## Examples

### Block writes outside the project

```bash
#!/usr/bin/env python3
import json, os, sys

payload = json.load(sys.stdin)
if payload.get("tool_name") not in ("Write", "Edit"):
    print(json.dumps({"action": "allow"}))
    sys.exit(0)

path = payload["tool_input"].get("file_path", "")
project_root = os.environ.get("PROJECT_ROOT", os.getcwd())
if os.path.realpath(path).startswith(project_root):
    print(json.dumps({"action": "allow"}))
else:
    print(json.dumps({
        "action": "deny",
        "message": f"Writes outside project root ({project_root}) are blocked.",
    }))
```

Save as `hooks/no-escape.py`, configure:
```json
{
  "hooks": {
    "pre_tool_use": [
      {"command": "python3 hooks/no-escape.py", "tools": ["Write", "Edit"]}
    ]
  }
}
```

### Log every tool call

```bash
#!/bin/bash
# Read stdin JSON, append to audit log, pass through.
read -r payload
echo "$(date -Is) $payload" >> ~/.alan-audit.log
echo '{"action": "allow"}'
```

```json
{
  "hooks": {
    "pre_tool_use": [
      {"command": "bash hooks/audit.sh", "shell": false}
    ]
  }
}
```

### Send a desktop notification on session end

```json
{
  "hooks": {
    "session_end": [
      {
        "command": "notify-send 'Alan session ended'",
        "shell": false
      }
    ]
  }
}
```

## Security model

- **Commands run with your user's permissions** — same trust boundary as any other shell script in your dotfiles.
- **`shell: false` is the default** — no shell metachar interpretation, argv-style exec. Makes `create_subprocess_exec` the safe path.
- **`.alan/settings.json` is trusted** — if someone compromises that file, they can configure arbitrary commands. Treat it like your `.bashrc`: don't commit unchecked content, be cautious when pulling projects from untrusted sources.

## Inspecting hook output

Hooks' `stderr` is logged at DEBUG level. Enable verbose mode to see it:

```bash
alancode --verbose
```

Or set `"verbose": true` in settings.

When a `pre_tool_use` hook denies, the `message` field shows up in the agent's conversation as the tool result — fully visible in the GUI's Chat panel and LLM Perspective.

## Related

- [reference/settings.md](../reference/settings.md) — the `hooks` setting key.
- [concepts/tools-and-permissions.md](../concepts/tools-and-permissions.md) — where hooks fit in the permission pipeline.
- `alancode/hooks/registry.py` — the implementation.
