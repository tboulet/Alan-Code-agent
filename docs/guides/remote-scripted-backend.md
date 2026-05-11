# Remote-scripted backend — be the model

The `RemoteScriptedProvider` lets an external caller act as Alan's LLM over HTTP. Use it to:

- Drive an agent step-by-step by hand to debug a system prompt, tool wiring, or framework integration.
- Have a second program impersonate the model deterministically without writing scripted rules ahead of time.
- Smoke-test embedding setups (GameAgents, custom orchestrators) without paying for tokens.

The backend lives in `alancode/providers/remote_scripted_provider.py`. Selected via:

```bash
alancode --backend scripted --model remote
```

Or from Python:

```python
agent = AlanCodeAgent(backend="scripted", model="remote", ...)
```

When the agent starts you'll see two lines on stdout:

```
[remote-scripted] LLM endpoint: http://127.0.0.1:8430
[remote-scripted] bound to session <sid8> (cwd=...)
```

Port `8430` is the default; if it's taken the provider scans upward (up to `8450`).

## Endpoints

All endpoints live under `http://127.0.0.1:<port>`. There is no auth — the server only binds to `127.0.0.1`.

| Method | Path             | Purpose |
|--------|------------------|---------|
| GET    | `/api/health`    | `{"ok": true}` once the server is up. |
| GET    | `/api/session`   | Session metadata: `session_id`, `cwd`, `model`, `port`, `calls_served`. |
| GET    | `/api/pending`   | The LLM call currently waiting for a response. Returns `204 No Content` when idle. |
| POST   | `/api/respond`   | Submit the assistant's response. Unblocks `stream()` and returns `{"accepted": true}`. |

### Pending payload (GET `/api/pending`)

```json
{
  "request_id": "remote-req-3-a1b2c3d4",
  "turn": 3,
  "model": "remote",
  "max_tokens": 16000,
  "thinking": {"type": "disabled", "budget_tokens": null},
  "stop_sequences": null,
  "system": ["You are a coding agent..."],
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": [...]},
    {"role": "tool", "tool_call_id": "...", "content": "..."}
  ],
  "tools": [
    {"name": "Bash", "description": "...", "input_schema": {...}},
    ...
  ],
  "session_id": "ce907458de4a4307...",
  "cwd": "/path/to/work_dir"
}
```

The same payload is mirrored to `<cwd>/.alan/sessions/<session_id>/remote_inbox.json` at the moment the call goes pending. Read either path — they're identical. The file persists after the run ends, holding the last pending snapshot.

### Response payload (POST `/api/respond`)

```json
{
  "text": "I'll list the directory.",
  "tool_calls": [
    {"name": "Bash", "input": {"command": "ls -la"}}
  ],
  "thinking": null,
  "stop_reason": "tool_use",
  "usage": {"input_tokens": 100, "output_tokens": 50}
}
```

All fields are optional:

- `text` — assistant text (plain string).
- `tool_calls` — list of `{"name": "ToolName", "input": {...}}`. Each gets an auto-generated `tool_use` id; pass `"id": "toolu_..."` to override.
- `thinking` — optional thinking text (emitted as a `StreamThinkingDelta`).
- `stop_reason` — `"end_turn"`, `"tool_use"`, `"max_tokens"`, etc. Auto-inferred from the body if omitted (`tool_use` when `tool_calls` is non-empty, else `end_turn`).
- `usage` — token counts. Optional; defaults to zeros if omitted.

Errors:

```json
{"error": "rate limit hit", "error_type": "overloaded", "status_code": 529}
```

Produces a `StreamError` event for the agent, exactly like a real provider failure.

## Shell macros

`scripts/alan-remote-macros.sh` ships a set of shell functions wrapping the endpoints. Source it once per terminal:

```bash
source ~/projects/Alan-Code-agent/scripts/alan-remote-macros.sh
alan-help    # list available commands
```

Then drive an experiment with:

```bash
alan-pending-last       # latest message Alan got
alan-bash 'ls -la'      # call the Bash tool
alan-wait               # block until next pending call
alan-text "I'm done."   # text-only turn
alan-exit               # ExitTask, ends the experiment
```

The default port is `8430`; override with `ALAN_PORT=8431` before sourcing.

## Typical interaction loop

```bash
PORT=8430

# 1. See what the model is being asked.
curl -s http://127.0.0.1:$PORT/api/pending | jq

# 2. Send a text-only response (ends the turn).
curl -s -X POST http://127.0.0.1:$PORT/api/respond \
  -H 'Content-Type: application/json' \
  -d '{"text": "acknowledged"}'

# 3. Or call a tool.
curl -s -X POST http://127.0.0.1:$PORT/api/respond \
  -H 'Content-Type: application/json' \
  -d '{"text": "running ls", "tool_calls": [{"name": "Bash", "input": {"command": "ls"}}]}'

# 4. Poll until the next call is pending.
while [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/api/pending)" != "200" ]; do
  sleep 0.3
done
```

`GET /api/pending` returns `204` while the previous response is still being processed (tool execution, framework bookkeeping). It flips to `200` when the next LLM call goes pending. The simplest pattern is to poll until `200`.

## Session shutdown

The provider's HTTP server lives on a daemon thread. It shuts down when:

- The agent calls `agent.close()` (normal lifecycle).
- The process exits.

After shutdown the port goes free, and `/api/health` becomes unreachable. That's the signal that the session ended.

## Concurrent sessions

If two agents both ask for the same port, the second picks the next free one (8431, 8432, …). Each agent's server is independent. The session id and cwd are exposed at `/api/session` so you can disambiguate which agent you're talking to.

## Inside Python

The provider can also be inspected and driven directly without HTTP:

```python
agent = AlanCodeAgent(backend="scripted", model="remote", ...)
provider = agent._provider  # RemoteScriptedProvider instance
print(provider._port)
```

…but the HTTP API is the supported surface and what tooling should target.

## Related

- [reference/python-api.md](../reference/python-api.md) — the `AlanCodeAgent` constructor.
- [CHANGELOG.md](../../CHANGELOG.md) — entry from 2026-05-11.
