# Using the GUI

`alancode --gui` launches a local browser interface at `http://localhost:8420/`. It shares the same agent core as the CLI — anything that works in the terminal works in the browser — but adds two panels for richer views.

## Launch

```bash
alancode --gui
```

You'll see:

```
  GUI: http://localhost:8420/<project-slug>/

  Open the URL in your browser. All interaction happens there.
```

Open the URL. The tab may say "Connecting…" briefly while the WebSocket handshake completes — this includes any first-time LiteLLM import (~1.5 s on cold Python).

## The two panels

### Chat

The main interaction panel. Same flow as the CLI:

- Type at the bottom, press Enter.
- Assistant responses stream token by token.
- Tool calls render as titled boxes.
- Tool results render inline — Edit/Write show a green/red unified diff with line numbers.
- Cost summary after each turn.

Shortcut: **Shift+Enter** inserts a newline, **Enter** submits.

### LLM Perspective

Shows Alan's **normalized pre-backend view** for each model call - the system prompt plus the `messages=[...]` list after filtering, role merging, and optional thinking re-injection. This is the most useful conversation-level debugging view when the agent's response surprises you:

- "Why did it call that tool?" → check the system prompt section for tools.
- "Why did it forget what we talked about?" → check if compaction happened (look for a `COMPACT_BOUNDARY` system message).
- "What conversation did Alan hand to the backend?" → read the rendered messages.

Useful when tuning skills, diagnosing hallucinations, or reverse-engineering weird model behaviour.

This is not a wire capture. It does not show tool schemas, the output budget, stop sequences, cache markers, or the Anthropic/LiteLLM backend's final provider-specific reshaping.

## Showing and hiding panels

Top-bar toggle buttons let you hide either panel.

## Permission prompts

When a tool needs approval, a modal appears with:
- Tool name and the input dict.
- **Allow / Deny** buttons.
- For Bash: an **Allow always "<prefix> *" commands** third option, recording the pattern to `.alan/allow_rules.json`.
- A free-text field: type your own answer to send to the model as the "tool result".

Ctrl+C (on the terminal running `alancode`) or closing the tab aborts the turn cleanly.

## Reconnecting

If you close the tab and reopen it, the browser reconnects via WebSocket and the server replays the current session's chat and LLM perspective history.

If you restart `alancode` without refreshing the tab, the new server's history replaces the old one (the frontend gets a `reset` event first). Hard-refresh (**Ctrl+Shift+R**) is needed when `app.js` / `style.css` change between launches — static assets are cached aggressively.

## Known limitations

- **Background tab timers are throttled** by browsers. If you leave the GUI tab in the background and restart `alancode`, the "Disconnected — reconnecting…" state can take 10–60 s to retry. Click back into the tab to force an immediate reconnect.
- **No CORS check currently.** The server binds to `127.0.0.1` only, but if you SSH-forward the port, anyone on the SSH client's host can connect. Don't expose to untrusted networks.
- **No authentication.** Anyone who can reach `localhost:8420` on your machine can drive your agent.

## Shutting down

- `/exit` in the Chat panel — clean shutdown. GUI closes, `alancode` process exits.
- Close the tab — server keeps running; reopen the URL to reconnect.
- Ctrl+C in the terminal — force-quit. May print a traceback (see the known-issue note in [reference/cli.md](../reference/cli.md)).

## Related

- [reference/slash-commands.md](../reference/slash-commands.md) — all slash commands.
- [reference/cli.md](../reference/cli.md) — `--gui`, `--resume`, other flags.
