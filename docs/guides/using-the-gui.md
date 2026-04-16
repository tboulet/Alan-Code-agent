# Using the GUI

`alancode --gui` launches a local browser interface at `http://localhost:8420/`. It shares the same agent core as the CLI — anything that works in the terminal works in the browser — but adds three panels for richer views.

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

## The three panels

### Chat

The main interaction panel. Same flow as the CLI:

- Type at the bottom, press Enter.
- Assistant responses stream token by token.
- Tool calls render as titled boxes.
- Tool results render inline — Edit/Write show a green/red unified diff with line numbers.
- Cost summary after each turn.

Shortcut: **Shift+Enter** inserts a newline, **Enter** submits.

### LLM Perspective

Shows the **exact payload** Alan sent to the model on each turn — the system prompt plus the full `messages=[...]` list. This is the definitive debugging view when the agent's response surprises you:

- "Why did it call that tool?" → check the system prompt section for tools.
- "Why did it forget what we talked about?" → check if compaction happened (look for a `COMPACT_BOUNDARY` system message).
- "What context did the model actually see?" → read the rendered messages.

Useful when tuning skills, diagnosing hallucinations, or reverse-engineering weird model behaviour.

### Git Tree

Visualises the commit graph and the agent's trajectory through it. Click any node to select; four action buttons light up:

- **Move to commit** → `/move <sha>`
- **Revert repo to** → `/revert-to <sha>` (destructive)
- **Revert conv. to** → `/convrevert` (conversation only)
- **Revert all to** → `/allrevert` (both)

See [concepts/git-tree.md](../concepts/git-tree.md) for the colour legend and semantics.

Top-right has a **curvature slider** for the branch-jump arrows and a legend showing what each colour means.

## Showing and hiding panels

Top-bar toggle buttons let you hide any panel — useful when you want a wide Chat view without the Git Tree taking space.

## Permission prompts

When a tool needs approval, a modal appears with:
- Tool name and the input dict.
- **Allow / Deny** buttons.
- For Bash: an **Allow always "<prefix> *" commands** third option, recording the pattern to `.alan/allow_rules.json`.
- A free-text field: type your own answer to send to the model as the "tool result".

Ctrl+C (on the terminal running `alancode`) or closing the tab aborts the turn cleanly.

## Reconnecting

If you close the tab and reopen it, the browser reconnects via WebSocket and the server replays the current session's event history — chat, LLM perspective, and git tree all repopulate automatically.

If you restart `alancode` without refreshing the tab, the new server's history replaces the old one (the frontend gets a `reset` event first). Hard-refresh (**Ctrl+Shift+R**) is needed when `app.js` / `style.css` change between launches — static assets are cached aggressively.

## Known limitations

- **Background tab timers are throttled** by browsers. If you leave the GUI tab in the background and restart `alancode`, the "Disconnected — reconnecting…" state can take 10–60 s to retry. Click back into the tab to force an immediate reconnect.
- **No CORS check currently.** The server binds to `127.0.0.1` only, but if you SSH-forward the port, anyone on the SSH client's host can connect. Don't expose to untrusted networks.
- **No authentication.** Anyone who can reach `localhost:8420` on your machine can drive your agent.
- **The Git Tree is still in development** — some edge cases around merges and detached HEAD aren't fully handled. Works well for linear history and simple branches.

## Shutting down

- `/exit` in the Chat panel — clean shutdown. GUI closes, `alancode` process exits.
- Close the tab — server keeps running; reopen the URL to reconnect.
- Ctrl+C in the terminal — force-quit. May print a traceback (see the known-issue note in [reference/cli.md](../reference/cli.md)).

## Related

- [concepts/git-tree.md](../concepts/git-tree.md) — AGT semantics.
- [reference/slash-commands.md](../reference/slash-commands.md) — all slash commands.
- [reference/cli.md](../reference/cli.md) — `--gui`, `--resume`, other flags.
