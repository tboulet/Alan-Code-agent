# Web backend - drive a chat assistant's website as the model

The `web` backend lets Alan use a web chat assistant (Claude, ChatGPT, ...) as
its model by automating the assistant's **browser tab** instead of calling an
API. No API key, no separate server, no browser extension. Useful when you have
a chat subscription but not API credits.

There are two transports:

- **`cdp` (default, recommended)** - talks to Chrome over the DevTools Protocol:
  it types into the composer element and reads replies straight from the message
  DOM node. Robust, and clean enough that a real tool-using agent works (see
  [What works](#what-works)).
- **`x11` (legacy)** - automates the desktop with xdotool (keystrokes, clipboard,
  screen copy). Fragile; kept only as a fallback.

## Requirements

- **Linux / X11 desktop** and **Google Chrome**.
- CDP transport: Chrome's remote-debugging port. Chrome >=136 refuses it on the
  default profile, so the driver runs a **separate** profile at
  `--user-data-dir=/tmp/chrome-cdp --remote-debugging-port=9222`, launching it
  automatically if the port is closed. **You must be logged into the assistant
  in that profile** (open `google-chrome --user-data-dir=/tmp/chrome-cdp` once
  and sign in, or copy your cookies over).
- x11 transport only: `sudo apt install xdotool xclip`.

## Quick start

```bash
alancode --backend web --model claude --tool-call-format alan
```

- `--model` is the **assistant name**, not a model id (see the table below).
- `--tool-call-format alan` is **required**: the web backend can't use native
  tool calling, so tools travel as a text protocol the model writes back.

Verified working end-to-end (a real file is created on disk):

```console
$ alancode --backend web --model claude --tool-call-format alan --permission-mode yolo \
    --print "Use the Bash tool to create /tmp/demo.txt containing PINEAPPLE, then confirm."
Created and verified - /tmp/demo.txt exists and contains PINEAPPLE.
```

## What works

Whether a given assistant works as an agent is **two independent questions**:
does the *model* cooperate with the injected tool protocol, and is the *per-site
DOM flow* (composer / message selectors, new-chat) tuned. Current state:

| assistant | model cooperates? | site flow tuned? | agent works? |
| --- | --- | --- | --- |
| `claude` | yes | yes | **yes - verified** |
| `kimi` | yes (emits real tool calls) | partial | flow needs tuning |
| `chatgpt` | **no** - fabricates tool results | yes | no |
| `gemini` | no - declines / answers in prose | n/a | no |
| `perplexity` | no - search-oriented | n/a | no |
| `mistral` | no - declines | n/a | no |
| `copilot` | unknown | send not landing yet | not yet |
| `deepseek` | unknown | composer selector unconfirmed | not yet |
| `grok` | unknown | composer selector unconfirmed | not yet |
| `qwen` | unknown | composer selector unconfirmed | not yet |

**Simple single-tool tasks work** on Claude and Kimi: given "create this file",
they emit a proper tool call, wait for the real result, and confirm it -
verified with real files on disk. **ChatGPT does not work**: told to use tools
it fabricates results (claims a file was written, invents output) with nothing
actually run - its safety posture resists the injected agent role, and that is
not worked around.

**Real multi-step agentic work is not reliable yet**, for two reasons beyond
per-site plumbing:

- **Web UIs change and break the integration.** Claude's composer (Opus 5) stopped
  accepting programmatic input mid-project after working earlier - exactly the
  fragility inherent to driving a UI rather than an API.
- **Models conflate their own tools with the injected protocol.** On a multi-step
  task Kimi started reasoning about using *its* native ipython/web tools "from a
  different system" instead of Alan's text protocol. Getting a chat model to
  stay in the injected-agent frame across many turns is unreliable.

Treat this backend as usable for light, mostly single-step tasks on a cooperative
assistant - not as a drop-in replacement for an API-backed agent.

## How the CDP transport works

1. **Connect.** Ensure a debuggable Chrome (auto-launched) has a tab for the
   assistant; connect to it over the DevTools websocket.
2. **Compact prompt.** Alan's multi-KB system prompt + tool schemas are rebuilt
   into a ~2 KB brief (honest description of the relay + one-line tool
   signatures + Alan's exact tool-call format). No `<answer>` tags are needed -
   the reply is read from the DOM directly.
3. **Send.** Focus the composer (`cdp_composer_css`), insert the text as a
   trusted input event, press Enter.
4. **Read.** Poll the assistant message nodes (`cdp_message_css`); when a new
   one appears and its text stops changing, return it. Alan parses any
   `<tool_use>` block, executes it for real, and sends the result back.

## Adding or tuning an assistant

Each assistant needs three CSS/URL values on its `WebAssistant` entry in
`alancode/providers/web/assistant.py`:

- `cdp_composer_css` - the input to type into.
- `cdp_message_css` - matches assistant message nodes (last = latest reply).
- `cdp_new_chat_url` - navigated to for a fresh conversation.

Discover them against the live tab with the dev client, e.g.:

```bash
perso_dev/cdp eval https://claude.ai "document.querySelector('[data-testid=\"chat-input\"]') && 'found'"
```

## Limitations

- **Per-site flow is bespoke.** Composer/message selectors and the new-chat
  action differ per assistant and must be tuned individually.
- **Model cooperation is not guaranteed.** Several assistants decline the agent
  role or (ChatGPT) fabricate. Verify any assistant with a real side-effect
  check (does a uniquely-named file actually appear?), never a guessable echo.
- **Auth in a separate profile.** CDP uses `/tmp/chrome-cdp`; you must be logged
  in there. A dedicated, persistent debug profile is cleaner than copying
  cookies each time.
- **Large tool outputs.** Very large results pasted into a composer may be
  truncated or rejected by some sites.
