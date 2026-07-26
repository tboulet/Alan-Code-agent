"""Chrome DevTools Protocol client for the web backend.

An alternative to the xdotool driver: instead of automating the desktop
(focus, keystrokes, screen copy), this talks to Chrome's debug port and reads
and writes the page's DOM directly. That removes every fragile part of the
xdotool approach - no window focus, no coordinate guessing, no Ctrl+A page
copy polluted by UI chrome - and is far more portable across assistant sites,
since each one only needs a few CSS selectors.

Requires Chrome started with ``--remote-debugging-port`` on a NON-default
``--user-data-dir`` (Chrome >=136 refuses remote debugging on the default
profile). See ``docs/guides/web-backend.md``.

Usable as a library (``CDPClient``) and as a small CLI for development:

    python -m alancode.providers.web.cdp tabs
    python -m alancode.providers.web.cdp verify https://chatgpt.com
    python -m alancode.providers.web.cdp eval https://chatgpt.com "document.title"
"""

from __future__ import annotations

import asyncio
import json
import urllib.request
from typing import Any

import websockets

DEFAULT_PORT = 9222


class CDPError(RuntimeError):
    """A CDP call failed or the target tab was not found."""


class CDPClient:
    """Minimal async CDP client bound to one page target."""

    def __init__(self, port: int = DEFAULT_PORT):
        self.port = port
        self._ws: Any = None
        self._id = 0

    def list_targets(self) -> list[dict]:
        url = f"http://localhost:{self.port}/json"
        with urllib.request.urlopen(url, timeout=5) as r:
            return json.load(r)

    def find_page(self, url_prefix: str) -> dict | None:
        for t in self.list_targets():
            if t.get("type") == "page" and t.get("url", "").startswith(url_prefix):
                return t
        return None

    async def connect(self, url_prefix: str) -> dict:
        target = self.find_page(url_prefix)
        if not target:
            raise CDPError(f"No open page tab matching {url_prefix!r} on port {self.port}")
        self._ws = await websockets.connect(target["webSocketDebuggerUrl"], max_size=None)
        return target

    async def close(self) -> None:
        if self._ws is not None:
            await self._ws.close()
            self._ws = None

    async def cmd(self, method: str, params: dict | None = None) -> dict:
        self._id += 1
        mid = self._id
        await self._ws.send(json.dumps({"id": mid, "method": method, "params": params or {}}))
        while True:
            msg = json.loads(await self._ws.recv())
            if msg.get("id") == mid:
                if "error" in msg:
                    raise CDPError(msg["error"])
                return msg.get("result", {})
            # otherwise it is an event - ignore

    async def evaluate(self, expression: str, await_promise: bool = False) -> Any:
        r = await self.cmd("Runtime.evaluate", {
            "expression": expression,
            "returnByValue": True,
            "awaitPromise": await_promise,
        })
        if r.get("exceptionDetails"):
            raise CDPError(r["exceptionDetails"].get("text", "JS exception"))
        return r.get("result", {}).get("value")

    # ── High-level page operations ───────────────────────────────────────

    async def focus(self, css: str) -> bool:
        return bool(await self.evaluate(
            f"(()=>{{const e=document.querySelector({json.dumps(css)});"
            f"if(!e)return false;e.focus();e.click&&e.click();return true}})()"
        ))

    async def click(self, css: str) -> bool:
        return bool(await self.evaluate(
            f"(()=>{{const e=document.querySelector({json.dumps(css)});"
            f"if(!e)return false;e.click();return true}})()"
        ))

    async def insert_text(self, text: str) -> None:
        """Insert *text* into the focused element as a trusted input event."""
        await self.cmd("Input.insertText", {"text": text})

    async def type_text(self, text: str) -> None:
        """Type *text* as individual key char events.

        Slower than insert_text, but rich editors (ProseMirror/Lexical) that
        ignore a bulk insert often register per-character key events, so their
        model - and the send button that depends on it - updates correctly.
        """
        for ch in text:
            await self.cmd("Input.dispatchKeyEvent", {"type": "char", "text": ch})

    async def press_enter(self) -> None:
        for t in ("keyDown", "keyUp"):
            await self.cmd("Input.dispatchKeyEvent", {
                "type": t, "key": "Enter", "code": "Enter",
                "windowsVirtualKeyCode": 13, "nativeVirtualKeyCode": 13,
            })

    async def last_text(self, css: str) -> str | None:
        return await self.evaluate(
            f"(()=>{{const e=document.querySelectorAll({json.dumps(css)});"
            f"return e.length?e[e.length-1].innerText:null}})()"
        )

    async def count(self, css: str) -> int:
        return int(await self.evaluate(f"document.querySelectorAll({json.dumps(css)}).length"))


# ── Development CLI ──────────────────────────────────────────────────────────

async def _cli(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 1
    action = argv[0]
    client = CDPClient()
    if action == "tabs":
        for t in client.list_targets():
            print(t.get("type"), "|", (t.get("title") or "")[:40], "|", t.get("url", "")[:60])
        return 0

    url = argv[1]
    await client.connect(url)
    try:
        if action == "verify":
            info = await client.evaluate(
                "JSON.stringify({"
                "hasInput: !!document.querySelector('textarea, [contenteditable=\"true\"]'),"
                "login: /\\b(log ?in|sign ?in|sign ?up|se connecter|connexion)\\b/i"
                ".test(document.body.innerText.slice(0,400)),"
                "title: document.title,"
                "snippet: document.body.innerText.trim().slice(0,140)})"
            )
            print(info)
        elif action == "eval":
            print(await client.evaluate(argv[2]))
        elif action == "typetest":
            composer = argv[2]
            word = argv[3] if len(argv) > 3 else "TYPEDOK"
            await client.focus(composer)
            await client.evaluate(
                "document.execCommand('selectAll');document.execCommand('delete')"
            )
            await client.focus(composer)
            await client.type_text(word)
            await asyncio.sleep(0.5)
            has_btn = await client.evaluate(
                '!!document.querySelector(\'button[aria-label*="Send" i]\')'
            )
            comp = await client.evaluate(
                f"(document.querySelector({composer!r})||{{}}).innerText"
            )
            print({"send_btn": has_btn, "composer": (comp or "")[:40]})
        elif action == "send":
            text = argv[2]
            composer = argv[3] if len(argv) > 3 else 'textarea, [contenteditable="true"]'
            if not await client.focus(composer):
                print(f"composer not found: {composer!r}")
                return 1
            await client.insert_text(text)
            await asyncio.sleep(0.3)
            await client.press_enter()
            print("sent")
        elif action == "sendfile":
            with open(argv[2], encoding="utf-8") as f:
                text = f.read()
            composer = argv[3] if len(argv) > 3 else 'textarea, [contenteditable="true"]'
            if not await client.focus(composer):
                print(f"composer not found: {composer!r}")
                return 1
            await client.insert_text(text)
            await asyncio.sleep(0.3)
            await client.press_enter()
            print("sent")
        elif action == "read":
            css = argv[2] if len(argv) > 2 else None
            if not css:
                print("read needs a CSS selector for the message element")
                return 1
            print(json.dumps({"count": await client.count(css),
                              "last": await client.last_text(css)}))
        else:
            print(f"unknown action {action!r}")
            return 1
    finally:
        await client.close()
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(asyncio.run(_cli(sys.argv[1:])))
