"""CDP transport driver for the web backend.

Drives an assistant tab through the Chrome DevTools Protocol instead of the
desktop (xdotool). It sends by inserting text into the composer element and
reads replies straight from the assistant message DOM node - no window focus,
no coordinates, no page-chrome pollution, and no need for the ``<answer>`` tag
trick the xdotool path relies on. Adding an assistant is just two selectors
(``cdp_composer_css`` / ``cdp_message_css`` on ``WebAssistant``).

Chrome must run with ``--remote-debugging-port`` on a non-default profile;
``ensure_ready`` launches it (see ``CDP_USER_DATA_DIR``) if the port is closed.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
import time
import urllib.request

from alancode.providers.web.assistant import WebAssistant
from alancode.providers.web.cdp import CDPClient, CDPError

logger = logging.getLogger(__name__)

CDP_PORT = 9222
CDP_USER_DATA_DIR = "/tmp/chrome-cdp"
LAUNCH_TIMEOUT_S = 25.0
PAGE_LOAD_WAIT_S = 3.0
REPLY_TIMEOUT_S = 300.0
POLL_S = 1.0
STABLE_POLLS = 3
SUBMIT_ATTEMPTS = 5
SUBMIT_VERIFY_S = 1.0


class CDPDriver:
    """One assistant tab, driven over CDP."""

    def __init__(
        self,
        assistant: WebAssistant,
        *,
        port: int = CDP_PORT,
        browser_command: str = "google-chrome",
        user_data_dir: str = CDP_USER_DATA_DIR,
    ):
        self.assistant = assistant
        self.port = port
        self.browser_command = browser_command
        self.user_data_dir = user_data_dir
        self.client = CDPClient(port)
        self._connected = False

    # ── Connection / lifecycle ───────────────────────────────────────────

    @property
    def _origin(self) -> str:
        # Match tabs by origin (scheme://host), ignoring path/conversation id.
        parts = self.assistant.url.split("/", 3)
        return "/".join(parts[:3])

    def _port_open(self) -> bool:
        try:
            urllib.request.urlopen(f"http://localhost:{self.port}/json/version", timeout=2)
            return True
        except OSError:
            return False

    def _launch(self, url: str) -> None:
        subprocess.Popen(
            [
                self.browser_command,
                f"--user-data-dir={self.user_data_dir}",
                f"--remote-debugging-port={self.port}",
                "--no-first-run", "--no-default-browser-check", url,
            ],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

    async def ensure_ready(self) -> None:
        """Ensure a debuggable Chrome with this assistant's tab, and connect."""
        if not self._port_open():
            logger.info("cdp: launching debuggable Chrome for %s", self.assistant.name)
            self._launch(self.assistant.url)
            deadline = time.monotonic() + LAUNCH_TIMEOUT_S
            while time.monotonic() < deadline and not self._port_open():
                await asyncio.sleep(0.5)
            if not self._port_open():
                raise CDPError(f"Chrome debug port {self.port} did not open")
            await asyncio.sleep(PAGE_LOAD_WAIT_S)

        if not self.client.find_page(self._origin):
            # Instance is up but this assistant has no tab: open one.
            self._launch(self.assistant.url)
            deadline = time.monotonic() + LAUNCH_TIMEOUT_S
            while time.monotonic() < deadline and not self.client.find_page(self._origin):
                await asyncio.sleep(0.5)
            await asyncio.sleep(PAGE_LOAD_WAIT_S)

        await self.client.connect(self._origin)
        self._connected = True

    async def close(self) -> None:
        await self.client.close()
        self._connected = False

    # ── Conversation operations ──────────────────────────────────────────

    async def new_chat(self) -> None:
        url = self.assistant.cdp_new_chat_url or self.assistant.url
        await self.client.evaluate(f"location.href = {url!r}")
        # Wait for the composer to be present on the fresh page.
        deadline = time.monotonic() + LAUNCH_TIMEOUT_S
        while time.monotonic() < deadline:
            await asyncio.sleep(0.5)
            try:
                if await self.client.evaluate(
                    f"!!document.querySelector({self.assistant.cdp_composer_css!r})"
                ):
                    return
            except CDPError:
                # Navigation can briefly drop the execution context.
                continue
        raise CDPError(f"composer never appeared after new chat for {self.assistant.name}")

    async def message_count(self) -> int:
        return await self.client.count(self.assistant.cdp_message_css)

    async def _composer_holds(self, marker: str) -> bool:
        css = self.assistant.cdp_composer_css
        return bool(await self.client.evaluate(
            f"(()=>{{const e=document.querySelector({css!r});"
            f"return e ? (e.value||e.innerText||'').includes({marker!r}) : false}})()"
        ))

    async def _submit_once(self) -> None:
        submit = self.assistant.cdp_submit
        if submit and submit != "enter":
            if not await self.client.click(submit):
                await self.client.press_enter()  # fall back if the button moved
        else:
            await self.client.press_enter()

    async def send(self, text: str) -> None:
        """Type *text* into the composer and submit, verifying it cleared.

        The first submit (Enter or button) often does not fire on a fresh, large
        draft, so it is retried until the composer no longer holds the text.
        """
        composer = self.assistant.cdp_composer_css
        marker = text.strip()[:30]
        if not await self.client.focus(composer):
            raise CDPError(f"composer not found: {composer!r}")
        # Clear any leftover draft first: a failed submit persists the text
        # (localStorage), and a fresh insert would append to that junk.
        await self.client.evaluate(
            "document.execCommand('selectAll');document.execCommand('delete')"
        )
        await self.client.focus(composer)
        await self.client.insert_text(text)
        await asyncio.sleep(0.4)
        for _ in range(SUBMIT_ATTEMPTS):
            await self.client.focus(composer)
            await self._submit_once()
            await asyncio.sleep(SUBMIT_VERIFY_S)
            if not await self._composer_holds(marker):
                return
        raise CDPError(f"message did not submit for {self.assistant.name}")

    async def read_reply(self, baseline_count: int) -> str:
        """Wait for a new assistant message to finish, return its text.

        Completion is detected by the last message's text going stable for a
        few polls (works while streaming grows the text). No tags needed - the
        message element IS the reply.
        """
        deadline = time.monotonic() + REPLY_TIMEOUT_S
        last = None
        stable = 0
        while time.monotonic() < deadline:
            await asyncio.sleep(POLL_S)
            if await self.message_count() <= baseline_count:
                continue
            text = await self.client.last_text(self.assistant.cdp_message_css)
            if text and text == last:
                stable += 1
                if stable >= STABLE_POLLS:
                    return text
            else:
                stable = 0
                last = text
        raise CDPError(f"no reply from {self.assistant.name} within {REPLY_TIMEOUT_S:.0f}s")

    async def send_and_read(self, text: str) -> str:
        baseline = await self.message_count()
        await self.send(text)
        return await self.read_reply(baseline)
