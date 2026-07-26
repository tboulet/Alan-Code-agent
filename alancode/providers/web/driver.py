"""X11 driver - window, keyboard and clipboard automation via xdotool/xclip.

Synchronous by design; ``WebProvider`` runs every call in a worker thread.
Works against any X11 display, including a virtual one (Xvfb) via $DISPLAY.

While a request is in flight the driver owns the display's focus and
clipboard: user keyboard/mouse activity can corrupt a send or a page copy.
"""

from __future__ import annotations

import logging
import subprocess
import time

from alancode.providers.web.assistant import WebAssistant

logger = logging.getLogger(__name__)

ACTIVATE_ATTEMPTS = 25
ACTIVATE_POLL_S = 0.2
LAUNCH_TIMEOUT_S = 20.0
KEY_SETTLE_S = 0.2
PASTE_SETTLE_S = 0.5
DISMISS_ATTEMPTS = 5
SUBMIT_ATTEMPTS = 4
SUBMIT_VERIFY_S = 1.5
PASTE_ATTEMPTS = 4
CLIPBOARD_SET_ATTEMPTS = 20
# Distinctive tail of the outgoing text, used to confirm the paste landed.
PASTE_CHECK_LEN = 30
# A single paste larger than a few hundred chars is turned by ChatGPT into a
# "Pasted text" file attachment, which the model treats as an opaque document
# to analyze rather than as the message. Pasting in sub-threshold chunks that
# append into the composer keeps a small message inline.
PASTE_CHUNK_CHARS = 400
CHUNK_PASTE_S = 0.25
# ChatGPT refuses to send an inline message beyond ~4 KB (the send control
# stays disabled). Messages up to this go inline (the compact system prompt is
# built to fit); larger ones fall back to an attachment plus a short inline
# directive - but assistants tend to distrust attached instructions, so the
# compact prompt (see providers/web/compact_prompt.py) is what makes this work.
INLINE_MAX_CHARS = 3500
ATTACHMENT_PASTE_S = 2.0
# Submission is detected by watching the tail of the page: an unsent draft
# leaves the pasted text as the last thing on the page, while submitting
# clears the composer (placeholder/reply take its place). Comparing only the
# tail avoids false positives from incidental churn elsewhere in a huge page.
SUBMIT_TAIL_CHARS = 120

# A GNOME file-chooser portal ("attach file") can appear over the assistant
# window - e.g. if focus lands on the composer's attach control and it gets
# activated. Being modal, it blocks activation of the assistant window, so
# the driver Escapes it away before every activation.
PORTAL_CLASSNAME = "xdg-desktop-portal"


class WebDriverError(RuntimeError):
    """The browser window could not be driven (missing, unfocusable, ...)."""


class X11Driver:
    """Drives one assistant's browser app window."""

    def __init__(self, assistant: WebAssistant, browser_command: str = "google-chrome"):
        self.assistant = assistant
        self.browser_command = browser_command
        self._window: str | None = None

    @staticmethod
    def _out(*args: str) -> str:
        return subprocess.check_output(args, text=True).strip()

    @staticmethod
    def _out_or_empty(*args: str) -> str:
        try:
            return subprocess.check_output(
                args, text=True, stderr=subprocess.DEVNULL
            ).strip()
        except subprocess.CalledProcessError:
            return ""

    @staticmethod
    def _run(*args: str) -> None:
        subprocess.check_call(args)

    def _key(self, spec: str) -> None:
        self._run("xdotool", "key", "--clearmodifiers", spec)

    @staticmethod
    def get_clipboard() -> str:
        return subprocess.run(
            ["xclip", "-selection", "clipboard", "-o"],
            capture_output=True, text=True,
        ).stdout

    def set_clipboard(self, text: str) -> None:
        """Own the clipboard with *text*, verifying the write took.

        xclip hands off asynchronously, and a preceding in-page Ctrl+C leaves
        the browser owning the selection; pasting before xclip has taken over
        would paste the stale page copy. Reading it back confirms ownership.
        """
        for _ in range(CLIPBOARD_SET_ATTEMPTS):
            subprocess.run(
                ["xclip", "-selection", "clipboard"],
                input=text, text=True, check=True,
            )
            time.sleep(KEY_SETTLE_S)
            if self.get_clipboard() == text:
                return
        raise WebDriverError("Could not set the clipboard (xclip ownership)")

    # ── Window management ────────────────────────────────────────────────

    def _find_window(self) -> str | None:
        try:
            ids = self._out(
                "xdotool", "search", "--onlyvisible",
                "--classname", self.assistant.window_classname,
            ).split()
        except subprocess.CalledProcessError:  # no match
            return None
        return ids[0] if ids else None

    def _window_alive(self) -> bool:
        if not self._window:
            return False
        return subprocess.call(
            ["xdotool", "getwindowname", self._window],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        ) == 0

    def ensure_window(self) -> None:
        """Find the assistant's app window, launching it if needed."""
        if self._window_alive():
            return
        self._window = self._find_window()
        if self._window:
            return
        logger.info("web driver: launching %s app window", self.assistant.name)
        subprocess.Popen(
            [self.browser_command, f"--app={self.assistant.url}"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        deadline = time.monotonic() + LAUNCH_TIMEOUT_S
        while time.monotonic() < deadline:
            time.sleep(ACTIVATE_POLL_S)
            self._window = self._find_window()
            if self._window:
                time.sleep(self.assistant.page_load_wait_s)
                return
        raise WebDriverError(
            f"No window with classname '{self.assistant.window_classname}' "
            f"appeared within {LAUNCH_TIMEOUT_S:.0f}s "
            f"(command: {self.browser_command} --app={self.assistant.url})"
        )

    def _dismiss_stray_dialogs(self) -> None:
        """Escape any modal file-chooser portal dialog blocking the window.

        The Escape must be a real key event to the focused dialog (not an
        XSendEvent via --window, which GTK ignores), so we activate the
        dialog first, then send a global key.
        """
        for _ in range(DISMISS_ATTEMPTS):
            ids = self._out_or_empty(
                "xdotool", "search", "--onlyvisible",
                "--classname", PORTAL_CLASSNAME,
            ).split()
            if not ids:
                return
            subprocess.call(
                ["xdotool", "windowactivate", ids[0]], stderr=subprocess.DEVNULL
            )
            time.sleep(KEY_SETTLE_S)
            self._key("Escape")
            time.sleep(KEY_SETTLE_S)

    def _activate(self) -> None:
        # Freshly created windows are not immediately activatable (the WM
        # has not adopted them yet), so verify instead of fire-and-forget.
        for _ in range(ACTIVATE_ATTEMPTS):
            self._dismiss_stray_dialogs()
            subprocess.call(
                ["xdotool", "windowactivate", self._window],
                stderr=subprocess.DEVNULL,
            )
            time.sleep(ACTIVATE_POLL_S)
            try:
                if self._out("xdotool", "getactivewindow") == self._window:
                    return
            except subprocess.CalledProcessError:
                pass
        raise WebDriverError(f"Could not focus window {self._window}")

    # ── High-level operations used by WebProvider ────────────────────────

    def new_chat(self) -> None:
        self.ensure_window()
        self._activate()
        self._key(self.assistant.new_chat_keys)
        time.sleep(1.0)

    @staticmethod
    def _paste_marker(text: str) -> str:
        for line in reversed(text.splitlines()):
            if line.strip():
                return line.strip()[-PASTE_CHECK_LEN:]
        return text.strip()[-PASTE_CHECK_LEN:]

    @staticmethod
    def _chunks(text: str) -> list[str]:
        return [
            text[i:i + PASTE_CHUNK_CHARS]
            for i in range(0, len(text), PASTE_CHUNK_CHARS)
        ] or [""]

    def _focus_composer(self) -> None:
        self._activate()
        self._key(self.assistant.focus_composer_keys)
        time.sleep(KEY_SETTLE_S)

    def _paste_inline(self, text: str) -> bool:
        """Paste *text* inline in sub-threshold chunks; return True if it landed.

        Ctrl+A before the first chunk selects any pre-existing draft, which
        that chunk's paste overwrites (drafts persist in localStorage and
        would otherwise corrupt the message); later chunks append.
        """
        marker = self._paste_marker(text)
        for _ in range(PASTE_ATTEMPTS):
            self._focus_composer()
            self._key("ctrl+a")
            time.sleep(KEY_SETTLE_S)
            for chunk in self._chunks(text):
                self.set_clipboard(chunk)
                self._key("ctrl+v")
                time.sleep(CHUNK_PASTE_S)
            time.sleep(PASTE_SETTLE_S)
            if marker in self.copy_page():
                return True
        return False

    def _paste_as_attachment(self, bulk: str, directive: str) -> bool:
        """Send *bulk* as a "Pasted text" attachment plus an inline *directive*.

        One big paste exceeds ChatGPT's inline threshold and is collapsed into
        a file attachment (so it also dodges the inline send-length limit); the
        directive, pasted inline after it, tells the model to follow the
        attached content. Returns True once the directive is inline and the
        bulk is not (i.e. it really became an attachment).
        """
        bulk_marker = self._paste_marker(bulk)
        dir_marker = self._paste_marker(directive)
        for _ in range(PASTE_ATTEMPTS):
            self._focus_composer()
            self._key("ctrl+a")
            time.sleep(KEY_SETTLE_S)
            self.set_clipboard(bulk)
            self._key("ctrl+v")
            time.sleep(ATTACHMENT_PASTE_S)
            self.set_clipboard(directive)
            self._key("ctrl+v")
            time.sleep(PASTE_SETTLE_S)
            page = self.copy_page()
            if dir_marker in page and bulk_marker not in page:
                return True
        return False

    def _submit(self) -> None:
        """Press Return until the composer clears (the message is sent).

        The first Return after a paste often does not submit (the send control
        is briefly not ready). Submission is detected by the page tail changing
        from the unsent draft - robust to churn elsewhere in a large page.
        """
        before_tail = self.copy_page().rstrip()[-SUBMIT_TAIL_CHARS:]
        for _ in range(SUBMIT_ATTEMPTS):
            self._focus_composer()
            self._key("Return")
            time.sleep(SUBMIT_VERIFY_S)
            if self.copy_page().rstrip()[-SUBMIT_TAIL_CHARS:] != before_tail:
                return
        raise WebDriverError("Message did not submit (composer never cleared)")

    def send_message(self, text: str, directive: str = "") -> None:
        """Paste *text* into the composer and submit it.

        Small messages go inline; ones beyond INLINE_MAX_CHARS are sent as an
        attachment plus *directive* (a short inline instruction to follow the
        attached content), since ChatGPT will not send a large inline message.
        """
        self.ensure_window()
        if len(text) <= INLINE_MAX_CHARS:
            landed = self._paste_inline(text)
        else:
            landed = self._paste_as_attachment(text, directive)
        if not landed:
            raise WebDriverError("Paste did not land in the composer (focus lost)")
        self._submit()

    def copy_page(self) -> str:
        """Return the page's full visible text via select-all copy."""
        self.ensure_window()
        self._activate()
        # Move focus out of the composer first: Ctrl+A inside it selects
        # its own (empty) content instead of the page.
        self._key("shift+Tab")
        self._key("ctrl+a")
        self._key("ctrl+c")
        time.sleep(KEY_SETTLE_S)
        return self.get_clipboard()
