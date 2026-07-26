"""WebProvider - an ``LLMProvider`` that drives an online assistant's web UI.

How a request maps onto the web conversation:

- The web conversation itself holds the history. Each ``stream()`` call
  relays only the messages not sent yet; assistant turns are skipped since
  the site already displays them. When the incoming history is shorter than
  what was already relayed (rewritten history, e.g. after compaction), a
  fresh web conversation is opened and everything is resent.
- The first payload of a conversation is prefixed with an operating-context
  preamble plus the system prompt: the preamble tells the assistant it is a
  coding agent's engine, that its own web-side tools are unavailable, and
  that every reply must be wrapped in ``<answer>`` tags.
- The reply is detected by polling the page text: a complete tag block that
  was not on the page before the send (and is not the preamble's example)
  is the answer. The closing tag doubles as the finished-generating signal.

Tool calling must use a text format (``tool_call_format`` setting): this
provider only ever deals in plain text, and never receives tool schemas.
"""

from __future__ import annotations

import asyncio
import logging
import re
import subprocess
import time
from typing import Any, AsyncGenerator

from alancode.providers.base import (
    LLMProvider,
    ModelInfo,
    ProviderStreamEvent,
    StreamError,
    StreamMessageDelta,
    StreamMessageStart,
    StreamMessageStop,
    StreamTextDelta,
    ThinkingConfig,
    ToolSchema,
)
from alancode.providers.web.assistant import ASSISTANTS
from alancode.providers.web.cdp import CDPError
from alancode.providers.web.compact_prompt import build_compact_system
from alancode.providers.web.driver import WebDriverError, X11Driver

logger = logging.getLogger(__name__)

ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
PLACEHOLDER = "YOUR ENTIRE REPLY HERE"
POLL_S = 2.0
DEFAULT_TIMEOUT_S = 600.0
# Consecutive unchanged polls before an untagged reply is considered complete.
STABILITY_CHECKS = 3

PREAMBLE = f"""\
OPERATING CONTEXT - read carefully, it applies to this entire conversation:
You are the reasoning engine of a coding agent framework that communicates
with you through this chat interface. You are NOT talking to a human. The
capabilities of this web interface (web browsing, image generation, code
interpreter, file uploads, canvas, ...) are NOT available to the framework.
The ONLY tools you can use are the ones described in the system prompt
below, and you invoke them by writing the documented text format in your
reply; the framework executes them and sends you the results as follow-up
messages.
Formatting contract: wrap EVERY reply entirely in answer tags, exactly like
<answer>{PLACEHOLDER}</answer> - text outside the tags is discarded, and a
reply without the tags is never seen by the framework. Never use these tags
anywhere else, and never wrap them in a code block."""

SYSTEM_HEADER = "=== SYSTEM PROMPT ==="
CONVERSATION_HEADER = "=== CONVERSATION ==="
REMINDER = "(Reply entirely wrapped in the answer tags.)"

# Short, honest preamble used with the compact prompt. It describes the relay
# truthfully and asks the assistant to opt in, rather than trying to override
# it - the whole payload stays small enough to send inline as an ordinary
# message (no oversized paste, no file attachment).
COMPACT_PREAMBLE = f"""\
You're connected to a relay that lets a command-line coding assistant use you \
as its reasoning engine. Your replies are read by a program, not a person, and \
any tool call you emit is really executed on the user's machine, with the real \
result sent back to you as the next message. If you're happy to help with this \
ordinary coding workflow, reply to each message with your ENTIRE response \
wrapped in <answer></answer> tags (for example <answer>{PLACEHOLDER}</answer>); \
text outside the tags is dropped. Your role, tools and the request follow."""

# Nothing WE send may contain a literal answer tag - only the model's genuine
# reply should. Otherwise the page-side extractor mis-parses: a stray opening
# tag (e.g. a system prompt saying "wrap in <answer></answer>") swallows text
# up to the model's real closing tag. A zero-width space after '<' breaks both
# tags for the regex while staying invisible to the assistant. The preamble's
# own example is exempt: it is a balanced block the extractor filters by its
# placeholder content, and the model needs to see real tags there.
_TAGS = ("<answer>", "</answer>")
_DEFUSED = ("<\u200banswer>", "<\u200b/answer>")


def _defuse_tags(text: str) -> str:
    for tag, safe in zip(_TAGS, _DEFUSED):
        text = text.replace(tag, safe)
    return text


# Inline text sent alongside the attachment when a payload is too large to go
# inline (see driver INLINE_MAX_CHARS). It must restate the reply format: the
# model reads this directly, whereas the format rule is buried in the attached
# file. Its own tags are defused so they do not pollute answer extraction.
ATTACHMENT_DIRECTIVE = _defuse_tags(
    "The attached file contains the full context (system prompt, tools, and "
    "latest message) for this turn. Please read it and respond to the latest "
    "message, wrapping your entire reply in <answer></answer> tags."
)


class WebProvider(LLMProvider):
    """Drives a web assistant (ChatGPT, ...) as an Alan Code backend."""

    def __init__(
        self,
        assistant: str = "chatgpt",
        *,
        driver: Any = None,
        transport: str = "cdp",
        timeout_s: float = DEFAULT_TIMEOUT_S,
        poll_s: float = POLL_S,
        compact: bool = True,
        **_: Any,
    ):
        if driver is None and assistant not in ASSISTANTS:
            raise ValueError(
                f"Unknown web assistant '{assistant}'. "
                f"Available: {', '.join(sorted(ASSISTANTS))}"
            )
        self.assistant_name = assistant
        self.timeout_s = timeout_s
        self.poll_s = poll_s
        # Compact prompt: replace Alan's multi-KB system prompt with a tiny
        # equivalent so the first message fits inline (a web chat will not send
        # a large inline message, and an attachment is treated as untrusted).
        self.compact = compact
        # Transport: "cdp" reads/writes the DOM via Chrome DevTools (robust,
        # tag-free); "x11" automates the desktop via xdotool (legacy, fragile).
        self.transport = "x11" if driver is not None else transport
        if driver is not None:
            self._driver = driver
        elif self.transport == "cdp":
            from alancode.providers.web.cdp_driver import CDPDriver
            self._driver = CDPDriver(ASSISTANTS[assistant])
        else:
            self._driver = X11Driver(ASSISTANTS[assistant])
        # Number of history messages already relayed to the web chat;
        # None until a web conversation exists.
        self._sent_count: int | None = None

    # ── Message rendering ────────────────────────────────────────────────

    @staticmethod
    def _block_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif isinstance(block, dict) and block.get("type") == "image_url":
                    parts.append("[image omitted: unsupported by the web backend]")
                else:
                    parts.append(str(block))
            return "\n".join(p for p in parts if p)
        return "" if content is None else str(content)

    @classmethod
    def _render_message(cls, msg: dict[str, Any]) -> str | None:
        role = msg.get("role")
        if role == "assistant":
            return None
        text = _defuse_tags(cls._block_text(msg.get("content")))
        if role == "tool":
            tool_id = msg.get("tool_call_id")
            label = f"[tool result: {tool_id}]" if tool_id else "[tool result]"
            return f"{label}\n{text}"
        if role == "system":
            return f"[system]\n{text}"
        return text

    def _build_payload(
        self, messages: list[dict[str, Any]], system: list[str], fresh: bool
    ) -> str:
        start = 0 if fresh else (self._sent_count or 0)
        rendered = [
            r for r in (self._render_message(m) for m in messages[start:]) if r
        ]
        if fresh and self.compact:
            system_text = _defuse_tags(build_compact_system(system))
            parts = [COMPACT_PREAMBLE, system_text, CONVERSATION_HEADER, *rendered]
        elif fresh:
            system_text = _defuse_tags("\n\n".join(system))
            parts = [PREAMBLE, SYSTEM_HEADER, system_text, CONVERSATION_HEADER, *rendered]
        else:
            parts = list(rendered)
        # The reminder is the last thing the model reads, right after the
        # question - decisive for format compliance, including on turn one
        # (the preamble's contract alone is too far up to reliably stick).
        parts.append(REMINDER)
        return "\n\n".join(p for p in parts if p and p.strip())

    def _build_cdp_payload(
        self, messages: list[dict[str, Any]], system: list[str], fresh: bool
    ) -> str:
        """Payload for the CDP transport - no answer tags or reminder.

        The reply is read from the assistant's message DOM node, so none of the
        tag/marker machinery the xdotool path needs applies. Tags in relayed
        content are left intact (the tool-call format example in the system must
        survive verbatim, and the DOM read never mis-parses it).
        """
        start = 0 if fresh else (self._sent_count or 0)
        rendered = []
        for msg in messages[start:]:
            if msg.get("role") == "assistant":
                continue
            text = self._block_text(msg.get("content"))
            role = msg.get("role")
            if role == "tool":
                tid = msg.get("tool_call_id")
                text = f"[tool result{': ' + tid if tid else ''}]\n{text}"
            elif role == "system":
                text = f"[system]\n{text}"
            rendered.append(text)
        if fresh:
            head = build_compact_system(system) if self.compact else "\n\n".join(system)
            parts = [head, *rendered]
        else:
            parts = rendered
        return "\n\n".join(p for p in parts if p and p.strip())

    # ── Answer extraction ────────────────────────────────────────────────

    @staticmethod
    def _answers_in(page: str) -> list[str]:
        return [m.strip() for m in ANSWER_RE.findall(page)]

    def _raw_reply(self, page: str) -> str | None:
        """Best-effort reply text when the model omits answer tags.

        Our message ends with REMINDER, so the reply is what follows the last
        occurrence of it on the page, truncated at the first known chrome marker
        (footer, composer placeholder, side panels) so interface text is never
        returned. Best-effort - only used when the tag path finds nothing.
        """
        idx = page.rfind(REMINDER)
        if idx < 0:
            return None
        raw = page[idx + len(REMINDER):]
        markers = ASSISTANTS[self.assistant_name].reply_stop_markers \
            if self.assistant_name in ASSISTANTS else ()
        cuts = [raw.find(m) for m in markers if m in raw]
        if cuts:
            raw = raw[: min(cuts)]
        raw = raw.strip()
        return raw or None

    async def _wait_answer(self, baseline: set[str]) -> str:
        """Return the assistant's reply once it is complete.

        Primary path: a new, complete ``<answer>...</answer>`` block - its
        closing tag is itself the done signal. Fallback (model ignored the tag
        contract): once the raw text after our message stops changing for a few
        polls, treat generation as finished and return that raw text.
        """
        deadline = time.monotonic() + self.timeout_s
        last_raw: str | None = None
        stable = 0
        while time.monotonic() < deadline:
            await asyncio.sleep(self.poll_s)
            page = await asyncio.to_thread(self._driver.copy_page)
            for block in reversed(self._answers_in(page)):
                if block and block != PLACEHOLDER and block not in baseline:
                    return block
            raw = self._raw_reply(page)
            if raw and raw == last_raw:
                stable += 1
                if stable >= STABILITY_CHECKS:
                    logger.warning(
                        "web: '%s' reply had no answer tags; using raw page "
                        "text (may include UI chrome)", self.assistant_name,
                    )
                    return raw
            else:
                stable = 0
                last_raw = raw
        raise TimeoutError(
            f"No answer from '{self.assistant_name}' within "
            f"{self.timeout_s:.0f}s"
        )

    # ── LLMProvider interface ────────────────────────────────────────────

    async def stream(
        self,
        messages: list[dict[str, Any]],
        system: list[str],
        tools: list[ToolSchema],
        *,
        model: str | None = None,
        max_tokens: int | None = None,
        thinking: ThinkingConfig | None = None,
        stop_sequences: list[str] | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[ProviderStreamEvent, None]:
        if tools:
            yield StreamError(
                error=(
                    "The web backend has no native tool calling. Set a text "
                    "format, e.g.: alancode --backend web --tool-call-format alan"
                ),
                error_type="invalid_request",
            )
            return

        fresh = self._sent_count is None or self._sent_count > len(messages)
        if self._sent_count is not None and self._sent_count > len(messages):
            logger.info(
                "web: history shorter than already relayed (%s < %s), "
                "starting a fresh web conversation",
                len(messages), self._sent_count,
            )
        try:
            if self.transport == "cdp":
                payload = self._build_cdp_payload(messages, system, fresh)
                await self._driver.ensure_ready()
                if fresh:
                    await self._driver.new_chat()
                answer = await self._driver.send_and_read(payload)
            else:
                payload = self._build_payload(messages, system, fresh)
                if fresh:
                    await asyncio.to_thread(self._driver.new_chat)
                baseline_page = await asyncio.to_thread(self._driver.copy_page)
                baseline = set(self._answers_in(baseline_page))
                await asyncio.to_thread(
                    self._driver.send_message, payload, ATTACHMENT_DIRECTIVE
                )
                answer = await self._wait_answer(baseline)
            self._sent_count = len(messages)
        except (WebDriverError, CDPError, subprocess.CalledProcessError, OSError) as exc:
            yield StreamError(error=f"web driver: {exc}")
            return
        except TimeoutError as exc:
            yield StreamError(error=str(exc))
            return

        yield StreamMessageStart(model=f"web/{self.assistant_name}")
        yield StreamTextDelta(text=answer)
        # No usage data exists on a web UI; estimate so budgeting stays sane.
        yield StreamMessageDelta(
            stop_reason="end_turn",
            usage={
                "input_tokens": max(1, len(payload) // 4),
                "output_tokens": max(1, len(answer) // 4),
            },
        )
        yield StreamMessageStop()

    def get_model_info(self, model: str | None = None) -> ModelInfo:
        return ModelInfo(
            context_window=400_000,
            max_output_tokens=100_000,
            supports_thinking=False,
            cw_source="override",
        )
