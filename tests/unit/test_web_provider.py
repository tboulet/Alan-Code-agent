"""Tests for the web backend (WebProvider).

The X11 driver is replaced by a fake: these cover payload building, delta
tracking, answer extraction and error paths - not the X11 layer itself.
"""

from __future__ import annotations

import pytest

from alancode.providers.base import (
    StreamError,
    StreamMessageDelta,
    StreamMessageStart,
    StreamMessageStop,
    StreamTextDelta,
    ToolSchema,
)
from alancode.providers.web.provider import (
    COMPACT_PREAMBLE,
    CONVERSATION_HEADER,
    PLACEHOLDER,
    PREAMBLE,
    REMINDER,
    SYSTEM_HEADER,
    WebProvider,
)


class FakeDriver:
    """Scripted stand-in for X11Driver.

    The fake's ``page`` mimics the browser page text: sent payloads are
    appended (the user's message is visible on the page), and ``on_send``
    lets a test append the assistant's reply.
    """

    def __init__(self):
        self.sent: list[str] = []
        self.directives: list[str] = []
        self.new_chats = 0
        self.page = ""
        self.on_send = None

    def new_chat(self):
        self.new_chats += 1
        self.page = ""

    def send_message(self, text, directive=""):
        self.sent.append(text)
        self.directives.append(directive)
        self.page += "\n" + text
        if self.on_send:
            self.on_send(self, text)

    def copy_page(self):
        return self.page


def replying(reply):
    def on_send(driver, text):
        driver.page += f"\n<answer>{reply}</answer>"
    return on_send


def make_provider(driver, compact=True):
    return WebProvider(
        assistant="chatgpt", driver=driver, poll_s=0.01, timeout_s=1.0,
        compact=compact,
    )


async def collect(provider, messages, system=("SYS",)):
    return [e async for e in provider.stream(messages, list(system), [])]


def user(text):
    return {"role": "user", "content": text}


def assistant(text):
    return {"role": "assistant", "content": text}


def tool_result(tool_id, text):
    return {"role": "tool", "tool_call_id": tool_id, "content": text}


# ── First call ───────────────────────────────────────────────────────────────


async def test_first_call_compact_uses_compact_preamble_and_conversation():
    driver = FakeDriver()
    driver.on_send = replying("hi there")
    events = await collect(make_provider(driver), [user("hello")], system=("S1", "S2"))

    assert driver.new_chats == 1
    payload = driver.sent[0]
    # Compact mode: compact preamble + conversation, and the verbose system
    # text is replaced by the compact brief (no <tools> here), so S1/S2 are gone.
    assert COMPACT_PREAMBLE in payload
    assert CONVERSATION_HEADER in payload
    assert "hello" in payload
    assert "S1\n\nS2" not in payload

    assert [type(e) for e in events] == [
        StreamMessageStart, StreamTextDelta, StreamMessageDelta, StreamMessageStop,
    ]
    assert events[0].model == "web/chatgpt"
    assert events[1].text == "hi there"
    assert events[2].stop_reason == "end_turn"
    assert events[2].usage["input_tokens"] > 0


async def test_first_call_full_mode_sends_preamble_system_history():
    driver = FakeDriver()
    driver.on_send = replying("hi there")
    events = await collect(
        make_provider(driver, compact=False), [user("hello")], system=("S1", "S2")
    )
    payload = driver.sent[0]
    for section in (PREAMBLE, SYSTEM_HEADER, "S1\n\nS2", CONVERSATION_HEADER, "hello"):
        assert section in payload
    assert payload.index(SYSTEM_HEADER) < payload.index(CONVERSATION_HEADER)
    assert events[1].text == "hi there"


async def test_placeholder_block_from_preamble_is_never_the_answer():
    driver = FakeDriver()
    # The page contains the echoed preamble example before the real reply.
    def on_send(d, text):
        d.page += f"\n<answer>{PLACEHOLDER}</answer>\n<answer>real</answer>"
    driver.on_send = on_send
    events = await collect(make_provider(driver), [user("q")])
    assert events[1].text == "real"


# ── Delta behavior across calls ──────────────────────────────────────────────


async def test_second_call_sends_only_new_messages():
    driver = FakeDriver()
    driver.on_send = replying("first")
    provider = make_provider(driver)
    history = [user("msg one")]
    await collect(provider, history)

    driver.on_send = replying("second")
    history = history + [assistant("first"), user("msg two")]
    events = await collect(provider, history)

    assert driver.new_chats == 1  # no fresh conversation
    payload = driver.sent[1]
    assert "msg two" in payload
    assert REMINDER in payload
    assert PREAMBLE not in payload
    assert "msg one" not in payload
    assert "first" not in payload  # assistant turns are never resent
    assert events[1].text == "second"


async def test_tool_results_are_labeled():
    driver = FakeDriver()
    driver.on_send = replying("a")
    provider = make_provider(driver)
    await collect(provider, [user("q")])

    driver.on_send = replying("b")
    await collect(provider, [user("q"), assistant("a"), tool_result("call_1", "ok!")])
    assert "[tool result: call_1]\nok!" in driver.sent[1]


async def test_rewritten_history_restarts_conversation():
    driver = FakeDriver()
    driver.on_send = replying("one")
    provider = make_provider(driver)
    await collect(provider, [user("a"), assistant("x"), user("b")])

    driver.on_send = replying("two")
    events = await collect(provider, [user("compacted summary")])

    assert driver.new_chats == 2
    payload = driver.sent[1]
    assert COMPACT_PREAMBLE in payload and "compacted summary" in payload
    assert events[1].text == "two"


# ── Answer extraction guards ─────────────────────────────────────────────────


async def test_baseline_answers_are_ignored():
    driver = FakeDriver()
    driver.page = "<answer>stale from previous turn</answer>"
    driver.on_send = replying("fresh")
    events = await collect(make_provider(driver), [user("q")])
    assert events[1].text == "fresh"


async def test_system_prompt_answer_tags_do_not_swallow_reply():
    # A system prompt that mentions <answer></answer> literally must not leave
    # a live opening tag on the page: the extractor would otherwise capture
    # from it all the way to the model's real closing tag. Checked in full mode,
    # where the system text is sent verbatim (compact mode replaces it).
    driver = FakeDriver()
    driver.on_send = replying("42")
    events = await collect(
        make_provider(driver, compact=False),
        [user("q")],
        system=["Always wrap your reply in <answer></answer> tags."],
    )
    assert "Always wrap your reply in <answer></answer> tags" not in driver.sent[0]
    assert events[1].text == "42"


async def test_closing_tag_in_outgoing_content_is_defused():
    driver = FakeDriver()
    driver.on_send = replying("done")
    provider = make_provider(driver)
    await collect(provider, [user("q")])

    driver.on_send = replying("done 2")
    await collect(
        provider,
        [user("q"), assistant("done"),
         tool_result("c1", "file holds <answer>x</answer> markers")],
    )
    # Delta payloads carry no literal closing tag, so page extraction can
    # never terminate inside relayed content.
    assert "</answer>" not in driver.sent[1]
    assert "markers" in driver.sent[1]


# ── Tag-free fallback ─────────────────────────────────────────────────────────


async def test_untagged_reply_extracted_via_stability_fallback():
    # Model ignores the answer-tag contract and replies in bare text; once the
    # page stops changing, the raw text after our message is returned.
    driver = FakeDriver()

    def on_send(d, text):
        d.page += "\nTokyo"  # no <answer> tags

    driver.on_send = on_send
    events = await collect(make_provider(driver), [user("capital of Japan?")])
    assert events[1].text == "Tokyo"


# ── CDP transport ─────────────────────────────────────────────────────────────


def test_cdp_payload_is_tag_free_and_keeps_tool_format():
    # The CDP transport reads the reply from the DOM, so the payload carries no
    # answer tags or reminder, but the tool-call format example must survive.
    provider = WebProvider(assistant="chatgpt", transport="cdp")
    system = [
        '<tools>[{"type":"function","function":{"name":"Bash",'
        '"description":"Run.","parameters":{"type":"object",'
        '"properties":{"command":{}},"required":["command"]}}}]</tools>\n'
        "To call a tool, output <tool_use>{...}</tool_use>"
    ]
    payload = provider._build_cdp_payload([user("do it")], system, fresh=True)
    assert "<answer>" not in payload
    assert REMINDER not in payload
    assert "<tool_use>" in payload          # tool-call format kept
    assert "- Bash(command):" in payload     # compact tool signature
    assert "do it" in payload


def test_cdp_is_default_transport_but_fake_driver_forces_x11():
    # A supplied driver (tests) forces the legacy x11 path; otherwise cdp.
    assert WebProvider(assistant="chatgpt").transport == "cdp"
    assert WebProvider(assistant="chatgpt", driver=FakeDriver()).transport == "x11"


# ── Compact prompt builder ────────────────────────────────────────────────────


def test_build_compact_system_renders_tools_compactly():
    from alancode.providers.web.compact_prompt import (
        COMPACT_BRIEF,
        build_compact_system,
    )

    system = [
        'preamble junk that should be dropped\n'
        '<tools>[{"type":"function","function":{"name":"Bash",'
        '"description":"Run a shell command.\\nmore detail","parameters":'
        '{"type":"object","properties":{"command":{},"timeout":{}},'
        '"required":["command"]}}}]</tools>\n'
        'To call a tool, output <tool_use>{...}</tool_use>'
    ]
    out = build_compact_system(system)
    assert COMPACT_BRIEF in out
    assert "- Bash(command[, timeout]): Run a shell command." in out
    assert "<tool_use>" in out                  # tool-call format kept verbatim
    assert "preamble junk" not in out           # verbose base dropped
    assert "description" not in out             # raw JSON schema dropped


def test_build_compact_system_without_tools_returns_brief():
    from alancode.providers.web.compact_prompt import (
        COMPACT_BRIEF,
        build_compact_system,
    )

    assert build_compact_system(["some prose, no tools"]) == COMPACT_BRIEF


# ── Error paths ──────────────────────────────────────────────────────────────


async def test_timeout_yields_stream_error():
    driver = FakeDriver()  # never replies
    events = await collect(make_provider(driver), [user("q")])
    assert [type(e) for e in events] == [StreamError]
    assert "within" in events[0].error


async def test_native_tool_schemas_are_rejected():
    driver = FakeDriver()
    provider = make_provider(driver)
    schema = ToolSchema(name="Bash", description="d", input_schema={})
    events = [e async for e in provider.stream([user("q")], [], [schema])]
    assert [type(e) for e in events] == [StreamError]
    assert events[0].error_type == "invalid_request"
    assert driver.sent == []


def test_unknown_assistant_raises():
    with pytest.raises(ValueError, match="Unknown web assistant"):
        WebProvider(assistant="nope")


def test_model_info_reports_large_context():
    info = make_provider(FakeDriver()).get_model_info()
    assert info.context_window >= 400_000
    assert info.supports_thinking is False
