"""ScriptedUI — test SessionUI implementation with scripted inputs.

``ScriptedUI`` mirrors the ``ScriptedProvider`` pattern: it supports
sequential and reactive modes for scripting user interactions in tests.

**Sequential** — inputs consumed in FIFO order::

    ui = ScriptedUI.from_inputs(["Fix the bug", "yes", "/exit"])

**Reactive** — inputs chosen by rules that inspect the event log::

    ui = ScriptedUI(rules=[
        ui_rule(turn=0, respond="Fix the bug"),
        ui_rule(
            input_type="ask",
            condition=lambda ctx: "permission" in ctx.last_question.lower(),
            respond="yes",
        ),
        ui_rule(respond="/exit"),
    ])

Both modes can be combined.  All events and interactions are logged
for assertions in tests.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Any, Callable

from rich.console import Console

from alancode.gui.base import SessionUI
from alancode.messages.types import Message, StreamEvent, Usage


# ═══════════════════════════════════════════════════════════════════════════════
# Context — structured accessors for rule conditions
# ═══════════════════════════════════════════════════════════════════════════════


class UIContext:
    """Parsed view of the UI state for writing rule conditions.

    Passed to condition functions so they can inspect what has happened
    in the session without manually parsing logs.
    """

    def __init__(
        self,
        event_log: list[dict[str, Any]],
        input_log: list[dict[str, Any]],
        console_log: list[str],
        input_count: int,
        current_prompt: str,
        current_question: str,
        current_options: list[str],
    ) -> None:
        self.event_log = event_log
        self.input_log = input_log
        self.console_log = console_log
        self.input_count = input_count
        self.current_prompt = current_prompt
        self.current_question = current_question
        self.current_options = current_options

    @property
    def last_question(self) -> str:
        """The question from the current or most recent ask_user call."""
        return self.current_question

    @property
    def last_prompt(self) -> str:
        """The prompt from the current get_input call."""
        return self.current_prompt

    @property
    def event_count(self) -> int:
        return len(self.event_log)

    @property
    def last_event_type(self) -> str:
        """Type string of the most recent event (empty if none)."""
        if not self.event_log:
            return ""
        return self.event_log[-1].get("type", "")

    @property
    def last_console_output(self) -> str:
        """Most recent console.print() output (empty if none)."""
        return self.console_log[-1] if self.console_log else ""

    def console_output_contains(self, substring: str) -> bool:
        """Check if any console output contains a substring."""
        return any(substring in line for line in self.console_log)

    def event_type_count(self, event_type: str) -> int:
        """Count events of a specific type."""
        return sum(1 for e in self.event_log if e.get("type") == event_type)


# ═══════════════════════════════════════════════════════════════════════════════
# Rule system
# ═══════════════════════════════════════════════════════════════════════════════


UIConditionFn = Callable[[UIContext], bool]


@dataclass
class UIRule:
    """A conditional input rule.

    Rules are evaluated in order; the first match wins.  A rule matches when
    ALL its conditions are satisfied:

    - ``turn`` — matches a specific input count (0-indexed)
    - ``input_type`` — matches the type of input request (``"prompt"`` or ``"ask"``)
    - ``condition`` — a callable receiving ``UIContext``
    - If none are set, the rule always matches (default fallback).

    Special responses:
    - ``respond=EOFError`` — raises EOFError (simulates Ctrl+D / session end)
    """

    respond: str | type
    turn: int | None = None
    input_type: str | None = None  # "prompt" or "ask"
    condition: UIConditionFn | None = None
    _consumed: bool = field(default=False, repr=False)

    def matches(self, ctx: UIContext, request_type: str) -> bool:
        if self.turn is not None and self.turn != ctx.input_count:
            return False
        if self.input_type is not None and self.input_type != request_type:
            return False
        if self.condition is not None and not self.condition(ctx):
            return False
        return True


def ui_rule(
    respond: str | type,
    *,
    turn: int | None = None,
    input_type: str | None = None,
    condition: UIConditionFn | None = None,
) -> UIRule:
    """Create a UI input rule."""
    return UIRule(respond=respond, turn=turn, input_type=input_type, condition=condition)


# ═══════════════════════════════════════════════════════════════════════════════
# ScriptedUI
# ═══════════════════════════════════════════════════════════════════════════════


class ScriptedUI(SessionUI):
    """Test UI with scripted inputs — sequential or reactive.

    All events, inputs, and console output are logged for assertions.

    Parameters
    ----------
    rules : list[UIRule], optional
        Rules for determining input responses.
    """

    def __init__(self, rules: list[UIRule] | None = None) -> None:
        self._rules: list[UIRule] = list(rules or [])
        self._input_count: int = 0

        # ── Logs (for test assertions) ───────────────────────────────
        self.event_log: list[dict[str, Any]] = []
        self.input_log: list[dict[str, Any]] = []
        self.cost_log: list[dict[str, Any]] = []
        self.console_log: list[str] = []
        self.lifecycle_log: list[str] = []  # "agent_start", "agent_done", "reset_stream"
        self.tree_update_log: list[dict[str, Any]] = []  # AGT tree updates

        # ── Console ──────────────────────────────────────────────────
        self._buf = io.StringIO()
        self._console = _ScriptedConsole(self)

    # ── Factory methods ──────────────────────────────────────────────

    @classmethod
    def from_inputs(
        cls,
        inputs: list[str | type],
        *,
        fallback: str | type | None = None,
    ) -> ScriptedUI:
        """Create a UI with FIFO inputs (and optional fallback).

        Each input is consumed in order.  Use ``EOFError`` to signal
        session end.  If a fallback is provided, it's used when the
        queue is exhausted.

        Example::

            ui = ScriptedUI.from_inputs([
                "Fix the bug in main.py",
                "yes",
                EOFError,  # End session
            ])
        """
        rules = [UIRule(respond=r, turn=i) for i, r in enumerate(inputs)]
        if fallback is not None:
            rules.append(UIRule(respond=fallback))
        return cls(rules=rules)

    # ── Convenience ──────────────────────────────────────────────────

    def add_rule(self, r: UIRule) -> None:
        """Append a rule."""
        self._rules.append(r)

    # ── SessionUI: Input ─────────────────────────────────────────────

    async def get_input(self, prompt: str = "\n> ") -> str:
        """Return next scripted input, or raise EOFError."""
        response = self._resolve("prompt", prompt=prompt, question="", options=[])
        self.input_log.append({
            "type": "prompt",
            "prompt": prompt,
            "response": str(response) if response is not EOFError else "<<EOF>>",
            "turn": self._input_count - 1,
        })
        if response is EOFError:
            raise EOFError("ScriptedUI: end of inputs")
        return response

    async def ask_user(self, question: str, options: list[str]) -> str:
        """Return next scripted answer to a question."""
        response = self._resolve("ask", prompt="", question=question, options=options)
        self.input_log.append({
            "type": "ask",
            "question": question,
            "options": options,
            "response": str(response) if response is not EOFError else "<<EOF>>",
            "turn": self._input_count - 1,
        })
        if response is EOFError:
            raise EOFError("ScriptedUI: end of inputs (ask)")
        return response

    def _resolve(
        self,
        request_type: str,
        *,
        prompt: str,
        question: str,
        options: list[str],
    ) -> str | type:
        """Find matching rule and return the response."""
        ctx = UIContext(
            event_log=self.event_log,
            input_log=self.input_log,
            console_log=self.console_log,
            input_count=self._input_count,
            current_prompt=prompt,
            current_question=question,
            current_options=options,
        )

        self._input_count += 1

        for r in self._rules:
            if r.matches(ctx, request_type):
                return r.respond

        # No matching rule — end session
        return EOFError

    # ── SessionUI: Agent event output ────────────────────────────────

    async def on_agent_event(self, event: StreamEvent | Message) -> None:
        """Log the event for later assertions."""
        entry: dict[str, Any] = {"type": type(event).__name__}
        # Extract useful fields for assertions
        if hasattr(event, "text"):
            entry["text"] = event.text
        if hasattr(event, "content"):
            entry["content"] = event.content
        if hasattr(event, "stop_reason"):
            entry["stop_reason"] = event.stop_reason
        if hasattr(event, "model"):
            entry["model"] = event.model
        self.event_log.append(entry)

    async def on_cost(
        self,
        usage: Usage,
        cost_usd: float,
        cost_unknown: bool,
        conversation_tokens: int = 0,
        context_window: int = 0,
    ) -> None:
        """Log cost info."""
        self.cost_log.append({
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cost_usd": cost_usd,
            "cost_unknown": cost_unknown,
            "conversation_tokens": conversation_tokens,
            "context_window": context_window,
        })

    # ── SessionUI: Lifecycle ─────────────────────────────────────────

    def on_agent_start(self) -> None:
        self.lifecycle_log.append("agent_start")

    def on_agent_done(self) -> None:
        self.lifecycle_log.append("agent_done")

    def reset_stream_state(self, assume_thinking: bool = False) -> None:
        self.lifecycle_log.append(f"reset_stream(thinking={assume_thinking})")

    # ── SessionUI: Git Tree ──────────────────────────────────────────

    def on_git_tree_update(self, tree_data: dict) -> None:
        """Log tree updates for test assertions."""
        self.tree_update_log.append(tree_data)

    # ── SessionUI: Console ───────────────────────────────────────────

    @property
    def console(self) -> Console:
        return self._console

    # ── Assertion helpers ────────────────────────────────────────────

    @property
    def prompt_responses(self) -> list[str]:
        """All responses given to get_input() calls."""
        return [
            e["response"] for e in self.input_log if e["type"] == "prompt"
        ]

    @property
    def ask_responses(self) -> list[str]:
        """All responses given to ask_user() calls."""
        return [
            e["response"] for e in self.input_log if e["type"] == "ask"
        ]

    @property
    def events_by_type(self) -> dict[str, list[dict[str, Any]]]:
        """Events grouped by type name."""
        result: dict[str, list[dict[str, Any]]] = {}
        for e in self.event_log:
            result.setdefault(e["type"], []).append(e)
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# _ScriptedConsole — captures console.print() output
# ═══════════════════════════════════════════════════════════════════════════════


class _ScriptedConsole(Console):
    """Rich Console subclass that captures output for test assertions.

    Does NOT write to the terminal.
    """

    def __init__(self, scripted_ui: ScriptedUI) -> None:
        self._inner_buf = io.StringIO()
        super().__init__(file=self._inner_buf, width=120, no_color=True)
        self._scripted_ui = scripted_ui

    def print(self, *objects: Any, **kwargs: Any) -> None:  # type: ignore[override]
        self._inner_buf.truncate(0)
        self._inner_buf.seek(0)
        super().print(*objects, **kwargs)
        text = self._inner_buf.getvalue().rstrip()
        self._inner_buf.truncate(0)
        self._inner_buf.seek(0)
        if text:
            self._scripted_ui.console_log.append(text)
