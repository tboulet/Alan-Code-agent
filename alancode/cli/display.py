"""Rich display helpers for the CLI.

Handles rendering of stream events, welcome banners, cost summaries,
and tool results using the Rich library.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from alancode.__version__ import __version__
from alancode.budget import ConfigError
from alancode.cli.mascot import render_mascot
from alancode.memory.memdir import ALAN_MD
from alancode.messages.types import (
    AssistantMessage,
    AttachmentMessage,
    Message,
    ProgressMessage,
    RequestStartEvent,
    StreamEvent,
    SystemMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

logger = logging.getLogger(__name__)

# ── Streaming text filter state ──────────────────────────────────────────────
# Tracks whether we're inside <think> or <tool_call> tags during streaming.
# Owned by the UI instance (CLIUI) and passed explicitly - never global.


def new_stream_state() -> dict:
    """Fresh streaming display state (one per turn, owned by the UI)."""
    return {
        "in_thinking": False,
        "in_tool_call": False,
        "buffer": "",
        # True once a "thinking" header has been printed for the current
        # thinking section. Cleared when the section ends so the header is
        # printed again on the next entry. Lets thinking be visually distinct
        # from other dim/italic content without re-printing the label per char.
        "thinking_active": False,
    }


_THINKING_LABEL = "[magenta]▎ Thinking:[/magenta] "


def _begin_thinking_section(console: Console, state: dict) -> None:
    """Print the thinking header if not already active for this section."""
    if not state["thinking_active"]:
        console.print(_THINKING_LABEL, end="")
        state["thinking_active"] = True


def _end_thinking_section(console: Console, state: dict) -> None:
    """Close an active thinking section with a newline."""
    if state["thinking_active"]:
        console.print()
        state["thinking_active"] = False


def _stream_text_delta(text: str, console, state: dict) -> None:
    """Display a streaming text delta, filtering out think/tool_call markup.

    - Text inside ``<think>...</think>`` is displayed in dim italic.
    - Text inside ``<tool_call>...</tool_call>`` is suppressed (shown as
      structured tool call blocks after parsing).
    - Regular text is displayed normally.
    """
    buf = state["buffer"] + text
    state["buffer"] = ""

    if state["in_thinking"]:
        _begin_thinking_section(console, state)

    i = 0
    while i < len(buf):
        # Check for tag openings
        if buf[i] == "<":
            # Check if we have a complete tag or need to buffer
            remaining = buf[i:]

            # <think>
            if remaining.startswith("<think>"):
                state["in_thinking"] = True
                _begin_thinking_section(console, state)
                i += len("<think>")
                continue
            # </think>
            if remaining.startswith("</think>"):
                state["in_thinking"] = False
                _end_thinking_section(console, state)
                i += len("</think>")
                continue
            # <tool_call>
            if remaining.startswith("<tool_call>"):
                state["in_tool_call"] = True
                i += len("<tool_call>")
                continue
            # </tool_call>
            if remaining.startswith("</tool_call>"):
                state["in_tool_call"] = False
                i += len("</tool_call>")
                continue

            # Might be a partial tag at the end — buffer it
            if len(remaining) < 13:  # max tag length: </tool_call>
                state["buffer"] = remaining
                return
            # Not a known tag — print the '<' and continue
            pass

        char = buf[i]
        if state["in_tool_call"]:
            pass  # Suppress tool call markup
        elif state["in_thinking"]:
            console.print(f"[dim italic]{char}[/dim italic]", end="")
        else:
            console.print(char, end="", highlight=False)
        i += 1


def _short_tokens(count: int) -> str:
    """Render a token count as 128K / 1.0M."""
    if count >= 1_000_000:
        return f"{count / 1_000_000:.1f}M"
    if count >= 1_000:
        return f"{count // 1_000}K"
    return str(count)


def _git_branch(cwd: str) -> str | None:
    """Current branch name, or None outside a repo / on a detached HEAD.

    Read straight from .git/HEAD: the banner must not spawn a subprocess on
    every startup, and a worktree's .git is a file pointing elsewhere.
    """
    try:
        git = Path(cwd, ".git")
        if git.is_file():
            git = Path(git.read_text().split("gitdir:", 1)[1].strip())
        head = (git / "HEAD").read_text().strip()
    except (OSError, IndexError):
        return None
    return head.split("refs/heads/", 1)[1] if "refs/heads/" in head else None


def _context_line(agent: Any) -> str:
    """Window size plus the fraction of it that triggers compaction."""
    from alancode.budget import resolve_context_budget

    budget = resolve_context_budget(
        agent._backend.get_model_info(agent._model), agent._settings,
    )
    percent = round(100 * budget.threshold_compaction / budget.context_window)
    return f"Context: {_short_tokens(budget.context_window)} window, compacts at ~{percent}%"


def display_welcome(console: Console, agent: Any) -> None:
    """Show the welcome banner at the start of a session."""
    model = agent._model
    session_short = agent.session_id[:8]
    cwd = agent._cwd or ""
    has_alan_md = Path(cwd, ALAN_MD).is_file() if cwd else False

    branch = _git_branch(cwd) if cwd else None
    where = cwd.replace(str(Path.home()), "~", 1) if cwd else "(no directory)"
    if branch:
        where = f"{where} ({branch})"

    # A bad budget configuration fails every turn, so say so here rather than
    # letting the first prompt be the one to discover it.
    config_error: str | None = None
    try:
        context = _context_line(agent)
    except ConfigError as e:
        config_error = str(e)
        context = "Context: [red]invalid budget configuration[/red]"
    except Exception:
        # A banner must never be the reason a session fails to start.
        logger.debug("Could not resolve context for the banner", exc_info=True)
        context = "Context: resolving on first call"

    memory = agent._memory_mode
    project = ALAN_MD if has_alan_md else f"no {ALAN_MD}, /init to add one"

    endpoint = agent._settings.get("base_url") or agent._settings.get("api_base")
    backend = agent._settings.get("backend") or "auto"
    backend_line = f"Backend: {backend}"
    if endpoint:
        backend_line += f" -> {endpoint}"

    info = Text.from_markup(
        f"Session: {session_short}... | Model: {model}\n"
        f"Dir: {where}\n"
        f"{context}\n"
        f"Memory: {memory} | {project}\n"
        f"{backend_line}\n"
        f"[dim]/help for commands, /exit to quit, Ctrl+C interrupts a turn[/dim]"
    )

    # Rich already decided whether this console gets color, so the mascot is
    # told rather than sniffing the stream itself.
    mascot = Text.from_ansi(
        render_mascot(force_color=console.is_terminal and not console.no_color)
    )

    body = Table.grid(padding=(0, 2))
    body.add_column(vertical="middle")
    body.add_column(vertical="middle")
    body.add_row(mascot, info)

    layout = Table.grid()
    layout.add_column()
    layout.add_row(
        Text.from_markup(
            f"[bold blue]Alan Code {__version__}[/bold blue] -- Open-source coding agent"
        )
    )
    layout.add_row(body)

    console.print(Panel.fit(layout, border_style="blue"))
    if config_error:
        console.print(f"[red]{config_error}[/red] Fix it in .alan/settings.json.")


def display_event(
    event: StreamEvent | Message, console: Console, stream_state: dict,
) -> None:
    """Display a stream event or message to the console.

    Routing logic:
    - AssistantMessage with hide_in_api=True: streaming text delta, print inline.
    - AssistantMessage with hide_in_api=False: final assembled message, render as Markdown.
    - UserMessage: if it contains tool results, show them in panels.
    - SystemMessage: dim informational text.
    - RequestStartEvent: thinking indicator.
    - AttachmentMessage: show attachment info.
    - ProgressMessage: show progress info.
    """
    if isinstance(event, RequestStartEvent):
        # Don't print "Thinking..." — it's noisy and misleading for non-thinking models.
        # The streaming text will appear soon enough.
        return

    if isinstance(event, AssistantMessage):
        _display_assistant_message(event, console, stream_state)
        return

    if isinstance(event, UserMessage):
        _display_user_message(event, console)
        return

    if isinstance(event, SystemMessage):
        _display_system_message(event, console)
        return

    if isinstance(event, AttachmentMessage):
        _display_attachment_message(event, console)
        return

    if isinstance(event, ProgressMessage):
        _display_progress_message(event, console)
        return


def _display_assistant_message(
    msg: AssistantMessage, console: Console, state: dict,
) -> None:
    """Render an assistant message.

    Display order: thinking (dim italic) → text → tool calls.
    Streaming deltas (hide_in_api=True) show text/thinking inline.
    Final messages show only tool calls (text was already streamed).
    """
    # Render API error messages in red, with details if available.
    if msg.is_api_error_message:
        for block in msg.content:
            if isinstance(block, TextBlock) and block.text.strip():
                console.print(f"[red]{block.text.strip()}[/red]")
        if msg.error_details:
            console.print(f"[dim]{msg.error_details}[/dim]")
        return

    if msg.hide_in_api:
        # Streaming delta — print inline without trailing newline.
        for block in msg.content:
            if isinstance(block, TextBlock):
                # If thinking was active (e.g. native ThinkingBlock streaming
                # before any text), close the section so text starts on its
                # own line.
                _end_thinking_section(console, state)
                _stream_text_delta(block.text, console, state)
            elif isinstance(block, ThinkingBlock) and block.thinking.strip():
                _begin_thinking_section(console, state)
                console.print(f"[dim italic]{block.thinking}[/dim italic]", end="")
        return

    # Final assembled message — text and thinking were already streamed,
    # so only show tool calls and any thinking that wasn't streamed.
    has_text = any(isinstance(b, TextBlock) and b.text.strip() for b in msg.content)
    has_thinking = any(
        isinstance(b, ThinkingBlock) and b.thinking.strip() for b in msg.content
    )

    # Close the streaming line if there was streamed content
    if has_text or has_thinking:
        _end_thinking_section(console, state)
        console.print()

    # Show thinking first (if not already streamed — i.e., extracted post-stream)
    for block in msg.content:
        if isinstance(block, ThinkingBlock) and block.thinking.strip():
            # Only show if this is a non-streaming context (text tool parser extracted it)
            if not has_text:
                # Thinking IS the response — show full
                console.print(
                    f"{_THINKING_LABEL}[dim italic]{block.thinking.strip()}[/dim italic]"
                )
            # If has_text, thinking was already streamed or will be shown as preview
            # via the streaming path — don't duplicate

    # Show tool calls
    for block in msg.content:
        if isinstance(block, ToolUseBlock):
            display_tool_use(block.name, block.input, console)


def _display_user_message(msg: UserMessage, console: Console) -> None:
    """Render a user message -- typically tool results."""
    if msg.hide_in_ui or msg.hide_in_api:
        return

    if isinstance(msg.content, str):
        # Plain user text -- usually not displayed in the REPL since the
        # user already typed it, but handle it gracefully.
        return

    for block in msg.content:
        if isinstance(block, ToolResultBlock):
            result_text = (
                block.content
                if isinstance(block.content, str)
                else "".join(b.text for b in block.content if isinstance(b, TextBlock))
            )
            display_tool_result(
                tool_name=block.tool_use_id,
                result_text=result_text,
                is_error=block.is_error,
                console=console,
            )


def _display_system_message(msg: SystemMessage, console: Console) -> None:
    """Render a system message in dim style."""
    style_map = {
        "info": "dim",
        "warning": "yellow",
        "error": "red",
    }
    style = style_map.get(msg.level, "dim")
    console.print(f"  [{style}]{msg.content}[/{style}]")


def _display_attachment_message(msg: AttachmentMessage, console: Console) -> None:
    """Show an attachment notification."""
    att = msg.attachment
    label = att.type.replace("_", " ").title()
    preview = att.content[:120] + "..." if len(att.content) > 120 else att.content
    console.print(
        Panel(
            preview or "[dim]<no content>[/dim]",
            title=f"[cyan]Attachment: {label}[/cyan]",
            border_style="cyan",
            expand=False,
        )
    )


def _display_progress_message(msg: ProgressMessage, console: Console) -> None:
    """Show a progress update."""
    data = msg.data
    label = data.get("label", "Progress")
    console.print(f"  [dim]{label}[/dim]")


def display_tool_use(tool_name: str, tool_input: dict, console: Console) -> None:
    """Display a tool invocation header."""
    # Show a compact summary of the tool call.
    input_summary = ", ".join(
        f"{k}={_truncate(str(v), 60)}" for k, v in tool_input.items()
    )
    console.print(
        Panel.fit(
            f"[bold]{tool_name}[/bold]({input_summary})",
            border_style="green",
            title="[green]Tool Call[/green]",
        )
    )


DIFF_SENTINEL = "[ALAN-DIFF]"


def display_tool_result(
    tool_name: str,
    result_text: str,
    is_error: bool,
    console: Console,
) -> None:
    """Display a tool result in a styled panel.

    Edit/Write results prefixed with ``[ALAN-DIFF]`` are rendered as a
    colorized unified diff with line numbers instead of a plain panel.
    """
    if not is_error and result_text.startswith(DIFF_SENTINEL):
        _display_diff_result(result_text, console)
        return

    border = "red" if is_error else "green"
    title_label = "Error" if is_error else "Result"
    title_style = "red" if is_error else "green"

    # Truncate very long results for display.
    display_text = _truncate(result_text, 2000)

    console.print(
        Panel(
            display_text or "[dim]<empty>[/dim]",
            title=f"[{title_style}]{title_label}: {tool_name}[/{title_style}]",
            border_style=border,
            expand=False,
        )
    )


def _display_diff_result(result_text: str, console: Console) -> None:
    """Render a unified-diff tool result with line numbers and colors.

    Expected format (produced by FileEditTool / FileWriteTool):

        [ALAN-DIFF]
        --- /path/to/file
        +++ /path/to/file
        @@ -start,len +start,len @@
         context
        -removed
        +added
        ...
        <plain-text summary line>
    """
    body = result_text[len(DIFF_SENTINEL):].lstrip("\n")
    lines = body.splitlines()

    # Locate the end of the diff body (everything up to the trailing
    # summary line that doesn't look like diff content).
    diff_end = len(lines)
    for i in range(len(lines) - 1, -1, -1):
        ln = lines[i]
        if ln.startswith((" ", "+", "-", "@", "\\")):
            diff_end = i + 1
            break
    diff_lines = lines[:diff_end]
    summary = "\n".join(lines[diff_end:]).strip()

    # Parse header to extract file path.
    file_path = ""
    for ln in diff_lines[:4]:
        if ln.startswith("+++ "):
            file_path = ln[4:].strip()
            break

    # Count adds/removes (skip +++/--- header lines).
    added = sum(
        1 for ln in diff_lines
        if ln.startswith("+") and not ln.startswith("+++")
    )
    removed = sum(
        1 for ln in diff_lines
        if ln.startswith("-") and not ln.startswith("---")
    )

    rendered = _render_diff_lines(diff_lines)

    title = f"[bold cyan]● Update[/bold cyan]([cyan]{file_path}[/cyan])"
    subtitle_parts: list[str] = []
    if added:
        subtitle_parts.append(f"[green]+{added}[/green]")
    if removed:
        subtitle_parts.append(f"[red]-{removed}[/red]")
    subtitle = "  ".join(subtitle_parts) if subtitle_parts else "[dim]no change[/dim]"

    console.print(
        Panel(
            rendered,
            title=f"{title}   {subtitle}",
            border_style="cyan",
            expand=False,
        )
    )
    if summary:
        console.print(f"[dim]{summary}[/dim]")


def _render_diff_lines(lines: list[str]) -> Text:
    """Turn unified-diff lines into a Rich Text with line numbers + colors."""
    text = Text()
    new_num = 0
    old_num = 0
    # Column width for line numbers — grows as we see hunk headers.
    width = 3

    for ln in lines:
        if ln.startswith("---") or ln.startswith("+++"):
            # File header — skip.
            continue
        if ln.startswith("@@"):
            # Hunk header like "@@ -172,3 +172,8 @@"
            old_num, new_num = _parse_hunk_header(ln)
            width = max(width, len(str(new_num + len(lines))))
            text.append(f"{ln}\n", style="dim cyan")
            continue
        if ln.startswith("\\"):
            # "\ No newline at end of file" — dim.
            text.append(f"{'':>{width}}  {ln}\n", style="dim")
            continue
        if ln.startswith("+"):
            text.append(f"{new_num:>{width}} + ", style="green")
            text.append(ln[1:] + "\n", style="green")
            new_num += 1
        elif ln.startswith("-"):
            text.append(f"{old_num:>{width}} - ", style="red")
            text.append(ln[1:] + "\n", style="red")
            old_num += 1
        else:
            # Context line (starts with space).
            content = ln[1:] if ln.startswith(" ") else ln
            text.append(f"{new_num:>{width}}   ", style="dim")
            text.append(content + "\n")
            old_num += 1
            new_num += 1
    return text


def _parse_hunk_header(header: str) -> tuple[int, int]:
    """Extract (old_start, new_start) from ``@@ -a,b +c,d @@``."""
    m = re.match(r"@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@", header)
    if not m:
        return (1, 1)
    return (int(m.group(1)), int(m.group(2)))


def display_cost(agent: Any, console: Console) -> None:
    """Show token usage and cost summary after a turn."""
    usage = agent.usage
    token_str = f"  [dim]Tokens: {usage.total_input:,} in / {usage.output_tokens:,} out"
    if agent.cost_unknown:
        console.print(f"{token_str}[/dim]")
    else:
        cost = agent.cost_usd
        console.print(f"{token_str} | Estimated cost: ${cost:.4f}[/dim]")


def _truncate(text: str, max_len: int) -> str:
    """Truncate a string, appending '...' if it exceeds max_len."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def display_replay_message(msg: Message, console: Console) -> None:
    """Render a single message for replay (e.g. on session resume).

    Unlike :func:`display_event`, this does NOT assume text was already
    streamed live. Assistant text is rendered as Markdown; user prompts
    and tool results are shown; hidden-in-UI items are skipped.
    """
    # Assistant messages: show thinking → text → tool calls (full).
    if isinstance(msg, AssistantMessage):
        if msg.hide_in_api:
            # Streaming delta stored in history — skip, the final message
            # sibling carries the same content.
            return
        for block in msg.content:
            if isinstance(block, ThinkingBlock) and block.thinking.strip():
                console.print(
                    f"{_THINKING_LABEL}[dim italic]{block.thinking.strip()}[/dim italic]"
                )
        for block in msg.content:
            if isinstance(block, TextBlock) and block.text.strip():
                console.print(Markdown(block.text))
        for block in msg.content:
            if isinstance(block, ToolUseBlock):
                display_tool_use(block.name, block.input, console)
        return

    # User messages: plain text prompts OR tool results.
    if isinstance(msg, UserMessage):
        if msg.hide_in_ui or msg.hide_in_api:
            return
        if isinstance(msg.content, str):
            content = msg.content
            if content.startswith("<system-reminder>"):
                return
            console.print(f"\n[dim]> {content}[/dim]")
            return
        for block in msg.content:
            if isinstance(block, ToolResultBlock):
                result_text = (
                    block.content
                    if isinstance(block.content, str)
                    else "".join(
                        b.text for b in block.content if isinstance(b, TextBlock)
                    )
                )
                display_tool_result(
                    tool_name=block.tool_use_id,
                    result_text=result_text,
                    is_error=block.is_error,
                    console=console,
                )
        return

    if isinstance(msg, SystemMessage):
        # Same styling as live.
        _display_system_message(msg, console)
        return

    # AttachmentMessage / ProgressMessage / unknown: skip in replay.


def display_replay(
    messages: list[Message], console: Console, *, limit: int = 20
) -> None:
    """Replay the tail of a message list to the console on session resume."""
    if not messages:
        return
    total = len(messages)
    tail = messages[-limit:] if total > limit else messages
    omitted = total - len(tail)
    if omitted > 0:
        console.print(
            f"\n[dim]… {omitted} earlier message(s) omitted. "
            f"Showing last {len(tail)} of {total}.[/dim]"
        )
    for msg in tail:
        display_replay_message(msg, console)
