"""Session loop for Alan Code.

UI-agnostic: works identically with CLIUI (terminal) or GUIUI (browser).
All I/O goes through the :class:`SessionUI` interface.
Slash commands use ``ui.console`` (Rich Console or GUIConsole).
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
from pathlib import Path

from rich.syntax import Syntax
from rich.table import Table

from alancode.agent import AlanCodeAgent
from alancode.cli.display import display_welcome
from alancode.cli.errors import classify_error
from alancode.compact.compact_auto import compaction_auto
from alancode.git_tree.layout import compute_layout
from alancode.git_tree.memory_snapshots import get_memory_diff
from alancode.git_tree.operations import (
    agt_all_revert,
    agt_conv_revert,
    agt_move,
    agt_revert,
    agt_revert_to,
)
from alancode.git_tree.parser import parse_git_tree
from alancode.gui.base import SessionUI
from alancode.memory.memdir import (
    ALAN_MD,
    ensure_memory_structure,
    ensure_project_instructions,
    get_global_memory_dir,
    get_memory_dir,
    load_global_memory_index,
    load_global_project_instructions,
    load_memory_index,
    load_project_instructions,
)
from alancode.memory.prompt import build_memory_section, get_save_command_prompt
from alancode.messages.factory import create_user_message
from alancode.messages.types import Usage
from alancode.prompt.system_prompt import get_system_prompt
from alancode.settings import (
    coerce_value,
    get_settings_path,
    load_settings,
)
from alancode.utils.env import is_git_repo as _is_git_repo

logger = logging.getLogger(__name__)

SLASH_COMMANDS: dict[str, str] = {
    "/help": "Show available commands",
    "/clear": "Clear conversation and start fresh",
    "/compact": "Manually trigger conversation compaction",
    "/model": "Show or change the current model",
    "/provider": "Show or change the current provider",
    "/exit": "Exit Alan Code",
    "/init": "Create ALAN.md in the project root with a starter template",
    "/diff": "Show git diff of all uncommitted changes",
    "/status": "Show session info (model, tokens, cost, etc.)",
    "/settings": "Show or update session settings (key=value)",
    "/settings-project": "Show or update project settings in .alan/settings.json",
    "/save": "Ask the agent to save noteworthy info from this conversation to memory",
    "/memory": "Show or change memory mode (on, off, intensive)",
    "/commit": "Stage and commit changes with an AI-generated commit message",
    "/name": "Set a name for this session (displayed in listings and GUI)",
    "/revert": "Revert N commits back (default 1). Discards uncommitted changes.",
    "/move": "Move agent to a commit SHA or branch name",
    "/convrevert": "Revert N steps in conversation (agent forgets, repo unchanged)",
    "/allrevert": "Revert both position and conversation by N steps",
    "/memodiff": "Show memory diff with last commit",
    "/skill": "Invoke a skill: /skill <name> [args] | /skill list | /skill create",
}


async def run_session(
    agent: AlanCodeAgent,
    ui: SessionUI,
    resumed_session_id: str | None = None,
) -> None:
    """Run the interactive session loop.

    Works with any :class:`SessionUI` implementation (CLI or GUI).
    """
    console = ui.console

    display_welcome(console, agent)

    # Show a one-line resume announcement (applies to both CLI and GUI).
    # The UI itself replays the conversation tail via on_initial_conversation.
    if resumed_session_id and agent._messages:
        session_name = agent._session.session_name
        label = session_name or resumed_session_id[:12] + "..."
        console.print(
            f"[dim]Session {label} resumed " f"({len(agent._messages)} messages)[/dim]"
        )

    # Send initial data to GUI panels (so they're not empty before first turn)
    _send_git_tree_update(agent, ui)
    if agent._messages:
        ui.on_initial_conversation(agent._messages)
    try:
        mem_dir = get_memory_dir(agent._cwd)
        global_mem_dir = get_global_memory_dir()
        memory_section = build_memory_section(
            agent._memory_mode,
            str(mem_dir),
            load_memory_index(cwd=agent._cwd),
            global_memory_dir=str(global_mem_dir),
            global_memory_index=load_global_memory_index(),
        )
        global_instr = load_global_project_instructions()
        project_instr = load_project_instructions(agent._cwd)
        append_parts = [p for p in (global_instr, project_instr) if p]
        append_prompt = "\n\n".join(append_parts) if append_parts else None
        sp, _boundary = get_system_prompt(
            tools=agent._tools,
            skills=agent._skill_registry.list_all(),
            model=agent._model,
            cwd=agent._cwd,
            append_prompt=append_prompt,
            memory_section=memory_section,
            scratchpad_dir=str(agent._scratchpad_dir),
        )
        ui.on_initial_system_prompt("\n\n".join(sp))
    except Exception:
        pass

    while True:
        try:
            user_input = await ui.get_input("\n> ")

            if not user_input:
                continue

            # Slash commands
            if user_input.startswith("/"):
                should_exit = await _handle_slash_command(
                    user_input,
                    agent,
                    console,
                    ui,
                )
                if should_exit:
                    break
                continue

            # Regular prompt
            await _handle_prompt(agent, user_input, ui)

        except EOFError:
            console.print("\nGoodbye!", style="dim")
            break
        except (KeyboardInterrupt, asyncio.CancelledError):
            console.print()
            continue

    await agent.close()


async def _handle_prompt(
    agent: AlanCodeAgent,
    prompt: str,
    ui: SessionUI,
) -> None:
    """Send a prompt to the agent and display events through the UI."""
    tool_call_format = agent._settings.get("tool_call_format")
    ui.reset_stream_state(assume_thinking=tool_call_format is not None)
    ui.on_agent_start()

    interrupted = False
    try:
        async for event in agent.query_events_async(prompt):
            await ui.on_agent_event(event)

    except (KeyboardInterrupt, asyncio.CancelledError):
        # Ctrl+C in CLI
        interrupted = True
        agent.abort()
    except Exception as e:
        logger.exception("Error during prompt handling")
        _display_error(e, ui.console)
    finally:
        # Check if abort was triggered (Stop button in GUI sets the event
        # without raising an exception — the loop just ends).
        if agent._abort_event.is_set():
            interrupted = True
            agent._abort_event.clear()

        if interrupted:
            ui.console.print("[yellow]Turn interrupted.[/yellow]")

        try:
            # Conversation size = last call's authoritative usage
            # (input + output). Zero on a fresh session before any call
            # completes, or when the provider didn't populate `usage`.
            lu = agent.last_usage
            conv_tokens = lu.input_tokens + lu.output_tokens
            try:
                model_info = agent._provider.get_model_info(agent._model)
                ctx_window = model_info.context_window
            except Exception:
                ctx_window = 0
            await ui.on_cost(
                agent.usage,
                agent.cost_usd,
                agent.cost_unknown,
                conversation_tokens=conv_tokens,
                context_window=ctx_window,
            )
        except Exception:
            pass
        ui.on_agent_done()
        _send_git_tree_update(agent, ui)


def _send_git_tree_update(agent: AlanCodeAgent, ui: SessionUI) -> None:
    """Send git tree layout to the UI (non-critical, errors ignored)."""
    try:
        if not _is_git_repo(agent.cwd):
            return

        # Sync agent_position with actual HEAD (defensive — catches missed updates)
        _sync_agent_position(agent)

        tree = parse_git_tree(
            agent.cwd,
            alan_commits=set(agent._session.alan_commits),
        )
        layout = compute_layout(
            tree,
            conv_path=agent._session.conv_path,
            compaction_markers=agent._session.compaction_markers,
            agent_position=agent._session.agent_position_sha,
            session_root=agent._session.session_root_sha,
        )
        ui.on_git_tree_update(layout.to_json())
    except Exception:
        pass


def _sync_agent_position(agent: AlanCodeAgent) -> None:
    """Sync agent_position_sha with HEAD if an external change happened.

    Only adds HEAD to conv_path when agent_position_sha is DIFFERENT
    from HEAD (meaning something external moved it).  If they match
    but HEAD isn't in conv_path, that's intentional (e.g., after
    /convrevert) and we don't interfere.
    """
    try:
        import subprocess

        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=agent.cwd,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return
        head = result.stdout.strip()
        state = agent._session

        if state.agent_position_sha and state.agent_position_sha != head:
            # HEAD changed externally — update position and conv_path
            state.agent_position_sha = head
            if head not in state.conv_path:
                state.add_to_conv_path(head)
        elif not state.agent_position_sha:
            # No position set yet — initialize
            state.agent_position_sha = head
            if head not in state.conv_path:
                state.add_to_conv_path(head)
    except Exception:
        pass


def _display_error(error: Exception, console) -> None:
    """Classify and display errors with helpful messages."""

    message, hint = classify_error(error)
    console.print(f"[red]{message}[/red]")
    if hint:
        console.print(f"[dim]{hint}[/dim]")


async def _handle_slash_command(
    command: str,
    agent: AlanCodeAgent,
    console,
    ui: SessionUI,
) -> bool:
    """Handle slash commands. Returns True if session should exit."""
    parts = command.strip().split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""

    if cmd == "/exit" or cmd == "/quit":
        console.print("Goodbye!", style="dim")
        return True

    if cmd == "/help":
        _show_help(console)
        return False

    if cmd == "/clear":
        _handle_clear(agent, console)
        return False

    if cmd == "/compact":
        await _handle_compact(agent, console, arg)
        _send_git_tree_update(agent, ui)
        return False

    if cmd == "/model":
        _handle_model(agent, console, arg)
        return False

    if cmd == "/provider":
        _handle_provider(agent, console, arg)
        return False

    if cmd == "/init":
        _handle_init(agent, console)
        return False

    if cmd == "/diff":
        _handle_diff(agent, console)
        return False

    if cmd == "/status":
        _handle_status(agent, console)
        return False

    if cmd == "/settings":
        _handle_settings(agent, console, arg)
        return False

    if cmd == "/settings-project":
        _handle_settings_project(agent, console, arg)
        return False

    if cmd == "/save":
        await _handle_save(agent, console, arg, ui)
        return False

    if cmd == "/memory":
        _handle_memory(agent, console, arg)
        return False

    if cmd == "/commit":
        await _handle_commit(agent, console, arg, ui)
        return False

    if cmd == "/name":
        _handle_name(agent, console, arg)
        return False

    if cmd == "/revert":
        await _handle_revert(agent, console, arg, ui)
        return False

    if cmd == "/move":
        await _handle_move(agent, console, arg, ui)
        return False

    if cmd == "/convrevert":
        await _handle_conv_revert(agent, console, arg, ui)
        return False

    if cmd == "/allrevert":
        await _handle_all_revert(agent, console, arg, ui)
        return False

    if cmd == "/memodiff":
        _handle_memodiff(agent, console, arg)
        return False

    if cmd == "/skill":
        await _handle_skill(agent, console, arg, ui)
        return False

    console.print(f"[yellow]Unknown command: {cmd}[/yellow]  (type /help for a list)")
    return False


# ── Slash command implementations ──────────────────────────────────────────


def _show_help(console) -> None:
    """Print the full slash-commands table from :data:`SLASH_COMMANDS`."""
    table = Table(title="Available Commands", show_header=True, header_style="bold")
    table.add_column("Command", style="cyan", no_wrap=True)
    table.add_column("Description")
    for cmd, desc in SLASH_COMMANDS.items():
        table.add_row(cmd, desc)
    console.print(table)


def _handle_clear(agent: AlanCodeAgent, console) -> None:
    """Drop the in-memory conversation and reset last-usage counters.

    The session file on disk is preserved (so ``--resume`` still works if
    the user wants to go back), but the in-memory messages list is
    emptied and the persisted ``last_usage`` counters are zeroed so the
    next turn's "Conversation" figure starts from 0.
    """
    agent._messages.clear()
    agent._last_usage = Usage()
    with agent._session.batch():
        agent._session.last_input_tokens = 0
        agent._session.last_output_tokens = 0
        agent._session.last_cache_read_tokens = 0
        agent._session.last_cache_write_tokens = 0
    console.print("[green]Conversation cleared.[/green]")


async def _handle_compact(agent: AlanCodeAgent, console, arg: str = "") -> None:
    """Manually trigger a Layer C (forked-agent) compaction.

    Args:
        arg: Optional extra instructions appended to the summarizer
            prompt (e.g. ``/compact focus on the bug we fixed``).
    """
    msg_count = len(agent._messages)
    if msg_count <= 2:
        console.print(
            "[dim]Conversation is too short (less than 2 messages), nothing to compact.[/dim]"
        )
        return

    custom_instructions = arg.strip() if arg.strip() else None
    console.print(f"[dim]Compacting conversation ({msg_count} messages)...[/dim]")

    try:
        result = await compaction_auto(
            agent._messages,
            agent._provider,
            model=agent._model,
            custom_instructions=custom_instructions,
            session_id=agent.session_id,
            memory_mode=agent._memory_mode,
            settings=agent._settings,
        )
        if result:
            agent._messages = [result.boundary_message] + result.summary_messages
            console.print("[green]Conversation compacted successfully.[/green]")
            # AGT: record compaction marker
            if _is_git_repo(agent.cwd) and agent._session.agent_position_sha:
                agent._session.add_compaction_marker(agent._session.agent_position_sha)
        else:
            console.print("[red]Compaction failed.[/red]")
    except Exception as e:
        logger.exception("Compaction error")
        console.print(f"[red]Compaction failed: {e}[/red]")


def _handle_model(agent: AlanCodeAgent, console, arg: str) -> None:
    """Show or change the current model.

    With no argument, prints the current model. With an argument, validates
    it against the settings validator, recreates the provider if the change
    is accepted, and injects a ``<system-reminder>`` so the agent knows
    later messages may have come from a different model.
    """
    if arg:
        old_model = agent._model
        error = agent.update_session_setting("model", arg)
        if error:
            console.print(f"[red]{error}[/red]")
        else:
            console.print(f"[green]Model changed to: {arg}[/green]")
            console.print(
                "[dim]Note: You might want to change provider too "
                "with '/provider <name>'.[/dim]"
            )

            agent._messages.append(
                create_user_message(
                    f"<system-reminder>Model changed from {old_model} to {arg}. "
                    f"Previous messages may have been generated by a different model.</system-reminder>",
                    hide_in_ui=True,
                )
            )
    else:
        console.print(f"Current model: [bold]{agent._model}[/bold]")


def _handle_provider(agent: AlanCodeAgent, console, arg: str) -> None:
    """Show or change the current provider.

    Unlike ``/model``, no ``<system-reminder>`` is injected — the provider
    is backend routing and doesn't affect what the model sees.
    """
    current = agent._settings.get("provider")
    if arg:
        error = agent.update_session_setting("provider", arg)
        if error:
            console.print(f"[red]{error}[/red]")
        else:
            console.print(f"[green]Provider changed to: {arg}[/green]")
            console.print(
                "[dim]Note: You might want to change model too "
                "with '/model <name>'.[/dim]"
            )
    else:
        console.print(f"Current provider: [bold]{current}[/bold]")


def _handle_init(agent: AlanCodeAgent, console) -> None:
    """Create a starter ``ALAN.md`` in the project root.

    Refuses if the file already exists so an existing file is never
    overwritten. Users edit the generated template to taste.
    """
    cwd = agent.cwd
    path = Path(cwd) / ALAN_MD
    if path.exists():
        console.print(f"[yellow]{ALAN_MD} already exists at {path}[/yellow]")
        return
    result_path = ensure_project_instructions(cwd)
    console.print(f"[green]Created {ALAN_MD} at {result_path}[/green]")


def _handle_diff(agent: AlanCodeAgent, console) -> None:
    """Display staged + unstaged git diff with syntax highlighting."""
    cwd = agent.cwd
    if not _is_git_repo(cwd):
        console.print("[yellow]Not a git repository.[/yellow]")
        return

    try:
        unstaged = subprocess.run(
            ["git", "diff"], cwd=cwd, capture_output=True, text=True
        )
        staged = subprocess.run(
            ["git", "diff", "--staged"], cwd=cwd, capture_output=True, text=True
        )
    except FileNotFoundError:
        console.print("[red]git is not installed or not on PATH.[/red]")
        return

    combined = ""
    if staged.stdout.strip():
        combined += "# Staged changes\n" + staged.stdout
    if unstaged.stdout.strip():
        if combined:
            combined += "\n"
        combined += "# Unstaged changes\n" + unstaged.stdout

    if not combined.strip():
        console.print("[dim]No uncommitted changes.[/dim]")
        return

    syntax = Syntax(combined, "diff", theme="monokai", line_numbers=False)
    console.print(syntax)


def _handle_status(agent: AlanCodeAgent, console) -> None:
    """Print a full session-status table.

    Includes provider, model, session ID, turn + message counts, the
    token breakdown (input / cache-creation / cache-read / output),
    estimated cost, cwd, and whether ``ALAN.md`` /
    ``.alan/settings.json`` exist.
    """
    cwd = agent.cwd
    usage = agent.usage
    model = agent._model
    session_name = agent._session.session_name
    session_id_short = agent.session_id[:12]
    turn_count = agent.turn_count
    msg_count = len(agent._messages)

    alan_md_exists = (Path(cwd) / ALAN_MD).exists()
    settings_exists = get_settings_path(cwd).exists()

    provider = agent._settings.get("provider")

    table = Table(title="Session Status", show_header=False)
    table.add_column("Key", style="cyan")
    table.add_column("Value", justify="right")
    table.add_row("Provider", str(provider))
    table.add_row("Model", str(model))
    table.add_row("Session ID", session_id_short)
    if session_name:
        table.add_row("Session name", session_name)
    table.add_row("Turns", str(turn_count))
    table.add_row("Messages", str(msg_count))
    table.add_row("Input tokens", f"{usage.input_tokens:,}")
    table.add_row("Cache creation tokens", f"{usage.cache_creation_input_tokens:,}")
    table.add_row("Cache read tokens", f"{usage.cache_read_input_tokens:,}")
    table.add_row("Total input", f"{usage.total_input:,}")
    table.add_row("Output tokens", f"{usage.output_tokens:,}")
    if agent.cost_unknown:
        table.add_row("Estimated cost", "unknown (model not in pricing registry)")
    else:
        table.add_row("Estimated cost", f"${agent.cost_usd:.4f}")
    table.add_row("Working directory", cwd)
    table.add_row(
        "ALAN.md", "[green]yes[/green]" if alan_md_exists else "[dim]no[/dim]"
    )
    table.add_row(
        ".alan/settings.json",
        "[green]yes[/green]" if settings_exists else "[dim]no[/dim]",
    )
    console.print(table)


def _handle_settings(agent: AlanCodeAgent, console, arg: str) -> None:
    """Show or update session settings.

    With no argument, prints the effective settings dict as JSON. With
    ``key=value``, validates and applies the change. Provider-related
    keys (``provider``, ``model``, ``api_key``, ``base_url``)
    trigger provider recreation.
    """
    if not arg:
        formatted = json.dumps(agent._settings, indent=2, default=str)
        syntax = Syntax(formatted, "json", theme="monokai", line_numbers=False)
        console.print(syntax)
        return

    if "=" not in arg:
        console.print(
            "[yellow]Usage: /settings key=value[/yellow]  (e.g. /settings model=openai/gpt-4o)"
        )
        return

    key, _, raw_value = arg.partition("=")
    key = key.strip()
    value = coerce_value(raw_value.strip())
    error = agent.update_session_setting(key, value)
    if error:
        console.print(f"[red]{error}[/red]")
    else:
        console.print(f"[green]Session setting updated: {key} = {value!r}[/green]")


def _handle_settings_project(agent: AlanCodeAgent, console, arg: str) -> None:
    """Show or update project settings in ``.alan/settings.json``.

    Unlike ``/settings``, changes here do NOT affect the current session —
    they update the on-disk defaults that future sessions will pick up.
    """
    cwd = agent.cwd
    if not arg:
        settings = load_settings(cwd)
        if not settings:
            console.print("[dim]No .alan/settings.json found. Using defaults.[/dim]")
            return
        formatted = json.dumps(settings, indent=2, default=str)
        syntax = Syntax(formatted, "json", theme="monokai", line_numbers=False)
        console.print(syntax)
        return

    if "=" not in arg:
        console.print("[yellow]Usage: /settings-project key=value[/yellow]")
        return

    key, _, raw_value = arg.partition("=")
    key = key.strip()
    value = coerce_value(raw_value.strip())
    error = agent.update_project_setting(key, value)
    if error:
        console.print(f"[red]{error}[/red]")
    else:
        console.print(
            f"[green]Project setting updated: {key} = {value!r} in .alan/settings.json[/green]"
        )


async def _handle_save(
    agent: AlanCodeAgent, console, arg: str = "", ui: SessionUI | None = None
) -> None:
    """Ask the agent to review the conversation and persist to memory.

    No-op (with a warning) if memory mode is ``off``. Otherwise injects the
    ``/save`` prompt (see ``alancode/memory/prompt.py``) appended with any
    extra context the user supplied, and runs a turn through the UI so the
    save happens and the tool calls render.
    """
    if agent._memory_mode == "off":
        console.print(
            "[yellow]Memory is disabled. Use '/memory [on/intensive]' to enable it first.[/yellow]"
        )
        return

    console.print(
        "[dim]Asking agent to review conversation and save to memory...[/dim]"
    )
    prompt = get_save_command_prompt()
    if arg.strip():
        prompt += (
            f"\n\nAdditional context from user memory update request: {arg.strip()}"
        )
    if ui:
        await _handle_prompt(agent, prompt, ui)


def _handle_memory(agent: AlanCodeAgent, console, arg: str) -> None:
    """Show or change the memory mode (``off``, ``on``, ``intensive``).

    With no argument, prints the current mode with a hint about the
    available options. With an argument, validates via the settings
    validator, applies the change, and injects a ``<system-reminder>``
    so the agent knows the mode switched.
    """
    if not arg:
        console.print(f"Memory mode: [bold]{agent._memory_mode}[/bold]")
        console.print(
            "[dim]Options: on (use memory, save on request), off (disabled), "
            "intensive (use memory, proactive saves)[/dim]"
        )
        return

    mode = arg.strip().lower()
    old_mode = agent._memory_mode
    error = agent.update_session_setting("memory", mode)
    if error:
        console.print(f"[red]{error}[/red]")
    else:
        console.print(f"[green]Memory mode changed to: {mode}[/green]")

        agent._messages.append(
            create_user_message(
                f"<system-reminder>Memory mode changed from '{old_mode}' to '{mode}'.</system-reminder>",
                hide_in_ui=True,
            )
        )

    if mode != "off":
        ensure_memory_structure(agent.cwd)


async def _handle_commit(
    agent: AlanCodeAgent, console, arg: str = "", ui: SessionUI | None = None
) -> None:
    """Ask the agent to draft and commit via the ``GitCommit`` tool.

    Refuses outside a git repo or when there are no changes to commit.
    Otherwise runs a turn where the prompt tells the agent to inspect
    diffs, draft a concise message in the repo's style, and commit.

    Args:
        arg: Optional user guidance (e.g. ``/commit note that this
            fixes the PTL retry bug``) appended to the prompt.
    """
    cwd = agent._cwd

    if not _is_git_repo(cwd):
        console.print("[red]Not a git repository.[/red]")
        return

    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if not result.stdout.strip():
            console.print("[dim]No changes to commit.[/dim]")
            return
    except Exception:
        console.print("[red]Failed to check git status.[/red]")
        return

    prompt = (
        "The user wants to commit their changes. Please:\n"
        "1. Inspect the changes with `git status`, `git diff`, and "
        "`git diff --staged`.\n"
        "2. Look at recent commit message style with `git log --oneline -5`.\n"
        "3. Draft a concise commit message (1-2 sentences) that follows the "
        "repo's style and captures the 'why' rather than the 'what'.\n"
        "4. Call the GitCommit tool with that message. GitCommit stages the "
        "changes, adds the Co-Authored-By trailer, and records the commit "
        "in the agent's tracked commits.\n"
        "5. Confirm the result with `git log --oneline -1`."
    )
    if arg.strip():
        prompt += f"\n\nUser's additional guidance for this commit: {arg.strip()}"

    console.print("[dim]Asking Alan to commit...[/dim]")
    if ui:
        await _handle_prompt(agent, prompt, ui)


def _handle_name(agent: AlanCodeAgent, console, arg: str) -> None:
    """Show or set the session's human-readable name.

    The name is displayed in session listings (``alancode --continue``)
    and in the GUI tab title. Not used anywhere functional — purely for
    the user's convenience.
    """
    if not arg:
        name = agent._session.session_name
        if name:
            console.print(f"Session name: [bold]{name}[/bold]")
        else:
            console.print(
                "[dim]No session name set. Use /name <text> to set one.[/dim]"
            )
        return

    agent._session.session_name = arg.strip()
    console.print(f"[green]Session named: {arg.strip()}[/green]")


# ── AGT movement commands ────────────────────────────────────────────────────


async def _handle_revert(
    agent: AlanCodeAgent,
    console,
    arg: str = "",
    ui: SessionUI | None = None,
) -> None:
    """Revert repo state.  Accepts N (integer steps) or a SHA/branch target."""
    if not _is_git_repo(agent.cwd):
        console.print("[yellow]Requires a git repository.[/yellow]")
        return

    arg = arg.strip()
    if not arg:

        result = agt_revert(agent.cwd, agent._session, 1)
    elif arg.isdigit():
        n = int(arg)
        if n < 1:
            console.print("[yellow]N must be at least 1.[/yellow]")
            return

        result = agt_revert(agent.cwd, agent._session, n)
    else:
        # SHA or branch target — destructive revert to that point
        target_sha = _resolve_sha(agent.cwd, arg)
        if not target_sha:
            console.print(f"[red]Cannot resolve '{arg}'[/red]")
            return

        result = agt_revert_to(agent.cwd, agent._session, target_sha)

    if result.success:
        console.print(f"[green]{result.description}[/green]")
        agent._messages.append(
            create_user_message(
                f"<system-reminder>User reverted repo. {result.description}\n"
                "Re-read files before making assumptions about their current state.</system-reminder>",
                hide_in_ui=True,
            )
        )
        if ui:
            _send_git_tree_update(agent, ui)
    else:
        console.print(f"[red]{result.description}[/red]")


async def _handle_move(
    agent: AlanCodeAgent,
    console,
    arg: str = "",
    ui: SessionUI | None = None,
) -> None:
    """Move the agent to a different commit or branch.

    Safe (non-destructive): checks out the target, updates the agent
    position, and injects a ``<system-reminder>`` explaining what
    happened so the model knows the working tree changed.

    Args:
        arg: Commit SHA or branch name.
    """
    if not _is_git_repo(agent.cwd):
        console.print("[yellow]Requires a git repository.[/yellow]")
        return

    target = arg.strip()
    if not target:
        console.print("[yellow]Usage: /move <commit-sha-or-branch>[/yellow]")
        return

    # Resolve branch names to SHAs
    import subprocess as _sp

    result = _sp.run(
        ["git", "rev-parse", target],
        cwd=agent.cwd,
        capture_output=True,
        text=True,
        timeout=5,
    )
    if result.returncode != 0:
        console.print(f"[red]Cannot resolve '{target}': {result.stderr.strip()}[/red]")
        return
    target_sha = result.stdout.strip()


    move = agt_move(agent.cwd, agent._session, target_sha)

    if move.success:
        console.print(f"[green]{move.description}[/green]")
        short_sha = target_sha[:10]
        ref_hint = (
            f" (ref '{target}')" if target != target_sha else ""
        )
        agent._messages.append(
            create_user_message(
                f"<system-reminder>User ran /move, checking out commit "
                f"{short_sha}{ref_hint}. The working tree now reflects that "
                f"commit — files on disk may have changed compared to what "
                f"you saw earlier. {move.description} Re-read files before "
                f"making assumptions about their current state.</system-reminder>",
                hide_in_ui=True,
            )
        )
        if ui:
            _send_git_tree_update(agent, ui)
    else:
        console.print(f"[red]{move.description}[/red]")


async def _handle_conv_revert(
    agent: AlanCodeAgent,
    console,
    arg: str = "",
    ui: SessionUI | None = None,
) -> None:
    """Revert conversation to the state it was at a specific commit.

    Accepts N (steps back in conv_path) or a SHA/branch target.
    Truncates messages to exactly where they were when that commit was made.
    """
    if not _is_git_repo(agent.cwd):
        console.print("[yellow]Requires a git repository.[/yellow]")
        return

    arg = arg.strip()
    if not arg:
        n = 1
    elif arg.isdigit():
        n = int(arg)
    else:
        # SHA target — compute N as steps from end of conv_path to this SHA
        conv = agent._session.conv_path
        target_sha = _resolve_sha(agent.cwd, arg)
        if not target_sha:
            console.print(f"[red]Cannot resolve '{arg}'[/red]")
            return
        if target_sha not in conv:
            console.print(
                f"[yellow]{arg[:7]} is not in the conversation path.[/yellow]"
            )
            return
        idx = len(conv) - 1 - conv[::-1].index(target_sha)
        n = len(conv) - 1 - idx
        if n <= 0:
            console.print("[dim]Already at that point in conversation.[/dim]")
            return

    # Find the target SHA we're reverting to
    conv = agent._session.conv_path
    target_idx = max(0, len(conv) - 1 - n)
    target_sha = conv[target_idx] if target_idx < len(conv) else None


    result = agt_conv_revert(agent.cwd, agent._session, n)

    if result.success:
        console.print(f"[green]{result.description}[/green]")
        # Truncate messages precisely using commit_message_indices
        if result.steps_reverted > 0 and target_sha:
            _truncate_messages_to_commit(agent, target_sha)
        agent._messages.append(
            create_user_message(
                f"<system-reminder>User ran /convrevert. {result.description} "
                "The recent conversation history has been truncated and those "
                "earlier messages are gone from your context. The working tree "
                "is unchanged — this only affects the conversation.</system-reminder>",
                hide_in_ui=True,
            )
        )
        if ui:
            _send_git_tree_update(agent, ui)
    else:
        console.print(f"[red]{result.description}[/red]")


async def _handle_all_revert(
    agent: AlanCodeAgent,
    console,
    arg: str = "",
    ui: SessionUI | None = None,
) -> None:
    """Revert both repo and conversation.  Accepts N (steps) or SHA target."""
    if not _is_git_repo(agent.cwd):
        console.print("[yellow]Requires a git repository.[/yellow]")
        return

    arg = arg.strip()
    if not arg:
        n = 1
    elif arg.isdigit():
        n = int(arg)
    else:
        # SHA target — use /move for repo + convrevert for conv
        target_sha = _resolve_sha(agent.cwd, arg)
        if not target_sha:
            console.print(f"[red]Cannot resolve '{arg}'[/red]")
            return
        # Move repo
        await _handle_move(agent, console, target_sha, ui)
        # Also revert conv to that point
        await _handle_conv_revert(agent, console, target_sha, ui)
        return


    result = agt_all_revert(agent.cwd, agent._session, n)

    if result.success:
        console.print(f"[green]{result.description}[/green]")
        # Truncate messages to the target commit
        target_sha = result.new_sha
        if target_sha:
            _truncate_messages_to_commit(agent, target_sha)
        agent._messages.append(
            create_user_message(
                f"<system-reminder>User ran /allrevert. {result.description} "
                "Both the working tree and the conversation were reverted: "
                "earlier messages are gone from your context, and the files "
                "on disk now reflect the earlier commit. Re-read files if "
                "you need them.</system-reminder>",
                hide_in_ui=True,
            )
        )
        if ui:
            _send_git_tree_update(agent, ui)
    else:
        console.print(f"[red]{result.description}[/red]")


def _truncate_messages_to_commit(agent: AlanCodeAgent, target_sha: str) -> None:
    """Truncate messages to exactly where they were when *target_sha* was committed.

    Uses ``commit_message_indices`` from session state for precision.
    Falls back to heuristic if no index is recorded.
    """
    indices = agent._session.commit_message_indices
    if target_sha in indices:
        cutoff = indices[target_sha]
        if 0 < cutoff < len(agent._messages):
            agent._messages = agent._messages[:cutoff]
            return

    # Fallback: no index recorded — try to find the GitCommit tool result
    # for target_sha in messages and truncate after it
    for i in range(len(agent._messages) - 1, -1, -1):
        msg = agent._messages[i]
        if hasattr(msg, "content") and isinstance(msg.content, list):
            for block in msg.content:
                if hasattr(block, "name") and block.name == "GitCommit":
                    # Check if this tool call produced the target commit
                    if hasattr(block, "input") and isinstance(block.input, dict):
                        # Can't reliably match — keep looking
                        pass
        # Check tool results mentioning the SHA
        if hasattr(msg, "content") and isinstance(msg.content, str):
            if target_sha[:7] in msg.content:
                agent._messages = agent._messages[: i + 1]
                return


def _resolve_sha(cwd: str, target: str) -> str | None:
    """Resolve a branch/tag/short-SHA to a full SHA."""
    import subprocess as _sp

    result = _sp.run(
        ["git", "rev-parse", target],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=5,
    )
    return result.stdout.strip() if result.returncode == 0 else None


async def _handle_skill(
    agent: AlanCodeAgent,
    console,
    arg: str,
    ui: SessionUI,
) -> None:
    """Handle /skill <name> [args] | /skill list | /skill create."""
    # Re-scan skill directories so newly-created skills are picked up
    # without requiring a session restart.
    agent._skill_registry.reload(agent._cwd)

    parts = arg.strip().split(maxsplit=1)
    if not parts:
        console.print(
            "[yellow]Usage: /skill <name> [args] | /skill list | /skill create[/yellow]"
        )
        return

    subcmd = parts[0]
    skill_args = parts[1] if len(parts) > 1 else ""

    if subcmd == "list":
        _show_skills_list(agent, console)
        return

    # Look up skill (includes built-in "create" and any discovered skills)
    skill = agent._skill_registry.get(subcmd)
    if skill is None:
        console.print(f"[yellow]Unknown skill: {subcmd}[/yellow]")
        console.print("[dim]Use /skill list to see available skills[/dim]")
        return

    expanded = agent._skill_registry.expand(subcmd, skill_args)
    # Set tool restriction if skill defines allowed-tools
    if skill.allowed_tools:
        agent._active_skill_filter = skill.allowed_tools
    console.print(f"[dim]Invoking skill: {skill.name}[/dim]")
    await _handle_prompt(agent, expanded, ui)


def _show_skills_list(agent: AlanCodeAgent, console) -> None:
    """Display all discovered skills in a table."""
    skills = agent._skill_registry.list_all()
    if not skills:
        console.print("[dim]No skills discovered.[/dim]")
        console.print(
            "[dim]Create one with /skill create or add SKILL.md files "
            "to .alan/skills/<name>/[/dim]"
        )
        return

    table = Table(title="Available Skills", show_header=True, header_style="bold")
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Description")
    table.add_column("Source", style="dim")
    for skill in skills:
        source = "builtin" if skill.source_path == "<builtin>" else "disk"
        table.add_row(skill.name, skill.description, source)
    console.print(table)
    console.print("[dim]Invoke with: /skill <name> [args][/dim]")


def _handle_memodiff(agent: AlanCodeAgent, console, arg: str = "") -> None:
    if not _is_git_repo(agent.cwd):
        console.print("[yellow]Requires a git repository.[/yellow]")
        return


    current = agent._session.agent_position_sha
    if not current:
        console.print("[dim]No AGT position tracked yet.[/dim]")
        return

    # Find the previous alan commit to diff against
    prev_commits = agent._session.alan_commits
    if len(prev_commits) < 2:
        console.print("[dim]Not enough commits to show a memory diff.[/dim]")
        return

    prev = prev_commits[-2]
    diff = get_memory_diff(agent.cwd, prev, current)
    if diff:
        console.print(f"[bold]Memory diff ({prev[:7]} → {current[:7]}):[/bold]")
        console.print(diff)
    else:
        console.print("[dim]No memory differences found.[/dim]")
