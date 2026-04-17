"""Alan Code CLI entry point.

Usage:
    alancode                                    # Start interactive session
    alancode --resume                           # Resume last session
    alancode --model openrouter/google/gemini-2.5-flash
    alancode --print "fix the bug in main.py"   # Non-interactive
    alancode --version
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

from alancode.agent import AlanCodeAgent
from alancode.cli.repl import run_session
from alancode.messages.types import AssistantMessage, TextBlock
from alancode.session.session import get_last_session_id, find_session_by_prefix
from alancode.__version__ import __version__
from alancode.settings import get_settings_path


def main() -> None:
    """The ``alancode`` entry point.

    Parses CLI arguments, resolves which mode to run
    (interactive CLI, browser GUI, or non-interactive print), runs
    first-run setup if needed, and dispatches to the matching runner.

    This is what ``pip install alancode`` binds to the ``alancode``
    executable via the ``[project.scripts]`` entry in ``pyproject.toml``.
    """
    parser = argparse.ArgumentParser(
        description="Alan Code -- Open-source Coding Agent (CLI mode)"
    )

    # Settings args — all default to None so we can detect "not passed"
    parser.add_argument("--provider", default=None, help="LLM provider (litellm, anthropic, scripted)")
    parser.add_argument("--model", default=None, help="Model to use")
    parser.add_argument("--api-key", default=None, help="API key")
    parser.add_argument("--base-url", default=None, help="API base URL (for local servers: http://localhost:8000/v1)")
    parser.add_argument("--tool-call-format", default=None, choices=["hermes", "glm", "alan"],
                        help="Text-based tool call format for models without native tool calling")
    parser.add_argument("--permission-mode", default=None, choices=["yolo", "edit", "safe"])
    parser.add_argument(
        "--max-iterations-per-turn",
        type=int,
        default=None,
        help="Max API calls (iterations) per user message before the agent stops",
    )
    parser.add_argument("--max-output-tokens", type=int, default=None)
    parser.add_argument("--memory", default=None, choices=["on", "off", "intensive"])
    parser.add_argument("--verbose", default=None, action="store_true")

    # Non-settings args
    parser.add_argument("--print", dest="print", default=None, metavar="PROMPT",
                        help="Non-interactive: run prompt and exit")
    parser.add_argument("--resume", action="store_true", help="Resume last session")
    parser.add_argument("--continue", dest="continue_session", default=None, nargs="?", const="__LIST__",
                        metavar="SESSION_PREFIX",
                        help="Continue a session. Without arg: list recent sessions. With arg: resume by prefix match.")
    parser.add_argument("--version", action="store_true", help="Show version")
    parser.add_argument("--gui", action="store_true", default=False,
                        help="Launch browser GUI alongside CLI (http://localhost:8420/)")

    # Parse and pop non-settings
    args = parser.parse_args()
    all_args = vars(args)

    print_instructions = all_args.pop("print")
    do_resume = all_args.pop("resume")
    continue_prefix = all_args.pop("continue_session")
    do_show_version = all_args.pop("version", None)
    do_gui = all_args.pop("gui", False)

    # Show version
    if do_show_version:
        print(f"alancode {__version__}")
        sys.exit(0)

    # Resolve session
    cwd = os.getcwd()
    session_id = None
    if do_resume:
        session_id = get_last_session_id(cwd=cwd)
        if not session_id:
            print("Error: No previous session found.", file=sys.stderr)
            sys.exit(1)
    elif continue_prefix == "__LIST__":
        _list_recent_sessions(cwd)
        sys.exit(0)
    elif continue_prefix:
        session_id = find_session_by_prefix(cwd, continue_prefix)
        if not session_id:
            print(f"Error: No unique session matching '{continue_prefix}'.", file=sys.stderr)
            sys.exit(1)

    # CLI settings = non-None args, coerced to proper types
    from alancode.settings import coerce_value
    settings_cli = {}
    for k, v in all_args.items():
        if v is not None:
            settings_cli[k] = coerce_value(v) if isinstance(v, str) else v

    # Logging
    if settings_cli.get("verbose"):
        logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
    else:
        logging.basicConfig(level=logging.WARNING)

    # First-run setup: detect API keys and configure defaults
    if not get_settings_path(cwd).exists():
        _first_run_setup(cwd)

    if print_instructions is not None:
        # Non-interactive mode — no UI, just run and print
        from alancode.cli.user_input import ask_user_cli
        ask_cb = ask_user_cli if sys.stdin.isatty() else None
        agent = AlanCodeAgent(
            session_id=session_id,
            ask_callback=ask_cb,
            **settings_cli,
        )
        asyncio.run(_run_print_mode(agent, prompt=print_instructions))
    elif do_gui:
        # GUI mode — browser-based UI
        asyncio.run(_run_gui_mode(session_id, settings_cli, cwd))
    else:
        # CLI mode (default) — terminal UI
        asyncio.run(_run_cli_mode(session_id, settings_cli))


async def _run_cli_mode(session_id, settings_cli):
    """Standard terminal mode."""
    from alancode.gui.cli_ui import CLIUI
    ui = CLIUI()
    agent = AlanCodeAgent(
        session_id=session_id,
        ask_callback=ui.ask_user,
        **settings_cli,
    )
    await run_session(agent, ui, resumed_session_id=session_id)


async def _run_gui_mode(session_id, settings_cli, cwd):
    """Browser GUI mode."""
    from alancode.gui.gui_ui import GUIUI
    agent = AlanCodeAgent(
        session_id=session_id,
        ask_callback=None,  # Will be set to ui.ask_user below
        **settings_cli,
    )
    ui = GUIUI(agent, cwd)
    agent._ask_callback = ui.ask_user
    agent._llm_perspective_callback = ui.set_llm_perspective

    url = await ui.start()
    print(f"\n  GUI: {url}\n")
    print(f"  Open the URL in your browser. All interaction happens there.\n")

    try:
        await run_session(agent, ui, resumed_session_id=session_id)
    finally:
        await ui.stop()


# ── Session listing ──────────────────────────────────────────────────────────


def _list_recent_sessions(cwd: str, max_sessions: int = 10) -> None:
    """List recent sessions with timestamps and last user message."""
    import json

    sessions_dir = Path(cwd) / ".alan" / "sessions"
    if not sessions_dir.is_dir():
        print("No sessions found.")
        return

    # Collect sessions with transcript info
    sessions = []
    for session_dir in sessions_dir.iterdir():
        if not session_dir.is_dir():
            continue
        transcript = session_dir / "transcript.jsonl"
        if not transcript.is_file():
            continue

        sid = session_dir.name
        created_at = None
        last_time = None
        last_user_msg = ""

        try:
            with open(transcript, "r") as f:
                lines = f.readlines()

            for line in lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Metadata line
                if "_metadata" in d:
                    meta = d["_metadata"]
                    created_at = meta.get("created_at", "")
                    continue

                # Track last user message (text content only)
                msg_type = d.get("type") or d.get("role")
                if msg_type == "user":
                    content = d.get("content", "")
                    if isinstance(content, str) and content.strip() and not content.startswith("<system-reminder>"):
                        last_user_msg = content.strip()

            # Last modified time of transcript as end time
            last_time = transcript.stat().st_mtime

        except (OSError, json.JSONDecodeError):
            continue

        sessions.append({
            "id": sid,
            "created_at": created_at or "",
            "last_time": last_time or 0,
            "last_user_msg": last_user_msg,
        })

    if not sessions:
        print("No sessions found.")
        return

    # Sort by last modification time, newest first
    sessions.sort(key=lambda s: s["last_time"], reverse=True)
    sessions = sessions[:max_sessions]

    print()
    print(f"  Recent sessions ({len(sessions)}):")
    print()

    from datetime import datetime as dt

    for s in sessions:
        sid_short = s["id"][:12]

        # Format start time from ISO created_at
        try:
            created = dt.fromisoformat(s["created_at"]).strftime("%Y-%m-%d %H:%M") if s["created_at"] else "?"
        except (ValueError, TypeError):
            created = "?"

        # Format end time from file mtime
        try:
            ended = dt.fromtimestamp(s["last_time"]).strftime("%H:%M") if s["last_time"] else "?"
        except (OSError, ValueError):
            ended = "?"

        time_display = f"{created} - {ended}" if created != "?" else "?"

        # Truncate last message
        msg = s["last_user_msg"]
        if len(msg) > 60:
            msg = msg[:57] + "..."
        msg_display = f'  "{msg}"' if msg else ""

        print(f"    {sid_short}  | {time_display} |{msg_display}")

    print()
    print("  Use: alancode --continue <session_id_prefix>")
    print()


# ── First-run setup ──────────────────────────────────────────────────────────


def _first_run_setup(cwd: str) -> None:
    """Detect API keys and show first-run guidance.

    Called once per project (when .alan/settings.json doesn't exist yet).
    Prints a welcome message with configuration suggestions based on
    detected API keys in the environment.
    """
    print()
    print("=" * 60)
    print("  Welcome to Alan Code!")
    print("=" * 60)
    print()
    print("  Default provider and model of current project:")
    print("    anthropic / claude-sonnet-4-6")
    print()

    # Detect API keys
    detections = _detect_api_keys()
    if detections:
        print("  Suggested configurations based on your API keys:")
        print()
        for d in detections:
            print(f"    {d['label']}")
            print(f"      /settings-project provider={d['provider']}")
            print(f"      /settings-project model={d['model']}")
            print()

    print("  To change default settings, use /settings-project. All future")
    print("  Alan Code sessions in this project will use these settings by")
    print("  default. To change this session's settings, use /settings.")
    print("  To override session and project settings, use")
    print("  'alancode --<key> <value>' on the command line when starting")
    print("  or resuming an Alan Code session.")
    print()
    print("=" * 60)
    print()


def _detect_api_keys() -> list[dict]:
    """Detect available API keys and suggest provider/model configs."""
    detections = []

    if os.environ.get("ANTHROPIC_API_KEY"):
        detections.append({
            "label": "ANTHROPIC_API_KEY detected (recommended)",
            "provider": "anthropic",
            "model": "claude-sonnet-4-6",
        })

    if os.environ.get("OPENROUTER_API_KEY"):
        label = "OPENROUTER_API_KEY detected"
        # Try to check balance
        balance = _check_openrouter_balance()
        if balance is not None:
            label += f" (${balance:.2f} remaining)"
            if balance < 1.0:
                label += " [low balance]"
        detections.append({
            "label": label,
            "provider": "litellm",
            "model": "openrouter/anthropic/claude-sonnet-4",
        })

    if os.environ.get("OPENAI_API_KEY"):
        detections.append({
            "label": "OPENAI_API_KEY detected",
            "provider": "litellm",
            "model": "openai/gpt-4o",
        })

    return detections


def _check_openrouter_balance() -> float | None:
    """Check OpenRouter API key balance. Returns remaining $ or None on failure."""
    try:
        import requests
        resp = requests.get(
            "https://openrouter.ai/api/v1/key",
            headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
            timeout=5,
        )
        if resp.status_code == 200:
            return float(resp.json()["data"]["limit_remaining"])
    except Exception:
        pass
    return None


# ── Print mode ───────────────────────────────────────────────────────────────


async def _run_print_mode(agent: AlanCodeAgent, prompt: str) -> None:
    """Run one turn non-interactively and print the answer to stdout.

    Used by ``alancode --print "some prompt"`` — stream assistant
    text to stdout as it arrives, catch Ctrl+C for a clean 130 exit
    code, and surface any other exception via :func:`_display_error_stderr`
    with exit code 1.

    Args:
        agent: Pre-configured agent (usually with ``yolo`` or a provided
            ``ask_callback`` — interactive prompts are awkward in pipe mode).
        prompt: The single user message to run.
    """
    try:
        async for event in agent.query_events_async(prompt):
            # Only print virtual (streaming) messages to avoid duplication
            if isinstance(event, AssistantMessage) and event.hide_in_api:
                for block in event.content:
                    if isinstance(block, TextBlock):
                        print(block.text, end="", flush=True)
    except KeyboardInterrupt:
        agent.abort()
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        _display_error_stderr(e)
        sys.exit(1)
    finally:
        await agent.close()
    print()


def _display_error_stderr(error: Exception) -> None:
    from alancode.cli.errors import classify_error
    message, _ = classify_error(error)
    print(f"\n{message}", file=sys.stderr)


if __name__ == "__main__":
    main()
