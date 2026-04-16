"""User interaction helpers for the CLI."""

import asyncio

from prompt_toolkit import PromptSession
from rich.console import Console


async def ask_user_cli(
    question: str,
    options: list[str],
    session: PromptSession | None = None,
) -> str:
    """Display a question with numbered options and get user's choice.

    Shows:
        ? <question>
        1) <option 1>
        2) <option 2>
        Or type your own answer

    Uses prompt_toolkit so Ctrl+C unwinds cleanly (as ``KeyboardInterrupt``
    from inside the executor thread) instead of orphaning a blocked
    ``input()`` call that would fight the main REPL over stdin.

    Returns the selected option text, or the custom text typed by the user.
    """
    console = Console()
    console.print(f"\n[bold yellow]? {question}[/bold yellow]")
    for i, opt in enumerate(options, 1):
        console.print(f"  [cyan]{i})[/cyan] {opt}")
    if options:
        console.print("  [dim]Or type your own answer[/dim]")

    pt_session = session if session is not None else PromptSession()

    loop = asyncio.get_running_loop()
    try:
        choice = await loop.run_in_executor(
            None, lambda: pt_session.prompt("\nYour choice: ").strip()
        )
    except (KeyboardInterrupt, EOFError):
        # Convert to CancelledError so the REPL's interrupt handler treats
        # this as "Turn interrupted" instead of letting KeyboardInterrupt
        # escape the event loop (asyncio.gather return_exceptions=True
        # does not catch BaseException).
        raise asyncio.CancelledError("User interrupted permission prompt")

    try:
        idx = int(choice)
        if 1 <= idx <= len(options):
            return options[idx - 1]
    except ValueError:
        pass

    return choice if choice else "No answer provided."
