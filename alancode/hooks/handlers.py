"""Hook handler utilities."""

from typing import Any

from alancode.hooks.registry import HookType, run_hooks


async def on_session_start(
    cwd: str, session_id: str, model: str, settings: dict[str, Any] | None = None,
) -> None:
    """Fire SessionStart hooks."""
    payload = {
        "hook_type": HookType.SESSION_START.value,
        "cwd": cwd,
        "session_id": session_id,
        "model": model,
    }
    await run_hooks(HookType.SESSION_START, payload, settings=settings)


async def on_session_end(
    session_id: str, total_cost: float, turn_count: int, settings: dict[str, Any] | None = None,
) -> None:
    """Fire SessionEnd hooks."""
    payload = {
        "hook_type": HookType.SESSION_END.value,
        "session_id": session_id,
        "total_cost": total_cost,
        "turn_count": turn_count,
    }
    await run_hooks(HookType.SESSION_END, payload, settings=settings)
