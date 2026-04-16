"""Hook registry -- lifecycle event hooks.

Hooks are user-defined shell commands that execute at lifecycle points.
Configured in .alan/settings.json under "hooks".

Example config:
{
  "hooks": {
    "PreToolUse": [
      {"command": "python check_safety.py", "tools": ["Bash"]}
    ],
    "PostToolUse": [
      {"command": "python auto_lint.py", "tools": ["Edit", "Write"]}
    ],
    "SessionStart": [
      {"command": "echo 'Alan Code session started'"}
    ],
    "SessionEnd": [
      {"command": "echo 'Session ended'"}
    ]
  }
}

Hook commands receive a JSON payload on stdin with context about the event.
They can return JSON on stdout to influence behavior:
  - PreToolUse: {"action": "allow"} | {"action": "deny", "message": "..."} | {"action": "ask"}
  - PostToolUse: (return value ignored, fire-and-forget)
  - SessionStart/End: (return value ignored)
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class HookType(str, Enum):
    PRE_TOOL_USE = "PreToolUse"
    POST_TOOL_USE = "PostToolUse"
    POST_TOOL_USE_FAILURE = "PostToolUseFailure"
    SESSION_START = "SessionStart"
    SESSION_END = "SessionEnd"


HOOK_TIMEOUT_SECONDS = 30  # Max time a hook can run


@dataclass
class HookConfig:
    """A single hook definition from settings.

    ``command`` is tokenized via ``shlex.split`` and executed with
    :func:`asyncio.create_subprocess_exec` by default — no shell,
    no metacharacter interpretation. Set ``shell: true`` in the
    settings entry to opt into shell interpretation (for pipes,
    redirects, etc.).
    """
    command: str
    tools: list[str] | None = None  # None = all tools, list = only these tools
    timeout: int = HOOK_TIMEOUT_SECONDS
    shell: bool = False


@dataclass
class HookResult:
    """Result from executing a hook."""
    action: str = "allow"  # 'allow', 'deny', 'ask', 'passthrough'
    message: str = ""
    hook_name: str = ""
    exit_code: int = 0
    stdout: str = ""
    stderr: str = ""


def load_hooks_from_settings(settings: dict[str, Any]) -> dict[HookType, list[HookConfig]]:
    """Load hook configs from settings dict."""
    hooks_raw = settings.get("hooks", {})
    if not isinstance(hooks_raw, dict):
        logger.warning("Invalid 'hooks' config: expected dict, got %s", type(hooks_raw).__name__)
        return {}

    result: dict[HookType, list[HookConfig]] = {}

    for type_name, hook_list in hooks_raw.items():
        # Resolve HookType from string
        try:
            hook_type = HookType(type_name)
        except ValueError:
            logger.warning("Unknown hook type '%s', skipping", type_name)
            continue

        if not isinstance(hook_list, list):
            logger.warning("Hooks for '%s' should be a list, got %s", type_name, type(hook_list).__name__)
            continue

        configs: list[HookConfig] = []
        for entry in hook_list:
            if isinstance(entry, str):
                # Shorthand: just a command string
                configs.append(HookConfig(command=entry))
            elif isinstance(entry, dict):
                command = entry.get("command")
                if not command:
                    logger.warning("Hook entry in '%s' missing 'command', skipping", type_name)
                    continue
                configs.append(HookConfig(
                    command=command,
                    tools=entry.get("tools"),
                    timeout=entry.get("timeout", HOOK_TIMEOUT_SECONDS),
                    shell=bool(entry.get("shell", False)),
                ))
            else:
                logger.warning("Invalid hook entry in '%s': %r", type_name, entry)

        if configs:
            result[hook_type] = configs

    return result


async def execute_hook(
    hook_type: HookType,
    hook: HookConfig,
    payload: dict[str, Any],
) -> HookResult:
    """Execute a single hook command.

    Sends payload as JSON on stdin.
    Parses stdout as JSON for the result (PreToolUse).
    Respects timeout.
    """
    import shlex

    result = HookResult(hook_name=hook.command)
    payload_bytes = json.dumps(payload).encode()
    # On failure of a PreToolUse hook, fall back to ASK (surface to the user)
    # instead of ALLOW. A crashing safety-critical hook must not be
    # indistinguishable from a successful allow.
    safe_failure_action = "ask" if hook_type == HookType.PRE_TOOL_USE else "allow"

    try:
        if hook.shell:
            # Opt-in shell execution. Documented as the risky path.
            proc = await asyncio.create_subprocess_shell(
                hook.command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        else:
            # Default: argv-style exec. No shell metachar interpretation.
            try:
                argv = shlex.split(hook.command)
            except ValueError as exc:
                logger.warning(
                    "Hook '%s' failed to tokenize: %s. Set 'shell: true' if "
                    "shell interpretation is intended.",
                    hook.command, exc,
                )
                result.exit_code = -1
                result.stderr = f"Tokenization failed: {exc}"
                result.action = safe_failure_action
                return result
            if not argv:
                result.exit_code = -1
                result.stderr = "Empty command"
                result.action = safe_failure_action
                return result
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(input=payload_bytes),
                timeout=hook.timeout,
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            logger.warning("Hook '%s' timed out after %ds", hook.command, hook.timeout)
            result.exit_code = -1
            result.stderr = f"Hook timed out after {hook.timeout}s"
            result.action = safe_failure_action
            return result

        result.exit_code = proc.returncode or 0
        result.stdout = stdout_bytes.decode(errors="replace").strip()
        result.stderr = stderr_bytes.decode(errors="replace").strip()

        if result.stderr:
            logger.debug("Hook '%s' stderr: %s", hook.command, result.stderr)

        # Parse stdout as JSON for PreToolUse hooks
        if hook_type == HookType.PRE_TOOL_USE and result.stdout:
            try:
                data = json.loads(result.stdout)
                if isinstance(data, dict):
                    result.action = data.get("action", "allow")
                    result.message = data.get("message", "")
            except json.JSONDecodeError:
                logger.debug(
                    "Hook '%s' stdout is not valid JSON, treating as allow: %s",
                    hook.command, result.stdout[:200],
                )

        # Non-zero exit code on PreToolUse => deny
        if hook_type == HookType.PRE_TOOL_USE and result.exit_code != 0 and result.action == "allow":
            result.action = "deny"
            if not result.message:
                result.message = f"Hook '{hook.command}' exited with code {result.exit_code}"

    except Exception as exc:
        logger.warning("Failed to execute hook '%s': %s", hook.command, exc)
        result.exit_code = -1
        result.stderr = str(exc)
        result.action = safe_failure_action

    return result


async def run_hooks(
    hook_type: HookType,
    payload: dict[str, Any],
    settings: dict[str, Any] | None = None,
    tool_name: str | None = None,
) -> list[HookResult]:
    """Run all hooks of a given type.

    If tool_name is provided, only hooks matching that tool are run.
    Returns list of results.
    """
    if settings is None:
        settings = {}

    hooks_by_type = load_hooks_from_settings(settings)
    hooks = hooks_by_type.get(hook_type, [])

    if not hooks:
        return []

    # Filter by tool_name if specified
    if tool_name is not None:
        hooks = [
            h for h in hooks
            if h.tools is None or tool_name in h.tools
        ]

    if not hooks:
        return []

    results: list[HookResult] = []
    for hook in hooks:
        result = await execute_hook(hook_type, hook, payload)
        results.append(result)

    return results


async def run_pre_tool_hooks(
    tool_name: str,
    tool_input: dict[str, Any],
    settings: dict[str, Any] | None = None,
) -> HookResult | None:
    """Run PreToolUse hooks. Returns first deny/ask result, or None if all allow."""
    payload = {
        "hook_type": HookType.PRE_TOOL_USE.value,
        "tool_name": tool_name,
        "tool_input": tool_input,
    }

    results = await run_hooks(
        HookType.PRE_TOOL_USE,
        payload,
        settings=settings,
        tool_name=tool_name,
    )

    for result in results:
        if result.action in ("deny", "ask"):
            return result

    return None


async def run_post_tool_hooks(
    tool_name: str,
    tool_input: dict[str, Any],
    tool_output: str,
    is_error: bool = False,
    settings: dict[str, Any] | None = None,
) -> None:
    """Run PostToolUse hooks (fire-and-forget)."""
    hook_type = HookType.POST_TOOL_USE_FAILURE if is_error else HookType.POST_TOOL_USE

    payload = {
        "hook_type": hook_type.value,
        "tool_name": tool_name,
        "tool_input": tool_input,
        "tool_output": tool_output,
        "is_error": is_error,
    }

    # Also run the general PostToolUse hooks even on failure
    await run_hooks(hook_type, payload, settings=settings, tool_name=tool_name)
    if is_error:
        await run_hooks(HookType.POST_TOOL_USE, payload, settings=settings, tool_name=tool_name)
