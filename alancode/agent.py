"""AlanCodeAgent — the main public interface for Alan Code.

Query API (2x2 matrix)::

    agent = AlanCodeAgent(provider="litellm", model="openrouter/google/gemini-2.5-flash")

    answer = agent.query("Fix the bug")                  # sync, text
    events = agent.query_events("Fix the bug")            # sync, events list
    answer = await agent.query_async("Fix the bug")       # async, text
    async for e in agent.query_events_async("Fix bug"):   # async, event stream
        ...
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import os
import queue
from enum import Enum
from pathlib import Path
from typing import Any, AsyncGenerator, Callable
from uuid import uuid4

from alancode.api.cost_tracker import CostTracker
from alancode.memory.memdir import (
    cleanup_old_scratchpads,
    ensure_memory_structure,
    get_global_memory_dir,
    get_memory_dir,
    load_global_memory_index,
    load_global_project_instructions,
    load_memory_index,
    load_project_instructions,
)
from alancode.memory.prompt import build_memory_section
from alancode.messages.factory import create_user_message
from alancode.messages.types import (
    AssistantMessage,
    AttachmentMessage,
    Message,
    StreamEvent,
    SystemMessage,
    Usage,
    UserMessage,
)
from alancode.permissions.context import PermissionBehavior, PermissionMode, PermissionResult, ToolPermissionContext
from alancode.permissions.pipeline import check_permissions
from alancode.prompt.system_prompt import get_system_prompt
from alancode.providers.base import LLMProvider
from alancode.session.state import SessionState
from alancode.session.session import (
    load_session_settings,
    save_session_settings,
)
from alancode.session.transcript import (
    load_transcript,
    record_transcript,
)
from alancode.hooks.handlers import on_session_start, on_session_end
from alancode.query.loop import QueryParams, query_loop
from alancode.settings import (
    SETTINGS_DEFAULTS,
    load_projects_settings_and_maybe_init,
    validate_setting,
    load_settings,
    save_settings,
)
from alancode.skills.registry import SkillRegistry
from alancode.tools.base import ToolUseContext
from alancode.tools.registry import get_enabled_tools
from alancode.tools.text_tool_parser import get_tool_format_system_prompt

logger = logging.getLogger(__name__)


class AgentState(str, Enum):
    """Lifecycle state of the agent."""

    WAITING = "waiting"
    RUNNING = "running"
    ERROR = "error"


# ── Helpers ──────────────────────────────────────────────────────────────────


def _ensure_alan_gitignored(cwd: str) -> None:
    """Ensure ``.alan/`` is listed in ``.gitignore``.

    Critical for ``git clean -fd`` safety during AGT operations.
    Without this, ``git clean`` would delete session state.
    """
    gitignore = Path(cwd) / ".gitignore"
    if gitignore.exists():
        content = gitignore.read_text()
        if ".alan" in content:
            return  # Already there
        # Append
        if not content.endswith("\n"):
            content += "\n"
        content += ".alan/\n"
        gitignore.write_text(content)
    else:
        gitignore.write_text(".alan/\n")


# ── Provider resolution ──────────────────────────────────────────────────────


def _resolve_provider(
    provider: str | LLMProvider,
    *,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    force_supports_tools: bool | None = None,
    force_supports_streaming: bool | None = None,
    force_supports_vision: bool | None = None,
    **kwargs: Any,
) -> LLMProvider:
    """Resolve a provider string or instance into an LLMProvider.

    If *provider* is already an ``LLMProvider`` instance, returns it as-is.
    If it's a string, creates the corresponding provider:

    - ``"litellm"`` → ``LiteLLMProvider``
    - ``"anthropic"`` → ``AnthropicProvider``
    - ``"scripted"`` → ``ScriptedProvider`` (for testing)
    """
    if isinstance(provider, LLMProvider):
        return provider

    if model is None:
        raise ValueError(
            "No model configured. Set a model via:\n"
            "  - CLI: alancode --model <model_name>\n"
            "  - Settings: /settings-project model=<model_name>\n"
            "  - Constructor: AlanCodeAgent(model='<model_name>')"
        )

    name = provider.lower()

    if name == "litellm":
        from alancode.providers.litellm_provider import LiteLLMProvider

        return LiteLLMProvider(
            model=model,
            api_key=api_key,
            api_base=base_url,
            force_supports_tools=force_supports_tools,
            force_supports_streaming=force_supports_streaming,
            force_supports_vision=force_supports_vision,
            **kwargs,
        )

    if name == "anthropic":
        from alancode.providers.anthropic_provider import AnthropicProvider

        return AnthropicProvider(api_key=api_key, model=model, base_url=base_url, **kwargs)

    if name == "scripted":
        from alancode.providers.scripted_provider import ScriptedProvider

        return ScriptedProvider(**kwargs)

    raise ValueError(
        f"Unknown provider '{provider}'. "
        f"Supported: 'litellm', 'anthropic', 'scripted', or pass an LLMProvider instance."
    )


def _create_provider_from_settings(settings: dict[str, Any], **extra) -> LLMProvider:
    """Create a provider from a settings dict. Used by __init__ and update_session_setting."""
    return _resolve_provider(
        settings.get("provider", "litellm"),
        model=settings.get("model"),
        api_key=settings.get("api_key"),
        base_url=settings.get("base_url"),
        force_supports_tools=settings.get("force_supports_tools"),
        force_supports_streaming=settings.get("force_supports_streaming"),
        force_supports_vision=settings.get("force_supports_vision"),
        **extra,
    )


# ── The agent ────────────────────────────────────────────────────────────────


class AlanCodeAgent:
    """Main interface for Alan Code sessions.

    All configuration is passed directly — no separate config object needed.

    Parameters
    ----------
    provider : str or LLMProvider
        LLM provider. Pass a string (``"litellm"``, ``"anthropic"``,
        ``"scripted"``) or an ``LLMProvider`` instance.
    model : str, optional
        Model to use. If None, uses the provider's default.
    api_key : str, optional
        API key. If None, read from environment variables.
    cwd : str, optional
        Working directory. Defaults to ``os.getcwd()``.
    permission_mode : str
        Permission mode: ``"yolo"``, ``"edit"``, ``"safe"``.
    max_iterations_per_turn : int, optional
        Maximum agentic iterations per turn.
    max_output_tokens : int, optional
        Max tokens per LLM response.
    session_id : str, optional
        Explicit session ID (pre-resolved by CLI or caller). Auto-generated if None.
    ask_callback : callable, optional
        Async callback for user prompts (permission questions, tool input).
        Signature: ``async (question: str, options: list[str]) -> str``.
        If None, permission prompts default to DENY.
    verbose : bool
        Enable debug logging.
    **provider_kwargs
        Extra keyword arguments passed to the provider constructor
        (only when *provider* is a string).
    """

    def __init__(
        self,
        provider: str | LLMProvider | None = None,
        *,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        cwd: str | None = None,
        permission_mode: str | None = None,
        max_iterations_per_turn: int | None = None,
        max_output_tokens: int | None = None,
        memory: str | None = None,
        tool_call_format: str | None = None,
        force_supports_tools: bool | None = None,
        force_supports_streaming: bool | None = None,
        force_supports_vision: bool | None = None,
        session_id: str | None = None,
        ask_callback: Callable | None = None,
        verbose: bool = False,
        **provider_kwargs: Any,
    ) -> None:
        self._cwd = cwd or os.getcwd()

        # Resolve session ID
        if session_id:
            self._session_id = session_id
        else:
            self._session_id = uuid4().hex

        # Load settings base (project or session) for merging with CLI overrides
        if session_id:
            settings_base = load_session_settings(self._cwd, session_id)
            if not settings_base:
                settings_base = load_projects_settings_and_maybe_init(self._cwd)
        else:
            settings_base = load_projects_settings_and_maybe_init(self._cwd)

        # Merge: defaults settings <- session settings <- constructor kwargs (non-None only)
        self._settings: dict[str, Any] = dict(SETTINGS_DEFAULTS)
        self._settings.update({k: v for k, v in settings_base.items()})

        constructor_overrides: dict[str, Any] = {
            "provider": provider,
            "model": model,
            "api_key": api_key,
            "base_url": base_url,
            "permission_mode": permission_mode,
            "max_iterations_per_turn": max_iterations_per_turn,
            "max_output_tokens": max_output_tokens,
            "memory": memory,
            "tool_call_format": tool_call_format,
            "force_supports_tools": force_supports_tools,
            "force_supports_streaming": force_supports_streaming,
            "force_supports_vision": force_supports_vision,
        }
        for k, v in constructor_overrides.items():
            if v is not None:
                self._settings[k] = v

        if verbose: # verbose=True should override; verbose=False (the default) should not
            self._settings["verbose"] = True

        # Resolve key fields
        self._provider = _create_provider_from_settings(self._settings, **provider_kwargs)
        self._model = self._settings.get("model")
        self._permission_mode = self._settings.get("permission_mode", "edit")
        self._max_iterations_per_turn = self._settings.get("max_iterations_per_turn")
        self._max_output_tokens = self._settings.get("max_output_tokens")
        self._memory_mode: str = self._settings.get("memory") or "off"
        self._verbose = self._settings.get("verbose", False)

        # Session state (disk-attached — all persistent state lives here)
        self._session = SessionState(
            session_id=self._session_id,
            cwd=self._cwd,
        )

        # Cost tracker (pricing logic, delegates totals to SessionState)
        self._cost_tracker = CostTracker(session=self._session)

        # Last completed API call's usage. Used for the display's
        # "Conversation: N / M" figure (authoritative post-turn) and as the
        # floor in the pre-call compaction estimate. Reset on /clear.
        # Seeded from persisted SessionState when resuming a session so
        # the first post-resume turn has a usage-based floor.
        self._last_usage = Usage(
            input_tokens=self._session.last_input_tokens,
            output_tokens=self._session.last_output_tokens,
            cache_read_input_tokens=self._session.last_cache_read_tokens,
            cache_creation_input_tokens=self._session.last_cache_write_tokens,
        )

        # Event listeners (for FrontendBridge / GUI integration)
        self._event_listeners: list[Callable] = []
        # LLM perspective callback (set by GUI bridge to receive api_messages snapshots)
        self._llm_perspective_callback: Callable | None = None

        # Skills
        self._skill_registry = SkillRegistry(self._cwd)

        # Tools, abort, message queue
        self._state = AgentState.WAITING
        self._messages: list[Message] = []
        self._tools = get_enabled_tools()
        # Append the Skill tool (needs registry reference)
        from alancode.tools.builtin.skill_tool import SkillTool
        self._tools.append(SkillTool(self._skill_registry))
        self._abort_event = asyncio.Event()
        self._message_queue: queue.SimpleQueue[str] = queue.SimpleQueue()
        self._permission_context = ToolPermissionContext(
            mode=PermissionMode(self._permission_mode),
        )
        self._session_start_fired = False
        self._ask_callback = ask_callback
        # Active skill tool filter (set by /skill command, cleared after turn)
        self._active_skill_filter: list[str] | None = None

        # Save session settings snapshot
        save_session_settings(self._cwd, self._session_id, self._settings)

        # Memory and scratchpad setup
        if self._memory_mode != "off":
            ensure_memory_structure(self._cwd)

        self._scratchpad_dir = (
            Path(self._cwd) / ".alan" / "sessions" / self._session_id / "scratchpad"
        )
        self._scratchpad_dir.mkdir(parents=True, exist_ok=True)

        max_scratch = self._settings.get("max_scratchpad_sessions", 5)
        cleanup_old_scratchpads(self._cwd, max_sessions=max_scratch)

        # Load transcript from previous session if resuming
        if session_id:
            messages = _run_async_safe(load_transcript(session_id, cwd=self._cwd))
            if messages:
                self._messages = messages
                logger.info(
                    "Resumed session %s (%d messages)", session_id, len(messages)
                )
            # Restore permission allow rules from session state
            self._restore_allow_rules()

    # ── Query API (2x2 matrix: text/events × sync/async) ───────────────────
    #
    #   |            | sync (default)       | async                         |
    #   |------------|----------------------|-------------------------------|
    #   | text       | query(msg) → str     | query_async(msg) → str        |
    #   | events     | query_events(msg)    | query_events_async(msg)       |
    #   |            | → list[Event]        | → AsyncGenerator[Event]       |

    def query(self, message: str) -> str:
        """Send a message and return the final assistant text.

        This is the simplest way to use Alan Code. Blocks until the full
        turn completes (including tool execution).

        Example::

            agent = AlanCodeAgent(provider="litellm", model="gemini-2.5-flash")
            answer = agent.query("What files are in this project?")
            print(answer)
        """
        return _run_async(self._query_text_async(message))

    def query_events(self, message: str) -> list:
        """Send a message and return the complete list of events.

        Blocks until the full turn completes. Returns every event
        (streaming deltas, tool calls, tool results, final messages).

        Example::

            events = agent.query_events("Fix the bug")
            for event in events:
                print(type(event).__name__)
        """
        return _run_async(self._query_events_collect_async(message))

    async def query_async(self, message: str) -> str:
        """Send a message and return the final assistant text (async).

        Like :meth:`query` but non-blocking — for use inside async code
        (web servers, async scripts, etc.).

        Example::

            answer = await agent.query_async("Fix the bug")
            return {"answer": answer}
        """
        return await self._query_text_async(message)

    async def query_events_async(
        self, message: str
    ) -> AsyncGenerator[StreamEvent | Message, None]:
        """Send a message and yield events as they stream (async generator).

        For real-time streaming to a UI, WebSocket, or custom handler.

        Example::

            async for event in agent.query_events_async("Fix the bug"):
                send_to_websocket(event)
        """
        if self._state == AgentState.RUNNING:
            raise RuntimeError(
                "Agent is already running. Use inject_message() to inject "
                "a message into the active loop."
            )

        self._state = AgentState.RUNNING
        self._abort_event.clear()

        # Fire SessionStart hook once
        if not self._session_start_fired:
            self._session_start_fired = True
            try:
                await on_session_start(
                    cwd=self._cwd,
                    session_id=self._session.session_id,
                    model=self._model,
                    settings=self._settings,
                )
            except Exception:
                logger.debug("SessionStart hook error (ignored)", exc_info=True)

            # Initialize AGT session root (once, on first turn)
            self._init_agt_root()

        try:
            # --- user message ---
            user_msg = create_user_message(message)
            self._messages.append(user_msg)

            # --- system prompt ---
            mem_dir = get_memory_dir(self._cwd)
            global_mem_dir = get_global_memory_dir()
            memory_index = load_memory_index(cwd=self._cwd)
            global_memory_index = load_global_memory_index()
            memory_section_text = build_memory_section(
                self._memory_mode,
                str(mem_dir),
                memory_index,
                global_memory_dir=str(global_mem_dir),
                global_memory_index=global_memory_index,
            )
            global_instructions = load_global_project_instructions()
            project_instructions = load_project_instructions(self._cwd)
            # Combine global + project instructions (project wins on conflicts)
            append_parts = [p for p in (global_instructions, project_instructions) if p]
            append_prompt = "\n\n".join(append_parts) if append_parts else None
            system_prompt = get_system_prompt(
                tools=self._tools,
                skills=self._skill_registry.list_all(),
                model=self._model,
                cwd=self._cwd,
                append_prompt=append_prompt,
                memory_section=memory_section_text,
                scratchpad_dir=str(self._scratchpad_dir),
            )

            # --- text-based tool calling instructions ---
            tool_call_format = self._settings.get("tool_call_format")
            model_info = self._provider.get_model_info(self._model)
            if tool_call_format and not model_info.supports_tool_use:
                tool_schemas = [
                    {
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.input_schema,
                        },
                    }
                    for t in self._tools
                    if t.is_enabled()
                ]
                system_prompt.append(
                    get_tool_format_system_prompt(tool_call_format, tool_schemas)
                )
                logger.info(
                    "Text-based tool calling enabled (format=%s, %d tools)",
                    tool_call_format, len(tool_schemas),
                )

            # --- tool context ---
            context = ToolUseContext(
                cwd=self._cwd,
                messages=self._messages,
                settings=self._settings,
                abort_signal=self._abort_event,
                ask_user_callback=self._ask_callback,
                session_state=self._session,
            )

            # --- permission callback ---
            # Wraps check_permissions with the agent's permission context.
            # Follows CC's pattern: canUseTool is built once per turn
            # and threaded through query loop -> orchestration -> execution.
            _perm_ctx = self._permission_context
            _ask_cb = self._ask_callback
            _session = self._session

            # Mutable container to pass custom message from prompt to result
            _permission_custom_message: list[str | None] = [None]

            async def _prompt_user_permission(
                tool_name: str, description: str, tool_input: dict,
            ) -> PermissionBehavior:
                """Prompt the user for permission via ask_callback."""
                if _ask_cb is None:
                    return PermissionBehavior.DENY

                # Build "Allow always" option from the command prefix
                allow_always_label = None
                allow_always_pattern = None
                if tool_name == "Bash" and "command" in tool_input:
                    cmd = tool_input["command"]
                    prefix = cmd.split()[0] if cmd.strip() else ""
                    if prefix:
                        allow_always_pattern = prefix
                        allow_always_label = f'Allow always "{prefix} *" commands'

                options = ["Allow", "Deny"]
                if allow_always_label:
                    options.append(allow_always_label)

                try:
                    answer = await _ask_cb(
                        f"Allow {tool_name}?\n{description}",
                        options,
                    )
                except asyncio.CancelledError:
                    # User hit Ctrl+C at the permission prompt — signal
                    # the whole turn to abort, then re-raise.
                    self._abort_event.set()
                    raise
                if answer == "Allow":
                    return PermissionBehavior.ALLOW
                if answer == "Deny":
                    return PermissionBehavior.DENY
                if answer == allow_always_label and allow_always_pattern:
                    # Add a session-scoped allow rule for this command prefix
                    from alancode.permissions.context import PermissionRule
                    rule = PermissionRule(
                        tool_name="Bash",
                        rule_content=f"{allow_always_pattern} *",
                        behavior=PermissionBehavior.ALLOW,
                        source="session",
                    )
                    _perm_ctx.allow_rules.append(rule)
                    # Persist to session state immediately
                    _session.add_allow_rule({
                        "tool_name": rule.tool_name,
                        "rule_content": rule.rule_content,
                        "source": rule.source,
                    })
                    logger.info("Added session allow rule: Bash(%s *)", allow_always_pattern)
                    return PermissionBehavior.ALLOW
                # Custom text — store it so the model sees the user's feedback
                _permission_custom_message[0] = answer
                return PermissionBehavior.DENY

            async def _permission_callback(
                tool, args, ctx,
            ) -> PermissionResult:
                _permission_custom_message[0] = None
                result = await check_permissions(
                    tool, args, ctx, _perm_ctx,
                    prompt_user=_prompt_user_permission,
                )
                if result.behavior == PermissionBehavior.DENY and _permission_custom_message[0]:
                    result.message = f"User response: {_permission_custom_message[0]}"
                return result

            # --- query loop ---
            # Apply skill tool filter if active
            effective_tools = self._tools
            if self._active_skill_filter is not None:
                from alancode.skills.tool_filter import filter_tools_for_skill
                effective_tools = filter_tools_for_skill(self._tools, self._active_skill_filter)

            params = QueryParams(
                messages=self._messages,
                system_prompt=system_prompt,
                provider=self._provider,
                tools=effective_tools,
                context=context,
                cost_tracker=self._cost_tracker,
                model=self._model,
                max_iterations_per_turn=self._max_iterations_per_turn,
                max_output_tokens=self._max_output_tokens,
                abort_event=self._abort_event,
                message_queue=self._message_queue,
                memory_mode=self._memory_mode,
                settings=self._settings,
                permission_callback=_permission_callback,
                last_input_tokens_seed=self._last_usage.input_tokens,
                last_output_tokens_seed=self._last_usage.output_tokens,
                llm_perspective_callback=self._llm_perspective_callback,
            )

            async for event in query_loop(params):
                # Capture the last final-assistant-message's usage so the
                # display and next-iteration pre-call estimate can use it.
                if (
                    isinstance(event, AssistantMessage)
                    and not getattr(event, "hide_in_api", False)
                    and event.usage.input_tokens > 0
                ):
                    self._last_usage = event.usage
                if isinstance(
                    event,
                    (UserMessage, AssistantMessage, SystemMessage, AttachmentMessage),
                ) and not getattr(event, "hide_in_api", False):
                    if event is not user_msg:
                        self._messages.append(event)
                # Notify event listeners (GUI bridge, etc.)
                for listener in self._event_listeners:
                    try:
                        await listener(event)
                    except Exception:
                        logger.debug("Event listener error", exc_info=True)
                yield event

            # Persist transcript
            await record_transcript(
                self._session.session_id, self._messages, cwd=self._cwd
            )

        except GeneratorExit:
            # Generator abandoned (Ctrl+C in REPL) — save state before cleanup
            logger.info("Turn interrupted by user")
            try:
                await record_transcript(
                    self._session.session_id, self._messages, cwd=self._cwd
                )
            except Exception:
                logger.debug("Failed to save state on interrupt", exc_info=True)
        except Exception:
            self._state = AgentState.ERROR
            logger.exception("Agent error")
            raise
        finally:
            self._state = AgentState.WAITING
            # Best-effort flush of turn-boundary state. Runs even under
            # cancellation because the flush is synchronous (no awaits),
            # but we still wrap in try/except so a disk error on shutdown
            # never masks the actual exception being propagated.
            try:
                with self._session.batch():
                    self._session.turn_count += 1
                    self._session.last_input_tokens = self._last_usage.input_tokens
                    self._session.last_output_tokens = self._last_usage.output_tokens
                    self._session.last_cache_read_tokens = (
                        self._last_usage.cache_read_input_tokens
                    )
                    self._session.last_cache_write_tokens = (
                        self._last_usage.cache_creation_input_tokens
                    )
            except Exception as exc:
                logger.warning("Failed to persist turn-boundary state: %s", exc)
            # Clear active skill filter after turn completes
            self._active_skill_filter = None

    async def close(self) -> None:
        """Fire SessionEnd hooks. Call once when the session is over."""
        try:
            await on_session_end(
                session_id=self._session.session_id,
                total_cost=self._session.total_cost_usd,
                turn_count=self._session.turn_count,
                settings=self._settings,
            )
        except Exception:
            logger.debug("SessionEnd hook error (ignored)", exc_info=True)

    # ── Allow rules persistence ──────────────────────────────────────────────

    def _restore_allow_rules(self) -> None:
        """Restore permission allow rules from session state."""
        from alancode.permissions.context import PermissionRule
        for rule_data in self._session.allow_rules:
            self._permission_context.allow_rules.append(
                PermissionRule(
                    tool_name=rule_data["tool_name"],
                    rule_content=rule_data.get("rule_content"),
                    behavior=PermissionBehavior.ALLOW,
                    source=rule_data.get("source", "session"),
                )
            )
        if self._session.allow_rules:
            logger.info(
                "Restored %d allow rules from session state",
                len(self._session.allow_rules),
            )

    def _init_agt_root(self) -> None:
        """Initialize AGT session root SHA (once, on session start).

        If we're in a git repo and session_root_sha is not yet set,
        record HEAD as the starting point for this session.
        Also ensures ``.alan/`` is gitignored (critical for ``git clean``
        safety during AGT move operations).
        """
        if self._session.session_root_sha:
            return  # Already initialized (resumed session)
        try:
            from alancode.utils.env import is_git_repo
            if not is_git_repo(self._cwd):
                return

            # Ensure .alan is gitignored (prevents git clean from nuking session data)
            _ensure_alan_gitignored(self._cwd)

            import subprocess
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self._cwd,
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                sha = result.stdout.strip()
                with self._session.batch():
                    self._session.session_root_sha = sha
                    self._session.agent_position_sha = sha
                    self._session.add_to_conv_path(sha)
                    self._session.record_commit_message_index(
                        sha, len(self._messages),
                    )
                logger.debug("AGT root initialized: %s", sha[:7])
        except Exception:
            logger.debug("AGT root init failed (non-critical)", exc_info=True)

    # ── Control API ────────────────────────────────────────────────────────────

    def add_event_listener(self, callback: Callable) -> None:
        """Register a callback that receives every event from query_events_async.

        Used by the FrontendBridge/GUI to observe events without consuming
        the generator.  For programmatic GUI use::

            agent = AlanCodeAgent(...)
            gui = AlanGUI(agent)  # calls add_event_listener internally
        """
        self._event_listeners.append(callback)

    def remove_event_listener(self, callback: Callable) -> None:
        """Remove a previously registered event listener."""
        if callback in self._event_listeners:
            self._event_listeners.remove(callback)

    def inject_message(self, message: str) -> None:
        """Inject a message while the agent is running.

        The message is queued and picked up on the next loop iteration.
        """
        self._message_queue.put(message)

    def abort(self) -> None:
        """Signal the agent to stop processing as soon as possible."""
        self._abort_event.set()

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def state(self) -> AgentState:
        """Current :class:`AgentState` (``WAITING``, ``RUNNING``, ``ERROR``)."""
        return self._state

    @property
    def messages(self) -> list[Message]:
        """Copy of the current conversation messages. Safe to mutate."""
        return list(self._messages)

    @property
    def usage(self) -> Usage:
        """Cumulative token usage across the whole session.

        Returns:
            :class:`Usage` with input / output / cache-creation / cache-read
            totals summed from all API calls since the session began.
        """
        s = self._session
        return Usage(
            input_tokens=s.total_input_tokens,
            output_tokens=s.total_output_tokens,
            cache_read_input_tokens=s.total_cache_read_tokens,
            cache_creation_input_tokens=s.total_cache_write_tokens,
        )

    @property
    def last_usage(self) -> Usage:
        """Usage reported by the most recent completed API call.

        Zero on a fresh session before any call has completed.
        """
        return self._last_usage

    @property
    def session_id(self) -> str:
        """Hex-encoded session ID. Used as the key in ``.alan/sessions/``."""
        return self._session.session_id

    @property
    def cost_usd(self) -> float:
        """Cumulative estimated session cost in USD.

        See :attr:`cost_unknown` — the value is ``0.0`` when pricing
        isn't available for the model.
        """
        return self._session.total_cost_usd

    @property
    def cost_unknown(self) -> bool:
        """``True`` if the model's pricing isn't in the registry.

        When ``True``, :attr:`cost_usd` is not a meaningful dollar figure
        (typically for local models or very new releases).
        """
        return self._session.cost_unknown

    @property
    def cwd(self) -> str:
        """Working directory the agent operates in."""
        return self._cwd

    @property
    def turn_count(self) -> int:
        """Number of user messages processed in this session."""
        return self._session.turn_count

    # ── Async internals ───────────────────────────────────────────────────

    async def _query_text_async(self, message: str) -> str:
        """Consume the event stream and return just the final text."""
        last_text = ""
        async for event in self.query_events_async(message):
            if isinstance(event, AssistantMessage) and not event.hide_in_api:
                last_text = event.text
        return last_text

    async def _query_events_collect_async(self, message: str) -> list:
        """Consume the event stream into a list."""
        events: list = []
        async for event in self.query_events_async(message):
            events.append(event)
        return events

    def update_session_setting(self, key: str, value: Any) -> str | None:
        """Validate, update a setting for this session in-memory + on disk.

        All settings can be changed mid-session. Provider-related settings
        trigger provider recreation. All others take effect on the next turn.

        Returns an error message string if validation fails, or None on success.
        """
        from alancode.settings import PROVIDER_SETTINGS

        if key not in SETTINGS_DEFAULTS:
            return f"Unknown setting '{key}'."

        error = validate_setting(key, value)
        if error:
            return error

        self._settings[key] = value

        # Sync the corresponding self._* field
        field_map = {
            "model": "_model",
            "permission_mode": "_permission_mode",
            "max_iterations_per_turn": "_max_iterations_per_turn",
            "max_output_tokens": "_max_output_tokens",
            "memory": "_memory_mode",
            "verbose": "_verbose",
        }
        attr = field_map.get(key)
        if attr:
            setattr(self, attr, value)

        # Recreate provider if a provider-related setting changed
        if key in PROVIDER_SETTINGS:
            try:
                self._provider = _create_provider_from_settings(self._settings)
                logger.info("Provider recreated: %s / %s",
                           self._settings.get("provider"), self._settings.get("model"))
            except Exception as e:
                return f"Failed to create provider: {e}"

        save_session_settings(self._cwd, self._session_id, self._settings)
        return None

    def update_project_setting(self, key: str, value: Any) -> str | None:
        """Validate and update a setting in the project's .alan/settings.json.

        Does NOT modify in-memory agent state — only the on-disk project defaults.

        Returns an error message string if validation fails, or None on success.
        """
        if key not in SETTINGS_DEFAULTS:
            return f"Unknown setting '{key}'."

        error = validate_setting(key, value)
        if error:
            return error

        settings = load_settings(self._cwd)
        settings[key] = value
        save_settings(settings, self._cwd)
        return None


# ── Async helpers ────────────────────────────────────────────────────────────


def _run_async(coro):
    """Run an async coroutine from synchronous code.

    Handles the case where an event loop is already running (e.g., Jupyter).
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        # Inside an existing async context (Jupyter, nested async).
        # Create a new thread to run the coroutine.
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()

    return asyncio.run(coro)


def _run_async_safe(coro):
    """Like _run_async but returns None on failure instead of raising."""
    try:
        return _run_async(coro)
    except Exception:
        logger.debug("Async operation failed (ignored)", exc_info=True)
        return None
