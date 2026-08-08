"""AlanCodeAgent — the main public interface for Alan Code.

Query API (2x2 matrix)::

    agent = AlanCodeAgent(model="openrouter/google/gemini-2.5-flash")

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
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, AsyncGenerator, Callable
from uuid import uuid4

from alancode.api.cost_tracker import CostTracker
from alancode.budget import resolve_context_budget
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
from alancode.messages.factory import create_system_message, create_user_message
from alancode.messages.types import (
    AssistantMessage,
    AttachmentMessage,
    Message,
    StreamEvent,
    SystemMessage,
    Usage,
    UserMessage,
)
from alancode.permissions.context import (
    PermissionBehavior,
    PermissionMode,
    PermissionResult,
    PermissionRule,
    ToolPermissionContext,
)
from alancode.permissions.pipeline import check_permissions
from alancode.permissions.project_rules import (
    add_project_allow_rule,
    load_project_allow_rules,
)
from alancode.prompt.system_prompt import get_system_prompt
from alancode.backends.base import LLMBackend
from alancode.session.state import SessionState
from alancode.session.session import (
    load_session_settings,
    save_session_settings,
)
from alancode.session.transcript import (
    append_transcript_message,
    load_transcript,
    record_transcript,
)
from alancode.hooks.handlers import on_session_start, on_session_end
from alancode.query.loop import QueryParams, query_loop
from alancode.settings import (
    BACKEND_SETTINGS,
    SETTINGS_DEFAULTS,
    infer_backend,
    get_settings_path,
    load_projects_settings_and_maybe_init,
    validate_setting,
    load_settings,
    save_settings,
)
from alancode.skills.registry import SkillRegistry
from alancode.skills.tool_filter import filter_tools_for_skill
from alancode.tools.base import ToolUseContext
from alancode.tools.builtin.skill_tool import SkillTool
from alancode.tools.registry import get_enabled_tools, get_programmatic_tool_set
from alancode.tools.text_tool_parser import get_tool_format_system_prompt
from alancode.utils.atomic_io import interprocess_lock

logger = logging.getLogger(__name__)


def _close_backend_soon(backend: LLMBackend) -> None:
    """Close a replaced backend from synchronous settings APIs."""
    async def close_and_log() -> None:
        try:
            await backend.close()
        except Exception:
            logger.warning("Failed to close replaced backend", exc_info=True)

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(close_and_log())
    else:
        loop.create_task(close_and_log())


class AgentState(str, Enum):
    """Lifecycle state of the agent."""

    WAITING = "waiting"
    RUNNING = "running"
    ERROR = "error"


# ── Backend resolution ──────────────────────────────────────────────────────


def _resolve_backend(
    backend: str | LLMBackend,
    *,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    request_timeout: int | str | None = None,
    **kwargs: Any,
) -> LLMBackend:
    """Resolve a backend string (or pre-built ``LLMBackend``) into an
    ``LLMBackend`` instance the agent can stream against.

    If *backend* is already an ``LLMBackend``, return it unchanged — this
    is the escape hatch for users who want to wire their own transport.
    Otherwise, look up the backend name in the registry:

    - ``"auto"``             → ``LiteLLMBackend`` (universal, prefix-routed).
    - ``"anthropic-native"`` → ``AnthropicBackend`` (direct SDK).
    - ``"scripted"``         → ``ScriptedBackend`` (tests).
    """
    if isinstance(backend, LLMBackend):
        return backend

    if model is None:
        raise ValueError(
            "No model configured. Set a model via:\n"
            "  - CLI: alancode --model <model_name>\n"
            "  - Settings: /settings-project model=<model_name>\n"
            "  - Constructor: AlanCodeAgent(model='<model_name>')"
        )

    name = backend.lower() if isinstance(backend, str) else backend

    if name == "auto":
        from alancode.backends.litellm_backend import LiteLLMBackend

        return LiteLLMBackend(
            model=model,
            api_key=api_key,
            api_base=base_url,
            request_timeout=request_timeout,
            **kwargs,
        )

    if name == "anthropic-native":
        from alancode.backends.anthropic_backend import AnthropicBackend

        return AnthropicBackend(
            api_key=api_key,
            model=model,
            base_url=base_url,
            request_timeout=request_timeout,
            **kwargs,
        )

    if name == "scripted":
        # ``model="remote"`` selects the HTTP-driven impersonation backend;
        # any other model name (or None) uses the in-memory ScriptedBackend.
        if isinstance(model, str) and model.lower() == "remote":
            from alancode.backends.remote_scripted_backend import (
                RemoteScriptedBackend,
            )
            return RemoteScriptedBackend(**kwargs)
        from alancode.backends.scripted_backend import ScriptedBackend

        return ScriptedBackend(**kwargs)

    raise ValueError(
        f"Unknown backend '{backend}'. "
        f"Supported: 'auto', 'anthropic-native', 'scripted', "
        f"or pass an LLMBackend instance."
    )


def _create_backend_from_settings(settings: dict[str, Any], **extra) -> LLMBackend:
    """Create the ``LLMBackend`` instance described by *settings*.

    Used by ``__init__`` and by ``update_session_setting`` when a
    backend-related key changes mid-session.
    """
    return _resolve_backend(
        settings.get("backend", "auto"),
        model=settings.get("model"),
        api_key=settings.get("api_key"),
        base_url=settings.get("base_url"),
        request_timeout=settings.get("request_timeout"),
        **extra,
    )


# ── The agent ────────────────────────────────────────────────────────────────


class AlanCodeAgent:
    """Main interface for Alan Code sessions.

    All configuration is passed directly — no separate config object needed.

    Parameters
    ----------
    backend : str or LLMBackend, optional
        Transport backend (advanced). Either a string
        (``"auto"`` — universal LiteLLM transport;
        ``"anthropic-native"`` — direct Anthropic SDK with cache_control,
        thinking, and native tool_use; ``"scripted"`` — internal/tests)
        or a pre-built ``LLMBackend`` instance. When not set, the
        backend is inferred from *model* (bare ``claude-*`` →
        ``"anthropic-native"``, anything else → ``"auto"``).
    model : str, optional
        Model to use. Accepts bare names (``"gpt-4o"``,
        ``"claude-sonnet-4-6"``) or LiteLLM-style ``provider/model``
        prefixes (``"ollama/llama3.1"``,
        ``"openrouter/google/gemini-2.5-pro"``).
    api_key : str, optional
        API key. If None, read from environment variables.
    base_url : str, optional
        Custom API endpoint, typically an OpenAI-compatible local server.
    request_timeout : int or "auto", optional
        Model request timeout. Custom endpoints use one hour in auto mode.
    context_window : int or "auto", optional
        Override the model/server context-window resolution.
    cwd : str, optional
        Working directory. Defaults to ``os.getcwd()``.
    permission_mode : str
        Permission mode: ``"yolo"``, ``"edit"``, ``"safe"``.
    max_iterations_per_turn : int, optional
        Maximum agentic iterations per turn.
    max_output_tokens : int, optional
        Max tokens per LLM response.
    custom_system_prompt : str, optional
        Replace Alan's normal system prompt.
    append_system_prompt : str, optional
        Add library/framework instructions after the normal system prompt.
    session_id : str, optional
        Explicit session ID (pre-resolved by CLI or caller). Auto-generated if None.
    ask_callback : callable, optional
        Async callback for user prompts (permission questions, tool input).
        Signature: ``async (question: str, options: list[str]) -> str``.
        If None, permission prompts default to DENY.
    verbose : bool
        Enable debug logging.
    **backend_kwargs
        Extra keyword arguments passed to the backend constructor
        (only when *backend* is a string).
    """

    def __init__(
        self,
        backend: str | LLMBackend | None = None,
        *,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        request_timeout: int | str | None = None,
        context_window: int | str | None = None,
        cwd: str | None = None,
        permission_mode: str | None = None,
        max_iterations_per_turn: int | None = None,
        max_output_tokens: int | None = None,
        memory: str | None = None,
        tool_call_format: str | None = None,
        session_id: str | None = None,
        ask_callback: Callable | None = None,
        verbose: bool = False,
        extra_tools: list | None = None,
        custom_system_prompt: str | None = None,
        append_system_prompt: str | None = None,
        gui_label: str | None = None,
        programmatic: bool = False,
        tools: list | None = None,
        disabled_tools: list[str] | None = None,
        **backend_kwargs: Any,
    ) -> None:
        self._gui_label = gui_label
        self._programmatic = programmatic
        self._closed = False
        self._session_started_at = (
            datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M")
        )

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

        # The constructor accepts an ``LLMBackend`` instance under ``backend``.
        # That instance can't be JSON-serialized into settings, so we
        # keep it aside and pass it directly to ``_resolve_backend`` later.
        backend_instance: LLMBackend | None = None
        backend_setting: str | None = None
        if isinstance(backend, LLMBackend):
            backend_instance = backend
        elif backend is not None:
            backend_setting = backend

        constructor_overrides: dict[str, Any] = {
            "backend": backend_setting,
            "model": model,
            "api_key": api_key,
            "base_url": base_url,
            "request_timeout": request_timeout,
            "context_window": context_window,
            "permission_mode": permission_mode,
            "max_iterations_per_turn": max_iterations_per_turn,
            "max_output_tokens": max_output_tokens,
            "memory": memory,
            "tool_call_format": tool_call_format,
            "custom_system_prompt": custom_system_prompt,
            "append_system_prompt": append_system_prompt,
        }
        backend_explicit = backend_setting is not None or backend_instance is not None
        for k, v in constructor_overrides.items():
            if v is not None:
                self._settings[k] = v

        # Inference: if the caller set ``model`` but not ``backend``, pick
        # the right backend for that model (bare claude-* → native; else auto).
        # Skip when an LLMBackend instance was passed — the user already
        # decided what transport to use.
        if backend_instance is None and not backend_explicit and model is not None:
            self._settings["backend"] = infer_backend(model)

        if verbose: # verbose=True should override; verbose=False (the default) should not
            self._settings["verbose"] = True

        # Resolve key fields
        if backend_instance is not None:
            self._backend = backend_instance
        else:
            self._backend = _create_backend_from_settings(self._settings, **backend_kwargs)
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

        # Optional opt-in hook: backends that want to know the session id
        # and cwd (e.g. the remote-scripted backend, which mirrors its
        # pending payload to the session directory) can implement
        # ``set_session_context(session_id, cwd)``.
        if hasattr(self._backend, "set_session_context"):
            self._backend.set_session_context(
                session_id=self._session_id, cwd=self._cwd,
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
        if tools is not None:
            base = list(tools)
        elif programmatic:
            base = get_programmatic_tool_set()
        else:
            base = get_enabled_tools()
            base.append(SkillTool(self._skill_registry))
        if disabled_tools:
            blocked = set(disabled_tools)
            base = [t for t in base if t.name not in blocked]
        if extra_tools:
            base.extend(extra_tools)
        self._tools = base
        self._custom_system_prompt = self._settings.get("custom_system_prompt")
        self._append_system_prompt = self._settings.get("append_system_prompt")
        self._abort_event = asyncio.Event()
        self._message_queue: queue.SimpleQueue[str] = queue.SimpleQueue()
        self._permission_context = ToolPermissionContext(
            mode=PermissionMode(self._permission_mode),
        )
        self._load_project_allow_rules()
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
            messages = load_transcript(session_id, cwd=self._cwd)
            if messages:
                self._messages = messages
                logger.info(
                    "Resumed session %s (%d messages)", session_id, len(messages)
                )
            # Trigger legacy allow-rules migration (older sessions stored
            # rules in state.json; accessing the property migrates them
            # to the project-level store which was already loaded above).
            _ = self._session.allow_rules

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

            agent = AlanCodeAgent(model="gemini/gemini-2.5-flash")
            answer = agent.query("What files are in this project?")
            print(answer)
        """
        return _run_async(self.query_async(message))

    def query_events(self, message: str) -> list:
        """Send a message and return the complete list of events.

        Blocks until the full turn completes. Returns every event
        (streaming deltas, tool calls, tool results, final messages).

        Example::

            events = agent.query_events("Fix the bug")
            for event in events:
                print(type(event).__name__)
        """
        async def _collect() -> list:
            return [event async for event in self.query_events_async(message)]

        return _run_async(_collect())

    async def query_async(self, message: str) -> str:
        """Send a message and return the final assistant text (async).

        Like :meth:`query` but non-blocking — for use inside async code
        (web servers, async scripts, etc.).

        Example::

            answer = await agent.query_async("Fix the bug")
            return {"answer": answer}
        """
        last_text = ""
        async for event in self.query_events_async(message):
            if isinstance(event, AssistantMessage) and not event.hide_in_api:
                last_text = event.text
        return last_text

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

        try:
            # --- context-window probe (unknown local models, one-time) ---
            # When get_model_info could not resolve the context window
            # (cw_source == "fallback"), actively probe the server once and
            # cache the result; every budget derivation depends on this
            # value being real. Best-effort: a failed probe leaves the
            # conservative fallback in effect.
            try:
                _mi = self._backend.get_model_info(self._model)
                if (
                    getattr(_mi, "cw_source", "registry") == "fallback"
                    and not isinstance(self._settings.get("context_window"), int)
                    and hasattr(self._backend, "probe_and_cache_context_window")
                    and not getattr(self._backend, "_cw_probe_attempted", False)
                ):
                    yield create_system_message(
                        "Context window unknown for this model - probing the "
                        "server (one-time, cached afterwards)..."
                    )
                    _detected = await self._backend.probe_and_cache_context_window(
                        self._model
                    )
                    if _detected:
                        yield create_system_message(
                            f"Context window detected: {_detected:,} tokens."
                        )
                    else:
                        yield create_system_message(
                            "Context window probe inconclusive - assuming "
                            "32,768 tokens. Set the 'context_window' setting "
                            "to override.",
                            level="warning",
                        )
            except Exception:
                logger.debug("CW probe skipped (non-critical)", exc_info=True)

            # --- user message ---
            user_msg = create_user_message(message)
            self._messages.append(user_msg)

            append_transcript_message(self._session.session_id, user_msg, cwd=self._cwd)

            # --- system prompt ---
            system_prompt, system_static_boundary = self.build_system_prompt()

            # --- tool context ---
            context = ToolUseContext(
                cwd=self._cwd,
                messages=self._messages,
                settings=self._settings,
                abort_signal=self._abort_event,
                ask_user_callback=self._ask_callback,
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
                    rule = PermissionRule(
                        tool_name="Bash",
                        rule_content=f"{allow_always_pattern} *",
                        behavior=PermissionBehavior.ALLOW,
                        source="project",
                    )
                    _perm_ctx.allow_rules.append(rule)
                    # Persist to project-level store (survives across sessions)
                    add_project_allow_rule({
                        "tool_name": rule.tool_name,
                        "rule_content": rule.rule_content,
                        "source": "project",
                    }, cwd=self._cwd)
                    logger.info("Added project allow rule: Bash(%s *)", allow_always_pattern)
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
                effective_tools = filter_tools_for_skill(self._tools, self._active_skill_filter)

            params = QueryParams(
                messages=self._messages,
                system_prompt=system_prompt,
                system_static_boundary=system_static_boundary,
                backend=self._backend,
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
                        append_transcript_message(
                            self._session.session_id, event, cwd=self._cwd
                        )
                # Notify event listeners (GUI bridge, etc.)
                for listener in self._event_listeners:
                    try:
                        await listener(event)
                    except Exception as exc:
                        logger.debug("Event listener error: %s", exc, exc_info=True)
                yield event

            # Full rewrite: reconciles out-of-band _messages mutations
            # (slash commands, reverts) that the incremental appends miss.
            record_transcript(
                self._session.session_id, self._messages, cwd=self._cwd
            )

        except GeneratorExit:
            # Generator abandoned (Ctrl+C in REPL) — save state before cleanup
            logger.info("Turn interrupted by user")
            try:
                record_transcript(
                    self._session.session_id, self._messages, cwd=self._cwd
                )
            except Exception as exc:
                logger.debug("Failed to save state on interrupt: %s", exc, exc_info=True)
        except Exception as exc:
            self._state = AgentState.ERROR
            logger.exception("Agent error: %s", exc)
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

    def build_system_prompt(self) -> tuple[list[str], int]:
        """Assemble the full system prompt exactly as sent to the API.

        The single source of truth for prompt assembly: used by
        ``query_events_async`` at the start of every turn, and by UIs for
        previews (the GUI "LLM Perspective" panel) so what is displayed
        can never drift from what is sent.

        Returns:
            (sections, static_boundary) - the list of system prompt
            sections and the index marking the end of the static
            (cacheable) prefix.
        """
        mem_dir = get_memory_dir(self._cwd)
        global_mem_dir = get_global_memory_dir()
        memory_index = load_memory_index(cwd=self._cwd)
        global_memory_index = (
            None if self._programmatic else load_global_memory_index()
        )
        memory_section_text = build_memory_section(
            self._memory_mode,
            str(mem_dir),
            memory_index,
            global_memory_dir=str(global_mem_dir),
            global_memory_index=global_memory_index,
        )
        if self._programmatic:
            global_instructions = None
            project_instructions = None
        else:
            global_instructions = load_global_project_instructions()
            project_instructions = load_project_instructions(self._cwd)
        # A custom prompt is a full replacement for Alan's prompt and project
        # context. The explicit append setting remains additive in either mode.
        inherited_parts = (
            []
            if self._custom_system_prompt is not None
            else [global_instructions, project_instructions]
        )
        append_parts = [
            p for p in (*inherited_parts, self._append_system_prompt) if p
        ]
        append_prompt = "\n\n".join(append_parts) if append_parts else None
        system_prompt, system_static_boundary = get_system_prompt(
            tools=self._tools,
            skills=self._skill_registry.list_all(),
            model=self._model,
            cwd=self._cwd,
            custom_prompt=self._custom_system_prompt,
            append_prompt=append_prompt,
            memory_section=memory_section_text,
            scratchpad_dir=str(self._scratchpad_dir),
            session_started_at=self._session_started_at,
        )

        # Text-based tool calling instructions (models without native
        # tool_use): schemas travel in the system prompt instead of the
        # API request.
        tool_call_format = self._settings.get("tool_call_format")
        if tool_call_format:
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

        return system_prompt, system_static_boundary

    async def close(self) -> None:
        """Fire SessionEnd hooks and release backend/session resources."""
        if self._closed:
            return
        self._closed = True
        try:
            await on_session_end(
                session_id=self._session.session_id,
                total_cost=self._session.total_cost_usd,
                turn_count=self._session.turn_count,
                settings=self._settings,
            )
        except Exception:
            logger.debug("SessionEnd hook error (ignored)", exc_info=True)
        try:
            await self._backend.close()
        except Exception:
            logger.warning("Backend shutdown failed", exc_info=True)
        finally:
            self._session.close()

    # ── Allow rules persistence ──────────────────────────────────────────────

    def _load_project_allow_rules(self) -> None:
        """Load project-level allow rules from ``.alan/allow_rules.json``."""
        rules = load_project_allow_rules(self._cwd)
        for rule_data in rules:
            self._permission_context.allow_rules.append(
                PermissionRule(
                    tool_name=rule_data["tool_name"],
                    rule_content=rule_data.get("rule_content"),
                    behavior=PermissionBehavior.ALLOW,
                    source="project",
                )
            )
        if rules:
            logger.info("Loaded %d project allow rules", len(rules))

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

    @property
    def context_window(self) -> int:
        """Resolved context window used by the current model and settings."""
        model_info = self._backend.get_model_info(self._model)
        return resolve_context_budget(model_info, self._settings).context_window

    def update_session_setting(self, key: str, value: Any) -> str | None:
        """Validate, update a setting for this session in-memory + on disk.

        All settings can be changed mid-session. Backend-related settings
        (``backend``, ``model``, ``api_key``, ``base_url``) trigger a
        fresh ``LLMBackend`` instance. All others take effect on the
        next turn.

        Updating ``model`` alone also re-infers ``backend`` (a bare Claude
        name flips the backend to ``anthropic-native``; anything else flips
        to ``auto``). Pass ``backend`` explicitly to override the
        inference.

        Returns an error message string if validation fails, or None on success.
        """
        if key not in SETTINGS_DEFAULTS:
            return f"Unknown setting '{key}'."

        error = validate_setting(key, value)
        if error:
            return error

        candidate_settings = dict(self._settings)
        candidate_settings[key] = value

        # Re-infer the backend when only the model changed. The new
        # backend may be the same as the old one (in which case this is
        # a no-op), or it may flip — e.g. switching from gpt-4o to
        # claude-sonnet-4-6 promotes auto → anthropic-native.
        if key == "model":
            inferred = infer_backend(value)
            if inferred != candidate_settings.get("backend"):
                candidate_settings["backend"] = inferred

        # Recreate the underlying LLMBackend if a backend-related setting changed
        candidate_backend: LLMBackend | None = None
        if key in BACKEND_SETTINGS:
            try:
                candidate_backend = _create_backend_from_settings(candidate_settings)
                if hasattr(candidate_backend, "set_session_context"):
                    candidate_backend.set_session_context(
                        session_id=self._session_id,
                        cwd=self._cwd,
                    )
            except Exception as e:
                if candidate_backend is not None:
                    _close_backend_soon(candidate_backend)
                return f"Failed to create backend: {e}"

        old_backend = self._backend
        self._settings = candidate_settings

        # Sync the corresponding self._* field only after backend creation
        # succeeds, so failed updates cannot leave a half-applied session.
        field_map = {
            "model": "_model",
            "permission_mode": "_permission_mode",
            "max_iterations_per_turn": "_max_iterations_per_turn",
            "max_output_tokens": "_max_output_tokens",
            "memory": "_memory_mode",
            "verbose": "_verbose",
            "custom_system_prompt": "_custom_system_prompt",
            "append_system_prompt": "_append_system_prompt",
        }
        attr = field_map.get(key)
        if attr:
            setattr(self, attr, value)

        if candidate_backend is not None:
            self._backend = candidate_backend
            _close_backend_soon(old_backend)
            logger.info(
                "Backend recreated: %s / %s",
                self._settings.get("backend"),
                self._settings.get("model"),
            )

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

        with interprocess_lock(get_settings_path(self._cwd)):
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
