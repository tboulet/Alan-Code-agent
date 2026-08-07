"""LiteLLM backend — supports 100+ LLM backends through a unified API.

Works with OpenRouter, OpenAI, Anthropic, local models (Ollama, vLLM),
and any backend supported by litellm.

Usage::

    from alancode.backends.litellm_backend import LiteLLMBackend

    # OpenRouter (free model)
    backend = LiteLLMBackend(model="openrouter/mistralai/devstral-2512:free")

    # OpenRouter (paid model, needs OPENROUTER_API_KEY env var)
    backend = LiteLLMBackend(model="openrouter/anthropic/claude-sonnet-4")

    # Local Ollama
    backend = LiteLLMBackend(model="ollama/llama3.1")

    # OpenAI
    backend = LiteLLMBackend(model="gpt-4o")
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, AsyncGenerator
from urllib.parse import urlsplit
from uuid import uuid4

from alancode.api.errors import is_prompt_too_long
from alancode.backends.base import (
    LLMBackend,
    ModelInfo,
    BackendStreamEvent,
    StreamError,
    StreamMessageDelta,
    StreamMessageStart,
    StreamMessageStop,
    StreamTextDelta,
    StreamThinkingDelta,
    StreamToolUseStart,
    StreamToolUseInputDelta,
    StreamToolUseStop,
    ThinkingConfig,
    ToolSchema,
)
from alancode.backends import cw_probe

logger = logging.getLogger(__name__)


DEFAULT_LOCAL_REQUEST_TIMEOUT_SECONDS = 3_600


def _load_litellm():
    """Import LiteLLM without its startup-time network cost-map fetch."""
    # Alan does not need a fresh pricing download to make a model request.
    # Respect an explicit user override, but default to the packaged map so
    # importing Alan remains deterministic on offline compute nodes.
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    import litellm

    litellm.suppress_debug_info = True
    litellm.print_verbose = lambda *args, **kwargs: None
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    logging.getLogger("LiteLLM Router").setLevel(logging.WARNING)
    logging.getLogger("LiteLLM Proxy").setLevel(logging.WARNING)
    return litellm


def _resolve_request_timeout(
    configured: int | str | None,
    api_base: str | None,
) -> int | None:
    """Resolve ``auto`` to a slow-local-server-friendly timeout."""
    if isinstance(configured, int) and not isinstance(configured, bool):
        if configured > 0:
            return configured
        raise ValueError("request_timeout must be a positive integer or 'auto'")
    if isinstance(configured, str) and configured.lower() == "auto":
        configured = "auto"
    if configured not in (None, "auto"):
        raise ValueError("request_timeout must be a positive integer or 'auto'")
    if api_base:
        return DEFAULT_LOCAL_REQUEST_TIMEOUT_SECONDS
    return None


def _finalize_tool_calls(
    current_tool_calls: dict[int, dict[str, Any]],
) -> tuple[list[StreamToolUseStop], str | None]:
    """Validate and finalize accumulated OpenAI streaming tool calls."""
    events: list[StreamToolUseStop] = []
    for index in sorted(current_tool_calls):
        call = current_tool_calls[index]
        name = call["name"]
        if not name:
            return [], f"Tool call at index {index} completed without a name."
        raw_input = call["arguments_json"]
        try:
            parsed_input = json.loads(raw_input) if raw_input else {}
        except json.JSONDecodeError as exc:
            return [], f"Tool call '{name}' returned invalid JSON arguments: {exc}."
        if not isinstance(parsed_input, dict):
            return [], f"Tool call '{name}' arguments must decode to an object."
        events.append(
            StreamToolUseStop(
                id=call["id"],
                name=name,
                input=parsed_input,
            )
        )
    return events, None


def _exception_status_code(exc: Exception) -> int | None:
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    return status if isinstance(status, int) else None

# Known context windows for common models (litellm handles most, this is fallback)
_KNOWN_CONTEXT_WINDOWS: dict[str, int] = {
    "claude-sonnet-4": 200_000,
    "claude-opus-4": 200_000,
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4.1": 1_000_000,
    "gpt-4.1-mini": 1_000_000,
    "gemini-2.5-flash": 1_000_000,
    "gemini-2.5-pro": 1_000_000,
    "devstral-2512": 128_000,
    "llama3.1": 128_000,
}


class LiteLLMBackend(LLMBackend):
    """LLM backend using litellm for multi-backend support.

    Supports any model string that litellm understands, including:
    - ``openrouter/anthropic/claude-sonnet-4``
    - ``openrouter/mistralai/devstral-2512:free``
    - ``openrouter/google/gemini-2.5-flash``
    - ``gpt-4o`` (OpenAI direct)
    - ``ollama/llama3.1`` (local)
    - ``anthropic/claude-sonnet-4`` (Anthropic direct)

    API keys are resolved from environment variables automatically by litellm
    (``OPENROUTER_API_KEY``, ``OPENAI_API_KEY``, ``ANTHROPIC_API_KEY``, etc.).
    """

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        api_base: str | None = None,
        context_window: int | None = None,
        max_output_tokens: int | None = None,
        request_timeout: int | str | None = "auto",
        extra_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        self._model = model
        self._api_key = api_key
        self._api_base = api_base
        self._context_window_override = context_window
        self._max_output_override = max_output_tokens
        self._request_timeout = _resolve_request_timeout(
            request_timeout, api_base,
        )
        self._extra_kwargs = dict(extra_kwargs or {})
        self._extra_kwargs.update(kwargs)
        self._cw_probe_attempted = False
        self._cw_fallback_warned: set[str] = set()

    def get_model_info(self, model: str | None = None) -> ModelInfo:
        """Get model capabilities.

        Context-window resolution chain (first hit wins):
        1. Constructor override        -> cw_source "override"
        2. litellm model registry      -> "registry"
        3. Server metadata endpoints   -> "server"   (vLLM/SGLang, Ollama)
        4. Known-models table          -> "known_table"
        5. Probe cache                 -> "cache"    (a past probe's result)
        6. Conservative 32k + warning  -> "fallback" (untrusted; the agent
           may trigger probe_and_cache_context_window to replace it)
        """
        m = model or self._model
        ctx = self._context_window_override
        source = "override" if ctx is not None else None
        max_out = self._max_output_override
        supports_thinking = False

        # Rung 2: litellm's registry (covers hundreds of cloud models).
        # Also the source of max_output/thinking regardless of the CW rung.
        try:
            litellm = _load_litellm()
            info = litellm.get_model_info(m)
            if ctx is None:
                ctx = info.get("max_input_tokens") or info.get("max_tokens")
                source = "registry" if ctx is not None else None
            if max_out is None:
                max_out = info.get("max_output_tokens")
            supports_thinking = info.get("supports_thinking", False)
        except Exception as exc:
            logger.debug(
                "Model '%s' not found in LiteLLM registry, trying server "
                "fallbacks: %s",
                m,
                exc,
            )

        # Rung 3: the serving endpoint's own metadata (local servers).
        if ctx is None and self._api_base:
            ctx = self._query_server_context_window(m)
            source = "server" if ctx is not None else None

        # Rung 4: known-models table.
        if ctx is None:
            for key, window in _KNOWN_CONTEXT_WINDOWS.items():
                if key in m.lower():
                    ctx = window
                    source = "known_table"
                    break

        # Rung 5: a previously probed value.
        if ctx is None:
            ctx = cw_probe.load_cached_context_window(m, self._api_base)
            source = "cache" if ctx is not None else None

        # Rung 6: conservative fallback - loudly, and only once per model.
        if ctx is None:
            ctx = 32_768
            source = "fallback"
            if m not in self._cw_fallback_warned:
                self._cw_fallback_warned.add(m)
                logger.warning(
                    "Context window of model '%s' is UNKNOWN (registry, server "
                    "metadata and cache all failed). Assuming a conservative "
                    "%d tokens. Set the 'context_window' setting if you know "
                    "the real value.", m, ctx,
                )

        return ModelInfo(
            context_window=ctx,
            max_output_tokens=max_out or 8_192,
            supports_thinking=supports_thinking,
            cw_source=source or "registry",
        )

    async def probe_and_cache_context_window(
        self, model: str | None = None,
    ) -> int | None:
        """Probe the server for the real context window and cache the result.

        Called by the agent (once per session at most) when
        ``get_model_info`` reported ``cw_source == "fallback"``. Returns the
        detected value, or None when probing failed or was distrusted -
        in which case the conservative fallback stays in effect.
        """
        if self._cw_probe_attempted:
            return None
        self._cw_probe_attempted = True

        m = model or self._model
        result = await cw_probe.probe_context_window(
            m, api_key=self._api_key, api_base=self._api_base,
        )
        if result.value:
            cw_probe.save_cached_context_window(
                m, self._api_base, result.value, result.method
            )
            logger.info(
                "Context window of '%s' probed: %d tokens (cached).",
                m, result.value,
            )
            return result.value

        logger.warning(
            "Context window probe for '%s' inconclusive (%s): %s. "
            "Keeping the conservative fallback.", m, result.method, result.detail,
        )
        return None

    def _query_server_context_window(self, model: str) -> int | None:
        """Query a local server's /v1/models or /api/tags for context window info."""
        import requests as http_requests

        base = self._api_base.rstrip("/")
        parsed = urlsplit(base)
        base_path = parsed.path.rstrip("/")
        if base_path.endswith("/v1"):
            root_path = base_path[:-3]
            root = parsed._replace(path=root_path, query="", fragment="").geturl().rstrip("/")
            openai_endpoints = [f"{base}/models"]
        else:
            root = base
            openai_endpoints = [f"{base}/v1/models", f"{base}/models"]

        # Try OpenAI-compatible /v1/models (vLLM, SGLang)
        requested_names = {model, model.split("/", 1)[-1], model.rsplit("/", 1)[-1]}
        for endpoint in openai_endpoints:
            try:
                resp = http_requests.get(endpoint, timeout=5)
                if resp.status_code == 200:
                    data = resp.json()
                    model_entries = [
                        item for item in data.get("data", [])
                        if isinstance(item, dict)
                    ]
                    matches = [
                        item for item in model_entries
                        if str(item.get("id") or item.get("name") or "")
                        in requested_names
                    ]
                    candidates = matches or (
                        model_entries if len(model_entries) == 1 else []
                    )
                    for m_info in candidates:
                        max_len = (
                            m_info.get("max_model_len")
                            or m_info.get("context_window")
                            or m_info.get("max_context_length")
                        )
                        if max_len:
                            value = int(max_len)
                            logger.info(
                                "Got context window %d for model %s from server %s",
                                value,
                                model,
                                endpoint,
                            )
                            return value
            except (OSError, ValueError, TypeError, http_requests.RequestException) as exc:
                logger.debug(
                    "Context metadata request failed for %s: %s", endpoint, exc,
                )
                continue

        # Try Ollama /api/show (POST with the model name). Note /api/tags
        # does NOT expose context_length - it lives in /api/show's
        # model_info under "<architecture>.context_length".
        try:
            ollama_base = root
            # "ollama/llama3.1" -> the server knows it as "llama3.1"
            server_model = model.split("/", 1)[1] if "/" in model else model
            resp = http_requests.post(
                f"{ollama_base}/api/show", json={"model": server_model}, timeout=5,
            )
            if resp.status_code == 200:
                data = resp.json()
                ctx = None
                for key, value in (data.get("model_info") or {}).items():
                    if key.endswith(".context_length"):
                        ctx = int(value)
                        break
                # The SERVED context may be lower than the model max: Ollama
                # runs with num_ctx (and silently truncates beyond it). If
                # the modelfile sets one, that is the real limit.
                params_str = data.get("parameters") or ""
                num_ctx_match = re.search(r"num_ctx\s+(\d+)", params_str)
                if num_ctx_match:
                    num_ctx = int(num_ctx_match.group(1))
                    ctx = min(ctx, num_ctx) if ctx else num_ctx
                if ctx:
                    logger.info("Got context window %d from Ollama /api/show", ctx)
                    return ctx
        except (OSError, ValueError, TypeError, http_requests.RequestException) as exc:
            logger.debug("Ollama context metadata request failed: %s", exc)

        return None

    async def stream(
        self,
        messages: list[dict[str, Any]],
        system: list[str],
        tools: list[ToolSchema],
        *,
        model: str | None = None,
        max_tokens: int | None = None,
        thinking: ThinkingConfig | None = None,
        stop_sequences: list[str] | None = None,
        **kwargs,
    ) -> AsyncGenerator[BackendStreamEvent, None]:
        """Stream from any litellm-supported backend."""
        try:
            litellm = _load_litellm()
        except ImportError:
            yield StreamError(
                error="litellm is not installed. Run: pip install litellm",
                error_type="configuration_error",
            )
            return

        resolved_model = model or self._model
        info = self.get_model_info(resolved_model)
        resolved_max_tokens = max_tokens or info.max_output_tokens
        static_boundary = kwargs.pop("system_static_boundary", None) or 0

        # Build system message (litellm uses the messages array, not a separate system param)
        litellm_messages: list[dict[str, Any]] = []
        if system:
            # Use structured content blocks so we can place cache_control
            # markers. LiteLLM passes cache_control through to backends
            # that support it (Anthropic, OpenRouter/Anthropic) and strips
            # it for backends that don't.
            blocks: list[dict[str, Any]] = []
            for i, s in enumerate(system):
                if not s:
                    continue
                block: dict[str, Any] = {"type": "text", "text": s}
                if i == static_boundary - 1 and static_boundary > 0:
                    block["cache_control"] = {"type": "ephemeral"}
                blocks.append(block)
            if blocks:
                blocks[-1]["cache_control"] = {"type": "ephemeral"}
                litellm_messages.append({"role": "system", "content": blocks})

        # Messages arrive in OpenAI format from the query loop — pass through.
        litellm_messages.extend(messages)

        # Prompt caching: mark last assistant message so the conversation
        # prefix is cached between consecutive API calls.
        for msg in reversed(litellm_messages):
            if msg.get("role") == "assistant":
                msg["cache_control"] = {"type": "ephemeral"}
                break

        # Build tools in OpenAI format (litellm uses OpenAI tool format)
        litellm_tools = None
        if tools:
            litellm_tools = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.input_schema,
                    },
                }
                for t in tools
            ]
            litellm_tools[-1]["cache_control"] = {"type": "ephemeral"}

        # Build completion kwargs
        completion_kwargs: dict[str, Any] = {
            "model": resolved_model,
            "messages": litellm_messages,
            "max_tokens": resolved_max_tokens,
            "stream": True,
            # Request usage stats in the stream (arrives as a final chunk)
            "stream_options": {"include_usage": True},
            **self._extra_kwargs,
            **kwargs,
        }
        if litellm_tools:
            completion_kwargs["tools"] = litellm_tools
        if stop_sequences:
            completion_kwargs["stop"] = stop_sequences
        if self._api_key:
            completion_kwargs["api_key"] = self._api_key
        if self._api_base:
            completion_kwargs["api_base"] = self._api_base
        if self._request_timeout is not None:
            completion_kwargs["timeout"] = self._request_timeout

        # OpenRouter-specific: use max_completion_tokens instead of max_tokens
        if "openrouter" in resolved_model:
            completion_kwargs["max_completion_tokens"] = completion_kwargs.pop("max_tokens")
            # Drop unsupported params
            completion_kwargs.setdefault("drop_params", True)

        request_id = str(uuid4())

        try:
            response = await litellm.acompletion(**completion_kwargs)
            # Do not announce a started message until the request has
            # successfully returned a stream. This keeps pre-stream failures
            # safe to retry without exposing partial response state.
            yield StreamMessageStart(model=resolved_model, request_id=request_id)

            # Track state for tool calls and usage
            current_tool_calls: dict[int, dict[str, Any]] = {}  # index → {id, name, arguments_json}
            final_usage: dict[str, int] | None = None
            stop_emitted = False
            mapped_stop_reason: str | None = None

            async for chunk in response:
                # Extract usage from ANY chunk — including the final
                # usage-only chunk that has no choices (empty list).
                # Must be checked BEFORE the choices guard below.
                if hasattr(chunk, "usage") and chunk.usage:
                    u = chunk.usage
                    cache_write = getattr(u, "cache_creation_input_tokens", 0) or 0
                    cache_read = getattr(u, "cache_read_input_tokens", 0) or 0
                    details = getattr(u, "prompt_tokens_details", None)
                    if details and (not cache_write and not cache_read):
                        cache_read = getattr(details, "cached_tokens", 0) or 0
                        cache_write = getattr(details, "cache_write_tokens", 0) or 0
                    final_usage = {
                        "input_tokens": getattr(u, "prompt_tokens", 0) or 0,
                        "output_tokens": getattr(u, "completion_tokens", 0) or 0,
                        "cache_creation_input_tokens": cache_write,
                        "cache_read_input_tokens": cache_read,
                    }

                delta = chunk.choices[0].delta if chunk.choices else None
                if delta is None:
                    continue

                finish_reason = chunk.choices[0].finish_reason if chunk.choices else None

                # Text content
                if delta.content:
                    yield StreamTextDelta(text=delta.content)

                # Thinking/reasoning (some backends support this)
                reasoning = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
                if reasoning:
                    yield StreamThinkingDelta(thinking=reasoning)

                # Tool calls (OpenAI format: delta.tool_calls is a list)
                tool_call_deltas = getattr(delta, "tool_calls", None)
                if tool_call_deltas:
                    for position, tc in enumerate(tool_call_deltas):
                        raw_index = getattr(tc, "index", None)
                        idx = raw_index if isinstance(raw_index, int) else position
                        function = getattr(tc, "function", None)

                        if idx not in current_tool_calls:
                            # New tool call starting
                            tool_id = getattr(tc, "id", None) or f"call_{uuid4().hex[:8]}"
                            tool_name = getattr(function, "name", None) or ""
                            current_tool_calls[idx] = {
                                "id": tool_id,
                                "name": tool_name,
                                "arguments_json": "",
                                "start_emitted": False,
                            }
                            if tool_name:
                                yield StreamToolUseStart(id=tool_id, name=tool_name)
                                current_tool_calls[idx]["start_emitted"] = True
                        else:
                            # Update name if we get it later
                            tool_name = getattr(function, "name", None)
                            if tool_name and not current_tool_calls[idx]["start_emitted"]:
                                current_tool_calls[idx]["name"] = tool_name
                                yield StreamToolUseStart(
                                    id=current_tool_calls[idx]["id"],
                                    name=tool_name,
                                )
                                current_tool_calls[idx]["start_emitted"] = True

                        # Accumulate arguments
                        arguments = getattr(function, "arguments", None)
                        if arguments:
                            if not isinstance(arguments, str):
                                arguments = json.dumps(arguments)
                            current_tool_calls[idx]["arguments_json"] += arguments
                            yield StreamToolUseInputDelta(
                                id=current_tool_calls[idx]["id"],
                                partial_json=arguments,
                            )

                # Check for finish — finalize pending tool calls
                if finish_reason and not stop_emitted:
                    final_events, tool_error = _finalize_tool_calls(
                        current_tool_calls,
                    )
                    if tool_error:
                        yield StreamError(
                            error=tool_error,
                            error_type="invalid_tool_call",
                        )
                        return
                    for final_event in final_events:
                        yield final_event
                    current_tool_calls.clear()
                    mapped_stop_reason = _map_finish_reason(finish_reason)
                    stop_emitted = True

            # Some OpenAI-compatible servers close the stream without a
            # finish_reason. Preserve complete accumulated tool calls instead
            # of silently dropping them.
            if current_tool_calls and not stop_emitted:
                final_events, tool_error = _finalize_tool_calls(
                    current_tool_calls,
                )
                if tool_error:
                    yield StreamError(
                        error=tool_error,
                        error_type="invalid_tool_call",
                    )
                    return
                for final_event in final_events:
                    yield final_event
                mapped_stop_reason = "tool_use"
                stop_emitted = True

            # Emit final delta with stop reason and usage AFTER the loop,
            # so we capture usage regardless of chunk ordering.
            if stop_emitted:
                yield StreamMessageDelta(
                    stop_reason=mapped_stop_reason,
                    usage=final_usage,
                )
            yield StreamMessageStop()

        except Exception as e:
            error_str = str(e)
            error_type = "api_error"
            status_code = _exception_status_code(e)

            # Classify common litellm exceptions
            if "AuthenticationError" in type(e).__name__ or "401" in error_str:
                error_type = "authentication_error"
            elif (
                "RateLimitError" in type(e).__name__
                or status_code == 429
                or "429" in error_str
            ):
                error_type = "rate_limit"
            elif (
                "ContextWindowExceededError" in type(e).__name__
                # Centralised matcher — covers OpenAI, vLLM, SGLang, TGI,
                # Anthropic, Ollama, Mistral and any other backend whose
                # context-overflow error phrasing we have seen in the wild.
                # Edit the pattern list in alancode/api/errors.py, not here.
                or is_prompt_too_long(error_str)
            ):
                error_type = "prompt_too_long"
            elif "Timeout" in type(e).__name__:
                error_type = "timeout"
            elif "Connection" in type(e).__name__:
                error_type = "connection_error"
            elif status_code is not None and 500 <= status_code <= 599:
                error_type = "server_error"

            logger.error(
                "LiteLLM error (identified as %s, status=%s): %s",
                error_type,
                status_code,
                error_str,
            )
            yield StreamError(
                error=error_str,
                error_type=error_type,
                status_code=status_code,
            )


def _map_finish_reason(reason: str | None) -> str:
    """Map backend-specific finish reasons to our standard reasons."""
    if reason is None:
        return "end_turn"
    mapping = {
        "stop": "end_turn",
        "end_turn": "end_turn",
        "tool_calls": "tool_use",
        "tool_use": "tool_use",
        "length": "max_tokens",
        "max_tokens": "max_tokens",
        "content_filter": "content_filter",
    }
    return mapping.get(reason, reason)


# ---------------------------------------------------------------------------
# Anthropic → OpenAI message format translation
