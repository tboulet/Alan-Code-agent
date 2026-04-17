"""LiteLLM provider — supports 100+ LLM providers through a unified API.

Works with OpenRouter, OpenAI, Anthropic, local models (Ollama, vLLM),
and any provider supported by litellm.

Usage::

    from alancode.providers.litellm_provider import LiteLLMProvider

    # OpenRouter (free model)
    provider = LiteLLMProvider(model="openrouter/mistralai/devstral-2512:free")

    # OpenRouter (paid model, needs OPENROUTER_API_KEY env var)
    provider = LiteLLMProvider(model="openrouter/anthropic/claude-sonnet-4")

    # Local Ollama
    provider = LiteLLMProvider(model="ollama/llama3.1")

    # OpenAI
    provider = LiteLLMProvider(model="gpt-4o")
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, AsyncGenerator
from uuid import uuid4

from alancode.providers.base import (
    LLMProvider,
    ModelInfo,
    ProviderStreamEvent,
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

logger = logging.getLogger(__name__)


# Suppress litellm's verbose debug logging
def _quiet_litellm() -> None:
    try:
        import litellm
        litellm.suppress_debug_info = True
        litellm.print_verbose = lambda *args, **kwargs: None
        logging.getLogger("LiteLLM").setLevel(logging.WARNING)
        logging.getLogger("LiteLLM Router").setLevel(logging.WARNING)
        logging.getLogger("LiteLLM Proxy").setLevel(logging.WARNING)
    except ImportError:
        pass


_quiet_litellm()

_CACHE_MARKER = {"type": "ephemeral"}


def _is_anthropic_model(model: str) -> bool:
    """Check if a model string routes to Anthropic's API via LiteLLM."""
    m = model.lower()
    return (
        m.startswith("anthropic/")
        or m.startswith("openrouter/anthropic/")
        or "claude" in m
    )


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


class LiteLLMProvider(LLMProvider):
    """LLM provider using litellm for multi-provider support.

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
        extra_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        self._model = model
        self._api_key = api_key
        self._api_base = api_base
        self._context_window_override = context_window
        self._max_output_override = max_output_tokens
        self._extra_kwargs = extra_kwargs or {}

    def get_model_info(self, model: str | None = None) -> ModelInfo:
        """Get model capabilities.

        Resolution order:
        1. Constructor overrides (context_window, max_output_tokens)
        2. litellm model registry
        3. Our fallback known-models table
        4. Safe defaults
        """
        m = model or self._model
        ctx = self._context_window_override
        max_out = self._max_output_override
        supports_thinking = False

        # Try litellm's registry (covers hundreds of cloud models)
        try:
            import litellm
            info = litellm.get_model_info(m)
            if ctx is None:
                ctx = info.get("max_input_tokens") or info.get("max_tokens")
            if max_out is None:
                max_out = info.get("max_output_tokens")
            supports_thinking = info.get("supports_thinking", False)
        except Exception:
            logger.debug(f"Model '{m}' not found in litellm registry, trying server fallbacks")

        # Fallback: query the server's /v1/models endpoint (local servers)
        if ctx is None and self._api_base:
            ctx = self._query_server_context_window(m)

        # Fallback: check our known-models table for context window
        if ctx is None:
            for key, window in _KNOWN_CONTEXT_WINDOWS.items():
                if key in m.lower():
                    ctx = window
                    break

        return ModelInfo(
            context_window=ctx or 200_000,
            max_output_tokens=max_out or 8_192,
            supports_thinking=supports_thinking,
        )

    def _query_server_context_window(self, model: str) -> int | None:
        """Query a local server's /v1/models or /api/tags for context window info."""
        import requests as http_requests

        base = self._api_base.rstrip("/")

        # Try OpenAI-compatible /v1/models (vLLM, SGLang)
        for endpoint in [f"{base}/models", f"{base.rstrip('/v1')}/v1/models"]:
            try:
                resp = http_requests.get(endpoint, timeout=5)
                if resp.status_code == 200:
                    data = resp.json()
                    for m_info in data.get("data", []):
                        max_len = m_info.get("max_model_len")
                        if max_len:
                            logger.info("Got context window %d from server %s", max_len, endpoint)
                            return max_len
            except Exception:
                continue

        # Try Ollama /api/tags
        try:
            ollama_base = base.replace("/v1", "")
            resp = http_requests.get(f"{ollama_base}/api/tags", timeout=5)
            if resp.status_code == 200:
                for m_info in resp.json().get("models", []):
                    details = m_info.get("details", {})
                    ctx = details.get("context_length")
                    if ctx:
                        logger.info("Got context window %d from Ollama", ctx)
                        return ctx
        except Exception:
            pass

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
    ) -> AsyncGenerator[ProviderStreamEvent, None]:
        """Stream from any litellm-supported provider."""
        try:
            import litellm
        except ImportError:
            yield StreamError(
                error="litellm is not installed. Run: pip install litellm",
                error_type="configuration_error",
            )
            return

        resolved_model = model or self._model
        info = self.get_model_info(resolved_model)
        resolved_max_tokens = max_tokens or info.max_output_tokens
        static_boundary = kwargs.pop("system_static_boundary", None)
        use_caching = static_boundary is not None and _is_anthropic_model(resolved_model)

        # Build system message (litellm uses the messages array, not a separate system param)
        litellm_messages: list[dict[str, Any]] = []
        if system:
            if use_caching:
                # Structured content blocks with cache_control for Anthropic
                content_blocks: list[dict[str, Any]] = []
                for i, s in enumerate(system):
                    if not s:
                        continue
                    block: dict[str, Any] = {"type": "text", "text": s}
                    # BP2: last static section
                    if i == static_boundary - 1 and static_boundary > 0:
                        block["cache_control"] = _CACHE_MARKER
                    content_blocks.append(block)
                # BP3: last system section (if different from BP2)
                if content_blocks:
                    last_idx = len([s for s in system if s]) - 1
                    if static_boundary <= 0 or last_idx != static_boundary - 1:
                        content_blocks[-1]["cache_control"] = _CACHE_MARKER
                litellm_messages.append({"role": "system", "content": content_blocks})
            else:
                system_text = "\n\n".join(s for s in system if s)
                if system_text:
                    litellm_messages.append({"role": "system", "content": system_text})

        # Messages arrive in OpenAI format from the query loop — pass through.
        litellm_messages.extend(messages)

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

        # BP1: last tool definition + BP4: last assistant message
        if use_caching:
            if litellm_tools:
                litellm_tools[-1]["cache_control"] = _CACHE_MARKER
            for msg in reversed(litellm_messages):
                if msg.get("role") == "assistant":
                    # For LiteLLM OpenAI format, cache_control on the message dict
                    # is forwarded to Anthropic by LiteLLM
                    msg["cache_control"] = _CACHE_MARKER
                    break

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

        # OpenRouter-specific: use max_completion_tokens instead of max_tokens
        if "openrouter" in resolved_model:
            completion_kwargs["max_completion_tokens"] = completion_kwargs.pop("max_tokens")
            # Drop unsupported params
            completion_kwargs.setdefault("drop_params", True)

        # Yield message start
        request_id = str(uuid4())
        yield StreamMessageStart(model=resolved_model, request_id=request_id)

        try:
            response = await litellm.acompletion(**completion_kwargs)

            # Track state for tool calls and usage
            current_tool_calls: dict[int, dict[str, Any]] = {}  # index → {id, name, arguments_json}
            final_usage: dict[str, int] | None = None
            stop_emitted = False
            mapped_stop_reason: str | None = None

            async for chunk in response:
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta is None:
                    continue

                finish_reason = chunk.choices[0].finish_reason if chunk.choices else None

                # Text content
                if delta.content:
                    yield StreamTextDelta(text=delta.content)

                # Thinking/reasoning (some providers support this)
                reasoning = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
                if reasoning:
                    yield StreamThinkingDelta(thinking=reasoning)

                # Tool calls (OpenAI format: delta.tool_calls is a list)
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index if hasattr(tc, "index") else 0

                        if idx not in current_tool_calls:
                            # New tool call starting
                            tool_id = tc.id or f"call_{uuid4().hex[:8]}"
                            tool_name = (tc.function.name if tc.function else None) or ""
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
                            if tc.function and tc.function.name and not current_tool_calls[idx]["start_emitted"]:
                                current_tool_calls[idx]["name"] = tc.function.name
                                yield StreamToolUseStart(
                                    id=current_tool_calls[idx]["id"],
                                    name=tc.function.name,
                                )
                                current_tool_calls[idx]["start_emitted"] = True

                        # Accumulate arguments
                        if tc.function and tc.function.arguments:
                            current_tool_calls[idx]["arguments_json"] += tc.function.arguments
                            yield StreamToolUseInputDelta(
                                id=current_tool_calls[idx]["id"],
                                partial_json=tc.function.arguments,
                            )

                # Extract usage from ANY chunk that carries it.
                # With stream_options={"include_usage": True}, most providers
                # send usage on a final chunk AFTER the finish_reason chunk.
                # Usage can arrive on the same chunk as finish_reason, or separately.
                if hasattr(chunk, "usage") and chunk.usage:
                    u = chunk.usage
                    final_usage = {
                        "input_tokens": getattr(u, "prompt_tokens", 0) or 0,
                        "output_tokens": getattr(u, "completion_tokens", 0) or 0,
                    }

                # Check for finish — finalize pending tool calls
                if finish_reason and not stop_emitted:
                    for tc_info in current_tool_calls.values():
                        try:
                            parsed_input = json.loads(tc_info["arguments_json"]) if tc_info["arguments_json"] else {}
                        except json.JSONDecodeError:
                            parsed_input = {}
                        yield StreamToolUseStop(
                            id=tc_info["id"],
                            name=tc_info["name"],
                            input=parsed_input,
                        )
                    current_tool_calls.clear()
                    mapped_stop_reason = _map_finish_reason(finish_reason)
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

            # Classify common litellm exceptions
            if "AuthenticationError" in type(e).__name__ or "401" in error_str:
                error_type = "authentication_error"
            elif "RateLimitError" in type(e).__name__ or "429" in error_str:
                error_type = "rate_limit"
            elif (
                "ContextWindowExceededError" in type(e).__name__
                # Narrower phrase list — the old `"context" in …` matched
                # unrelated errors like "error in context of X" and
                # over-triggered auto-compact.
                or "context length" in error_str.lower()
                or "context_length_exceeded" in error_str.lower()
                or "maximum context" in error_str.lower()
                or "prompt is too long" in error_str.lower()
            ):
                error_type = "prompt_too_long"
            elif "Timeout" in type(e).__name__:
                error_type = "timeout"

            logger.error(f"LiteLLM error ({error_type}): {error_str}")
            yield StreamError(error=error_str, error_type=error_type)


def _map_finish_reason(reason: str | None) -> str:
    """Map provider-specific finish reasons to our standard reasons."""
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
