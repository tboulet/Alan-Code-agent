"""Anthropic provider — wraps the official anthropic Python SDK.

Translates anthropic SDK stream events into Alan Code's ProviderStreamEvent types.
"""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncGenerator

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
    StreamToolUseInputDelta,
    StreamToolUseStart,
    StreamToolUseStop,
    ThinkingConfig,
    ToolSchema,
)

logger = logging.getLogger(__name__)

# ── Model capabilities lookup ─────────────────────────────────────────────────

from alancode.providers.anthropic_models import lookup_anthropic_model

_CACHE_MARKER = {"type": "ephemeral"}


def _inject_cache_breakpoints(
    system_blocks: list[dict[str, Any]],
    api_tools: list[dict[str, Any]] | None,
    messages: list[dict[str, Any]],
    static_boundary: int,
) -> None:
    """Add ``cache_control`` markers for Anthropic prompt caching.

    Uses up to 4 breakpoints (Anthropic's maximum):
    1. Last tool definition — caches all tool schemas
    2. Last static system section — caches tools + stable prompt
    3. Last system section — caches tools + full system prompt
    4. Last assistant message — caches entire conversation prefix
    """
    # BP1: last tool definition
    if api_tools:
        api_tools[-1]["cache_control"] = _CACHE_MARKER

    # BP2: last static system section
    if system_blocks and static_boundary > 0:
        idx = min(static_boundary, len(system_blocks)) - 1
        system_blocks[idx]["cache_control"] = _CACHE_MARKER

    # BP3: last system section (dynamic end)
    if system_blocks:
        last = len(system_blocks) - 1
        # Only add if different from BP2 (avoid wasting a breakpoint)
        if static_boundary <= 0 or last != min(static_boundary, len(system_blocks)) - 1:
            system_blocks[last]["cache_control"] = _CACHE_MARKER

    # BP4: last assistant message's last content block
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            content = msg.get("content")
            if isinstance(content, list) and content:
                content[-1]["cache_control"] = _CACHE_MARKER
            break


class AnthropicProvider(LLMProvider):
    """LLM provider using the official Anthropic Python SDK.

    Parameters
    ----------
    api_key : str | None
        Anthropic API key.  Falls back to ``ANTHROPIC_API_KEY`` env var.
    base_url : str | None
        Override the default API base URL.
    model : str
        Default model to use when none is specified per-request.
    **client_kwargs
        Additional keyword arguments forwarded to ``anthropic.AsyncAnthropic``.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str = "claude-sonnet-4-6",
        **client_kwargs: Any,
    ) -> None:
        import anthropic

        self._model = model
        self._client = anthropic.AsyncAnthropic(
            api_key=api_key,
            base_url=base_url,
            **client_kwargs,
        )

    # ── LLMProvider interface ──────────────────────────────────────────────────

    def get_model_info(self, model: str | None = None) -> ModelInfo:
        return lookup_anthropic_model(model or self._model)

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
        **kwargs: Any,
    ) -> AsyncGenerator[ProviderStreamEvent, None]:
        """Stream from the Anthropic API, translating raw events.

        Uses ``self._client.messages.stream()`` with raw event iteration so we
        receive tool_use events in addition to text deltas.
        """
        import anthropic

        resolved_model = model or self._model  # model param overrides constructor model
        info = self.get_model_info(resolved_model)
        resolved_max_tokens = max_tokens or info.max_output_tokens

        # Build system prompt blocks
        system_blocks: list[dict[str, Any]] = [
            {"type": "text", "text": s} for s in system if s
        ]

        # Build tool definitions
        api_tools: list[dict[str, Any]] | None = None
        if tools:
            api_tools = [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.input_schema,
                }
                for t in tools
            ]

        # Translate OpenAI-format messages to Anthropic format
        anthropic_messages = _openai_to_anthropic_messages(messages)

        # Prompt caching — place cache_control breakpoints.
        boundary = kwargs.pop("system_static_boundary", None)
        if boundary is not None:
            _inject_cache_breakpoints(
                system_blocks, api_tools, anthropic_messages, boundary
            )

        # Base request params
        params: dict[str, Any] = {
            "model": resolved_model,
            "messages": anthropic_messages,
            "max_tokens": resolved_max_tokens,
        }
        if system_blocks:
            params["system"] = system_blocks
        if api_tools:
            params["tools"] = api_tools
        if stop_sequences:
            params["stop_sequences"] = stop_sequences

        # Thinking configuration
        use_thinking = (
            thinking is not None
            and thinking.type != "disabled"
            and info.supports_thinking
        )
        if use_thinking:
            assert thinking is not None  # for type narrowing
            if thinking.type == "budget" and thinking.budget_tokens is not None:
                params["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": thinking.budget_tokens,
                }
            else:
                # "adaptive" or budget without explicit tokens — let the model
                # decide by using a sensible default budget.
                thinking_default = kwargs.pop("thinking_budget_default", 10_000)
                default_budget = min(thinking_default, resolved_max_tokens // 2)
                params["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": thinking.budget_tokens or default_budget,
                }

        # Merge any extra kwargs (e.g. metadata, temperature)
        params.update(kwargs)

        try:
            async with self._client.messages.stream(**params) as raw_stream:
                # State for accumulating tool input JSON across deltas
                current_tool_id: str | None = None
                current_tool_name: str | None = None
                accumulated_tool_json: str = ""

                async for event in raw_stream:
                    event_type = event.type

                    # ── message_start ────────────────────────────────────
                    if event_type == "message_start":
                        msg = event.message
                        usage_dict: dict[str, int] | None = None
                        if msg.usage is not None:
                            usage_dict = {
                                "input_tokens": msg.usage.input_tokens,
                                "output_tokens": msg.usage.output_tokens,
                                "cache_read_input_tokens": getattr(msg.usage, "cache_read_input_tokens", 0) or 0,
                                "cache_creation_input_tokens": getattr(msg.usage, "cache_creation_input_tokens", 0) or 0,
                            }
                        yield StreamMessageStart(
                            model=msg.model,
                            request_id=msg.id,
                            usage=usage_dict,
                        )

                    # ── content_block_start ──────────────────────────────
                    elif event_type == "content_block_start":
                        block = event.content_block
                        if block.type == "tool_use":
                            current_tool_id = block.id
                            current_tool_name = block.name
                            accumulated_tool_json = ""
                            yield StreamToolUseStart(
                                id=block.id,
                                name=block.name,
                            )
                        # text and thinking blocks just start; deltas carry
                        # the actual content.

                    # ── content_block_delta ──────────────────────────────
                    elif event_type == "content_block_delta":
                        delta = event.delta
                        if delta.type == "text_delta":
                            yield StreamTextDelta(text=delta.text)
                        elif delta.type == "thinking_delta":
                            yield StreamThinkingDelta(thinking=delta.thinking)
                        elif delta.type == "input_json_delta":
                            partial = delta.partial_json
                            accumulated_tool_json += partial
                            yield StreamToolUseInputDelta(
                                id=current_tool_id or "",
                                partial_json=partial,
                            )

                    # ── content_block_stop ───────────────────────────────
                    elif event_type == "content_block_stop":
                        # If we were accumulating a tool call, finalise it
                        if current_tool_id is not None:
                            parsed_input: dict[str, Any] = {}
                            if accumulated_tool_json:
                                try:
                                    parsed_input = json.loads(
                                        accumulated_tool_json
                                    )
                                except json.JSONDecodeError:
                                    logger.warning(
                                        "Failed to parse tool input JSON: %s",
                                        accumulated_tool_json,
                                    )
                                    yield StreamError(
                                        error=f"Malformed tool input JSON for {current_tool_name}: {accumulated_tool_json[:200]}",
                                        error_type="api_error",
                                        status_code=None,
                                    )
                            # Guard against empty id/name — emitting one
                            # would produce orphan tool_results downstream
                            # (the next turn's API call rejects with 400).
                            if current_tool_id and current_tool_name:
                                yield StreamToolUseStop(
                                    id=current_tool_id,
                                    name=current_tool_name,
                                    input=parsed_input,
                                )
                            else:
                                logger.warning(
                                    "Dropping tool_use_stop with empty id/name: "
                                    "id=%r, name=%r",
                                    current_tool_id, current_tool_name,
                                )
                            # Reset tool accumulation state
                            current_tool_id = None
                            current_tool_name = None
                            accumulated_tool_json = ""

                    # ── message_delta ────────────────────────────────────
                    elif event_type == "message_delta":
                        delta = event.delta
                        usage_out: dict[str, int] | None = None
                        if event.usage is not None:
                            usage_out = {
                                "output_tokens": event.usage.output_tokens,
                            }
                        yield StreamMessageDelta(
                            stop_reason=getattr(delta, "stop_reason", None),
                            usage=usage_out,
                        )

                    # ── message_stop ─────────────────────────────────────
                    elif event_type == "message_stop":
                        yield StreamMessageStop()

        except anthropic.AuthenticationError as exc:
            yield StreamError(
                error=f"Authentication failed: {exc.message}",
                error_type="invalid_request",
                status_code=exc.status_code,
            )
        except anthropic.RateLimitError as exc:
            yield StreamError(
                error=f"Rate limited: {exc.message}",
                error_type="overloaded",
                status_code=exc.status_code,
            )
        except anthropic.APIConnectionError as exc:
            yield StreamError(
                error=f"Connection error: {exc}",
                error_type="api_error",
                status_code=None,
            )
        except anthropic.APIError as exc:
            yield StreamError(
                error=f"API error: {exc.message}",
                error_type="api_error",
                status_code=exc.status_code,
            )
        except Exception as exc:
            logger.exception("Unexpected error during Anthropic streaming")
            yield StreamError(
                error=f"Unexpected error: {exc}",
                error_type="api_error",
                status_code=None,
            )


# ── OpenAI → Anthropic message translation ──────────────────────────────────


def _openai_to_anthropic_messages(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Translate OpenAI-format message dicts to Anthropic format.

    Key translations:
    - ``tool_calls`` on assistant → ``tool_use`` content blocks
    - ``role: "tool"`` → ``tool_result`` content blocks in a user message
    - Consecutive tool results are merged into one user message
    """
    result: list[dict[str, Any]] = []

    for msg in messages:
        role = msg.get("role", "")

        if role == "assistant":
            result.append(_convert_assistant_to_anthropic(msg))

        elif role == "tool":
            tool_result_block = {
                "type": "tool_result",
                "tool_use_id": msg.get("tool_call_id", ""),
                "content": msg.get("content", ""),
            }
            # Merge consecutive tool results into one user message
            if (
                result
                and result[-1].get("role") == "user"
                and isinstance(result[-1].get("content"), list)
                and any(
                    isinstance(b, dict) and b.get("type") == "tool_result"
                    for b in result[-1]["content"]
                )
            ):
                result[-1]["content"].append(tool_result_block)
            else:
                result.append({
                    "role": "user",
                    "content": [tool_result_block],
                })

        elif role == "user":
            result.append({"role": "user", "content": msg.get("content", "")})

        elif role == "system":
            # System messages handled separately — pass through if present
            result.append(msg)

        else:
            result.append(msg)

    return result


def _convert_assistant_to_anthropic(msg: dict[str, Any]) -> dict[str, Any]:
    """Convert an OpenAI assistant message to Anthropic content-block format."""
    content_blocks: list[dict[str, Any]] = []

    text = msg.get("content")
    if text:
        content_blocks.append({"type": "text", "text": text})

    for tc in msg.get("tool_calls") or []:
        func = tc.get("function", {})
        arguments = func.get("arguments", "{}")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except (json.JSONDecodeError, TypeError):
                arguments = {}
        content_blocks.append({
            "type": "tool_use",
            "id": tc.get("id", ""),
            "name": func.get("name", ""),
            "input": arguments,
        })

    if not content_blocks:
        content_blocks.append({"type": "text", "text": ""})

    return {"role": "assistant", "content": content_blocks}
