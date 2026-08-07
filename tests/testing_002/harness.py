"""Backend-boundary sensor for minimum compaction-output tests."""

from __future__ import annotations

from typing import Any, AsyncGenerator

from alancode.backends.base import (
    BackendStreamEvent,
    LLMBackend,
    ModelInfo,
    StreamMessageStop,
    StreamTextDelta,
    ThinkingConfig,
    ToolSchema,
)
from alancode.backends.scripted_backend import ScriptedBackend


class AuditedBackend(LLMBackend):
    """Assert every attempted request fits the declared context window."""

    def __init__(self, inner: ScriptedBackend, context_window: int) -> None:
        self.inner = inner
        self.cw = context_window
        self.calls: list[dict[str, Any]] = []

    def get_model_info(self, model: str | None = None) -> ModelInfo:
        return ModelInfo(context_window=self.cw, max_output_tokens=8_192)

    @staticmethod
    def _estimate(messages: list[dict], system: list[str]) -> int:
        chars = sum(len(str(message.get("content", ""))) for message in messages)
        return (chars + sum(len(section) for section in system)) // 4

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
    ) -> AsyncGenerator[BackendStreamEvent, None]:
        input_tokens = self._estimate(messages, system)
        entry = {
            "input": input_tokens,
            "max_tokens": max_tokens,
            "cw": self.cw,
        }
        self.calls.append(entry)
        assert max_tokens is not None
        assert input_tokens + max_tokens <= self.cw, entry

        if system and "summariz" in system[0].lower():
            yield StreamTextDelta(text="<summary>summary</summary>")
            yield StreamMessageStop()
            return

        async for event in self.inner.stream(
            messages,
            system,
            tools,
            model=model,
            max_tokens=max_tokens,
            thinking=thinking,
            stop_sequences=stop_sequences,
            **kwargs,
        ):
            yield event
