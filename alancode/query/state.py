"""Query loop state management."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class LoopState:
    """Mutable state carried between loop iterations."""
    messages: list  # Current message history
    max_output_tokens_recovery_count: int = 0
    max_output_tokens_override: int | None = None
    has_attempted_emergency_compact: bool = False
    iteration_count: int = 0  # Number of tool-use iterations completed
    transition: str | None = None  # Why previous iteration continued
    auto_compact_tracking: dict | None = None  # {compacted, iteration_counter, consecutive_failures}
    turns_since_memory_update: int = 0  # Iterations since last memory reminder (intensive mode)
    cached_model_info: Any = None  # Cached ModelInfo from provider (reset on model change)
    # Last API call's provider-reported usage, captured so the next
    # iteration's pre-call token estimate can use it as a floor.
    last_input_tokens: int = 0
    last_output_tokens: int = 0
    # len(state.messages) at the moment of the last call, so we can
    # delta-count messages added since then.
    messages_len_at_last_call: int = 0
