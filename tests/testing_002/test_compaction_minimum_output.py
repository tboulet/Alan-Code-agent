"""Proof for finding T002-001.  This deliberately fails until fixed."""

import pytest

from alancode.budget import resolve_context_budget
from alancode.compact.compact_auto import compaction_auto
from alancode.messages.types import UserMessage
from alancode.providers.base import ModelInfo
from alancode.providers.scripted_provider import ScriptedProvider
from perso_dev.testing_002.harness import AuditedProvider


@pytest.mark.asyncio
async def test_compaction_never_overrides_a_zero_output_legality_clamp():
    budget = resolve_context_budget(ModelInfo(context_window=8_192, max_output_tokens=8_192))
    provider = AuditedProvider(ScriptedProvider.from_responses([]), 8_192)
    # The compact prompt plus this history leaves fewer than 256 output tokens.
    messages = [UserMessage(content="x" * 45_000)]
    # No legal summarizer call exists for a single unshrinkable giant message
    # at this CW, so compaction must decline (return None) rather than emit an
    # over-budget call. The invariant is "never an illegal call", not "always
    # a call".
    result = await compaction_auto(messages, provider, budget=budget)
    assert result is None
    assert all(call["input"] + call["max_tokens"] <= call["cw"] for call in provider.calls), provider.calls
