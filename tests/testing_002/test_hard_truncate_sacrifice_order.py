"""Proof for finding T002-002.  This deliberately fails until fixed."""

from alancode.messages.factory import create_compact_boundary_message
from alancode.messages.types import UserMessage
from alancode.query.loop import _hard_truncate_fallback


def test_hard_truncation_keeps_compact_head_before_stale_tail_messages():
    boundary = create_compact_boundary_message("auto", 10_000, messages_summarized=5)
    summary = UserMessage(content="SUMMARY-MARKER", is_compact_summary=True)
    stale = [UserMessage(content="STALE-MARKER " + "x" * 3_000) for _ in range(8)]
    kept, _ = _hard_truncate_fallback([boundary, summary, *stale, UserMessage(content="RECENT-MARKER")], 400)
    rendered = str(kept)
    assert "SUMMARY-MARKER" in rendered
    assert "RECENT-MARKER" in rendered
    assert "STALE-MARKER" not in rendered
