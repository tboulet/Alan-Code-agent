"""Tests for change-set 6: liveness fallback helper + budget settings surface."""

from alancode.messages.factory import (
    create_compact_boundary_message,
    create_tool_result_message,
    create_user_message,
)
from alancode.messages.types import UserMessage
from alancode.query.loop import _hard_truncate_fallback
from alancode.settings import (
    SETTINGS_DEFAULTS,
    _DEPRECATED_KEYS,
    load_settings,
    save_settings,
    validate_setting,
)


# ---------------------------------------------------------------------------
# _hard_truncate_fallback (circuit-breaker liveness path)
# ---------------------------------------------------------------------------


class TestHardTruncateFallback:
    def test_noop_when_under_target(self):
        msgs = [create_user_message("hello"), create_user_message("world")]
        kept, dropped = _hard_truncate_fallback(msgs, target_tokens=999_999)
        assert kept == msgs
        assert dropped == 0

    def test_middle_dropped_head_and_tail_survive(self):
        msgs = [create_user_message(f"msg_{i} " + "pad " * 200) for i in range(20)]
        kept, dropped = _hard_truncate_fallback(msgs, target_tokens=2_000)
        assert dropped > 0
        assert len(kept) + dropped == 20
        # The opening task-statement message survives (head preservation)
        assert kept[0].content == msgs[0].content
        # The recent tail survives
        assert kept[-1].content == msgs[-1].content
        # The victims came from just after the head
        assert kept[1].content != msgs[1].content

    def test_compact_summary_head_survives(self):
        # After a past compaction the history starts with boundary + summary:
        # they compress the WHOLE earlier conversation and must be the last
        # to go - never the first.
        boundary = create_compact_boundary_message(trigger="auto", pre_tokens=50_000)
        summary = create_user_message(
            "SUMMARY-MARKER " + "s " * 200, is_compact_summary=True
        )
        middle = [create_user_message(f"mid_{i} " + "pad " * 200) for i in range(15)]
        tail = [create_user_message("recent question")]
        kept, dropped = _hard_truncate_fallback(
            [boundary, summary] + middle + tail, target_tokens=2_000
        )
        assert dropped > 0
        assert kept[0] is boundary
        assert "SUMMARY-MARKER" in kept[1].content
        assert kept[-1].content == "recent question"

    def test_seam_never_starts_on_orphan_tool_result(self):
        # History: text, tool result, text, tool result... After the head,
        # the seam must land on plain user text (an orphan tool_result
        # would be rejected by the API).
        msgs = [create_user_message("TASK-MARKER opening")]
        for i in range(10):
            msgs.append(create_user_message(f"text_{i} " + "pad " * 300))
            msgs.append(create_tool_result_message(f"tu_{i}", "output " * 300))
        kept, _ = _hard_truncate_fallback(msgs, target_tokens=2_000)
        assert kept, "should keep something"
        assert "TASK-MARKER" in kept[0].content  # head preserved
        seam = kept[1]
        assert isinstance(seam, UserMessage)
        assert isinstance(seam.content, str)

    def test_giant_head_escape_hatch(self):
        # A head that alone blows the target loses its protection:
        # survival wins over preservation.
        msgs = [
            create_user_message("giant paste " * 5_000),
            create_user_message("small recent 1"),
            create_user_message("small recent 2"),
        ]
        kept, dropped = _hard_truncate_fallback(msgs, target_tokens=500)
        assert dropped >= 1
        assert all("giant paste" not in m.content for m in kept)
        assert kept[-1].content == "small recent 2"


# ---------------------------------------------------------------------------
# Settings surface
# ---------------------------------------------------------------------------


class TestBudgetSettingsSurface:
    def test_deprecated_keys_removed_from_defaults(self):
        for key in _DEPRECATED_KEYS:
            assert key not in SETTINGS_DEFAULTS

    def test_budget_keys_default_to_auto(self):
        assert SETTINGS_DEFAULTS["context_window"] == "auto"
        assert SETTINGS_DEFAULTS["tool_result_max_chars"] == "auto"
        assert SETTINGS_DEFAULTS["compact_max_output_tokens"] == "auto"
        assert SETTINGS_DEFAULTS["compaction_threshold_percent"] == "auto"

    def test_auto_accepted_by_validators(self):
        for key in (
            "context_window",
            "max_output_tokens",
            "tool_result_max_chars",
            "compact_max_output_tokens",
            "compaction_threshold_percent",
        ):
            assert validate_setting(key, "auto") is None
            assert validate_setting(key, "AUTO") is None

    def test_explicit_ints_accepted(self):
        assert validate_setting("context_window", 32_768) is None
        assert validate_setting("tool_result_max_chars", 5_000) is None
        assert validate_setting("compaction_threshold_percent", 75) is None

    def test_invalid_values_rejected(self):
        assert validate_setting("context_window", 0) is not None
        assert validate_setting("context_window", -5) is not None
        assert validate_setting("context_window", "large") is not None
        assert validate_setting("compaction_threshold_percent", 100) is not None
        assert validate_setting("compaction_threshold_percent", 0) is not None
        assert validate_setting("tool_result_max_chars", True) is not None

    def test_load_strips_deprecated_keys(self, tmp_path):
        settings = dict(SETTINGS_DEFAULTS)
        settings["blocking_limit_buffer_tokens"] = 3_000
        settings["compact_clear_keep_recent"] = 10
        save_settings(settings, str(tmp_path))

        loaded = load_settings(str(tmp_path))
        assert "blocking_limit_buffer_tokens" not in loaded
        assert "compact_clear_keep_recent" not in loaded
        # The rest survives
        assert loaded["compaction_threshold_percent"] == "auto"
