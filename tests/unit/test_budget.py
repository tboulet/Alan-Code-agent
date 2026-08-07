"""Tests for alancode.budget - the token-budget resolution DAG.

Covers: the all-auto no-crash property over the CW range, the ordering
invariant, the auto formulas (worked examples pinned to the design doc),
explicit-override validation (ConfigError catalogue), and the call-time
output clamp.
"""

import logging

import pytest

from alancode.budget import (
    AUTO_OUTPUT_CW_FRACTION,
    ConfigError,
    ContextBudget,
    MIN_CONTEXT_WINDOW,
    WARN_CONTEXT_WINDOW,
    clamp_output_budget,
    resolve_context_budget,
)
from alancode.backends.base import ModelInfo


def info(cw: int = 200_000, max_out: int = 8_192) -> ModelInfo:
    return ModelInfo(context_window=cw, max_output_tokens=max_out)


# ---------------------------------------------------------------------------
# Property: all-auto never raises, ordering always holds
# ---------------------------------------------------------------------------


CW_SWEEP = [
    8_192,
    10_000,
    16_384,
    32_768,
    65_536,
    131_072,
    200_000,
    262_144,
    1_000_000,
    2_097_152,
]


class TestAllAutoProperty:
    @pytest.mark.parametrize("cw", CW_SWEEP)
    def test_all_auto_resolves(self, cw):
        b = resolve_context_budget(info(cw=cw))
        assert isinstance(b, ContextBudget)

    @pytest.mark.parametrize("cw", CW_SWEEP)
    def test_ordering_invariant(self, cw):
        b = resolve_context_budget(info(cw=cw))
        assert (
            0
            < b.threshold_compaction
            < b.tool_result_clear_target
            < b.max_context_size_allowed
            <= b.context_window
        )

    @pytest.mark.parametrize("cw", CW_SWEEP)
    def test_auto_output_capped_at_quarter_window(self, cw):
        b = resolve_context_budget(info(cw=cw))
        assert b.max_output_tokens <= cw // AUTO_OUTPUT_CW_FRACTION or (
            b.max_output_tokens == 8_192  # model default already below the cap
        )

    @pytest.mark.parametrize("cw", CW_SWEEP)
    def test_derived_values_positive(self, cw):
        b = resolve_context_budget(info(cw=cw))
        assert b.tool_result_cap_chars > 0
        assert b.max_tokens_for_summary > 0
        assert b.safety_margin >= 500

    @pytest.mark.parametrize("cw", CW_SWEEP)
    def test_huge_model_default_output_still_safe(self, cw):
        # A model advertising a 64k output capability must not eat the window.
        b = resolve_context_budget(info(cw=cw, max_out=64_000))
        assert b.max_output_tokens <= cw // AUTO_OUTPUT_CW_FRACTION
        assert (
            0
            < b.threshold_compaction
            < b.tool_result_clear_target
            < b.max_context_size_allowed
        )

    def test_auto_accepts_auto_string_and_none(self):
        for value in ("auto", "AUTO", None):
            b = resolve_context_budget(
                info(),
                {
                    "context_window": value,
                    "max_output_tokens": value,
                    "compaction_threshold_percent": value,
                    "tool_result_max_chars": value,
                    "compact_max_output_tokens": value,
                },
            )
            assert b.context_window == 200_000


# ---------------------------------------------------------------------------
# Worked examples - pinned to perso_dev/budget_redesign_decision_plan.md
# ---------------------------------------------------------------------------


class TestWorkedExamples:
    def test_big_model_200k(self):
        b = resolve_context_budget(info(cw=200_000, max_out=8_192))
        assert b.max_output_tokens == 8_192
        assert b.safety_margin == 2_000
        assert b.max_context_size_allowed == 189_808
        assert b.threshold_compaction == 151_846
        assert b.tool_result_clear_target == (151_846 + 189_808) // 2
        assert b.tool_result_cap_chars == 10_000  # absolute ceiling binds
        assert b.max_tokens_for_summary == 20_000

    def test_small_model_32k(self):
        b = resolve_context_budget(info(cw=32_768, max_out=8_192))
        assert b.max_output_tokens == 8_192
        assert b.safety_margin == 500
        assert b.max_context_size_allowed == 24_076
        assert b.threshold_compaction == 24_076 * 80 // 100  # 19_260
        assert b.tool_result_clear_target == (b.threshold_compaction + 24_076) // 2
        # 10% of T in tokens, times 3 chars - the scaled cap binds
        assert b.tool_result_cap_chars == int(0.10 * b.threshold_compaction * 3)
        assert b.tool_result_cap_chars < 10_000
        # Summarizer budget shrinks to fit the window
        assert b.max_tokens_for_summary == 32_768 - b.threshold_compaction - 500
        assert b.max_tokens_for_summary < 20_000

    def test_tiny_model_16k_auto_output_shrinks(self):
        b = resolve_context_budget(info(cw=16_384, max_out=8_192))
        # 8_192 default would be half the window - auto caps at CW/4
        assert b.max_output_tokens == 4_096
        assert (
            0
            < b.threshold_compaction
            < b.tool_result_clear_target
            < b.max_context_size_allowed
        )


# ---------------------------------------------------------------------------
# ConfigError catalogue
# ---------------------------------------------------------------------------


class TestConfigErrors:
    @pytest.mark.parametrize("cw", [0, -5, 1, 4_096, MIN_CONTEXT_WINDOW - 1])
    def test_cw_below_minimum(self, cw):
        with pytest.raises(ConfigError, match="minimum"):
            resolve_context_budget(info(cw=cw))

    def test_cw_override_below_minimum(self):
        with pytest.raises(ConfigError):
            resolve_context_budget(info(), {"context_window": 4_000})

    def test_cw_warn_band_logs_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger="alancode.budget"):
            resolve_context_budget(info(cw=WARN_CONTEXT_WINDOW - 1000))
        assert any("tight" in r.message for r in caplog.records)

    def test_explicit_output_at_least_window(self):
        with pytest.raises(ConfigError, match="max_output_tokens"):
            resolve_context_budget(info(), {"max_output_tokens": 200_000})

    def test_explicit_output_leaves_no_input(self):
        # 199_000 < CW so it passes the first check, but U goes negative.
        with pytest.raises(ConfigError, match="usable input"):
            resolve_context_budget(info(), {"max_output_tokens": 199_000})

    def test_explicit_output_honored_above_quarter_cap(self):
        # Explicit values are NOT subject to the auto CW/4 cap.
        b = resolve_context_budget(info(), {"max_output_tokens": 100_000})
        assert b.max_output_tokens == 100_000
        assert b.max_context_size_allowed == 200_000 - 100_000 - 2_000

    @pytest.mark.parametrize("pct", [0, 100, 150, -3])
    def test_threshold_percent_out_of_range(self, pct):
        with pytest.raises(ConfigError):
            resolve_context_budget(info(), {"compaction_threshold_percent": pct})

    @pytest.mark.parametrize(
        "key",
        [
            "max_output_tokens",
            "tool_result_max_chars",
            "compact_max_output_tokens",
            "context_window",
        ],
    )
    def test_non_int_values_rejected(self, key):
        for bad in ("big", 3.5, True, [1]):
            with pytest.raises(ConfigError):
                resolve_context_budget(info(), {key: bad})

    def test_explicit_summary_at_least_window(self):
        with pytest.raises(ConfigError, match="compact_max_output_tokens"):
            resolve_context_budget(
                info(cw=32_768), {"compact_max_output_tokens": 40_000}
            )

    def test_explicit_caps_honored(self):
        b = resolve_context_budget(
            info(),
            {
                "tool_result_max_chars": 123_456,
                "compact_max_output_tokens": 5_000,
                "compaction_threshold_percent": 50,
            },
        )
        assert b.tool_result_cap_chars == 123_456
        assert b.max_tokens_for_summary == 5_000
        assert b.threshold_compaction == b.max_context_size_allowed * 50 // 100


# ---------------------------------------------------------------------------
# Call-time output clamp (invariant I1)
# ---------------------------------------------------------------------------


class TestClampOutputBudget:
    def test_no_clamp_needed(self):
        b = resolve_context_budget(info())
        assert (
            clamp_output_budget(b, estimated_input_tokens=10_000)
            == b.max_output_tokens
        )

    def test_clamps_near_full_window(self):
        b = resolve_context_budget(info(cw=32_768, max_out=8_192))
        est = 30_000
        clamped = clamp_output_budget(b, est)
        assert clamped == 32_768 - 30_000 - b.safety_margin
        assert est + clamped + b.safety_margin <= b.context_window

    def test_zero_when_nothing_fits(self):
        b = resolve_context_budget(info(cw=32_768))
        assert clamp_output_budget(b, estimated_input_tokens=32_768) == 0

    def test_clamps_escalated_request(self):
        # Escalation asks for 64k on a 32k window: physically impossible,
        # the clamp grants what actually fits.
        b = resolve_context_budget(info(cw=32_768, max_out=8_192))
        clamped = clamp_output_budget(b, 10_000, requested_max_tokens=64_000)
        assert clamped == 32_768 - 10_000 - b.safety_margin

    @pytest.mark.parametrize("cw", CW_SWEEP)
    @pytest.mark.parametrize("est_frac", [0.0, 0.5, 0.8, 0.95, 1.0])
    def test_legality_property(self, cw, est_frac):
        b = resolve_context_budget(info(cw=cw))
        est = int(cw * est_frac)
        for requested in (None, 1, 8_192, 64_000, cw):
            clamped = clamp_output_budget(b, est, requested_max_tokens=requested)
            assert clamped >= 0
            if clamped > 0:
                assert est + clamped + b.safety_margin <= b.context_window
