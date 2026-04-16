"""Test the cost tracker (alancode/api/cost_tracker.py)."""

import tempfile
from unittest.mock import patch, MagicMock

import pytest

from alancode.api.cost_tracker import (
    ANTHROPIC_PRICING,
    CostTracker,
    _anthropic_cost,
    _litellm_cost,
)
from alancode.messages.types import Usage
from alancode.session.state import SessionState


def _make_tracker(tmp_path=None):
    """Create a CostTracker with a temporary SessionState."""
    if tmp_path is None:
        tmp_path = tempfile.mkdtemp()
    session = SessionState(session_id="test-session", cwd=str(tmp_path))
    return CostTracker(session=session), session


# ---------------------------------------------------------------------------
# Anthropic cost calculation
# ---------------------------------------------------------------------------


class TestAnthropicCost:

    def test_known_model_exact_match(self):
        """Known model name returns a non-None cost."""
        usage = Usage(input_tokens=1000, output_tokens=500)
        cost = _anthropic_cost(usage, "claude-sonnet-4-6")
        assert cost is not None
        prices = ANTHROPIC_PRICING["claude-sonnet-4-6"]
        expected = (1000 * prices["input"] + 500 * prices["output"]) / 1_000_000
        assert cost == pytest.approx(expected)

    def test_prefix_match(self):
        """Model name with date suffix matches via prefix."""
        usage = Usage(input_tokens=2000, output_tokens=1000)
        cost = _anthropic_cost(usage, "claude-sonnet-4-6-20250514")
        assert cost is not None
        # Should equal the cost for the base model
        cost_base = _anthropic_cost(usage, "claude-sonnet-4-6")
        assert cost == pytest.approx(cost_base)

    def test_cache_tokens_included(self):
        """Cache read and write tokens are accounted for in cost."""
        usage = Usage(
            input_tokens=500,
            output_tokens=200,
            cache_read_input_tokens=300,
            cache_creation_input_tokens=100,
        )
        cost = _anthropic_cost(usage, "claude-sonnet-4-6")
        assert cost is not None
        prices = ANTHROPIC_PRICING["claude-sonnet-4-6"]
        expected = (
            500 * prices["input"]
            + 200 * prices["output"]
            + 300 * prices["cache_read"]
            + 100 * prices["cache_write"]
        ) / 1_000_000
        assert cost == pytest.approx(expected)

    def test_unknown_model_returns_none(self):
        """Non-Anthropic model returns None from the Anthropic calculator."""
        usage = Usage(input_tokens=100, output_tokens=50)
        cost = _anthropic_cost(usage, "gpt-4o")
        assert cost is None

    def test_zero_usage_returns_zero(self):
        """Zero tokens yields zero cost."""
        usage = Usage()
        cost = _anthropic_cost(usage, "claude-sonnet-4-6")
        assert cost == 0.0


# ---------------------------------------------------------------------------
# litellm cost calculation
# ---------------------------------------------------------------------------


class TestLitellmCost:

    def test_litellm_cost_uses_litellm_module(self):
        """_litellm_cost calls litellm.cost_per_token and returns the sum."""
        usage = Usage(input_tokens=1000, output_tokens=500)
        mock_litellm = MagicMock()
        mock_litellm.cost_per_token.return_value = (0.005, 0.010)

        with patch.dict("sys.modules", {"litellm": mock_litellm}):
            cost = _litellm_cost(usage, "some-litellm-model")

        assert cost == pytest.approx(0.015)
        mock_litellm.cost_per_token.assert_called_once_with(
            model="some-litellm-model",
            prompt_tokens=1000,
            completion_tokens=500,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        )

    def test_litellm_exception_returns_none(self):
        """If litellm raises, _litellm_cost returns None."""
        usage = Usage(input_tokens=100, output_tokens=50)
        mock_litellm = MagicMock()
        mock_litellm.cost_per_token.side_effect = Exception("model not found")

        with patch.dict("sys.modules", {"litellm": mock_litellm}):
            cost = _litellm_cost(usage, "unknown-model")

        assert cost is None


# ---------------------------------------------------------------------------
# CostTracker
# ---------------------------------------------------------------------------


class TestCostTracker:

    def test_initial_state(self, tmp_path):
        tracker, session = _make_tracker(tmp_path)
        assert session.total_cost_usd == 0.0
        assert session.total_input_tokens == 0
        assert session.total_output_tokens == 0
        assert session.cost_unknown is False

    def test_add_usage_accumulates_tokens(self, tmp_path):
        """add_usage should accumulate token counts in session state."""
        tracker, session = _make_tracker(tmp_path)
        usage1 = Usage(input_tokens=100, output_tokens=50)
        usage2 = Usage(input_tokens=200, output_tokens=100)

        tracker.add_usage(usage1, "claude-sonnet-4-6")
        tracker.add_usage(usage2, "claude-sonnet-4-6")

        assert session.total_input_tokens == 300
        assert session.total_output_tokens == 150

    def test_add_usage_accumulates_cost(self, tmp_path):
        """add_usage should accumulate USD cost in session state."""
        tracker, session = _make_tracker(tmp_path)
        usage = Usage(input_tokens=1000, output_tokens=500)

        tracker.add_usage(usage, "claude-sonnet-4-6")

        assert session.total_cost_usd > 0.0
        expected_cost = _anthropic_cost(usage, "claude-sonnet-4-6")
        assert session.total_cost_usd == pytest.approx(expected_cost)

    def test_add_usage_tracks_per_model(self, tmp_path):
        """add_usage should track token counts per model."""
        tracker, _ = _make_tracker(tmp_path)
        usage1 = Usage(input_tokens=100, output_tokens=50)
        usage2 = Usage(input_tokens=200, output_tokens=100)

        tracker.add_usage(usage1, "claude-sonnet-4-6")
        tracker.add_usage(usage2, "claude-opus-4-6")

        assert "claude-sonnet-4-6" in tracker.model_usage
        assert "claude-opus-4-6" in tracker.model_usage
        assert tracker.model_usage["claude-sonnet-4-6"].input_tokens == 100
        assert tracker.model_usage["claude-opus-4-6"].input_tokens == 200

    def test_cost_unknown_flag_set_for_unknown_model(self, tmp_path):
        """cost_unknown should be True when a model has no pricing info."""
        tracker, session = _make_tracker(tmp_path)
        usage = Usage(input_tokens=100, output_tokens=50)

        # Use a model that is not in ANTHROPIC_PRICING and mock litellm to fail
        mock_litellm = MagicMock()
        mock_litellm.cost_per_token.side_effect = Exception("no pricing")

        with patch.dict("sys.modules", {"litellm": mock_litellm}):
            tracker.add_usage(usage, "totally-unknown-model-xyz")

        assert session.cost_unknown is True

    def test_cost_unknown_flag_not_set_for_known_model(self, tmp_path):
        """cost_unknown should remain False for a known Anthropic model."""
        tracker, session = _make_tracker(tmp_path)
        usage = Usage(input_tokens=100, output_tokens=50)
        tracker.add_usage(usage, "claude-sonnet-4-6")
        assert session.cost_unknown is False

    def test_add_usage_accumulates_duration(self, tmp_path):
        tracker, _ = _make_tracker(tmp_path)
        usage = Usage(input_tokens=100, output_tokens=50)
        tracker.add_usage(usage, "claude-sonnet-4-6", duration_ms=150.5)
        tracker.add_usage(usage, "claude-sonnet-4-6", duration_ms=250.0)
        assert tracker.total_api_duration_ms == pytest.approx(400.5)

    def test_get_summary_structure(self, tmp_path):
        """get_summary returns the expected dict shape."""
        tracker, _ = _make_tracker(tmp_path)
        usage = Usage(input_tokens=100, output_tokens=50)
        tracker.add_usage(usage, "claude-sonnet-4-6", duration_ms=100.0)

        summary = tracker.get_summary()
        assert "total_input_tokens" in summary
        assert "total_output_tokens" in summary
        assert "total_cost_usd" in summary
        assert "total_api_duration_ms" in summary
        assert "models" in summary
        assert "claude-sonnet-4-6" in summary["models"]

    def test_state_persisted_to_disk(self, tmp_path):
        """Verify that cost tracker updates are written to state.json."""
        tracker, session = _make_tracker(tmp_path)
        usage = Usage(input_tokens=500, output_tokens=200)
        tracker.add_usage(usage, "claude-sonnet-4-6")

        # Create a new SessionState pointing to the same path — should see the data
        session2 = SessionState(session_id="test-session", cwd=str(tmp_path))
        assert session2.total_input_tokens == 500
        assert session2.total_output_tokens == 200
        assert session2.total_cost_usd > 0.0
