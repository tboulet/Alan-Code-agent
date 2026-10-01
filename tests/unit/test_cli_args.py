"""Command-line flags of the ``alancode`` entry point."""

import sys

import pytest

from alancode.cli.main import main


def test_the_removed_escalation_flag_is_rejected(monkeypatch, capsys):
    """1.3.17 removed escalated_max_tokens, but the CLI kept the flag and
    passed it on as a backend kwarg: the client constructor crashed."""
    monkeypatch.setattr(sys, "argv", ["alancode", "--escalated-max-tokens", "5000"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    assert "--escalated-max-tokens" in capsys.readouterr().err
