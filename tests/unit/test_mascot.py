"""The jellyfish: terminal rendering, and the PNG the GUI and README point at."""

import struct
from pathlib import Path

import pytest
from rich.console import Console

from alancode.__version__ import __version__
from alancode.backends.base import ModelInfo
from alancode.cli.display import display_welcome
from alancode.cli.mascot import ALAN_PIXELS, render_mascot
from alancode.settings import SETTINGS_DEFAULTS

REPO_ROOT = Path(__file__).resolve().parents[2]
GUI_STATIC = REPO_ROOT / "alancode" / "gui" / "static"
MASCOT_PNG = "alan_mascot_pixelart.png"
TITLED_PNG = "alan_mascot_pixelart_with_text.png"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


class FakeBackend:
    def get_model_info(self, model):
        return ModelInfo(context_window=262_144, max_output_tokens=32_768)


class FakeAgent:
    _model = "claude-sonnet-4-6"
    session_id = "4c4a127a9f0b4e11"
    _cwd = ""
    _memory_mode = "off"
    _backend = FakeBackend()
    _settings = dict(SETTINGS_DEFAULTS, backend="auto")


def _banner(agent=None, width=110):
    console = Console(no_color=True, force_terminal=False, width=width, record=True)
    display_welcome(console, agent or FakeAgent())
    return console.export_text()


def test_two_pixel_rows_per_terminal_line():
    lines = render_mascot(force_color=False).split("\n")
    assert len(lines) == len(ALAN_PIXELS) // 2


def test_sprite_has_an_even_number_of_rows():
    # An odd row count would drop the last row: the renderer steps in pairs.
    assert len(ALAN_PIXELS) % 2 == 0


def test_plain_mode_emits_no_escape_codes():
    assert "\x1b" not in render_mascot(force_color=False)


def test_colored_mode_emits_truecolor():
    body = render_mascot(force_color=True, color="#38A8E8")
    assert "\x1b[38;2;56;168;232m" in body


def test_indent_shifts_every_line():
    lines = render_mascot(force_color=False, indent=4).split("\n")
    assert all(line.startswith("    ") for line in lines)


def test_negative_indent_is_rejected():
    with pytest.raises(ValueError):
        render_mascot(indent=-1)


def test_invalid_color_is_rejected():
    with pytest.raises(ValueError, match="invalid color"):
        render_mascot(color="not-a-color")


def test_welcome_banner_shows_mascot_and_every_adopted_line():
    out = _banner()

    assert "█" in out                     # the sprite made it into the panel
    assert f"Alan Code {__version__}" in out
    assert "4c4a127a..." in out
    assert "claude-sonnet-4-6" in out
    assert "Context: 262K window" in out
    assert "Memory: off" in out
    assert "Backend: auto" in out
    assert "/exit to quit" in out


def test_banner_no_longer_claims_ctrl_c_quits():
    # The old wording sent people to Ctrl+C, which only prints "Use /exit".
    out = _banner()
    assert "Ctrl+C to interrupt" not in out
    assert "Ctrl+C interrupts a turn" in out


def test_title_sits_above_the_mascot():
    lines = [ln for ln in _banner().splitlines() if ln.strip("| ")]
    title = next(i for i, ln in enumerate(lines) if "Open-source coding agent" in ln)
    sprite = next(i for i, ln in enumerate(lines) if "█" in ln)
    assert title < sprite


def test_endpoint_is_shown_only_when_one_is_configured():
    assert "->" not in _banner()

    class Remote(FakeAgent):
        _settings = dict(
            SETTINGS_DEFAULTS, backend="auto", base_url="http://localhost:8000/v1"
        )

    assert "Backend: auto -> http://localhost:8000/v1" in _banner(Remote())


def test_a_bad_budget_config_is_reported_not_hidden():
    # Regression: the banner used to swallow ConfigError behind a bland
    # placeholder, so the first prompt was what discovered the broken session.
    from alancode.budget import ConfigError

    class BadConfig(FakeAgent):
        class _backend:
            @staticmethod
            def get_model_info(model):
                raise ConfigError(
                    "compact_max_output_tokens (20000) must be smaller "
                    "than the context window (16384)."
                )

    out = _banner(BadConfig())
    assert "invalid budget configuration" in out
    assert "compact_max_output_tokens (20000)" in out


def test_an_unexpected_failure_does_not_stop_the_session():
    class Exploding(FakeAgent):
        class _backend:
            @staticmethod
            def get_model_info(model):
                raise RuntimeError("no model info")

    out = _banner(Exploding())
    assert "Context: resolving on first call" in out
    assert "4c4a127a..." in out           # the rest of the banner survived


def test_gui_ships_the_mascot_png():
    # The GUI serves this from /static; losing it silently breaks the topbar.
    png = GUI_STATIC / MASCOT_PNG
    assert png.is_file()
    assert png.read_bytes()[:8] == PNG_MAGIC


def test_gui_markup_references_the_mascot():
    index = (GUI_STATIC / "index.html").read_text()
    assert f"/static/{MASCOT_PNG}" in index
    assert 'rel="icon"' in index


def test_readme_uses_the_titled_mascot():
    # The GUI keeps the bare sprite; only the README carries the titled card.
    readme = (REPO_ROOT / "README.md").read_text()
    assert f"assets/images/{TITLED_PNG}" in readme
    assert (REPO_ROOT / "assets" / "images" / TITLED_PNG).is_file()


def test_titled_mascot_is_a_wide_rgba_png():
    # Read IHDR directly: Pillow is not an alancode dependency, and the two
    # facts that matter here are in the header.
    raw = (REPO_ROOT / "assets" / "images" / TITLED_PNG).read_bytes()
    assert raw[:8] == PNG_MAGIC
    width, height = struct.unpack(">II", raw[16:24])
    colour_type = raw[25]
    assert colour_type == 6                 # RGBA: the soft edge needs alpha
    assert width > height                   # sprite beside the text, not above it
