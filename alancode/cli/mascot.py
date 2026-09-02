"""The Alan Code jellyfish, rendered as terminal pixel art.

Two vertical pixels share one terminal cell via half-block characters, so the
12-row sprite prints in 6 lines - short enough to sit beside the welcome
banner's text.
"""

from __future__ import annotations

import os
import re
import sys
from typing import Final, TextIO

# W = white outline, B = blue jellyfish, space = transparent.
# Every row is a palindrome, so the mascot is exactly symmetrical.
ALAN_PIXELS: Final[tuple[str, ...]] = (
    "   WWWWW   ",
    " WWBBBBBWW ",
    "WWBBBBBBBWW",
    "WBBB B BBBW",
    "WBBBBBBBBBW",
    "WBBBBBBBBBW",
    "WWBBBBBBBWW",
    " WWWWWWWWW ",
    "  B B B B  ",
    "  B B B B  ",
    " B  B B  B ",
    " B  B  B B ",
)

DEFAULT_BODY: Final[str] = "#38A8E8"
DEFAULT_OUTLINE: Final[str] = "#FFFFFF"
RESET: Final[str] = "\x1b[0m"
_HEX_COLOR = re.compile(r"^#?([0-9a-fA-F]{6})$")


def _rgb(value: str) -> tuple[int, int, int]:
    """Convert '#RRGGBB' into an RGB tuple."""
    match = _HEX_COLOR.fullmatch(value)
    if not match:
        raise ValueError(f"invalid color {value!r}; expected '#RRGGBB'")

    value = match.group(1)
    return tuple(
        int(value[i : i + 2], 16) for i in (0, 2, 4)
    )  # type: ignore[return-value]


def _fg(rgb: tuple[int, int, int]) -> str:
    return f"\x1b[38;2;{rgb[0]};{rgb[1]};{rgb[2]}m"


def _bg(rgb: tuple[int, int, int]) -> str:
    return f"\x1b[48;2;{rgb[0]};{rgb[1]};{rgb[2]}m"


def _use_color(stream: TextIO, force_color: bool | None) -> bool:
    if force_color is not None:
        return force_color

    return "NO_COLOR" not in os.environ and stream.isatty()


def _plain_cell(top: bool, bottom: bool) -> str:
    if top and bottom:
        return "█"
    if top:
        return "▀"
    if bottom:
        return "▄"
    return " "


def _colored_cell(
    top: tuple[int, int, int] | None,
    bottom: tuple[int, int, int] | None,
) -> str:
    """Combine two colored vertical pixels into one terminal cell."""
    if top is None and bottom is None:
        return " "

    if top is None:
        return f"{_fg(bottom)}▄{RESET}"  # type: ignore[arg-type]

    if bottom is None:
        return f"{_fg(top)}▀{RESET}"

    if top == bottom:
        return f"{_fg(top)}█{RESET}"

    return f"{_fg(top)}{_bg(bottom)}▀{RESET}"


def render_mascot(
    *,
    color: str = DEFAULT_BODY,
    outline: str = DEFAULT_OUTLINE,
    indent: int = 0,
    force_color: bool | None = None,
    stream: TextIO = sys.stdout,
) -> str:
    """Return the six-line Alan mascot, ready to print.

    Color is enabled automatically for a terminal and disabled for pipes or
    when NO_COLOR is set. Pass force_color=True/False to override detection -
    the CLI does, because Rich already resolved the question.

    Lines are not right-padded, so a caller aligning text beside the mascot
    must pad to a common width itself.
    """
    if indent < 0:
        raise ValueError("indent cannot be negative")

    palette = {
        "B": _rgb(color),
        "W": _rgb(outline),
        " ": None,
    }

    colored = _use_color(stream, force_color)
    prefix = " " * indent
    lines: list[str] = []

    for y in range(0, len(ALAN_PIXELS), 2):
        top = ALAN_PIXELS[y]
        bottom = ALAN_PIXELS[y + 1]
        cells: list[str] = []

        for top_pixel, bottom_pixel in zip(top, bottom):
            if colored:
                cells.append(
                    _colored_cell(palette[top_pixel], palette[bottom_pixel])
                )
            else:
                cells.append(
                    _plain_cell(top_pixel != " ", bottom_pixel != " ")
                )

        lines.append(prefix + "".join(cells).rstrip())

    return "\n".join(lines)
