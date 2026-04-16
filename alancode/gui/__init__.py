"""Alan Code GUI — SessionUI interface with CLI and GUI implementations.

Usage::

    # CLI mode (default)
    from alancode.gui.cli_ui import CLIUI
    ui = CLIUI()

    # GUI mode (--gui)
    from alancode.gui.gui_ui import GUIUI
    ui = GUIUI(agent, cwd)
    await ui.start()

    # Testing
    from alancode.gui.scripted_ui import ScriptedUI
    ui = ScriptedUI.from_inputs(["Fix the bug", EOFError])
"""

from alancode.gui.base import SessionUI

__all__ = ["SessionUI"]
