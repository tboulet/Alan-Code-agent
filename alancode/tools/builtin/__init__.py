"""Built-in tools for Alan Code."""

from alancode.tools.builtin.bash import BashTool
from alancode.tools.builtin.file_read import FileReadTool
from alancode.tools.builtin.file_write import FileWriteTool
from alancode.tools.builtin.file_edit import FileEditTool
from alancode.tools.builtin.glob_tool import GlobTool
from alancode.tools.builtin.grep_tool import GrepTool
from alancode.tools.builtin.web_fetch import WebFetchTool
from alancode.tools.builtin.ask_user import AskUserQuestionTool
from alancode.tools.builtin.git_commit import GitCommitTool

ALL_BUILTIN_TOOLS = [
    BashTool(),
    FileReadTool(),
    FileWriteTool(),
    FileEditTool(),
    GlobTool(),
    GrepTool(),
    WebFetchTool(),
    AskUserQuestionTool(),
    GitCommitTool(),
]

__all__ = [
    "BashTool",
    "FileReadTool",
    "FileWriteTool",
    "FileEditTool",
    "GlobTool",
    "GrepTool",
    "WebFetchTool",
    "AskUserQuestionTool",
    "ALL_BUILTIN_TOOLS",
]
