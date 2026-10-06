"""ViewImageTool - show an image file to a model that can see."""

import base64
import os
from typing import Any

from alancode.messages.types import ImageBlock, TextBlock
from alancode.tools.base import Tool, ToolResult, ToolUseContext

# Checked on the file's leading bytes, not its name: a renamed file must not
# reach the server as a media type it is not.
_SIGNATURES: tuple[tuple[bytes, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
)
_WEBP_RIFF, _WEBP_TAG = b"RIFF", b"WEBP"

# The smallest per-image limit among the providers alancode talks to.
MAX_IMAGE_BYTES = 5 * 1024 * 1024

VIEW_IMAGE_TOOL_NAME = "ViewImage"


def _media_type(head: bytes) -> str | None:
    for signature, media_type in _SIGNATURES:
        if head.startswith(signature):
            return media_type
    if head[:4] == _WEBP_RIFF and head[8:12] == _WEBP_TAG:
        return "image/webp"
    return None


class ViewImageTool(Tool):
    """Return an image file to the model as an image, not as text."""

    @property
    def name(self) -> str:
        return VIEW_IMAGE_TOOL_NAME

    @property
    def description(self) -> str:
        return (
            "Shows you an image file (PNG, JPEG, GIF or WEBP) from the working "
            "directory. Use it to look at a picture; use the other tools for "
            "text files.\n\n"
            "Usage:\n"
            "- file_path is the image's path, absolute or relative to the "
            "working directory. The file must be inside the working directory.\n"
            f"- The file must be at most {MAX_IMAGE_BYTES // (1024 * 1024)} MB."
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path of the image file to look at.",
                },
            },
            "required": ["file_path"],
        }

    def permission_level(self, args: dict[str, Any]) -> str:
        return "read"

    async def call(self, args: dict[str, Any], context: ToolUseContext) -> ToolResult:
        if not (context.settings or {}).get("vision"):
            return ToolResult(
                data="Images cannot be shown in this session: the 'vision' "
                     "setting is off.",
                is_error=True,
            )
        file_path = args.get("file_path", "")
        if not isinstance(file_path, str) or not file_path.strip():
            return ToolResult(
                data="Error: 'file_path' parameter is required but was not provided.",
                is_error=True,
            )

        root = os.path.realpath(context.cwd)
        path = os.path.realpath(os.path.join(root, file_path))
        if os.path.commonpath([root, path]) != root:
            return ToolResult(
                data=f"Error: {file_path} is outside the working directory. "
                     "Only images inside it can be shown.",
                is_error=True,
            )
        if not os.path.isfile(path):
            return ToolResult(data=f"Error: no such file: {file_path}", is_error=True)
        size = os.path.getsize(path)
        if size > MAX_IMAGE_BYTES:
            return ToolResult(
                data=f"Error: {file_path} is {size:,} bytes; the limit is "
                     f"{MAX_IMAGE_BYTES:,}. Save a smaller version and view that.",
                is_error=True,
            )
        try:
            with open(path, "rb") as f:
                raw = f.read()
        except OSError as exc:
            return ToolResult(data=f"Error reading {file_path}: {exc}", is_error=True)

        media_type = _media_type(raw[:12])
        if media_type is None:
            return ToolResult(
                data=f"Error: {file_path} is not a PNG, JPEG, GIF or WEBP image.",
                is_error=True,
            )
        return ToolResult(data=[
            TextBlock(text=f"Image {os.path.relpath(path, root)} ({media_type}, {size:,} bytes):"),
            ImageBlock(source={
                "type": "base64",
                "media_type": media_type,
                "data": base64.b64encode(raw).decode("ascii"),
            }),
        ])
