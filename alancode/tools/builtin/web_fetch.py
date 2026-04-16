"""WebFetchTool — fetch URL content via httpx."""

import re
from typing import Any

from alancode.tools.base import Tool, ToolResult, ToolUseContext

_DEFAULT_MAX_LENGTH = 50_000
_TIMEOUT_SECONDS = 30


def _strip_html_tags(html: str) -> str:
    """Rough HTML-to-text: strip tags, collapse whitespace."""
    # Remove script and style blocks
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html, flags=re.DOTALL | re.IGNORECASE)
    # Remove all tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Decode common entities
    for entity, char in [("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"),
                          ("&quot;", '"'), ("&#39;", "'"), ("&nbsp;", " ")]:
        text = text.replace(entity, char)
    # Collapse whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


class WebFetchTool(Tool):
    """Fetch and return content from a URL, stripping HTML tags."""

    @property
    def name(self) -> str:
        return "WebFetch"

    @property
    def description(self) -> str:
        return (
            "Fetches content from a specified URL and returns it. "
            "HTML tags are stripped for cleaner output.\n\n"
            "Usage notes:\n"
            "- The URL must be a fully-formed valid URL (http:// or https://)\n"
            "- This tool is read-only and does not modify any files\n"
            "- Results may be truncated if the content is very large\n"
            "- For GitHub URLs, prefer using the gh CLI via Bash instead "
            "(e.g., gh pr view, gh issue view)"
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch content from (must start with http:// or https://).",
                },
                "max_length": {
                    "type": "integer",
                    "description": "Maximum number of characters to return. Default 50000.",
                },
            },
            "required": ["url"],
        }

    def permission_level(self, args: dict[str, Any]) -> str:
        return "read"

    async def call(self, args: dict[str, Any], context: ToolUseContext) -> ToolResult:
        url = args.get("url", "")
        max_length = args.get("max_length", _DEFAULT_MAX_LENGTH)

        if not url:
            given_keys = list(args.keys())
            return ToolResult(
                data=f"Error: 'url' parameter is required but was not provided. "
                     f"Got parameters: {given_keys}. "
                     f"Use <arg_key>url</arg_key><arg_value>https://example.com</arg_value>",
                is_error=True,
            )

        # Validate URL scheme
        if not url.startswith(("http://", "https://")):
            return ToolResult(data="Error: url must start with http:// or https://", is_error=True)

        try:
            import httpx
        except ImportError:
            return ToolResult(
                data="Error: httpx is not installed. Run 'pip install httpx' to enable web fetching.",
                is_error=True,
            )

        try:
            async with httpx.AsyncClient(
                follow_redirects=True,
                timeout=httpx.Timeout(_TIMEOUT_SECONDS),
            ) as client:
                response = await client.get(url)
        except httpx.TimeoutException:
            return ToolResult(data=f"Error: request timed out after {_TIMEOUT_SECONDS}s.", is_error=True)
        except httpx.TooManyRedirects:
            return ToolResult(data="Error: too many redirects.", is_error=True)
        except Exception as exc:
            return ToolResult(data=f"Error fetching URL: {exc}", is_error=True)

        if response.status_code >= 400:
            return ToolResult(
                data=f"HTTP {response.status_code}: {response.reason_phrase or 'Error'}",
                is_error=True,
            )

        content_type = response.headers.get("content-type", "")
        body = response.text

        # Strip HTML if the response looks like HTML
        if "html" in content_type.lower() or body.lstrip()[:15].lower().startswith(("<!doctype", "<html")):
            body = _strip_html_tags(body)

        # Truncate to max_length
        truncated = False
        if len(body) > max_length:
            body = body[:max_length]
            truncated = True

        header = f"URL: {url}\nStatus: {response.status_code}\nContent-Type: {content_type}\n\n"
        output = header + body
        if truncated:
            output += f"\n\n(Truncated to {max_length} characters.)"

        return ToolResult(data=output)
