"""Single tool execution with validation and permission checking."""

import logging
from typing import Callable, Awaitable

from alancode.tools.base import Tool, ToolResult, ToolUseContext
from alancode.messages.types import ToolUseBlock, UserMessage
from alancode.messages.factory import create_tool_result_message
from alancode.permissions.context import PermissionResult, PermissionBehavior
from alancode.hooks.registry import run_pre_tool_hooks, run_post_tool_hooks

logger = logging.getLogger(__name__)

# Type alias for the permission callback
PermissionCallback = Callable[
    [Tool, dict, ToolUseContext], Awaitable[PermissionResult]
]


async def run_tool_use(
    tool_use: ToolUseBlock,
    tool: Tool,
    context: ToolUseContext,
    permission_callback: PermissionCallback | None = None,
) -> UserMessage:
    """Execute a single tool call with validation and permission checking.

    Steps:
    1. Validate input via tool.validate_input()
    2. Check permissions via permission_callback (if provided)
    3. Execute tool via tool.call()
    4. Build and return tool_result message

    On any error, returns an error tool_result message.
    """
    tool_use_id = tool_use.id
    args = tool_use.input

    # 1. Validate input
    try:
        validation_error = tool.validate_input(args, context)
    except Exception as exc:
        logger.error("Validation crashed for tool %s: %s", tool.name, exc)
        return _error_result(tool_use_id, f"Input validation error: {exc}")

    if validation_error is not None:
        logger.warning(
            "Validation failed for tool %s: %s", tool.name, validation_error
        )
        return _error_result(tool_use_id, validation_error)

    # 2. Pre-tool-use hooks
    try:
        hook_result = await run_pre_tool_hooks(tool.name, args, settings=context.settings)
    except Exception as exc:
        logger.error("Pre-tool hook crashed for tool %s: %s", tool.name, exc)
        hook_result = None

    if hook_result is not None:
        if hook_result.action == "deny":
            message = hook_result.message or f"Blocked by hook: {hook_result.hook_name}"
            logger.info("Tool %s denied by hook: %s", tool.name, message)
            return _error_result(tool_use_id, message)
        # "ask" falls through to normal permission check below

    # 3. Check permissions (ASK from hooks falls through here)
    if permission_callback is not None:
        try:
            perm = await permission_callback(tool, args, context)
        except Exception as exc:
            logger.error(
                "Permission callback crashed for tool %s: %s", tool.name, exc
            )
            return _error_result(
                tool_use_id, f"Permission check error: {exc}"
            )

        if perm.behavior == PermissionBehavior.DENY:
            message = perm.message or "Permission denied."
            logger.info("Tool %s denied: %s", tool.name, message)
            return _error_result(tool_use_id, message)

        if perm.behavior == PermissionBehavior.ASK:
            # ASK without explicit ALLOW is treated as a denial at this layer.
            # Higher-level code should resolve ASK before calling run_tool_use.
            message = perm.message or "Tool use requires approval but was not approved."
            logger.info("Tool %s requires approval: %s", tool.name, message)
            return _error_result(tool_use_id, message)

        # If the permission hook modified the input, use the updated version
        if perm.updated_input is not None:
            args = perm.updated_input

    # 4. Execute the tool
    try:
        result: ToolResult = await tool.call(args, context)
    except Exception as exc:
        logger.error("Tool %s execution failed: %s", tool.name, exc, exc_info=True)
        # Fire post-tool hooks for failure (fire-and-forget)
        try:
            await run_post_tool_hooks(
                tool.name, args, str(exc), is_error=True, settings=context.settings,
            )
        except Exception:
            logger.debug("Post-tool hook error (ignored)", exc_info=True)
        return _error_result(tool_use_id, f"Tool execution error: {exc}")

    # 5. Post-tool-use hooks (fire-and-forget)
    content = _result_to_str(result)
    try:
        await run_post_tool_hooks(
            tool.name, args, content, is_error=result.is_error, settings=context.settings,
        )
    except Exception:
        logger.debug("Post-tool hook error (ignored)", exc_info=True)

    # 6. Build the tool_result message
    return create_tool_result_message(
        tool_use_id=tool_use_id,
        content=content,
        is_error=result.is_error,
    )


def _error_result(tool_use_id: str, message: str) -> UserMessage:
    """Build an error tool_result message."""
    return create_tool_result_message(
        tool_use_id=tool_use_id,
        content=message,
        is_error=True,
    )


def _result_to_str(result: ToolResult) -> str:
    """Convert a ToolResult's data to a string suitable for the API."""
    if isinstance(result.data, str):
        return result.data
    if result.data is None:
        return ""
    return str(result.data)
