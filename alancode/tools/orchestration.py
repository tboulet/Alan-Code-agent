"""Tool call orchestration — concurrent/serial batching.

Read-only tools run concurrently; mutating tools run serially.
Permission callback is threaded from the query loop through to each tool call.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import AsyncGenerator

from alancode.messages.factory import create_tool_result_message
from alancode.messages.types import ToolUseBlock, UserMessage
from alancode.tools.base import Tool, ToolUseContext
from alancode.tools.execution import PermissionCallback, run_tool_use
from alancode.tools.registry import find_tool_by_name

logger = logging.getLogger(__name__)


@dataclass
class ToolUpdate:
    """Result of a single tool execution within orchestration."""
    message: UserMessage | None = None  # The tool_result message
    tool_use_id: str = ""


@dataclass
class _Batch:
    """A group of tool-use blocks that share the same concurrency mode."""
    blocks: list[ToolUseBlock] = field(default_factory=list)
    is_concurrent: bool = False


# ---------------------------------------------------------------------------
# Partitioning
# ---------------------------------------------------------------------------


def partition_tool_calls(
    tool_use_blocks: list[ToolUseBlock],
    tools: list[Tool],
) -> list[_Batch]:
    """Split tool calls into consecutive batches of read-only (concurrent)
    and mutating (serial) calls.

    Consecutive read-only calls are grouped into a single concurrent batch.
    Each mutating call gets its own serial batch (is_concurrent=False).
    Unknown tools are treated as mutating to be safe.
    """
    if not tool_use_blocks:
        return []

    batches: list[_Batch] = []
    current_batch: _Batch | None = None

    for block in tool_use_blocks:
        tool = find_tool_by_name(tools, block.name)
        is_ro = tool is not None and tool.permission_level(block.input) == "read"

        if is_ro:
            # Extend or start a concurrent batch
            if current_batch is not None and current_batch.is_concurrent:
                current_batch.blocks.append(block)
            else:
                current_batch = _Batch(blocks=[block], is_concurrent=True)
                batches.append(current_batch)
        else:
            # Each mutating call is its own serial batch
            current_batch = _Batch(blocks=[block], is_concurrent=False)
            batches.append(current_batch)

    return batches


# ---------------------------------------------------------------------------
# Execution helpers
# ---------------------------------------------------------------------------


async def _execute_single_tool(
    block: ToolUseBlock,
    tools: list[Tool],
    context: ToolUseContext,
    permission_callback: PermissionCallback | None = None,
) -> ToolUpdate:
    """Find the tool, validate, execute, and wrap the result in a ToolUpdate."""
    tool = find_tool_by_name(tools, block.name)

    if tool is None:
        msg = create_tool_result_message(
            tool_use_id=block.id,
            content=f"Unknown tool: {block.name}",
            is_error=True,
        )
        return ToolUpdate(message=msg, tool_use_id=block.id)

    message = await run_tool_use(
        tool_use=block,
        tool=tool,
        context=context,
        permission_callback=permission_callback,
    )
    return ToolUpdate(message=message, tool_use_id=block.id)


async def _run_tools_concurrently(
    blocks: list[ToolUseBlock],
    tools: list[Tool],
    context: ToolUseContext,
    *,
    max_concurrency: int = 10,
    permission_callback: PermissionCallback | None = None,
) -> AsyncGenerator[ToolUpdate, None]:
    """Run a batch of read-only tool calls concurrently with a semaphore."""
    semaphore = asyncio.Semaphore(max_concurrency)

    async def _guarded(block: ToolUseBlock) -> ToolUpdate:
        async with semaphore:
            return await _execute_single_tool(block, tools, context, permission_callback)

    tasks = [asyncio.create_task(_guarded(b)) for b in blocks]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    aborted = (
        context.abort_signal is not None and context.abort_signal.is_set()
    )

    for i, result in enumerate(results):
        if isinstance(result, BaseException):
            block = blocks[i]
            if isinstance(result, asyncio.CancelledError) and aborted:
                # User cancelled (Ctrl+C at a UI prompt). Not a crash.
                logger.info("Tool %s cancelled by user", block.name)
                msg = create_tool_result_message(
                    tool_use_id=block.id,
                    content="Tool interrupted by user.",
                    is_error=True,
                )
            else:
                logger.error(
                    "Concurrent tool %s raised: %s", block.name, result, exc_info=result
                )
                msg = create_tool_result_message(
                    tool_use_id=block.id,
                    content=f"Tool execution error: {result}",
                    is_error=True,
                )
            yield ToolUpdate(message=msg, tool_use_id=block.id)
        else:
            yield result


async def _run_tools_serially(
    blocks: list[ToolUseBlock],
    tools: list[Tool],
    context: ToolUseContext,
    permission_callback: PermissionCallback | None = None,
) -> AsyncGenerator[ToolUpdate, None]:
    """Run tool calls one at a time."""
    for block in blocks:
        update = await _execute_single_tool(block, tools, context, permission_callback)
        yield update


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def run_tools(
    tool_use_blocks: list[ToolUseBlock],
    tools: list[Tool],
    context: ToolUseContext,
    *,
    max_concurrency: int = 10,
    permission_callback: PermissionCallback | None = None,
) -> AsyncGenerator[ToolUpdate, None]:
    """Execute tool calls with concurrent/serial batching.

    Partitions tool calls into batches:
    - Consecutive read-only tools -> run concurrently (up to *max_concurrency*)
    - Mutating tools -> run one at a time

    The *permission_callback* is called before each tool execution to check
    permissions (hooks run first, then the callback). See execution.py for
    the full order: validate -> hooks -> permissions -> tool.call() -> post-hooks.

    Yields a ToolUpdate for each completed tool.
    """
    for batch in partition_tool_calls(tool_use_blocks, tools):
        if batch.is_concurrent:
            async for update in _run_tools_concurrently(
                batch.blocks, tools, context,
                max_concurrency=max_concurrency,
                permission_callback=permission_callback,
            ):
                yield update
        else:
            async for update in _run_tools_serially(
                batch.blocks, tools, context,
                permission_callback=permission_callback,
            ):
                yield update
