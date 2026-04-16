"""Stream assistant text and tool calls in real time.

Useful when plugging AlanCodeAgent into a custom UI (web app, TUI, WebSocket
bridge, ...) — you receive events as the agent produces them, instead of
waiting for the final answer.
"""

import asyncio

from alancode import AlanCodeAgent
from alancode.messages.types import AssistantMessage, TextBlock, ToolUseBlock


async def main() -> None:
    agent = AlanCodeAgent(permission_mode="yolo", provider="litellm", model="openrouter/google/gemini-2.5-flash")

    async for event in agent.query_events_async(
        "List the files here, then tell me what the project is about."
    ):
        if not isinstance(event, AssistantMessage):
            continue
        # Streaming deltas carry text chunks (hide_in_api=True).
        # Final assembled messages carry tool calls (hide_in_api=False).
        for block in event.content:
            if event.hide_in_api and isinstance(block, TextBlock):
                print(block.text, end="", flush=True)
            elif not event.hide_in_api and isinstance(block, ToolUseBlock):
                print(f"\n[tool: {block.name}({block.input})]")
    print()


if __name__ == "__main__":
    asyncio.run(main())
