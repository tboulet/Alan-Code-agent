"""Minimal interactive CLI agent built on AlanCodeAgent."""

import asyncio

from alancode import AlanCodeAgent

agent = AlanCodeAgent(backend="auto", model="openrouter/anthropic/claude-haiku-5.5")

try:
    while True:
        try:
            message = input("> ")
        except (EOFError, KeyboardInterrupt):
            break
        if message.strip():
            print(agent.query(message))
finally:
    asyncio.run(agent.close())
