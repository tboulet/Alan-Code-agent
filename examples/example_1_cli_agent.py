"""Minimal interactive CLI agent built on AlanCodeAgent."""

import asyncio

from alancode import AlanCodeAgent

agent = AlanCodeAgent(backend="auto", model="openrouter/google/gemini-2.5-flash")

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
