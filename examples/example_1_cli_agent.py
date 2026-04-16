"""Minimal interactive CLI agent built on AlanCodeAgent."""

from alancode import AlanCodeAgent

agent = AlanCodeAgent(provider="litellm", model="openrouter/google/gemini-2.5-flash")

while True:
    try:
        message = input("> ")
    except (EOFError, KeyboardInterrupt):
        break
    if message.strip():
        print(agent.query(message))
