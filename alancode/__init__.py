"""Alan Code — an open-source, provider-agnostic coding agent.

Simple usage::

    from alancode import AlanCodeAgent

    agent = AlanCodeAgent(provider="litellm", model="openrouter/google/gemini-2.5-flash")
    answer = agent.query("What files are in this project?")
    print(answer)

Streaming::

    for event in agent.query_events("Fix the bug"):
        print(event)

Async::

    async for event in agent.query_events_async("Fix the bug"):
        ...
"""

from alancode.__version__ import __version__
from alancode.agent import AlanCodeAgent

__all__ = ["AlanCodeAgent", "__version__"]
