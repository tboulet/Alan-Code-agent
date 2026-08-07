"""Alan Code — an open-source, backend-agnostic coding agent.

Simple usage::

    from alancode import AlanCodeAgent

    agent = AlanCodeAgent(model="openrouter/google/gemini-2.5-flash")
    answer = agent.query("What files are in this project?")
    print(answer)

The transport backend is inferred from the model string. A bare Claude
name (``claude-sonnet-4-6``) uses the native Anthropic SDK; any other
model goes through LiteLLM. Pass ``backend="anthropic-native" | "auto" |
"scripted"`` to override, or an ``LLMBackend`` instance for custom
transports.

Streaming::

    for event in agent.query_events("Fix the bug"):
        print(event)

Async::

    async for event in agent.query_events_async("Fix the bug"):
        ...
"""

import os

# LiteLLM otherwise attempts to refresh its model-cost map during import. Set
# the offline default at the package boundary so every Alan import path is
# deterministic, including utilities that import LiteLLM before a backend is
# constructed. An explicit caller-provided value still wins.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

from alancode.__version__ import __version__
from alancode.agent import AlanCodeAgent

__all__ = ["AlanCodeAgent", "__version__"]
