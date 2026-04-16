"""Shared test fixtures."""

import pytest
from alancode.providers.scripted_provider import ScriptedProvider, ScriptedResponse, text, tool_call

from alancode.tools.base import Tool, ToolResult, ToolUseContext


@pytest.fixture
def scripted_provider():
    return ScriptedProvider.from_responses([])


@pytest.fixture
def agent(scripted_provider):
    from alancode.agent import AlanCodeAgent
    return AlanCodeAgent(provider=scripted_provider, cwd="/tmp/test")


# ── Git test repo fixture ───────────────────────────────────────────────────

@pytest.fixture
def git_repo(tmp_path):
    """Create a temporary git repo for testing. Cleaned up automatically."""
    from tests.integration.git_helpers import GitTestRepo
    repo = GitTestRepo(tmp_path / "test_repo")
    repo.init()
    return repo


# A simple echo tool for testing tool execution
class EchoTool(Tool):
    """Test tool that echoes its input."""

    @property
    def name(self):
        return "Echo"

    @property
    def aliases(self):
        return ["echo", "EchoAlias"]

    @property
    def description(self):
        return "Echoes the input text"

    @property
    def input_schema(self):
        return {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        }

    async def call(self, args, context):
        return ToolResult(data=f"Echo: {args.get('text', '')}")

    def permission_level(self, args):
        return "read"


class MutateTool(Tool):
    """Test tool that pretends to mutate something (write level)."""

    @property
    def name(self):
        return "Mutate"

    @property
    def description(self):
        return "Mutates something"

    @property
    def input_schema(self):
        return {
            "type": "object",
            "properties": {"target": {"type": "string"}},
            "required": ["target"],
        }

    async def call(self, args, context):
        return ToolResult(data=f"Mutated: {args.get('target', '')}")

    def permission_level(self, args):
        return "write"


@pytest.fixture
def echo_tool():
    return EchoTool()


@pytest.fixture
def mutate_tool():
    return MutateTool()
