"""Tests for ScriptedUI and git test infrastructure.

Validates:
1. ScriptedUI sequential and reactive modes
2. ScriptedUI + ScriptedProvider + agent integration
3. ScriptedUI + run_session full loop
4. Git test repo helpers
5. Combined: ScriptedUI + agent + git repo (AGT test foundation)
"""

import asyncio
import pytest

from alancode.agent import AlanCodeAgent
from alancode.cli.repl import run_session
from alancode.gui.scripted_ui import ScriptedUI, UIContext, UIRule, ui_rule
from alancode.messages.types import AssistantMessage, RequestStartEvent, Usage
from alancode.providers.scripted_provider import (
    ScriptedProvider,
    rule,
    text,
    tool_call,
)
from tests.integration.git_helpers import GitTestRepo


# ═══════════════════════════════════════════════════════════════════════════════
# ScriptedUI unit tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestScriptedUISequential:
    """Test FIFO input mode."""

    @pytest.mark.asyncio
    async def test_basic_inputs(self):
        ui = ScriptedUI.from_inputs(["hello", "world", EOFError])
        assert await ui.get_input("> ") == "hello"
        assert await ui.get_input("> ") == "world"
        with pytest.raises(EOFError):
            await ui.get_input("> ")

    @pytest.mark.asyncio
    async def test_ask_user(self):
        ui = ScriptedUI.from_inputs(["prompt1", "yes", EOFError])
        assert await ui.get_input("> ") == "prompt1"
        assert await ui.ask_user("Continue?", ["yes", "no"]) == "yes"
        with pytest.raises(EOFError):
            await ui.get_input("> ")

    @pytest.mark.asyncio
    async def test_input_log(self):
        ui = ScriptedUI.from_inputs(["fix bug", "yes"])
        await ui.get_input("> ")
        await ui.ask_user("Approve?", ["yes", "no"])

        assert len(ui.input_log) == 2
        assert ui.input_log[0]["type"] == "prompt"
        assert ui.input_log[0]["response"] == "fix bug"
        assert ui.input_log[1]["type"] == "ask"
        assert ui.input_log[1]["question"] == "Approve?"
        assert ui.input_log[1]["options"] == ["yes", "no"]

    @pytest.mark.asyncio
    async def test_no_matching_rule_raises_eof(self):
        """When no rule matches, EOFError is raised."""
        ui = ScriptedUI.from_inputs([])  # No inputs at all
        with pytest.raises(EOFError):
            await ui.get_input("> ")

    @pytest.mark.asyncio
    async def test_fallback(self):
        ui = ScriptedUI.from_inputs(["first"], fallback="fallback_answer")
        assert await ui.get_input("> ") == "first"
        assert await ui.get_input("> ") == "fallback_answer"
        assert await ui.get_input("> ") == "fallback_answer"

    @pytest.mark.asyncio
    async def test_prompt_responses_helper(self):
        ui = ScriptedUI.from_inputs(["a", "b", "c"])
        await ui.get_input()
        await ui.ask_user("q?", [])
        await ui.get_input()
        assert ui.prompt_responses == ["a", "c"]
        assert ui.ask_responses == ["b"]


class TestScriptedUIReactive:
    """Test rule-based input mode."""

    @pytest.mark.asyncio
    async def test_turn_matching(self):
        ui = ScriptedUI(rules=[
            ui_rule("first", turn=0),
            ui_rule("second", turn=1),
            ui_rule(EOFError),
        ])
        assert await ui.get_input() == "first"
        assert await ui.get_input() == "second"
        with pytest.raises(EOFError):
            await ui.get_input()

    @pytest.mark.asyncio
    async def test_input_type_matching(self):
        ui = ScriptedUI(rules=[
            ui_rule("prompt_answer", input_type="prompt"),
            ui_rule("ask_answer", input_type="ask"),
        ])
        assert await ui.get_input() == "prompt_answer"
        assert await ui.ask_user("q?", []) == "ask_answer"
        # Second prompt still matches prompt rule
        assert await ui.get_input() == "prompt_answer"

    @pytest.mark.asyncio
    async def test_condition_matching(self):
        ui = ScriptedUI(rules=[
            ui_rule("first", turn=0),
            ui_rule("saw_permission",
                    input_type="ask",
                    condition=lambda ctx: "permission" in ctx.last_question.lower()),
            ui_rule("default_ask", input_type="ask"),
            ui_rule(EOFError),
        ])
        assert await ui.get_input() == "first"
        assert await ui.ask_user("Grant permission?", ["yes", "no"]) == "saw_permission"
        assert await ui.ask_user("Other question?", ["a", "b"]) == "default_ask"
        with pytest.raises(EOFError):
            await ui.get_input()


class TestScriptedUILogging:
    """Test event and lifecycle logging."""

    @pytest.mark.asyncio
    async def test_event_logging(self):
        ui = ScriptedUI.from_inputs([])
        event = AssistantMessage(
            content=[], model="test", stop_reason="end_turn",
        )
        await ui.on_agent_event(event)
        assert len(ui.event_log) == 1
        assert ui.event_log[0]["type"] == "AssistantMessage"

    @pytest.mark.asyncio
    async def test_cost_logging(self):
        ui = ScriptedUI.from_inputs([])
        usage = Usage(input_tokens=100, output_tokens=50)
        await ui.on_cost(usage, 0.01, False, 500, 200000)
        assert len(ui.cost_log) == 1
        assert ui.cost_log[0]["input_tokens"] == 100
        assert ui.cost_log[0]["conversation_tokens"] == 500

    def test_lifecycle_logging(self):
        ui = ScriptedUI.from_inputs([])
        ui.on_agent_start()
        ui.on_agent_done()
        ui.reset_stream_state(assume_thinking=True)
        assert ui.lifecycle_log == [
            "agent_start", "agent_done", "reset_stream(thinking=True)",
        ]

    def test_console_capture(self):
        ui = ScriptedUI.from_inputs([])
        ui.console.print("Hello world")
        ui.console.print("Line two")
        assert len(ui.console_log) == 2
        assert "Hello world" in ui.console_log[0]
        assert "Line two" in ui.console_log[1]


# ═══════════════════════════════════════════════════════════════════════════════
# ScriptedUI + Agent integration
# ═══════════════════════════════════════════════════════════════════════════════


class TestScriptedUIWithAgent:
    """Test ScriptedUI driving an actual agent with ScriptedProvider."""

    @pytest.mark.asyncio
    async def test_single_turn(self, tmp_path):
        provider = ScriptedProvider.from_responses([text("Hello!")])
        ui = ScriptedUI.from_inputs(["Hi there", EOFError])
        agent = AlanCodeAgent(provider=provider, cwd=str(tmp_path))

        # Simulate one turn
        user_input = await ui.get_input()
        assert user_input == "Hi there"

        ui.on_agent_start()
        async for event in agent.query_events_async(user_input):
            await ui.on_agent_event(event)
        await ui.on_cost(agent.usage, agent.cost_usd, agent.cost_unknown)
        ui.on_agent_done()

        # Verify events were logged
        event_types = [e["type"] for e in ui.event_log]
        assert "AssistantMessage" in event_types
        assert "RequestStartEvent" in event_types
        assert len(ui.cost_log) == 1
        assert ui.lifecycle_log == [
            "reset_stream(thinking=False)",  # Won't appear without _handle_prompt
            "agent_start", "agent_done",
        ] or ui.lifecycle_log == ["agent_start", "agent_done"]

    @pytest.mark.asyncio
    async def test_multi_turn(self, tmp_path):
        provider = ScriptedProvider.from_responses([
            text("First response."),
            text("Second response."),
        ])
        ui = ScriptedUI.from_inputs(["Turn 1", "Turn 2", EOFError])
        agent = AlanCodeAgent(provider=provider, cwd=str(tmp_path))

        for _ in range(2):
            user_input = await ui.get_input()
            async for event in agent.query_events_async(user_input):
                await ui.on_agent_event(event)

        assert provider._call_count == 2
        assert len(ui.input_log) == 2  # Two prompts consumed

    @pytest.mark.asyncio
    async def test_with_tool_call(self, tmp_path):
        """Agent makes a tool call, ScriptedUI logs all events."""
        provider = ScriptedProvider.from_responses([
            tool_call("Read", {"file_path": str(tmp_path / "test.txt")}),
            text("I read the file."),
        ])
        ui = ScriptedUI.from_inputs(["Read test.txt", EOFError])

        # Create the file so Read tool works
        (tmp_path / "test.txt").write_text("hello")

        agent = AlanCodeAgent(provider=provider, cwd=str(tmp_path))
        user_input = await ui.get_input()
        async for event in agent.query_events_async(user_input):
            await ui.on_agent_event(event)

        # Should have multiple events (request start, user msg, assistant msg, etc.)
        assert len(ui.event_log) >= 3
        assert provider._call_count == 2  # Two LLM calls (tool_call + final)

    @pytest.mark.asyncio
    async def test_ask_callback_integration(self, tmp_path):
        """Verify ask_callback wired to ScriptedUI.ask_user works."""
        ui = ScriptedUI(rules=[
            ui_rule("do it", turn=0, input_type="prompt"),
            ui_rule("yes", input_type="ask"),
            ui_rule(EOFError, input_type="prompt"),
        ])

        provider = ScriptedProvider.from_responses([text("Done.")])
        agent = AlanCodeAgent(
            provider=provider,
            cwd=str(tmp_path),
            ask_callback=ui.ask_user,
        )

        # Simulate asking the user through the callback
        result = await ui.ask_user("Allow write?", ["yes", "no"])
        assert result == "yes"
        assert ui.input_log[0]["question"] == "Allow write?"


# ═══════════════════════════════════════════════════════════════════════════════
# ScriptedUI + run_session (full loop)
# ═══════════════════════════════════════════════════════════════════════════════


class TestRunSessionWithScriptedUI:
    """Test the full session loop with ScriptedUI."""

    @pytest.mark.asyncio
    async def test_single_prompt_then_exit(self, tmp_path):
        provider = ScriptedProvider.from_responses([
            text("I'll help you fix that bug."),
        ])
        ui = ScriptedUI.from_inputs([
            "Fix the bug",
            EOFError,
        ])
        agent = AlanCodeAgent(provider=provider, cwd=str(tmp_path))

        await run_session(agent, ui)

        # Verify the session ran
        assert len(ui.input_log) == 2  # "Fix the bug" + EOFError
        assert ui.input_log[0]["response"] == "Fix the bug"

        # Verify events were produced
        event_types = [e["type"] for e in ui.event_log]
        assert "AssistantMessage" in event_types

        # Verify cost was reported
        assert len(ui.cost_log) == 1

        # Verify lifecycle
        assert "agent_start" in ui.lifecycle_log
        assert "agent_done" in ui.lifecycle_log

    @pytest.mark.asyncio
    async def test_slash_exit(self, tmp_path):
        provider = ScriptedProvider.from_responses([])
        ui = ScriptedUI.from_inputs(["/exit"])
        agent = AlanCodeAgent(provider=provider, cwd=str(tmp_path))

        await run_session(agent, ui)

        # /exit should not trigger an LLM call
        assert provider._call_count == 0
        assert ui.input_log[0]["response"] == "/exit"

    @pytest.mark.asyncio
    async def test_slash_help(self, tmp_path):
        provider = ScriptedProvider.from_responses([])
        ui = ScriptedUI.from_inputs(["/help", EOFError])
        agent = AlanCodeAgent(provider=provider, cwd=str(tmp_path))

        await run_session(agent, ui)

        # /help should produce console output
        assert provider._call_count == 0
        assert any("help" in line.lower() or "exit" in line.lower()
                    for line in ui.console_log)

    @pytest.mark.asyncio
    async def test_empty_input_skipped(self, tmp_path):
        provider = ScriptedProvider.from_responses([text("Response.")])
        ui = ScriptedUI.from_inputs(["", "", "actual prompt", EOFError])
        agent = AlanCodeAgent(provider=provider, cwd=str(tmp_path))

        await run_session(agent, ui)

        # Empty inputs should be skipped, only one LLM call
        assert provider._call_count == 1

    @pytest.mark.asyncio
    async def test_multiple_turns(self, tmp_path):
        provider = ScriptedProvider.from_responses([
            text("First answer."),
            text("Second answer."),
        ])
        ui = ScriptedUI.from_inputs(["Question 1", "Question 2", EOFError])
        agent = AlanCodeAgent(provider=provider, cwd=str(tmp_path))

        await run_session(agent, ui)

        assert provider._call_count == 2
        assert len(ui.cost_log) == 2


# ═══════════════════════════════════════════════════════════════════════════════
# Git test repo tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestGitTestRepo:
    """Test the git test infrastructure itself."""

    def test_init(self, git_repo: GitTestRepo):
        assert git_repo.current_branch() == "main"
        assert not git_repo.is_dirty()
        assert len(git_repo.log_hashes()) == 1  # Initial commit

    def test_linear_history(self, git_repo: GitTestRepo):
        shas = git_repo.build_linear_history(5)
        assert len(shas) == 5

        log = git_repo.log_hashes()
        # Newest first: shas[4], shas[3], ..., shas[0], initial
        assert log[0] == shas[4]
        assert log[4] == shas[0]

        # Parent chain
        assert git_repo.parent_sha(shas[2]) == shas[1]
        assert git_repo.parent_sha(shas[0]) is not None  # has initial commit parent

    def test_branching_history(self, git_repo: GitTestRepo):
        result = git_repo.build_branching_history()

        branches = git_repo.branches()
        assert "main" in branches
        assert "feature" in branches

        # HEAD should be the merge commit
        assert git_repo.head_sha() == result["c5"]
        assert "merge" in git_repo.commit_message(result["c5"]).lower()

    def test_dirty_detection(self, git_repo: GitTestRepo):
        assert not git_repo.is_dirty()
        git_repo.write_file("new.txt", "content")
        assert git_repo.is_dirty()
        git_repo.commit("Add new.txt")
        assert not git_repo.is_dirty()

    def test_checkout_and_branch(self, git_repo: GitTestRepo):
        sha1 = git_repo.commit("c1", add_all=False)
        git_repo.write_file("f.py", "x")
        sha2 = git_repo.commit("c2")

        # Create branch at sha1
        git_repo.checkout_new_branch("test-branch", sha1)
        assert git_repo.current_branch() == "test-branch"
        assert git_repo.head_sha() == sha1

        # Back to main
        git_repo.checkout("main")
        assert git_repo.head_sha() == sha2

    def test_memory_helpers(self, git_repo: GitTestRepo):
        git_repo.write_memory("notes.md", "# Notes")
        assert git_repo.read_memory("notes.md") == "# Notes"

        sha = git_repo.head_sha()
        snap_dir = git_repo.memory_snapshot_dir(sha)
        assert snap_dir.exists()
        assert snap_dir.parent.name == "memory_snapshots"

    def test_diff(self, git_repo: GitTestRepo):
        git_repo.write_file("a.py", "line1\n")
        git_repo.commit("add a")
        git_repo.write_file("a.py", "line1\nline2\n")

        diff = git_repo.diff()
        assert "+line2" in diff

    def test_commit_metadata(self, git_repo: GitTestRepo):
        git_repo.write_file("f.py", "x")
        sha = git_repo.commit("Test message")

        assert git_repo.commit_message(sha) == "Test message"
        assert git_repo.commit_author(sha) == "Test User"
        assert git_repo.commit_timestamp(sha)  # Not empty

    def test_all_branches_log(self, git_repo: GitTestRepo):
        git_repo.build_branching_history()
        all_hashes = git_repo.log_hashes(all_branches=True)
        main_hashes = git_repo.log_hashes(all_branches=False)
        # All branches should have at least as many commits
        assert len(all_hashes) >= len(main_hashes)


# ═══════════════════════════════════════════════════════════════════════════════
# Combined: ScriptedUI + Agent + Git Repo
# ═══════════════════════════════════════════════════════════════════════════════


class TestScriptedUIWithGitRepo:
    """Foundation tests for AGT: ScriptedUI + agent in a real git repo."""

    @pytest.mark.asyncio
    async def test_agent_in_git_repo(self, git_repo: GitTestRepo):
        """Agent runs in a git repo, can read files created by test."""
        git_repo.write_file("main.py", "print('hello')")
        git_repo.commit("Add main.py")

        provider = ScriptedProvider.from_responses([
            tool_call("Read", {"file_path": str(git_repo.path / "main.py")}),
            text("The file prints hello."),
        ])
        ui = ScriptedUI.from_inputs(["What does main.py do?", EOFError])

        agent = AlanCodeAgent(provider=provider, cwd=str(git_repo.path))
        await run_session(agent, ui)

        assert provider._call_count == 2
        event_types = [e["type"] for e in ui.event_log]
        assert "AssistantMessage" in event_types

    @pytest.mark.asyncio
    async def test_agent_modifies_git_repo(self, git_repo: GitTestRepo):
        """Agent writes a file in the git repo, we verify git sees it."""
        provider = ScriptedProvider.from_responses([
            tool_call("Write", {
                "file_path": str(git_repo.path / "output.txt"),
                "content": "generated content",
            }),
            text("I wrote the file."),
        ])
        ui = ScriptedUI.from_inputs(["Create output.txt", EOFError])

        agent = AlanCodeAgent(provider=provider, cwd=str(git_repo.path))
        await run_session(agent, ui)

        # Verify the file was created
        assert git_repo.file_exists("output.txt")
        assert git_repo.read_file("output.txt") == "generated content"

        # Verify git sees the change
        assert git_repo.is_dirty()
        assert "output.txt" in git_repo.status_porcelain()

    @pytest.mark.asyncio
    async def test_agent_with_commit_history(self, git_repo: GitTestRepo):
        """Agent runs in a repo with existing commit history."""
        shas = git_repo.build_linear_history(3)

        provider = ScriptedProvider.from_responses([
            tool_call("Bash", {"command": f"cd {git_repo.path} && git log --oneline -5"}),
            text("I see 3 commits."),
        ])
        ui = ScriptedUI.from_inputs(["Show git history", EOFError])

        agent = AlanCodeAgent(provider=provider, cwd=str(git_repo.path))
        await run_session(agent, ui)

        assert provider._call_count == 2

    @pytest.mark.asyncio
    async def test_run_session_in_branched_repo(self, git_repo: GitTestRepo):
        """Full session in a repo with branches — the AGT foundation scenario."""
        result = git_repo.build_branching_history()

        provider = ScriptedProvider.from_responses([
            text("I see a repo with main and feature branches."),
        ])
        ui = ScriptedUI.from_inputs(["Describe the repo", EOFError])

        agent = AlanCodeAgent(provider=provider, cwd=str(git_repo.path))
        await run_session(agent, ui)

        # Session completes cleanly in a branched repo
        assert provider._call_count == 1
        assert len(ui.cost_log) == 1

        # Git state unchanged
        assert git_repo.head_sha() == result["c5"]
        assert git_repo.current_branch() == "main"
