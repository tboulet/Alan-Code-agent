"""Permission system types — modes, rules, decision results."""
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal


class PermissionMode(str, Enum):
    """Permission mode controlling which tool categories require user approval."""
    YOLO = "yolo"      # Allow everything without asking
    EDIT = "edit"      # Allow read + write, ask for exec (Bash)
    SAFE = "safe"      # Allow read, ask for write + exec


class PermissionBehavior(str, Enum):
    """Outcome of a permission check for a single tool invocation."""
    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"
    PASSTHROUGH = "passthrough"  # No opinion


@dataclass
class PermissionRule:
    """A single allow/deny/ask rule matching a tool name and optional content pattern."""
    tool_name: str
    rule_content: str | None = None  # e.g., "git *" for "Bash(git *)"
    behavior: PermissionBehavior = PermissionBehavior.ALLOW
    source: str = "default"  # 'settings', 'cli_arg', 'command', 'session'


@dataclass
class PermissionResult:
    """Result returned by the permission pipeline for a tool call."""
    behavior: PermissionBehavior
    message: str = ""
    updated_input: dict | None = None  # Hook can modify input


@dataclass
class ToolPermissionContext:
    """Aggregated permission state (mode + rules) for the current session."""
    mode: PermissionMode = PermissionMode.EDIT
    allow_rules: list[PermissionRule] = field(default_factory=list)
    deny_rules: list[PermissionRule] = field(default_factory=list)
    ask_rules: list[PermissionRule] = field(default_factory=list)
    should_avoid_prompts: bool = False  # Background agents
