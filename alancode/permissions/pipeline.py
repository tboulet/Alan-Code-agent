"""Permission decision pipeline.

Resolves whether a tool call should be allowed, denied, or prompted to
the user. The pipeline walks allow-rules, deny-rules, mode defaults,
and falls back to the user's ``ask_callback`` when a decision requires
explicit approval.
"""

from __future__ import annotations

import logging
from typing import Callable, Awaitable

from alancode.tools.base import Tool, ToolUseContext
from alancode.permissions.context import (
    ToolPermissionContext,
    PermissionResult,
    PermissionBehavior,
    PermissionMode,
    PermissionRule,
)

logger = logging.getLogger(__name__)

# Type for the user prompt callback
PermissionPromptFn = Callable[[str, str, dict], Awaitable[PermissionBehavior]]
# Args: tool_name, description_message, tool_input -> returns allow/deny


def check_rule_match(
    rules: list[PermissionRule],
    tool: Tool,
    input: dict,
) -> PermissionRule | None:
    """Find the first rule that matches this tool + input.

    A rule matches if:
    - rule.tool_name matches the tool's name (exact match), AND
    - rule.rule_content is None (blanket match), OR
    - rule.rule_content matches a prefix of the relevant input value
      (e.g., for Bash, rule_content="git *" matches command starting with "git ")
    """
    for rule in rules:
        if not tool.matches_name(rule.tool_name):
            continue

        # Blanket rule (no content filter) — matches any input
        if rule.rule_content is None:
            return rule

        rule_pattern = rule.rule_content.rstrip("*").rstrip()

        # Route the pattern to the field that's semantically the "target"
        # of the tool, per-tool. Previously we scanned every string value
        # in the input dict — a rule like `Read: "config*"` would match
        # `limit="config_limit"` coerced as a string, which surprised users.
        _TOOL_FIELD_MAP = {
            "Bash": "command",
            "Read": "file_path",
            "Write": "file_path",
            "Edit": "file_path",
            "Glob": "pattern",
            "Grep": "pattern",
            "WebFetch": "url",
        }
        target_field = _TOOL_FIELD_MAP.get(tool.name)

        if target_field is not None:
            value = input.get(target_field)
            if isinstance(value, str) and (
                value == rule_pattern
                or value.startswith(rule_pattern + " ")
            ):
                return rule
        else:
            # Unknown tool — fall back to scanning all string values so
            # user-defined tools with allow-rules still work.
            for value in input.values():
                if isinstance(value, str) and (
                    value == rule_pattern
                    or value.startswith(rule_pattern + " ")
                ):
                    return rule

    return None


def get_deny_rule(
    context: ToolPermissionContext,
    tool: Tool,
) -> PermissionRule | None:
    """Check if tool is blanket-denied.

    Looks for a deny rule that matches the tool name (blanket or specific).
    """
    for rule in context.deny_rules:
        if tool.matches_name(rule.tool_name):
            return rule
    return None


def _mode_allows(mode: PermissionMode, level: str) -> bool:
    """Check if the permission mode allows a given permission level without asking.

    | Mode   | read | write | exec |
    |--------|------|-------|------|
    | yolo   | yes  | yes   | yes  |
    | edit   | yes  | yes   | no   |
    | safe   | yes  | no    | no   |
    """
    if mode == PermissionMode.YOLO:
        return True
    if mode == PermissionMode.EDIT:
        return level in ("read", "write")
    if mode == PermissionMode.SAFE:
        return level == "read"
    return False


async def check_permissions(
    tool: Tool,
    input: dict,
    context: ToolUseContext,
    permission_context: ToolPermissionContext,
    *,
    prompt_user: PermissionPromptFn | None = None,
) -> PermissionResult:
    """Run the permission decision pipeline.

    Step 1: Rule-based checks (deny rules, ask rules, tool-specific)
    Step 2: Mode check (yolo/edit/safe × read/write/exec)
    Step 3: Hooks (stubbed as passthrough — real hooks are in
            alancode/hooks/registry.py and fire from run_tool_use)
    Step 4: Classifier (reserved for a future ML auto-allow/deny layer)
    Step 5: User prompt (if all above say 'ask')
    """

    # ── Step 1: Rule-based checks ──────────────────────────────────────────

    # 1a. Check deny rules first
    deny_rule = check_rule_match(permission_context.deny_rules, tool, input)
    if deny_rule is not None:
        logger.info("Permission denied by rule: %s (source=%s)", deny_rule.tool_name, deny_rule.source)
        return PermissionResult(
            behavior=PermissionBehavior.DENY,
            message=f"Tool '{tool.name}' denied by rule from {deny_rule.source}",
        )

    # 1b. Check ask rules
    ask_rule = check_rule_match(permission_context.ask_rules, tool, input)
    if ask_rule is not None:
        logger.debug("Ask rule matched for tool '%s' (source=%s)", tool.name, ask_rule.source)
        # Don't return yet — fall through to step 5 (user prompt)
        # But mark that we need to ask
        must_ask = True
    else:
        must_ask = False

    # Note: tool.validate_input() is called in run_tool_use() before
    # check_permissions(), so we don't duplicate it here.

    # ── Step 2: Mode check ─────────────────────────────────────────────────

    level = tool.permission_level(input)

    if not must_ask and _mode_allows(permission_context.mode, level):
        logger.debug("Permission allowed by mode '%s' for tool '%s' (level=%s)",
                      permission_context.mode.value, tool.name, level)
        return PermissionResult(behavior=PermissionBehavior.ALLOW)

    # Also check explicit allow rules (override mode for specific tools)
    if not must_ask:
        allow_rule = check_rule_match(permission_context.allow_rules, tool, input)
        if allow_rule is not None:
            logger.debug(
                "Permission allowed by rule for tool '%s' (source=%s)",
                tool.name,
                allow_rule.source,
            )
            return PermissionResult(behavior=PermissionBehavior.ALLOW)

    # ── Step 3: Hooks (stub; real hooks fire from run_tool_use) ──────────

    hook_result = PermissionBehavior.PASSTHROUGH
    if hook_result not in (PermissionBehavior.PASSTHROUGH,):
        return PermissionResult(behavior=hook_result)

    # ── Step 4: Classifier (reserved for a future ML auto-allow/deny) ────

    classifier_result = PermissionBehavior.PASSTHROUGH
    if classifier_result not in (PermissionBehavior.PASSTHROUGH,):
        return PermissionResult(behavior=classifier_result)

    # ── Step 5: User prompt ────────────────────────────────────────────────

    # If we should avoid prompts (e.g., background agent), deny by default
    if permission_context.should_avoid_prompts:
        logger.info("Avoiding prompt for tool '%s' (background agent)", tool.name)
        return PermissionResult(
            behavior=PermissionBehavior.DENY,
            message=f"Tool '{tool.name}' requires permission but prompts are disabled",
        )

    if prompt_user is not None:
        description = f"Tool '{tool.name}' wants to execute with input: {input}"
        user_decision = await prompt_user(tool.name, description, input)
        logger.info("User decision for tool '%s': %s", tool.name, user_decision.value)

        if user_decision == PermissionBehavior.ALLOW:
            return PermissionResult(behavior=PermissionBehavior.ALLOW)
        elif user_decision == PermissionBehavior.DENY:
            return PermissionResult(
                behavior=PermissionBehavior.DENY,
                message="Denied by user",
            )

    # No prompt callback — return ASK so the caller knows to prompt
    return PermissionResult(behavior=PermissionBehavior.ASK)
