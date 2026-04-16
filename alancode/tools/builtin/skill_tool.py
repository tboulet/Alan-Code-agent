"""Skill tool — model-invoked skill execution.

Allows the model to proactively invoke discovered skills when it
recognizes a situation matching a skill's ``when_to_use`` description.
"""

from typing import Any, Literal

from alancode.tools.base import Tool, ToolResult, ToolUseContext


class SkillTool(Tool):
    """Execute a skill (prompt template) by name."""

    def __init__(self, skill_registry):
        # Avoid circular import — registry is passed at init time
        self._registry = skill_registry

    @property
    def name(self) -> str:
        return "Skill"

    @property
    def description(self) -> str:
        return (
            "Execute a skill (reusable prompt template) by name. "
            "Skills are markdown-based workflow recipes defined in "
            ".alan/skills/ or ~/.alan/skills/. Use /skill list to see available skills."
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "skill": {
                    "type": "string",
                    "description": "The skill name to invoke",
                },
                "args": {
                    "type": "string",
                    "description": "Optional arguments passed to the skill (replaces $ARGUMENTS in the template)",
                },
            },
            "required": ["skill"],
        }

    async def call(self, args: dict[str, Any], context: ToolUseContext) -> ToolResult:
        skill_name = args["skill"]
        skill_args = args.get("args", "")

        skill = self._registry.get(skill_name)
        if skill is None:
            available = ", ".join(s.name for s in self._registry.list_all())
            return ToolResult(
                data=f"Unknown skill: {skill_name!r}. Available skills: {available or 'none'}",
                is_error=True,
            )

        expanded = self._registry.expand(skill_name, skill_args)

        # Build the response with tool restriction hints if applicable
        parts = [f'<skill-prompt name="{skill_name}">']
        if skill.allowed_tools:
            tools_str = ", ".join(skill.allowed_tools)
            parts.append(
                f"IMPORTANT: While executing this skill, you may only use "
                f"the following tools: {tools_str}"
            )
        parts.append(expanded)
        parts.append("</skill-prompt>")

        return ToolResult(data="\n".join(parts))

    def permission_level(self, args: dict[str, Any]) -> Literal["read", "write", "exec"]:
        return "read"  # Loading a prompt template is read-only
