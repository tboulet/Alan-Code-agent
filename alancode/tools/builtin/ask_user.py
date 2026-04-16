"""AskUserQuestion tool -- lets the model ask the user a question with proposed answers.

The model provides a question and a list of proposed options.
The user selects one option, or chooses "Other" to type a custom response.
The selected answer is returned to the model as the tool result.
"""

from typing import Any

from alancode.tools.base import Tool, ToolResult, ToolUseContext


class AskUserQuestionTool(Tool):
    name = "AskUserQuestion"
    description = (
        "Ask the user a question with proposed answers. "
        "Use this when you need clarification, confirmation, or a decision from the user. "
        "Provide at least 1 option. The user can always choose 'Other' to give a custom answer.\n\n"
        "Usage:\n"
        "- Use sparingly -- only when you truly need user input to proceed\n"
        "- Frame questions clearly with actionable options\n"
        "- Prefer making reasonable assumptions over asking when the choice is low-risk"
    )
    input_schema = {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question to ask the user. Should be clear and specific.",
            },
            "options": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "List of proposed answers (at least 1). "
                    "The user can also type a custom answer by choosing 'Other'."
                ),
            },
        },
        "required": ["question", "options"],
    }

    def permission_level(self, args: dict[str, Any]) -> str:
        return "read"

    def validate_input(self, args: dict[str, Any], context: ToolUseContext) -> str | None:
        given_keys = list(args.keys())
        options = args.get("options", [])
        if not isinstance(options, list) or len(options) < 1:
            return (
                f"Error: 'options' parameter must be a list with at least 1 item. "
                f"Got parameters: {given_keys}. "
                f"Use <arg_key>question</arg_key><arg_value>YOUR_QUESTION</arg_value> "
                f"<arg_key>options</arg_key><arg_value>[\"Option A\", \"Option B\"]</arg_value>"
            )
        question = args.get("question", "")
        if not question or not question.strip():
            return (
                f"Error: 'question' parameter is required but was not provided. "
                f"Got parameters: {given_keys}. "
                f"Use <arg_key>question</arg_key><arg_value>YOUR_QUESTION</arg_value> "
                f"<arg_key>options</arg_key><arg_value>[\"Option A\", \"Option B\"]</arg_value>"
            )
        return None

    async def call(self, args: dict[str, Any], context: ToolUseContext) -> ToolResult:
        import asyncio

        question: str = args["question"]
        options: list[str] = args["options"]

        if context.ask_user_callback is None:
            return ToolResult(
                data="No user interaction available in this mode."
            )

        try:
            answer = await context.ask_user_callback(question, options)
        except asyncio.CancelledError:
            if context.abort_signal is not None:
                context.abort_signal.set()
            raise
        return ToolResult(data=answer)
