"""The Bash tool's schema, as every model sees it."""

import asyncio

from alancode.tools.base import ToolUseContext
from alancode.tools.builtin.bash import BashTool


def test_schema_offers_no_label_field_a_model_can_fill_instead_of_command():
    # A `purpose` field was documented as "shown to the user before approval"
    # but nothing ever displayed it. A model (MiMo-V2.6-Flash-RL) called Bash
    # with only {"purpose": ...} and repeated the call 13 times in a row.
    assert set(BashTool().input_schema["properties"]) == {"command", "timeout"}


def test_missing_command_error_says_what_to_put_where():
    result = asyncio.run(
        BashTool().call({"purpose": "Read note 3"}, ToolUseContext(cwd="/tmp", messages=[]))
    )
    assert result.is_error
    assert "Put the shell command itself in 'command'" in result.data
