"""The Bash tool's schema, as every model sees it."""

import asyncio

from alancode.tools.base import ToolUseContext
from alancode.tools.builtin.bash import OUTPUT_MAX_BYTES, BashTool


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


def _run(command):
    return asyncio.run(
        BashTool().call({"command": command}, ToolUseContext(cwd="/tmp", messages=[]))
    )


def test_huge_output_keeps_both_ends_and_says_what_was_dropped():
    # A MiMo run printed 147 MB on one line; the frame carrying it broke
    # regact's 64 MiB line reader and crashed the turn.
    size = 3 * OUTPUT_MAX_BYTES
    result = _run(f"""python3 -c "print('START' + 'x' * {size} + 'END', end='')"; exit 3""")
    assert result.is_error
    assert result.data.startswith("START")
    assert result.data.endswith("END\n\nExit code: 3")
    assert f"[{size + 8 - OUTPUT_MAX_BYTES} bytes of output omitted" in result.data
    assert len(result.data) < OUTPUT_MAX_BYTES + 200


def test_output_at_the_limit_is_returned_whole():
    result = _run(f"""python3 -c "print('x' * {OUTPUT_MAX_BYTES}, end='')" """)
    assert not result.is_error
    assert result.data == "x" * OUTPUT_MAX_BYTES


def test_multibyte_character_split_at_the_boundary_does_not_raise():
    result = _run(f"""python3 -c "print('\\u00e9' * {OUTPUT_MAX_BYTES}, end='')" """)
    assert "bytes of output omitted" in result.data
