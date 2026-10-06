"""The ViewImage tool and the path an image takes to a model that can see."""

import base64
import os
import struct

import pytest

from alancode.agent import AlanCodeAgent
from alancode.backends.anthropic_backend import _openai_to_anthropic_messages
from alancode.backends.scripted_backend import ScriptedBackend, text, tool_call
from alancode.compact.compact_truncate import compaction_truncate_tool_results
from alancode.messages.factory import (
    create_assistant_message,
    create_tool_result_message,
    create_user_message,
)
from alancode.messages.serialization import (
    IMAGE_PLACEHOLDER,
    messages_to_openai_dicts,
)
from alancode.messages.types import (
    ImageBlock,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from alancode.session.transcript import dict_to_message, message_to_dict
from alancode.tools.base import ToolUseContext
from alancode.tools.builtin import view_image
from alancode.tools.builtin.view_image import ViewImageTool
from alancode.utils.tokens import (
    IMAGE_TOKEN_CEILING,
    IMAGE_TOKEN_ESTIMATE,
    estimate_message_tokens,
    image_tokens,
)

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
PNG_B64 = base64.b64encode(PNG).decode("ascii")


def _context(tmp_path, vision=True):
    return ToolUseContext(cwd=str(tmp_path), messages=[], settings={"vision": vision})


def _image_result(call_id="call_1", note="Image frame.png:"):
    return create_tool_result_message(call_id, [
        TextBlock(text=note),
        ImageBlock(source={"type": "base64", "media_type": "image/png", "data": PNG_B64}),
    ])


def _call(call_id="call_1"):
    return create_assistant_message(
        [ToolUseBlock(id=call_id, name="ViewImage", input={"file_path": "frame.png"})]
    )


class TestViewImageTool:
    @pytest.mark.asyncio
    async def test_returns_the_image_with_a_line_of_text(self, tmp_path):
        (tmp_path / "frame.png").write_bytes(PNG)
        result = await ViewImageTool().call({"file_path": "frame.png"}, _context(tmp_path))

        assert not result.is_error
        note, image = result.data
        assert "frame.png" in note.text and "image/png" in note.text
        assert image.source == {
            "type": "base64", "media_type": "image/png", "data": PNG_B64,
        }

    @pytest.mark.asyncio
    async def test_refuses_when_vision_is_off(self, tmp_path):
        (tmp_path / "frame.png").write_bytes(PNG)
        result = await ViewImageTool().call(
            {"file_path": "frame.png"}, _context(tmp_path, vision=False),
        )
        assert result.is_error and isinstance(result.data, str)
        assert "vision" in result.data

    @pytest.mark.asyncio
    async def test_refuses_a_file_outside_the_working_directory(self, tmp_path):
        work = tmp_path / "work"
        work.mkdir()
        (tmp_path / "secret.png").write_bytes(PNG)
        os.symlink(tmp_path / "secret.png", work / "link.png")

        for path in ("../secret.png", str(tmp_path / "secret.png"), "link.png"):
            result = await ViewImageTool().call({"file_path": path}, _context(work))
            assert result.is_error, path
            assert "outside the working directory" in result.data

    @pytest.mark.asyncio
    async def test_refuses_a_file_that_is_not_an_image(self, tmp_path):
        (tmp_path / "notes.png").write_text("just text")
        result = await ViewImageTool().call({"file_path": "notes.png"}, _context(tmp_path))
        assert result.is_error and "not a PNG" in result.data

    @pytest.mark.asyncio
    async def test_refuses_an_oversized_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(view_image, "MAX_IMAGE_BYTES", 16)
        (tmp_path / "frame.png").write_bytes(PNG)
        result = await ViewImageTool().call({"file_path": "frame.png"}, _context(tmp_path))
        assert result.is_error and "limit" in result.data


class TestImageSerialization:
    def test_image_follows_the_tool_message_as_an_image_url_part(self):
        dicts = messages_to_openai_dicts([_call(), _image_result()])

        assert dicts[1] == {
            "role": "tool", "tool_call_id": "call_1", "content": "Image frame.png:",
        }
        assert dicts[2]["role"] == "user"
        kinds = [part["type"] for part in dicts[2]["content"]]
        assert kinds == ["text", "image_url"]
        assert dicts[2]["content"][1]["image_url"]["url"] == (
            f"data:image/png;base64,{PNG_B64}"
        )

    def test_images_come_after_every_tool_message_of_the_batch(self):
        """A user message between two tool messages breaks the pairing."""
        batch = create_user_message([
            *_image_result("call_1").content,
            ToolResultBlock(tool_use_id="call_2", content="plain"),
        ])
        dicts = messages_to_openai_dicts([batch])
        assert [d["role"] for d in dicts] == ["tool", "tool", "user"]

    def test_text_dialect_carries_the_image_in_the_result_message(self):
        dicts = messages_to_openai_dicts([_image_result()], text_dialect=True)
        assert len(dicts) == 1 and dicts[0]["role"] == "user"
        assert dicts[0]["content"][0] == {"type": "text", "text": "Image frame.png:"}
        assert dicts[0]["content"][1]["type"] == "image_url"

    def test_text_only_results_are_serialized_as_before(self):
        dicts = messages_to_openai_dicts(
            [create_tool_result_message("call_1", "file.txt")]
        )
        assert dicts == [
            {"role": "tool", "tool_call_id": "call_1", "content": "file.txt"},
        ]

    def test_images_can_be_left_out(self):
        dicts = messages_to_openai_dicts([_image_result()], include_images=False)
        assert len(dicts) == 1
        assert IMAGE_PLACEHOLDER in dicts[0]["content"]
        assert PNG_B64 not in str(dicts)

    def test_anthropic_gets_an_image_block_beside_its_tool_result(self):
        converted = _openai_to_anthropic_messages(
            messages_to_openai_dicts([_call(), _image_result()])
        )
        assert [m["role"] for m in converted] == ["assistant", "user"]
        blocks = converted[1]["content"]
        assert blocks[0]["type"] == "tool_result"
        assert blocks[-1] == {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": PNG_B64},
        }

    def test_transcript_round_trip_keeps_the_image(self):
        restored = dict_to_message(message_to_dict(_image_result()))
        image = restored.content[0].content[1]
        assert isinstance(image, ImageBlock) and image.source["data"] == PNG_B64


class TestImageBudget:
    @staticmethod
    def _png_block(width, height, padding=0):
        header = (
            b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR"
            + struct.pack(">II", width, height)
        )
        data = base64.b64encode(header + b"\x00" * padding).decode("ascii")
        return ImageBlock(
            source={"type": "base64", "media_type": "image/png", "data": data}
        )

    def test_an_image_costs_by_its_pixels(self):
        """Measured on Qwen3.8-27B under llama.cpp: about 100 tokens at
        256x256 and 300 at 512x512. A flat 1500 overcounted 5 to 15 times."""
        assert image_tokens(self._png_block(256, 256)) == 88
        assert image_tokens(self._png_block(512, 512)) == 350

    def test_a_heavy_file_of_few_pixels_stays_cheap(self):
        heavy = self._png_block(64, 64, padding=600_000)
        message = create_tool_result_message("call_1", [TextBlock(text="Image:"), heavy])
        assert estimate_message_tokens([message]) < 50

    def test_a_huge_image_stops_at_the_ceiling(self):
        assert image_tokens(self._png_block(8_000, 8_000)) == IMAGE_TOKEN_CEILING

    def test_unreadable_dimensions_fall_back_to_the_flat_estimate(self):
        jpeg = ImageBlock(source={
            "type": "base64", "media_type": "image/jpeg",
            "data": base64.b64encode(b"\xff\xd8\xff" + b"\x00" * 64).decode("ascii"),
        })
        assert image_tokens(jpeg) == IMAGE_TOKEN_ESTIMATE

    def test_truncating_a_long_result_keeps_its_image(self):
        message = create_tool_result_message("call_1", [
            TextBlock(text="x" * 5_000),
            ImageBlock(source={"type": "base64", "media_type": "image/png", "data": PNG_B64}),
        ])
        truncated, count = compaction_truncate_tool_results([message], max_chars=500)
        content = truncated[0].content[0].content
        assert count == 1
        assert len(content[0].text) <= 500
        assert isinstance(content[1], ImageBlock)


class TestVisionSetting:
    def test_view_image_is_offered_only_when_vision_is_on(self, tmp_path):
        backend = ScriptedBackend.from_responses([], fallback=text("ok"))
        blind = AlanCodeAgent(backend=backend, cwd=str(tmp_path))
        seeing = AlanCodeAgent(backend=backend, cwd=str(tmp_path), vision=True)

        assert "ViewImage" not in {t.name for t in blind._tools}
        assert "ViewImage" in {t.name for t in seeing._tools}

    def test_an_explicit_tool_list_is_kept_as_given(self, tmp_path):
        backend = ScriptedBackend.from_responses([], fallback=text("ok"))
        agent = AlanCodeAgent(
            backend=backend, cwd=str(tmp_path), tools=[ViewImageTool()],
        )
        assert [t.name for t in agent._tools] == ["ViewImage"]

    @pytest.mark.asyncio
    async def test_the_model_receives_the_image_it_asked_for(self, tmp_path):
        (tmp_path / "frame.png").write_bytes(PNG)
        backend = ScriptedBackend.from_responses(
            [tool_call("ViewImage", {"file_path": "frame.png"})],
            fallback=text("I see it."),
        )
        seen: list[list[dict]] = []
        stream = backend.stream

        async def recording_stream(messages, *args, **kwargs):
            seen.append(messages)
            async for event in stream(messages, *args, **kwargs):
                yield event

        backend.stream = recording_stream
        agent = AlanCodeAgent(
            backend=backend, cwd=str(tmp_path), programmatic=True,
            permission_mode="yolo", custom_system_prompt="t",
            tools=[ViewImageTool()], vision=True,
        )
        [event async for event in agent.query_events_async("look at frame.png")]

        assert len(seen) == 2
        parts = [
            part
            for message in seen[1] if isinstance(message["content"], list)
            for part in message["content"]
        ]
        assert any(
            part.get("type") == "image_url" and PNG_B64 in part["image_url"]["url"]
            for part in parts
        )
