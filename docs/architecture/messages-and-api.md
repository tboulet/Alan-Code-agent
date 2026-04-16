# Messages and the API payload

How Alan's internal `agent._messages` list becomes the bytes on the wire. This is where the mental model "what the user saw" diverges from "what the model saw".

## Pipeline

```
agent._messages  (raw, includes all history + hidden reminders)
      │
      ▼
boundary slice (drop everything before the last compact summary)
      │
      ▼
compaction layers A → B → C (if over threshold)
      │
      ▼
normalize_messages_for_api  (filter hidden, merge same-role, drop orphan tool_results)
      │
      ▼
messages_to_openai_dicts    (serialize to [{role, content}, ...])
      │
      ▼
provider envelope (Anthropic: convert to Anthropic shape; LiteLLM: pass through)
      │
      ▼
HTTP POST to the provider
```

Each step is covered below.

## Step 1 — Boundary slice

`alancode/messages/types.py::get_messages_after_compact_boundary` drops everything before the last `SystemMessage(subtype=COMPACT_BOUNDARY)`. After compaction, the pre-summary messages aren't sent — the summary replaces them.

Implementation: scan backwards through the list, find the last compact-boundary marker, return everything from there onwards. If no boundary exists (no compaction yet), return all messages.

## Step 2 — Compaction layers

Covered in [concepts/context-and-compaction.md](../concepts/context-and-compaction.md) and [architecture/query-loop.md#phase-2](query-loop.md#phase-2--compaction-pipeline).

Layers may mutate the message list in place (Layer A truncates tool_result content), return a new list (Layer B clears old results), or replace the entire history with a summary (Layer C).

## Step 3 — Normalization

`alancode/messages/normalization.py::normalize_messages_for_api`. Six steps:

### 3a. Drop ProgressMessages

Purely informational messages used for UI progress updates. Never sent.

### 3b. Drop messages with `hide_in_api=True`

These live in `agent._messages` for UI replay but are stripped before sending. Examples:

- `<system-reminder>` with date/time (injected each turn).
- `<system-reminder>` about model / provider / memory-mode changes.
- `<system-reminder>` about `/move`, `/convrevert`, `/allrevert`.
- Virtual "resume directly" recovery prompts.

### 3c. Drop SystemMessages

Alan's `SystemMessage` type is **internal** — `COMPACT_BOUNDARY`, informational `command_output`, error markers. These are not API-level system prompts. Exception: `local_command` subtype gets converted to a UserMessage.

### 3d. Convert AttachmentMessages to UserMessages

`AttachmentMessage` carries structured metadata (e.g. a file-diff attachment, a `max_iterations_per_turn_reached` marker). Converted to a UserMessage with the content stringified so the model sees it.

### 3e. Merge consecutive same-role messages

Required by:
- Many LLM APIs that enforce strict `user, assistant, user, assistant, ...` alternation (Anthropic, some Bedrock models).
- Bedrock specifically — rejects consecutive same-role.

`merge_user_messages` and `_merge_assistant_messages` concatenate content lists.

### 3f. Drop orphan tool_results

After merging, it's possible for a `tool_result` block to end up without a preceding `tool_use` (the referenced `tool_use_id` no longer exists in the conversation — maybe compacted away, maybe merged-out).

`_drop_orphan_tool_results` walks the merged list, tracks every `tool_use_id` it sees on assistant messages, and strips `tool_result` blocks whose id doesn't appear in that set. Logs a WARNING if it drops any.

Without this pass, the API 400's with "tool_use_id does not match any tool_use block".

## Step 4 — Serialization

`alancode/messages/serialization.py::messages_to_openai_dicts` converts to the universal OpenAI-compatible shape:

```json
[
  {"role": "user", "content": "..."},
  {"role": "assistant", "content": [{"type": "text", "text": "..."}, {"type": "tool_use", "id": "...", ...}]},
  {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "...", "content": "..."}]}
]
```

Content is a string when simple, a list of blocks when there are tool calls, images, or thinking blocks.

## Step 5 — Provider envelope

### Anthropic

`AnthropicProvider.stream` converts to Anthropic's specific shape:
- `system` parameter becomes a list of cache-scoped text blocks (see [prompt-caching.md](prompt-caching.md)).
- `messages` stays alternating user/assistant.
- `tool_use` blocks use Anthropic's schema (`{type: "tool_use", id, name, input}`).
- `tool_result` blocks too (`{type: "tool_result", tool_use_id, content}`).
- Headers: `anthropic-version`, and `anthropic-beta` for cache / thinking / cache-keys features.

### LiteLLM

`LiteLLMProvider.stream` passes the OpenAI dicts through to `litellm.acompletion(...)`, which handles provider-specific reshape internally. Our job is just to ensure the dicts are well-formed.

`stream_options={"include_usage": True}` is set so `usage` arrives in the final stream chunk — needed for our token accounting.

### Scripted

`ScriptedProvider` ignores the payload entirely and returns the Nth pre-canned response.

## The two sides diverge — what the user sees vs what the API sees

| Item | In user's chat panel? | Sent to API? |
|---|---|---|
| User's typed prompt | ✅ | ✅ |
| Assistant text (streamed) | ✅ | ✅ |
| Tool call blocks | ✅ | ✅ |
| Tool result panels | ✅ | ✅ |
| `<system-reminder>` for date/time | ❌ (hide_in_ui) | ✅ |
| `<system-reminder>` after `/move` / `/revert` | ❌ | ✅ |
| `ProgressMessage` (compaction started) | ✅ as informational line | ❌ |
| `SystemMessage(COMPACT_BOUNDARY)` | ✅ as subtle marker | ❌ (filtered at step 3c) |
| `AttachmentMessage(max_iterations_per_turn_reached)` | depends on UI | ✅ (converted to UserMessage) |
| Layer-B-cleared tool results | ✅ (original until next send) | ❌ (sent as `[cleared to free context]`) |

## Debugging — the LLM Perspective panel

The GUI's LLM Perspective panel shows the serialized messages (the output of step 4) along with the system prompt. This is the authoritative view of "what the model actually saw for this turn".

From Python, the same data is available via `llm_perspective_callback`:

```python
def on_perspective(messages_dicts, system_prompt):
    print(json.dumps(messages_dicts, indent=2))

agent = AlanCodeAgent(...)
agent._llm_perspective_callback = on_perspective
```

Called before each API call.

## Related

- [architecture/query-loop.md](query-loop.md) — where normalization happens (phase 4).
- [architecture/system-prompt.md](system-prompt.md) — how the system half of the payload is built.
- [concepts/context-and-compaction.md](../concepts/context-and-compaction.md) — how compaction reshapes the messages list.
- `alancode/messages/` — implementation.
