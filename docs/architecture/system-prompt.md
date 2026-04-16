# System prompt assembly

The system prompt Alan sends in every API call is assembled by `alancode/prompt/system_prompt.py::get_system_prompt`. It's built as a list of sections; the provider decides how to serialize them (Anthropic: separate cache blocks; OpenAI-compatible: joined with `\n\n`).

## Assembly order

| # | Section | Source function | Conditional? |
|---|---|---|---|
| 1 | Intro | `get_intro_section` | always |
| 2 | System rules | `get_system_section` | always |
| 3 | Doing tasks | `get_doing_tasks_section` | always |
| 4 | Executing actions with care | `get_actions_section` | always |
| 5 | Using your tools | `get_using_tools_section(tools)` | always |
| 6 | Tone and style | `get_tone_section` | always |
| 7 | Communicating with the user | `get_communication_section` | always |
| 8 | Session-specific guidance | `get_session_guidance_section` | always |
| 9 | Environment | `get_environment_section(model, cwd)` | always (content varies) |
| 10 | Available skills | `get_skills_section(skills)` | if ≥1 skill registered |
| 11 | Memory | `build_memory_section(memory_mode, …)` | always (short stub when off) |
| 12 | Scratchpad | `get_scratchpad_section(scratchpad_dir)` | always in normal runs |
| 13 | `ALAN.md` append | — | if `ALAN.md` exists (project or global) |
| 14 | Tool-format instructions | `get_tool_format_system_prompt(fmt, schemas)` | if `tool_call_format` is set |

Sections 1–8 are **static** — same bytes every call. Sections 9–14 vary per session / mode. This split is designed for Anthropic's prompt caching: section 1 alone, sections 2–8 together, section 9+ as a dynamic block. See [prompt-caching.md](prompt-caching.md).

## `custom_system_prompt` override

If the `custom_system_prompt` setting is set, **sections 1–9 are replaced** by that string. Sections 10–14 (skills, memory, scratchpad, ALAN.md, tool-format) still append. Use with care — you lose all the tool guidance, action safety rules, and session awareness built in.

Prefer `append_system_prompt` for additive modifications.

## What's in each section

### 1. Intro (always)

> You are Alan Code, an open-source coding agent. You are an interactive agent that helps users with software engineering tasks. […]
> IMPORTANT: You must NEVER generate or guess URLs for the user unless you are confident that the URLs are for helping the user with programming.

### 2. System rules (always)

Bullet list. Covers: "all text you output is displayed to the user", "tools run in a permission mode", "tool results may include `<system-reminder>` tags", "conversation automatically compressed near context limit".

### 3. Doing tasks (always)

The biggest section (~50 bullet points). Covers:
- Interpreting requests in the context of software engineering tasks.
- When to read files before proposing changes.
- Don't create files the user didn't ask for.
- No time estimates.
- When to diagnose before switching tactics.
- Don't add features beyond what was asked.
- Don't add error handling / validation for impossible cases.
- Default to no comments (only when the WHY is non-obvious).
- Don't remove existing comments unless removing the code.
- No backwards-compatibility hacks.
- Verify before claiming done.
- Don't run destructive commands without confirming.
- Report outcomes faithfully.
- Ask for clarification when instructions seem off.

### 4. Executing actions with care (always)

About reversibility and blast radius. Lists risky action categories (destructive ops, hard-to-reverse ops, shared-state ops, third-party uploads) and explains the default stance: confirm for risky, proceed for local reversible.

### 5. Using your tools (always)

- Don't use Bash when a dedicated tool exists (Read/Edit/Write/Glob/Grep).
- Multi-tool calls OK for independent ops.
- Trailing line: `Available tools: Bash, Read, Edit, Write, Glob, Grep, AskUserQuestion, WebFetch, GitCommit, Skill`.

### 6. Tone and style (always)

- Use `file_path:line_number` pattern when citing code.
- Don't end text with a colon before a tool call.

### 7. Communicating with the user (always)

Longer section. Writing for a person, not logging. Updates at load-bearing moments. Noticing user pace. Match prose style to task. Concise and direct.

### 8. Session-specific guidance (always)

- If the user denied a tool call you don't understand, ask them.
- Glob/Grep for directed searches.
- Break broader exploration into steps.

### 9. Environment (always, content varies)

```
# Environment
You have been invoked in the following environment:
 - Primary working directory: <cwd>
 - Is a git repository: Yes
 - Platform: linux
 - Shell: bash
 - OS Version: Linux 6.17.0-14-generic
 - Session started: 2026-04-15 18:42
 - Model: openrouter/google/gemini-2.5-flash

gitStatus:
 <output of `git status` + `git log --oneline -5`>
```

`gitStatus` block omitted in non-git directories.

### 10. Available skills (conditional)

Only present when skills are registered. Format:

```
# Available skills

Skills are reusable prompt templates. Users invoke them via `/skill <name> [args]`. You can invoke them via the Skill tool.

- **review-pr** <pr-number or branch>: Review a pull request for correctness...
  TRIGGER: When the user asks for a code review.
```

### 11. Memory (always, three variants)

**`memory=off`** (short stub):

> Memory is currently disabled for this session. Do not attempt to read or write memory files. If the user asks to save something, tell them they can enable memory with `/memory on` or `/memory intensive`.

**`memory=on`** or **`memory=intensive`** (full block):

- Intro explaining global vs project scope.
- `## Types of memory` — XML catalogue of user / feedback / project / reference / workflow.
- `## What NOT to save in memory`.
- `## When to save memories` — mode-specific (on: only on user request; intensive: proactive).
- `## How to save, update, and remove a memory` — the three-step process emphasizing **update in place** rather than append.
- `## When to access memories`.
- `## Before recommending from memory` — verify before acting.
- Then the full contents of `~/.alan/memory/MEMORY.md` (global) and `<cwd>/.alan/memory/MEMORY.md` (project), appended.

### 12. Scratchpad (conditional)

> You have a session-scoped scratchpad directory at `<cwd>/.alan/sessions/<id>/scratchpad`. Use it for temporary notes, draft plans, or intermediate work. This directory is session-specific and does not carry over.

### 13. ALAN.md append (conditional)

Contents of `~/.alan/ALAN.md` + `<cwd>/ALAN.md`, joined with `\n\n`. Only sent if at least one file exists.

### 14. Tool-format instructions (conditional)

For `--tool-call-format hermes|glm|alan`. Appended at the very end. See `alancode/tools/text_tool_parser.py`:

- **hermes**: `<tool_call>{"name": ..., "arguments": ...}</tool_call>`
- **glm**: `<tool_call>ToolName<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>` (closing tag now mandatory after an audit fix)
- **alan**: `<tool_use>{"name": ..., "input": ...}</tool_use>`

## Provider-specific assembly

### Anthropic

Sections are passed as a **list of cache blocks**, enabling fine-grained caching:

```python
system = [
    {"type": "text", "text": intro, "cache_control": {"type": "ephemeral"}},
    {"type": "text", "text": "\n\n".join(sections_2_to_8), "cache_control": {"type": "ephemeral"}},
    {"type": "text", "text": "\n\n".join(dynamic_sections_9_plus)},
]
```

Section 1 has its own cache breakpoint because `model_info.supports_extended_thinking` can change between calls, but the intro is the most stable block.

### LiteLLM

LiteLLM's `completion(...)` accepts a single `system` parameter (or a `system`-role message). Alan joins all sections with `\n\n` into one string:

```python
messages = [{"role": "system", "content": "\n\n".join(sections)}, ...user/assistant messages]
```

No per-block caching (most LiteLLM backends don't support it). Some backends (Anthropic via LiteLLM, some Gemini versions) do — LiteLLM handles the translation.

## Inspecting what was sent

In the GUI, the **LLM Perspective** panel shows the exact system prompt for the current turn. From Python:

```python
agent = AlanCodeAgent(...)
agent._llm_perspective_callback = lambda msgs, sys: print(sys)
# On the next turn, sys is the list of section strings.
```

## Related

- [concepts/memory.md](../concepts/memory.md) — section 11 details.
- [concepts/skills.md](../concepts/skills.md) — section 10 details.
- [concepts/project-context.md](../concepts/project-context.md) — section 13 details.
- [architecture/prompt-caching.md](prompt-caching.md) — how Anthropic cache blocks are arranged.
- [reference/settings.md](../reference/settings.md) — `custom_system_prompt`, `append_system_prompt`, `tool_call_format`.
