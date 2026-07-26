"""Build a compact system prompt for web assistants.

Alan's full system prompt (verbose base instructions + JSON tool schemas) runs
to tens of KB. A web chat UI will not send a message that large inline, and
delivering it as a file attachment makes assistants treat it as untrusted
content. This module rebuilds an equivalent-but-tiny prompt that fits inline:

- the verbose base instructions are replaced by a short, honest description of
  the relay setup (this is what the model actually needs to cooperate);
- each tool's JSON schema is collapsed to a one-line signature;
- the exact tool-call FORMAT block Alan emits is kept verbatim, so the model's
  output still parses with the configured ``tool_call_format``.

If no ``<tools>`` block is present (no text tool format configured) the base
text is returned trimmed, so the function is always safe to call.
"""

from __future__ import annotations

import json
import re

TOOLS_RE = re.compile(r"<tools>\s*(.*?)\s*</tools>(.*)", re.DOTALL)
DESC_MAX = 110

# Honest, minimal operating brief. It states the one fact assistants get wrong
# about this setup - that emitted tool calls are really executed and real
# results returned, so the model is not being asked to invent tool output.
COMPACT_BRIEF = (
    "You are the reasoning engine of \"Alan\", a command-line coding assistant. "
    "A relay program forwards the user's messages to you here and executes any "
    "tool call you emit, returning the real result as the next message. Answer "
    "the user's questions directly; to inspect or change their project, use the "
    "tools below and wait for each result before continuing. Keep replies "
    "concise. Do not invent a tool result you have not been sent."
)


def _render_tool(fn: dict) -> str:
    name = fn.get("name", "?")
    params = fn.get("parameters", {}) or {}
    props = list((params.get("properties") or {}).keys())
    required = params.get("required", []) or []
    optional = [p for p in props if p not in required]
    sig = ", ".join(required)
    if optional:
        sig += ("[, " if required else "[") + ", ".join(optional) + "]"
    desc = (fn.get("description") or "").split("\n", 1)[0].strip()
    if len(desc) > DESC_MAX:
        desc = desc[: DESC_MAX - 3].rstrip() + "..."
    return f"- {name}({sig}): {desc}"


def build_compact_system(system: list[str]) -> str:
    """Return a compact replacement for Alan's full system prompt."""
    full = "\n\n".join(s for s in system if s and s.strip())
    match = TOOLS_RE.search(full)
    if not match:
        return COMPACT_BRIEF

    tools_json, tail = match.group(1), match.group(2).strip()
    try:
        tools = json.loads(tools_json)
    except json.JSONDecodeError:
        return COMPACT_BRIEF

    lines = []
    for entry in tools:
        fn = entry.get("function", entry) if isinstance(entry, dict) else {}
        if fn:
            lines.append(_render_tool(fn))

    parts = [COMPACT_BRIEF, "Tools available to you:", "\n".join(lines)]
    if tail:  # Alan's exact tool-call format block - keep verbatim so it parses.
        parts.append(tail)
    return "\n\n".join(parts)
