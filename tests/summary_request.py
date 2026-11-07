from alancode.compact.prompt import NO_TOOLS_PREAMBLE


def is_summary_request(messages) -> bool:
    """A compaction call shares the main call's system prompt and tools; only
    its last message, the summary instruction, tells it apart."""
    return bool(messages) and str(messages[-1].get("content", "")).startswith(NO_TOOLS_PREAMBLE)
