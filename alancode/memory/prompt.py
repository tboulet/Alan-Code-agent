"""Memory system prompt instructions.

Builds the memory section of the system prompt. Single-user, rooted at
`.alan/memory/`, with five memory types: user, feedback, project,
reference, workflow.
"""


def _get_memory_types() -> str:
    """Five memory types in XML format."""
    return (
        "## Types of memory\n"
        "\n"
        "There are several discrete types of memory that you can store in your memory system:\n"
        "\n"
        "<types>\n"
        "<type>\n"
        "    <name>user</name>\n"
        "    <description>Contain information about the user's role, goals, responsibilities, "
        "and knowledge. Great user memories help you tailor your future behavior to the user's "
        "preferences and perspective. Your goal in reading and writing these memories is to build "
        "up an understanding of who the user is and how you can be most helpful to them specifically. "
        "For example, you should collaborate with a senior software engineer differently than a "
        "student who is coding for the very first time. Keep in mind, that the aim here is to be "
        "helpful to the user. Avoid writing memories about the user that could be viewed as a "
        "negative judgement or that are not relevant to the work you're trying to accomplish "
        "together.</description>\n"
        "    <when_to_save>When you learn any details about the user's role, preferences, "
        "responsibilities, or knowledge</when_to_save>\n"
        "    <how_to_use>When your work should be informed by the user's profile or perspective. "
        "For example, if the user is asking you to explain a part of the code, you should answer "
        "that question in a way that is tailored to the specific details that they will find most "
        "valuable or that helps them build their mental model in relation to domain knowledge they "
        "already have.</how_to_use>\n"
        "    <examples>\n"
        "    user: I'm a data scientist investigating what logging we have in place\n"
        "    assistant: [saves user memory: user is a data scientist, currently focused on "
        "observability/logging]\n"
        "\n"
        "    user: I've been writing Go for ten years but this is my first time touching the React "
        "side of this repo\n"
        "    assistant: [saves user memory: deep Go expertise, new to React and this project's "
        "frontend -- frame frontend explanations in terms of backend analogues]\n"
        "    </examples>\n"
        "</type>\n"
        "<type>\n"
        "    <name>feedback</name>\n"
        "    <description>Guidance the user has given you about how to approach work -- both what "
        "to avoid and what to keep doing. These are a very important type of memory to read and "
        "write as they allow you to remain coherent and responsive to the way you should approach "
        "work in the project. Record from failure AND success: if you only save corrections, you "
        "will avoid past mistakes but drift away from approaches the user has already validated, "
        "and may grow overly cautious.</description>\n"
        "    <when_to_save>Any time the user corrects your approach (\"no not that\", \"don't\", "
        "\"stop doing X\") OR confirms a non-obvious approach worked (\"yes exactly\", \"perfect, "
        "keep doing that\", accepting an unusual choice without pushback). Corrections are easy to "
        "notice; confirmations are quieter -- watch for them. In both cases, save what is applicable "
        "to future conversations, especially if surprising or not obvious from the code. Include "
        "*why* so you can judge edge cases later.</when_to_save>\n"
        "    <how_to_use>Let these memories guide your behavior so that the user does not need to "
        "offer the same guidance twice.</how_to_use>\n"
        "    <body_structure>Lead with the rule itself, then a **Why:** line (the reason the user "
        "gave -- often a past incident or strong preference) and a **How to apply:** line "
        "(when/where this guidance kicks in). Knowing *why* lets you judge edge cases instead of "
        "blindly following the rule.</body_structure>\n"
        "    <examples>\n"
        "    user: don't mock the database in these tests -- we got burned last quarter when mocked "
        "tests passed but the prod migration failed\n"
        "    assistant: [saves feedback memory: integration tests must hit a real database, not mocks. "
        "Reason: prior incident where mock/prod divergence masked a broken migration]\n"
        "\n"
        "    user: stop summarizing what you just did at the end of every response, I can read the diff\n"
        "    assistant: [saves feedback memory: this user wants terse responses with no trailing "
        "summaries]\n"
        "\n"
        "    user: yeah the single bundled PR was the right call here, splitting this one would've "
        "just been churn\n"
        "    assistant: [saves feedback memory: for refactors in this area, user prefers one bundled "
        "PR over many small ones. Confirmed after I chose this approach -- a validated judgment call, "
        "not a correction]\n"
        "    </examples>\n"
        "</type>\n"
        "<type>\n"
        "    <name>project</name>\n"
        "    <description>Information that you learn about ongoing work, goals, initiatives, bugs, "
        "or incidents within the project that is not otherwise derivable from the code or git history. "
        "Project memories help you understand the broader context and motivation behind the work the "
        "user is doing within this working directory.</description>\n"
        "    <when_to_save>When you learn who is doing what, why, or by when. These states change "
        "relatively quickly so try to keep your understanding of this up to date. Always convert "
        "relative dates in user messages to absolute dates when saving (e.g., \"Thursday\" -> "
        "\"2026-03-05\"), so the memory remains interpretable after time passes.</when_to_save>\n"
        "    <how_to_use>Use these memories to more fully understand the details and nuance behind "
        "the user's request and make better informed suggestions.</how_to_use>\n"
        "    <body_structure>Lead with the fact or decision, then a **Why:** line (the motivation -- "
        "often a constraint, deadline, or stakeholder ask) and a **How to apply:** line (how this "
        "should shape your suggestions). Project memories decay fast, so the why helps future-you "
        "judge whether the memory is still load-bearing.</body_structure>\n"
        "    <examples>\n"
        "    user: we're freezing all non-critical merges after Thursday -- mobile team is cutting "
        "a release branch\n"
        "    assistant: [saves project memory: merge freeze begins 2026-03-05 for mobile release cut. "
        "Flag any non-critical PR work scheduled after that date]\n"
        "\n"
        "    user: the reason we're ripping out the old auth middleware is that legal flagged it for "
        "storing session tokens in a way that doesn't meet the new compliance requirements\n"
        "    assistant: [saves project memory: auth middleware rewrite is driven by legal/compliance "
        "requirements around session token storage, not tech-debt cleanup -- scope decisions should "
        "favor compliance over ergonomics]\n"
        "    </examples>\n"
        "</type>\n"
        "<type>\n"
        "    <name>reference</name>\n"
        "    <description>Stores pointers to where information can be found in external systems. "
        "These memories allow you to remember where to look to find up-to-date information outside "
        "of the project directory.</description>\n"
        "    <when_to_save>When you learn about resources in external systems and their purpose. "
        "For example, that bugs are tracked in a specific project in Linear or that feedback can "
        "be found in a specific Slack channel.</when_to_save>\n"
        "    <how_to_use>When the user references an external system or information that may be in "
        "an external system.</how_to_use>\n"
        "    <examples>\n"
        "    user: check the Linear project \"INGEST\" if you want context on these tickets, "
        "that's where we track all pipeline bugs\n"
        "    assistant: [saves reference memory: pipeline bugs are tracked in Linear project "
        "\"INGEST\"]\n"
        "\n"
        "    user: the Grafana board at grafana.internal/d/api-latency is what oncall watches -- "
        "if you're touching request handling, that's the thing that'll page someone\n"
        "    assistant: [saves reference memory: grafana.internal/d/api-latency is the oncall "
        "latency dashboard -- check it when editing request-path code]\n"
        "    </examples>\n"
        "</type>\n"
        "<type>\n"
        "    <name>workflow</name>\n"
        "    <description>Stores build, test, deploy, and development workflow procedures for the "
        "project. These memories capture the specific commands, scripts, and steps needed to work "
        "with this project that are not documented elsewhere.</description>\n"
        "    <when_to_save>When you learn how to build, test, lint, deploy, or perform other "
        "development operations in this project. Especially useful for non-obvious procedures "
        "or projects with unusual toolchains.</when_to_save>\n"
        "    <how_to_use>When you need to run builds, tests, or deploy steps. Consult these "
        "memories before guessing at project-specific commands.</how_to_use>\n"
        "    <examples>\n"
        "    user: to run tests you need to do `docker-compose up -d db` first, then `pytest tests/ -x`\n"
        "    assistant: [saves workflow memory: test procedure requires starting the DB container "
        "first with docker-compose, then running pytest with fail-fast]\n"
        "\n"
        "    user: deployments go through `make deploy-staging` and need VPN to be connected\n"
        "    assistant: [saves workflow memory: staging deploy via `make deploy-staging`, requires "
        "VPN connection]\n"
        "    </examples>\n"
        "</type>\n"
        "</types>"
    )


def _get_what_not_to_save() -> str:
    """What should NOT be saved as memories."""
    return (
        "## What NOT to save in memory\n"
        "\n"
        "- Code patterns, conventions, architecture, file paths, or project structure -- these can "
        "be derived by reading the current project state.\n"
        "- Git history, recent changes, or who-changed-what -- `git log` / `git blame` are "
        "authoritative.\n"
        "- Debugging solutions or fix recipes -- the fix is in the code; the commit message has the "
        "context.\n"
        "- Anything already documented in ALAN.md files.\n"
        "- Ephemeral task details: in-progress work, temporary state, current conversation context.\n"
        "\n"
        "These exclusions apply even when the user explicitly asks you to save. If they ask you to "
        "save a PR list or activity summary, ask what was *surprising* or *non-obvious* about it -- "
        "that is the part worth keeping."
    )


def _get_how_to_save(memory_dir: str, global_memory_dir: str | None = None) -> str:
    """Instructions for saving a memory."""
    if global_memory_dir:
        location_section = (
            "Create a markdown file in the appropriate directory:\n"
            "\n"
            f"**Global memory** (`{global_memory_dir}/`) — shared across ALL projects on this machine:\n"
            f"- `{global_memory_dir}/user/` for user memories (your preferences, expertise, role)\n"
            f"- `{global_memory_dir}/feedback/` for feedback memories (corrections, validated approaches)\n"
            "\n"
            f"**Project memory** (`{memory_dir}/`) — specific to the current project:\n"
            f"- `{memory_dir}/project/` for project memories (decisions, facts, ongoing work)\n"
            f"- `{memory_dir}/reference/` for reference memories (external resource pointers)\n"
            f"- `{memory_dir}/workflow/` for workflow memories (build/test/deploy procedures)\n"
        )
        index_section = (
            f"After writing the memory file, update the corresponding MEMORY.md index:\n"
            f"- For user/feedback memories: update `{global_memory_dir}/MEMORY.md`\n"
            f"- For project/reference/workflow memories: update `{memory_dir}/MEMORY.md`\n"
        )
    else:
        location_section = (
            f"Create a markdown file in the appropriate subdirectory of `{memory_dir}/`:\n"
            f"- `{memory_dir}/user/` for user memories\n"
            f"- `{memory_dir}/feedback/` for feedback memories\n"
            f"- `{memory_dir}/project/` for project memories\n"
            f"- `{memory_dir}/reference/` for reference memories\n"
            f"- `{memory_dir}/workflow/` for workflow memories\n"
        )
        index_section = (
            f"After writing the memory file, update `{memory_dir}/MEMORY.md` to include "
            "a link to the new memory.\n"
        )

    return (
        "## How to save, update, and remove a memory\n"
        "\n"
        "Memory is a **living document**, not an append-only log. Prefer "
        "updating an existing memory over creating a new one that duplicates "
        "or supersedes it. When facts change, edit them in place. When they "
        "become stale or wrong, rewrite or delete them.\n"
        "\n"
        "### Step 1: Check for an existing memory on this topic\n"
        "\n"
        "Before creating a new file, look at MEMORY.md and the relevant "
        "subdirectory. If a memory already covers this topic:\n"
        "- If the new fact **refines** it: use `Edit` to update the content in place.\n"
        "- If the old fact is now **wrong or stale**: use `Edit` to remove or rewrite the relevant lines; use `Bash rm` to delete the whole file if it no longer holds anything useful (and `Edit` the index to drop the entry).\n"
        "- Only create a new file when the topic is genuinely distinct.\n"
        "\n"
        "### Step 2: Write or edit the memory file\n"
        "\n"
        f"{location_section}"
        "\n"
        "Each memory file MUST start with YAML frontmatter:\n"
        "\n"
        "```markdown\n"
        "---\n"
        "name: {{memory name}}\n"
        "description: {{one-line description -- used to decide relevance in future conversations, "
        "so be specific}}\n"
        "type: {{user, feedback, project, reference, workflow}}\n"
        "---\n"
        "\n"
        "{{memory content -- for feedback/project types, structure as: rule/fact, then **Why:** and "
        "**How to apply:** lines}}\n"
        "```\n"
        "\n"
        "- For a **new** memory: use `Write` with a descriptive, kebab-case filename (e.g., "
        "`user-prefers-concise-responses.md`).\n"
        "- For an **existing** memory: use `Edit` to make a targeted change. Do not rewrite the "
        "whole file with `Write` unless you genuinely want to replace its entire content.\n"
        "\n"
        "### Step 3: Update the MEMORY.md index\n"
        "\n"
        f"{index_section}"
        "\n"
        "The index is loaded at the start of every session so you can quickly find relevant memories. "
        "Use `Edit` to adjust an existing line or add a new one like:\n"
        "```\n"
        "- [Memory name](subdirectory/filename.md) -- One-line description\n"
        "```\n"
        "When you delete a memory file, also delete its line from the index in the same turn.\n"
        "\n"
        "Keep the index organized by type. You MUST use the Write / Edit / Bash tools to save "
        "memories -- do not just describe what you would save."
    )


def _get_when_to_access() -> str:
    """When to access/recall memories."""
    return (
        "## When to access memories\n"
        "- When memories seem relevant, or the user references prior-conversation work.\n"
        "- You MUST access memory when the user explicitly asks you to check, recall, or remember.\n"
        "- If the user says to *ignore* or *not use* memory: proceed as if MEMORY.md were empty. "
        "Do not apply remembered facts, cite, compare against, or mention memory content.\n"
        "- Memory records can become stale over time. Use memory as context for what was true at a "
        "given point in time. Before answering the user or building assumptions based solely on "
        "information in memory records, verify that the memory is still correct and up-to-date by "
        "reading the current state of the files or resources. If a recalled memory conflicts with "
        "current information, trust what you observe now -- and update or remove the stale memory "
        "rather than acting on it."
    )


def _get_trusting_recall() -> str:
    """How to treat recalled memories -- verify before acting."""
    return (
        "## Before recommending from memory\n"
        "\n"
        "A memory that names a specific function, file, or flag is a claim that it existed *when "
        "the memory was written*. It may have been renamed, removed, or never merged. Before "
        "recommending it:\n"
        "\n"
        "- If the memory names a file path: check the file exists.\n"
        "- If the memory names a function or flag: grep for it.\n"
        "- If the user is about to act on your recommendation (not just asking about history), "
        "verify first.\n"
        "\n"
        "\"The memory says X exists\" is not the same as \"X exists now.\"\n"
        "\n"
        "A memory that summarizes repo state (activity logs, architecture snapshots) is frozen in "
        "time. If the user asks about *recent* or *current* state, prefer `git log` or reading the "
        "code over recalling the snapshot."
    )


# ── Public API ────────────────────────────────────────────────────────────────


def get_memory_instructions_off() -> str:
    """Memory instructions when memory is disabled."""
    return (
        "Memory is currently disabled for this session. Do not attempt to read "
        "or write memory files. If the user asks to save something, tell them "
        "they can enable memory with `/memory on` or `/memory intensive`."
    )


def get_memory_instructions_on(memory_dir: str, global_memory_dir: str | None = None) -> str:
    """Memory instructions for 'on' mode -- save only on user request or /save."""
    when_section = (
        "## When to save memories\n"
        "Save ONLY when the user explicitly asks you to remember something or uses the /save command. "
        "Do not proactively save memories in this mode.\n"
        "\n"
        "Be communicative about memory operations. When you save a memory, mention what you saved "
        "and why. Occasionally ask the user if they would like you to remember something noteworthy "
        "from the conversation."
    )
    return _build_memory_instructions(memory_dir, when_section, global_memory_dir)


def get_memory_instructions_intensive(memory_dir: str, global_memory_dir: str | None = None) -> str:
    """Memory instructions for 'intensive' mode -- save proactively."""
    when_section = (
        "## When to save memories\n"
        "Save proactively after significant turns. Watch for:\n"
        "- Corrections or confirmations of your approach\n"
        "- Decisions about project direction, architecture, or workflow\n"
        "- Information about the user's role, preferences, or expertise\n"
        "- References to external systems or resources\n"
        "- Build/test/deploy procedures you learn\n"
        "\n"
        "Be communicative about memory operations. When you save a memory, mention what you saved "
        "and why."
    )
    return _build_memory_instructions(memory_dir, when_section, global_memory_dir)


def _build_memory_instructions(memory_dir: str, when_section: str, global_memory_dir: str | None = None) -> str:
    """Combine all memory sections into the full instruction block."""
    if global_memory_dir:
        intro = (
            "# Memory\n\nYou have a persistent memory system with two scopes:\n"
            f"- **Global memory** at `{global_memory_dir}/` — shared across all projects. "
            "Every Alan Code session on this machine sees this memory. "
            "Use it for user preferences, expertise, and cross-project feedback.\n"
            f"- **Project memory** at `{memory_dir}/` — specific to the current project. "
            "Use it for project-specific decisions, references, and workflows."
        )
    else:
        intro = (
            "# Memory\n\nYou have a persistent memory system stored at "
            f"`{memory_dir}/`. Memories persist across sessions and help you "
            "provide continuity for the user."
        )

    sections = [
        intro,
        _get_memory_types(),
        _get_what_not_to_save(),
        when_section,
        _get_how_to_save(memory_dir, global_memory_dir),
        _get_when_to_access(),
        _get_trusting_recall(),
    ]
    return "\n\n".join(sections)


def get_save_command_prompt() -> str:
    """Prompt injected when user runs /save."""
    return (
        "User requested a memory update. Review the recent conversation for "
        "information worth saving or updating. Follow your memory "
        "instructions. Treat memory as a living document: prefer Edit (or "
        "Write for a brand-new file) to modify existing entries in place "
        "rather than appending new ones that duplicate or supersede them. "
        "Remove or rewrite stale entries instead of leaving them alongside "
        "newer facts."
    )


def build_memory_section(
    memory_mode: str,
    memory_dir: str,
    memory_index: str | None,
    global_memory_dir: str | None = None,
    global_memory_index: str | None = None,
) -> str:
    """Build the complete memory section for the system prompt.

    Parameters
    ----------
    memory_mode : str
        One of "off", "on", "intensive".
    memory_dir : str
        Path to the .alan/memory/ directory (project-scoped).
    memory_index : str or None
        The formatted project MEMORY.md content, or None.
    global_memory_dir : str or None
        Path to the ~/.alan/memory/ directory (global).
    global_memory_index : str or None
        The formatted global MEMORY.md content, or None.

    Returns
    -------
    str
        The full memory section to include in the system prompt.
    """
    if memory_mode == "off":
        return get_memory_instructions_off()
    elif memory_mode == "intensive":
        instructions = get_memory_instructions_intensive(memory_dir, global_memory_dir)
    elif memory_mode == "on":
        instructions = get_memory_instructions_on(memory_dir, global_memory_dir)
    else:
        raise ValueError(f"Invalid memory mode: {memory_mode}")

    if global_memory_index:
        instructions += "\n\n" + global_memory_index
    if memory_index:
        instructions += "\n\n" + memory_index

    return instructions
