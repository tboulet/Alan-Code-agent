"""Built-in skills shipped with Alan Code.

These are SkillDefinition objects defined in Python rather than on disk.
They serve as defaults that can be overridden by user or project skills.
"""

from alancode.skills.parser import SkillDefinition


# ── /skill create — interactive skill generator ─────────────────────────────

CREATE_SKILL_BODY = """\
# Create a New Skill

You are helping the user create a new reusable skill (prompt template).

The user provided: "$ARGUMENTS"

## Interpret the user input

The input above may be:
- Empty — no instructions given, interview the user from scratch
- A skill name only (e.g. "deploy") — use it as the name, ask what it should do
- A name followed by a description (e.g. "deploy Build a deploy workflow that runs tests then ships to prod") — extract the name from the first word and use the rest as context for what the skill should do
- A full description without a clear name — extract intent, suggest a name

Use your judgment to parse this naturally. When in doubt, ask the user to clarify.

## Interview the user

If you don't have enough information yet, ask clarifying questions:
- What steps does the workflow involve?
- What tools are needed? (Bash, Read, Edit, Write, Glob, Grep, etc.)
- Does it take arguments? If so, what?
- When should this skill be triggered? (for model auto-invocation)

You can skip questions you already have answers to from the user input.

## Validate the skill name

Before writing, check the name for issues:

- **Reserved name `list`**: Warn the user — a skill named `list` can never be invoked
  via `/skill list` because that dispatches to the skill-listing subcommand. The skill
  would only be callable by the model via the Skill tool. Suggest a different name.

- **Reserved name `create`**: Warn the user — writing a skill named `create` will
  override the built-in skill creator (the very skill currently running). They would
  lose the ability to create new skills interactively. Ask them to confirm or pick
  a different name.

- **Existing skill**: Use the Read or Glob tool to check if
  `.alan/skills/<name>/SKILL.md` already exists in the project directory. If it does,
  show the user the existing skill's description and ask whether to overwrite, pick
  a different name, or abort.

## Generate the SKILL.md

Once you have enough information and the name is validated, generate a complete
SKILL.md file with:
- YAML frontmatter (name, description, allowed-tools, argument-hint, when_to_use)
- Markdown body with clear step-by-step instructions

Write it to `.alan/skills/<name>/SKILL.md` in the project directory.

Note: If the user wants a global skill (available across all projects), they should
manually place the SKILL.md at `~/.alan/skills/<name>/SKILL.md` instead. The /skill
create flow always writes to the project directory.

## Example output format

```markdown
---
name: example
description: Use when user asks to "do the example thing"
allowed-tools: [Bash, Read, Edit]
argument-hint: "[target]"
when_to_use: "When the user wants to run the example workflow"
---

# Example Skill

Do the example thing for $ARGUMENTS.

## Steps

1. First step
2. Second step
3. Third step
```

After writing the file, confirm to the user and suggest they try it with `/skill <name>`.
"""

BUILTIN_SKILLS: dict[str, SkillDefinition] = {
    "create": SkillDefinition(
        name="create",
        description="Create a new skill by interviewing the user about their workflow",
        body=CREATE_SKILL_BODY,
        source_path="<builtin>",
        allowed_tools=["Read", "Write", "Glob", "Bash"],
        argument_hint="[name] [description...]",
        when_to_use=None,  # never auto-invoked by model
    ),
}
