# Alan Code — Documentation

Welcome. Alan Code is a Python coding agent you can use as a **CLI**, a **browser GUI**, or a **Python library**. These docs are organised into four tracks, following the [Diátaxis framework](https://diataxis.fr/):

| Track | What it is | Read this when… |
|---|---|---|
| **Getting started** | A single linear walkthrough | You just installed Alan and want to see it work end-to-end. |
| **Concepts** | How things *are* | You want to understand — the mental model, terminology, what each subsystem does. |
| **Guides** | How to *do* a specific task | You have a concrete goal (set up a local model, build a custom agent, configure hooks). |
| **Reference** | Exhaustive lookup | You know what you need and want the exact name / default / schema. |
| **Architecture** | For contributors | You're reading the code or building on it. |
| **Contributing** | Process | You want to send a PR or cut a release. |

## Landing points

- **New to Alan?** → [Getting started](getting-started.md)
- **Looking up a slash command?** → [reference/slash-commands.md](reference/slash-commands.md)
- **Looking up a CLI flag?** → [reference/cli.md](reference/cli.md)
- **Hooking up a local model?** → [reference/local-models.md](reference/local-models.md)
- **Building an agent in Python?** → [guides/building-agents.md](guides/building-agents.md) and [reference/python-api.md](reference/python-api.md)

## Concepts

- [The agent loop](concepts/agent-loop.md) — turns, iterations, sessions, and how they relate
- [Tools and permissions](concepts/tools-and-permissions.md) — how tools run and when the user is asked
- [Context and compaction](concepts/context-and-compaction.md) — the 3 compaction layers and how to tune them
- [Memory](concepts/memory.md) — the `off` / `on` / `intensive` modes and what they store
- [Skills](concepts/skills.md) — user-defined prompt templates the agent can invoke
- [Project context (ALAN.md)](concepts/project-context.md) — per-project instructions auto-loaded into the system prompt
- [Git Tree (AGT)](concepts/git-tree.md) — `/move`, `/revert`, `/convrevert`, `/allrevert`

## Guides

- [Using the GUI](guides/using-the-gui.md)
- [Configuration](guides/configuration.md) — settings.json priority chain
- [Hooks](guides/hooks.md) — pre/post tool-use callbacks
- [Building agents](guides/building-agents.md) — using `AlanCodeAgent` as a library

## Reference

- [CLI flags](reference/cli.md)
- [Slash commands](reference/slash-commands.md)
- [Tools](reference/tools.md) — every built-in tool with schemas and examples
- [Settings](reference/settings.md) — every `.alan/settings.json` key
- [Providers](reference/providers.md) — Anthropic, LiteLLM, local
- [Local models](reference/local-models.md) — vLLM / SGLang / Ollama / llama.cpp
- [Cost tracking](reference/cost.md) — what the "Session" line means
- [Python API](reference/python-api.md) — `AlanCodeAgent` and related

## Architecture (for contributors)

- [Overview](architecture/overview.md)
- [The query loop](architecture/query-loop.md)
- [System prompt assembly](architecture/system-prompt.md)
- [Messages and the API payload](architecture/messages-and-api.md)
- [Prompt caching](architecture/prompt-caching.md)

## Contributing

- [Development setup](contributing/development.md)
- [Testing](contributing/testing.md)
- [Release process](contributing/release.md)
