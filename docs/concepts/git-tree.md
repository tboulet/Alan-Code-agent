# Git Tree (AGT)

The **Agentic Git Tree (AGT)** is Alan's treatment of git as first-class state. Unlike traditional dev tools that treat git as a side channel, AGT tracks where the agent is in your repo, which commits it has made, and lets you move or revert — **both the files and the conversation** — together.

AGT is in development. It's usable but the UX is still being refined, especially around the GUI's tree panel.

## Vocabulary

| Term | Definition |
|---|---|
| **Agent position** | The SHA Alan considers "current". Usually equal to `HEAD`, but can diverge after external commits or reverts. |
| **Session root** | The SHA `HEAD` was at when the session started. Anchors the tree view. |
| **Alan commits** | Commits made by the agent via the `GitCommit` tool during this session. Tracked separately so the GUI can colour them blue. |
| **Conv path** | An ordered list of SHAs the agent has "been at" — the trajectory through the commit graph. Used to compute "steps back" for `/convrevert`. |

All of the above are persisted in `.alan/sessions/<id>/state.json`.

## The four movement commands

### `/move <ref>`

Move the agent's position to a different commit or branch — essentially a `git checkout`, but Alan also updates its position tracking and injects a reminder so the agent knows the working tree changed:

```
<system-reminder>User ran /move, checking out commit abc1234def (ref 'main'). The working tree now reflects that commit — files on disk may have changed compared to what you saw earlier. … Re-read files before making assumptions about their current state.</system-reminder>
```

Safe: `/move` preserves commits. It's just a checkout.

### `/revert [N]`

Destructively revert N commits back on the current branch via `git reset --hard HEAD~N`. The commits are removed from the branch (still recoverable via `git reflog` for ~30 days).

Special case: if the working tree is dirty and `N=1`, just discards the uncommitted changes without touching commits.

The conversation is **not** affected — the agent still remembers what happened. Only the repo moves.

### `/convrevert [N]`

**Conversation** revert: drop the last N user↔agent exchanges from the conversation, but leave the working tree untouched. Useful for "we went down the wrong path, let me restart this turn with a different framing" without losing the files we touched.

### `/allrevert [N]`

Both at once: revert the working tree AND truncate the conversation by the equivalent number of steps. Use when you want a clean do-over — the agent's memory of the sidetrack and the files it produced both vanish.

## Why these are different tools, not flags

| Command | Repo state | Conversation |
|---|---|---|
| `/move <ref>` | → new commit | unchanged |
| `/revert N` | → N commits back | unchanged |
| `/convrevert N` | unchanged | → N steps back |
| `/allrevert N` | → N commits back | → N steps back |

Four corners of a 2×2. Each pair of axes is a legitimate use case: sometimes you want to explore a different branch without losing context, sometimes you want to forget the last exchange but keep the files, sometimes both.

## Safety: `.alan/` is never touched

All AGT operations filter `.alan/` from the working tree cleanup (`git clean -fd -e .alan`). So your session state, memory, skills, and allow rules survive reverts. Without this guard a `/revert` would destroy the session itself.

Alan also ensures `.alan/` is in your `.gitignore` on session start. If it's not, you'll see a `[WARNING]` and the session will refuse some destructive operations until it is.

## Memory snapshots

Before any destructive op, Alan takes a memory snapshot — a copy of `.alan/memory/` tagged with the SHA you're leaving. After the op, it restores the snapshot corresponding to the destination SHA (if one exists).

This means: if you edit memories, commit, then `/revert`, your memories return to what they were at the older commit. This is what makes "Alan commits" work well — the agent's mental state moves with the repo.

Stored at `.alan/sessions/<id>/memory_snapshots/<sha>/`. Cleaned up when the session ends.

## The GUI Git Tree panel

When you launch with `--gui`, the third panel shows the commit graph with:

- **Blue nodes**: Alan commits (made by `GitCommit` in this session).
- **Grey nodes**: external commits (made by you outside the session).
- **Dashed node**: uncommitted changes.
- **Blue line**: the conversation path (trajectory through commits).
- **Yellow ring**: session root or compaction marker.
- **White ring**: the agent's current position.
- **Pink ring**: selected commit (click any node).
- **Green labels**: branch names.

When you click a node, four buttons light up: **Move to commit**, **Revert repo to**, **Revert conv. to**, **Revert all to** — GUI equivalents of the four slash commands.

The **curvature slider** in the top-right legend tunes how curved the branch-jump arrows are.

## Commits made by Alan

When the agent uses `GitCommit` (typically via `/commit`), the commit gets a `Co-Authored-By: Alan Code` trailer and is tracked in `state.alan_commits`. The GUI colours those blue. Rich commit history: you can always tell at a glance which commits were yours vs Alan's.

## When to use AGT vs plain git

- Plain `git` commands (via `Bash`) still work, of course. AGT doesn't try to replace them.
- Use AGT when **agent mental state matters** — after a bad path, before a major refactor, when you want the agent to pick up from an older state.
- Use plain git for routine work that Alan doesn't need to know about.

AGT is about **keeping the agent's view of reality and git's view of reality in sync**, and making the divergence points explicit.

## Related

- [reference/slash-commands.md](../reference/slash-commands.md) — exact command syntax.
- [guides/using-the-gui.md](../guides/using-the-gui.md) — Git Tree panel walkthrough.
- [concepts/memory.md](memory.md) — why memory snapshots matter for AGT.
- `alancode/git_tree/` — the implementation.
