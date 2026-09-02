# Known limitations

Behaviors that surprise people, written down so nobody has to rediscover them by
debugging. Each entry says what actually happens, when it bites, and what to do
instead. These are current-design facts, not bugs on a fix list.

## `max_iterations_per_turn` counts tool cycles, not model calls

The cap counts **completed model -> tool cycles**. Recovery calls do not count:
output-truncation escalation and continuation, malformed-tool-call retries, and
the empty-response nudge (`empty_response_retries`) are extra billed calls that
leave the counter untouched. A turn with a low cap can therefore still make more
model calls than the cap suggests.

- Default is `None` (unlimited); a positive integer ends the turn with a
  `max_iterations_per_turn_reached` attachment message.
- If you need a hard ceiling on *spend*, watch the cost line rather than this
  setting - it is a loop-shape control, not a budget.

## Abort is cooperative

`agent.abort()` sets a flag the query loop checks at defined points. It does not
cancel a request already in flight or kill a tool already running: a ten-minute
`Bash` command finishes, then the abort takes effect at the next checkpoint.

- For a guaranteed deadline, an outer harness must own it - a process-level
  timeout, a scheduler walltime, or a supervisor that kills the child.
- `request_timeout` bounds one backend request only. It is not a turn deadline
  and not a task deadline.

## Token counts are estimates

Alan does not run the provider's tokenizer. It takes the maximum of the last
call's reported usage and its own estimate, then adds a safety margin. The
fallback estimator (characters / 3) under-counts scripts such as CJK.

The failure mode is deliberately one-sided: compaction fires slightly early
rather than letting a request overflow, and a genuine prompt-too-long is caught
and retried with truncation. Treat the displayed conversation size as accurate
to within a few percent, never as an exact number.

## A `PreToolUse` hook returning `ask` does not force a prompt

`ask` falls through to the ordinary permission pipeline. If the current
permission mode already auto-allows that tool level, the tool runs with **no
prompt at all**.

A hook that must stop a tool has to return `deny`. See
[guides/hooks.md](../guides/hooks.md).

## Skill `allowed-tools` is a name filter, not a sandbox

A skill's `allowed-tools` frontmatter restricts which tools are offered while
the skill is active, by **tool name only**. An argument clause such as
`Bash(git:*)` is parsed and logged, then only the `Bash` part is enforced - the
skill gets all of `Bash`, not just git commands.

Treat skill frontmatter as model guidance plus a tool-name allowlist. To
restrict what a tool may be called *with*, use permission rules
([concepts/tools-and-permissions.md](../concepts/tools-and-permissions.md)).

## Extended thinking cannot be switched on from Alan

There is no setting that enables reasoning on a model that is not already
serving it. `disable_thinking` only turns it *off*, and only on the `auto`
(LiteLLM) transport where a chat template exists to obey
`chat_template_kwargs={"enable_thinking": false}`; the native Anthropic backend
never receives it. `persist_thinking` only carries reasoning the model already
returned forward into later turns.

## Settings are not one atomic schema object

Constructor parameters are the checked surface. Everything else is applied
through `.alan/settings.json` or `update_session_setting()`. **An unrecognized
constructor keyword is not an error** - it is forwarded to backend construction,
which is how transport-level options reach the provider. A typo therefore
travels silently to the backend instead of raising.

When wiring a setting programmatically, prefer `update_session_setting()`, which
validates the key. See [reference/python-api.md](python-api.md).

## The GUI's LLM perspective is not the wire

The GUI renders what Alan is about to send *before* backend-specific shaping
(serialization, cache breakpoints, provider quirks), so it is the right tool for
debugging history and compaction and the wrong one for proving the exact
payload.

For the real request, set `ALANCODE_WIRE_LOG` to a file path. Every outgoing
provider request is appended there as one JSON line, credentials redacted:

```bash
ALANCODE_WIRE_LOG=/tmp/wire.jsonl alancode
jq '.request.messages[-1]' /tmp/wire.jsonl | tail
```

## Two custom-tool fields are inert

`ToolResult.new_messages` and `Tool.max_result_size_chars` exist so v1
custom-tool sources keep importing, but execution reads neither. Result size is
capped by the context budget instead (compaction Layer A), and extra content
must be returned inside the tool result itself. A tool that sets either one logs
a warning the first time it runs.

## Compaction is itself a model call

Layer C summarization spends real tokens, including the attempts a
prompt-too-long retry throws away. Those calls are counted in the session totals
alongside normal turns, so a session that compacts often costs more than its
visible turns suggest. See [reference/cost.md](cost.md).
