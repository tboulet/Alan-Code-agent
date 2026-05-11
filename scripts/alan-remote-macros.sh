# Shell macros for the remote-scripted backend.
#
# Source this file in your shell:
#
#     source ~/projects/Alan-Code-agent/scripts/alan-remote-macros.sh
#
# Then drive an Alan session from your terminal:
#
#     alan-pending-last           # see Alan's latest message
#     alan-bash 'ls -la'          # call the Bash tool
#     alan-wait                   # block until next pending call
#     alan-text "I'm done."       # text-only turn
#     alan-exit                   # ExitTask
#
# Default port is 8430. Override with ALAN_PORT=8431 before sourcing,
# or with ALAN_PORT=8431 prefixed on individual calls.
#
# Requires: curl, jq.

: "${ALAN_PORT:=8430}"
export ALAN_PORT

_alan_url() {
    echo "http://127.0.0.1:${ALAN_PORT}"
}

alan-help() {
    cat <<EOF
Remote-scripted Alan macros
---------------------------
  alan-pending             pretty-print the current pending payload
  alan-pending-last        just the most recent message
  alan-pending-system      just the system prompt array
  alan-pending-tools       just the available tool names
  alan-session             session metadata (id, cwd, port, calls_served)
  alan-health              {"ok": true} if the server is up
  alan-wait                poll /api/pending until 200; returns when ready

  alan-text TEXT           text-only response
  alan-tool NAME INPUT_JSON [TEXT]
                           generic tool call, e.g.
                           alan-tool Read '{"file_path":"solution.py"}'
  alan-bash 'cmd'          shortcut for Bash tool
  alan-read PATH           shortcut for Read tool
  alan-submit              SubmitSolution
  alan-exit                ExitTask (ends the session)

Port: ${ALAN_PORT} (override with ALAN_PORT=NNNN).
EOF
}

# ── Reads ────────────────────────────────────────────────────────────────

alan-health() {
    curl -s "$(_alan_url)/api/health"
    echo
}

alan-session() {
    curl -s "$(_alan_url)/api/session" | jq
}

alan-pending() {
    curl -s "$(_alan_url)/api/pending" | jq
}

alan-pending-last() {
    curl -s "$(_alan_url)/api/pending" | jq '.messages[-1]'
}

alan-pending-system() {
    curl -s "$(_alan_url)/api/pending" | jq '.system'
}

alan-pending-tools() {
    curl -s "$(_alan_url)/api/pending" | jq '.tools | map(.name)'
}

alan-wait() {
    local i=0
    while [ "$(curl -s -o /dev/null -w '%{http_code}' "$(_alan_url)/api/pending")" != "200" ]; do
        i=$((i + 1))
        if [ $i -gt 200 ]; then
            echo "[alan-wait] gave up after ~60s — server may be down" >&2
            return 1
        fi
        sleep 0.3
    done
    echo "[alan-wait] ready"
}

# ── Writes ────────────────────────────────────────────────────────────────

alan-text() {
    if [ "$#" -lt 1 ]; then
        echo "usage: alan-text 'message'" >&2
        return 2
    fi
    local payload
    payload=$(jq -n --arg t "$1" '{text:$t}')
    curl -s -X POST "$(_alan_url)/api/respond" \
        -H 'Content-Type: application/json' -d "$payload"
    echo
}

alan-tool() {
    if [ "$#" -lt 2 ]; then
        echo "usage: alan-tool TOOL_NAME 'JSON_INPUT' [TEXT]" >&2
        return 2
    fi
    local name="$1"
    local input="$2"
    local text="${3:-}"
    local payload
    payload=$(jq -n \
        --arg t "$text" \
        --arg n "$name" \
        --argjson i "$input" \
        '{text:$t, tool_calls:[{name:$n, input:$i}]}')
    curl -s -X POST "$(_alan_url)/api/respond" \
        -H 'Content-Type: application/json' -d "$payload"
    echo
}

alan-bash() {
    if [ "$#" -lt 1 ]; then
        echo "usage: alan-bash 'shell command' [text]" >&2
        return 2
    fi
    local cmd="$1"
    local text="${2:-running shell command}"
    local input
    input=$(jq -n --arg c "$cmd" '{command:$c}')
    alan-tool Bash "$input" "$text"
}

alan-read() {
    if [ "$#" -lt 1 ]; then
        echo "usage: alan-read /path/to/file" >&2
        return 2
    fi
    local path="$1"
    local input
    input=$(jq -n --arg p "$path" '{file_path:$p}')
    alan-tool Read "$input" "reading $path"
}

alan-submit() {
    alan-tool SubmitSolution '{}' "submitting for eval"
}

alan-exit() {
    alan-tool ExitTask '{}' "done"
}
