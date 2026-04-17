/**
 * Alan Code GUI — WebSocket client and event renderer.
 *
 * Connects to the server, receives OutputEvents, and renders them
 * into the chat panel. Handles input submission and panel toggling.
 */

// ── State ──────────────────────────────────────────────────────

let ws = null;
let isAgentRunning = false;
let pendingInputRequest = null;
let currentStreamEl = null; // Element for the current streaming text
let inputHistory = [];       // Past user inputs
let historyIndex = -1;       // Current position in history (-1 = new input)

// ── DOM refs ───────────────────────────────────────────────────

const chatMessages = document.getElementById("chat-messages");
const chatInput = document.getElementById("chat-input");
const btnSend = document.getElementById("btn-send");
const btnAbort = document.getElementById("btn-abort");
const askOverlay = document.getElementById("ask-overlay");
const askQuestion = document.getElementById("ask-question");
const askOptions = document.getElementById("ask-options");
const statusDot = document.getElementById("status-dot");
const statusText = document.getElementById("status-text");
const llmMessages = document.getElementById("llm-messages");

// Panel toggles
const toggleChat = document.getElementById("toggle-chat");
const toggleLlm = document.getElementById("toggle-llm");
const toggleGitTree = document.getElementById("toggle-git-tree");
const panelChat = document.getElementById("panel-chat");
const panelLlm = document.getElementById("panel-llm");
const panelGitTree = document.getElementById("panel-git-tree");

// ── WebSocket connection ───────────────────────────────────────

function connect() {
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    ws = new WebSocket(`${proto}//${location.host}/ws`);

    ws.onopen = () => {
        statusDot.className = "dot dot-connected";
        statusText.textContent = "Connected";
        setInputEnabled(true);
    };

    ws.onclose = () => {
        statusDot.className = "dot dot-disconnected";
        statusText.textContent = "Disconnected — reconnecting...";
        setInputEnabled(false);
        setTimeout(connect, 2000);
    };

    ws.onerror = () => {
        statusDot.className = "dot dot-disconnected";
        statusText.textContent = "Connection error";
    };

    ws.onmessage = (ev) => {
        try {
            const msg = JSON.parse(ev.data);
            handleServerMessage(msg);
        } catch (e) {
            console.error("Failed to parse message:", e);
        }
    };
}

// ── Message handling ───────────────────────────────────────────

function handleServerMessage(msg) {
    const kind = msg.kind;

    if (kind === "event") {
        handleEvent(msg.event);
    } else if (kind === "input_request") {
        handleInputRequest(msg.request);
    } else if (kind === "agent_start") {
        setAgentRunning(true);
    } else if (kind === "agent_done") {
        setAgentRunning(false);
    }
}

function handleEvent(event) {
    const type = event.type;
    const data = event.data;

    switch (type) {
        case "reset":
            // Server telling us to clear state before replaying history.
            // Prevents duplicated messages when the browser reconnects to
            // a restarted alancode (auto-reconnect keeps the DOM around).
            while (chatMessages.firstChild) {
                chatMessages.removeChild(chatMessages.firstChild);
            }
            currentStreamEl = null;
            break;

        case "request_start":
            // New API call starting — show agent is working
            setAgentRunning(true);
            currentStreamEl = null;
            break;

        case "assistant_delta":
            renderAssistantDelta(data);
            break;

        case "assistant_message":
            renderAssistantFinal(data);
            break;

        case "user_message":
            renderUserMessage(data);
            break;

        case "system_message":
            renderSystemMessage(data);
            break;

        case "cost_summary":
            renderCostSummary(data);
            setAgentRunning(false);
            break;

        case "llm_perspective":
            renderLlmPerspective(data.messages, data.system_prompt);
            break;

        case "local_output":
            appendMsg("msg-local-output", data.text || "");
            break;

        case "git_tree_update":
            if (typeof renderGitTree === "function") {
                renderGitTree(data);
            }
            break;

        default:
            // Ignore unknown events
            break;
    }
}

// ── Chat rendering ─────────────────────────────────────────────

function renderAssistantDelta(data) {
    // Streaming text chunk
    if (!data.content) return;

    for (const block of data.content) {
        if (block.type === "text" && block.text) {
            if (!currentStreamEl) {
                currentStreamEl = appendMsg("msg-assistant", "");
            }
            currentStreamEl.textContent += block.text;
            scrollToBottom();
        } else if (block.type === "thinking" && block.thinking) {
            const el = appendMsg("msg-thinking", block.thinking);
            scrollToBottom();
        }
    }
}

function renderAssistantFinal(data) {
    // Final assistant message — render text (if not already streamed) and tool calls.
    // During live streaming, currentStreamEl is set by renderAssistantDelta and text
    // is already on screen. On resume/replay or for synthetic messages (errors),
    // no deltas preceded this event so we must render text blocks here.
    const wasStreamed = currentStreamEl !== null;
    currentStreamEl = null;
    if (!data.content) return;

    for (const block of data.content) {
        if (block.type === "text" && block.text && !wasStreamed) {
            appendMsg("msg-assistant", block.text);
        } else if (block.type === "tool_use") {
            renderToolCall(block);
        }
    }
}

function renderToolCall(block) {
    const el = document.createElement("div");
    el.className = "msg-tool-call";

    const argsStr = Object.entries(block.input || {})
        .map(([k, v]) => `${k}=${truncate(String(v), 80)}`)
        .join(", ");

    el.innerHTML = `<span class="tool-name">${esc(block.name)}</span>`
        + `<span class="tool-args">(${esc(argsStr)})</span>`;

    chatMessages.appendChild(el);
    scrollToBottom();
}

function renderUserMessage(data) {
    if (data.hide_in_ui) return;

    const content = data.content;
    if (typeof content === "string") {
        // Skip system reminders (injected context, not user-typed).
        if (content.startsWith("<system-reminder>")) return;
        // Real user prompt — render with the same "> " style used for live input.
        appendMsg("msg-user", `> ${content}`);
        scrollToBottom();
        return;
    }

    // Tool results
    if (Array.isArray(content)) {
        for (const block of content) {
            if (block.type === "tool_result") {
                const text = typeof block.content === "string"
                    ? block.content
                    : (block.content || []).map(b => b.text || "").join("");
                // Edit / Write tools return a [ALAN-DIFF] sentinel followed by
                // a unified diff + summary line. Render as a styled diff view.
                if (!block.is_error && text.startsWith("[ALAN-DIFF]")) {
                    chatMessages.appendChild(renderDiffResult(text));
                } else {
                    const el = document.createElement("div");
                    el.className = "msg-tool-result" + (block.is_error ? " error" : "");
                    el.textContent = truncate(text, 1000);
                    chatMessages.appendChild(el);
                }
            }
        }
        scrollToBottom();
    }
}

// ── Diff rendering (Edit / Write tool results) ─────────────────

function renderDiffResult(text) {
    const body = text.slice("[ALAN-DIFF]".length).replace(/^\n+/, "");
    const lines = body.split("\n");

    // Separate diff body from the trailing plain-text summary line.
    let diffEnd = lines.length;
    for (let i = lines.length - 1; i >= 0; i--) {
        const ln = lines[i];
        if (ln.startsWith(" ") || ln.startsWith("+") || ln.startsWith("-")
            || ln.startsWith("@") || ln.startsWith("\\")) {
            diffEnd = i + 1;
            break;
        }
    }
    const diffLines = lines.slice(0, diffEnd);
    const summary = lines.slice(diffEnd).join("\n").trim();

    let filePath = "";
    for (let i = 0; i < Math.min(4, diffLines.length); i++) {
        if (diffLines[i].startsWith("+++ ")) {
            filePath = diffLines[i].slice(4).trim();
            break;
        }
    }

    let added = 0, removed = 0;
    for (const ln of diffLines) {
        if (ln.startsWith("+") && !ln.startsWith("+++")) added++;
        else if (ln.startsWith("-") && !ln.startsWith("---")) removed++;
    }

    const wrap = document.createElement("div");
    wrap.className = "msg-diff";

    const header = document.createElement("div");
    header.className = "msg-diff-header";
    header.innerHTML =
        `<span class="msg-diff-bullet">●</span> `
        + `<span class="msg-diff-label">Update</span>`
        + `(<span class="msg-diff-path">${esc(filePath)}</span>)`
        + (added
            ? ` <span class="msg-diff-added">+${added}</span>`
            : "")
        + (removed
            ? ` <span class="msg-diff-removed">-${removed}</span>`
            : "");
    wrap.appendChild(header);

    const body_el = document.createElement("div");
    body_el.className = "msg-diff-body";

    let oldNum = 0, newNum = 0;
    for (const ln of diffLines) {
        if (ln.startsWith("---") || ln.startsWith("+++")) continue;

        if (ln.startsWith("@@")) {
            const m = ln.match(/@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@/);
            if (m) {
                oldNum = parseInt(m[1], 10);
                newNum = parseInt(m[2], 10);
            }
            const row = document.createElement("div");
            row.className = "msg-diff-line msg-diff-hunk";
            row.textContent = ln;
            body_el.appendChild(row);
            continue;
        }

        const row = document.createElement("div");
        const numEl = document.createElement("span");
        numEl.className = "msg-diff-lineno";
        const markEl = document.createElement("span");
        markEl.className = "msg-diff-mark";
        const textEl = document.createElement("span");
        textEl.className = "msg-diff-text";

        if (ln.startsWith("+")) {
            row.className = "msg-diff-line msg-diff-add";
            numEl.textContent = String(newNum);
            markEl.textContent = "+";
            textEl.textContent = ln.slice(1);
            newNum++;
        } else if (ln.startsWith("-")) {
            row.className = "msg-diff-line msg-diff-remove";
            numEl.textContent = String(oldNum);
            markEl.textContent = "-";
            textEl.textContent = ln.slice(1);
            oldNum++;
        } else if (ln.startsWith("\\")) {
            row.className = "msg-diff-line msg-diff-nonewline";
            numEl.textContent = "";
            markEl.textContent = "";
            textEl.textContent = ln;
        } else {
            row.className = "msg-diff-line msg-diff-context";
            numEl.textContent = String(newNum);
            markEl.textContent = " ";
            textEl.textContent = ln.startsWith(" ") ? ln.slice(1) : ln;
            oldNum++;
            newNum++;
        }

        row.appendChild(numEl);
        row.appendChild(markEl);
        row.appendChild(textEl);
        body_el.appendChild(row);
    }

    wrap.appendChild(body_el);

    if (summary) {
        const foot = document.createElement("div");
        foot.className = "msg-diff-summary";
        foot.textContent = summary;
        wrap.appendChild(foot);
    }

    return wrap;
}

function renderSystemMessage(data) {
    if (data.hide_in_ui) return;
    const styleMap = { info: "msg-system", warning: "msg-system", error: "msg-error" };
    const cls = styleMap[data.level] || "msg-system";
    appendMsg(cls, data.content);
}

function renderCostSummary(data) {
    const inTokens = (data.input_tokens + (data.cache_read_tokens || 0)).toLocaleString();
    const outTokens = data.output_tokens.toLocaleString();
    let text = `Session: ${inTokens} in + ${outTokens} out`;
    if (!data.cost_unknown) {
        text += ` = $${data.cost_usd.toFixed(4)} (estimated)`;
    }
    if (data.context_window > 0 && data.conversation_tokens > 0) {
        const pct = Math.round(data.conversation_tokens * 100 / data.context_window);
        text += ` | Conversation: ${data.conversation_tokens.toLocaleString()} / ${data.context_window.toLocaleString()} (${pct}%)`;
    }
    appendMsg("msg-cost", text);
}

// ── Input handling ─────────────────────────────────────────────

function handleInputRequest(request) {
    pendingInputRequest = request;

    // Receiving an input request means the agent is done processing
    // (slash commands don't send agent_done, but they do re-request input)
    setAgentRunning(false);

    if (request.type === "prompt") {
        // Main prompt — just enable the input box
        setInputEnabled(true);
        chatInput.focus();
    } else if (request.type === "ask" || request.type === "confirm") {
        // Show ask overlay with options
        askQuestion.textContent = request.question;
        askOptions.innerHTML = "";

        for (const opt of request.options) {
            const btn = document.createElement("button");
            btn.className = "ask-option";
            btn.textContent = opt;
            btn.onclick = () => sendInputResponse(opt);
            askOptions.appendChild(btn);
        }

        askOverlay.classList.add("visible");
        setInputEnabled(true);
        chatInput.focus();
    }
}

function sendInputResponse(value) {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;

    if (isAgentRunning && !pendingInputRequest) {
        // Agent is running — this is a "btw" injection
        ws.send(JSON.stringify({ kind: "inject", text: value }));
        appendMsg("msg-user", `> [btw] ${value}`);
        return;
    }

    ws.send(JSON.stringify({
        kind: pendingInputRequest ? "input_response" : "prompt",
        request_id: pendingInputRequest ? pendingInputRequest.id : null,
        value: value,
        text: value,
    }));

    appendMsg("msg-user", `> ${value}`);

    // Save to input history
    inputHistory.push(value);
    historyIndex = -1;

    pendingInputRequest = null;
    askOverlay.classList.remove("visible");
    setInputEnabled(false);
    setAgentRunning(true);
}

// ── LLM Perspective rendering ──────────────────────────────────

function renderLlmPerspective(messages, systemPrompt) {
    llmMessages.innerHTML = "";

    // Render system prompt first (if provided)
    if (systemPrompt) {
        const sysEl = document.createElement("div");
        sysEl.className = "llm-msg llm-msg-system";
        const roleEl = document.createElement("div");
        roleEl.className = "llm-role llm-role-system";
        roleEl.textContent = "system";
        sysEl.appendChild(roleEl);
        const contentEl = document.createElement("div");
        contentEl.className = "llm-content";
        contentEl.textContent = systemPrompt;  // Full system prompt, no truncation
        sysEl.appendChild(contentEl);
        llmMessages.appendChild(sysEl);
    }

    if (!messages || messages.length === 0) {
        if (!systemPrompt) {
            llmMessages.innerHTML = '<div style="color: var(--text-dim); padding: 20px; text-align: center;">No messages yet</div>';
        }
        return;
    }

    for (const msg of messages) {
        const role = msg.role || "unknown";
        const el = document.createElement("div");
        el.className = `llm-msg llm-msg-${role}`;

        const roleEl = document.createElement("div");
        roleEl.className = `llm-role llm-role-${role}`;
        roleEl.textContent = role;
        el.appendChild(roleEl);

        const contentEl = document.createElement("div");
        contentEl.className = "llm-content";

        if (typeof msg.content === "string") {
            contentEl.textContent = truncate(msg.content, 2000);
        } else if (msg.tool_calls) {
            // Assistant with tool calls
            const text = msg.content || "";
            if (text) contentEl.textContent = truncate(text, 1000) + "\n\n";
            for (const tc of msg.tool_calls) {
                const fn = tc.function || {};
                contentEl.textContent += `[tool_call] ${fn.name}(${truncate(fn.arguments || "", 200)})\n`;
            }
        } else if (msg.tool_call_id) {
            // Tool result
            contentEl.textContent = truncate(msg.content || "", 1000);
        } else {
            contentEl.textContent = truncate(JSON.stringify(msg.content), 1000);
        }

        el.appendChild(contentEl);
        llmMessages.appendChild(el);
    }

    llmMessages.scrollTop = llmMessages.scrollHeight;
}

// ── Panel toggles ──────────────────────────────────────────────

function updatePanelVisibility() {
    const showChat = toggleChat.checked;
    const showLlm = toggleLlm.checked;
    const showGitTree = toggleGitTree ? toggleGitTree.checked : false;

    panelChat.style.display = showChat ? "flex" : "none";
    panelLlm.style.display = showLlm ? "flex" : "none";
    if (panelGitTree) panelGitTree.style.display = showGitTree ? "flex" : "none";

    // Equal flex distribution
    if (showChat) panelChat.style.flex = "1";
    if (showLlm) panelLlm.style.flex = "1";
    if (panelGitTree && showGitTree) panelGitTree.style.flex = "1";
}

toggleChat.addEventListener("change", updatePanelVisibility);
toggleLlm.addEventListener("change", updatePanelVisibility);
if (toggleGitTree) toggleGitTree.addEventListener("change", updatePanelVisibility);

// ── UI helpers ─────────────────────────────────────────────────

function appendMsg(className, text) {
    const el = document.createElement("div");
    el.className = `msg ${className}`;
    el.textContent = text;
    chatMessages.appendChild(el);
    scrollToBottom();
    return el;
}

function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function setInputEnabled(enabled) {
    chatInput.disabled = !enabled;
    btnSend.disabled = !enabled;
}

function setAgentRunning(running) {
    isAgentRunning = running;
    btnAbort.style.display = running ? "inline-block" : "none";
    // Keep input enabled during agent turns for "btw" messages
    setInputEnabled(true);
    chatInput.placeholder = running
        ? "Type a message to inject mid-turn... (Enter to send)"
        : "Type a message... (Enter to send, Shift+Enter for newline)";
}

function truncate(text, max) {
    if (!text) return "";
    return text.length > max ? text.slice(0, max - 3) + "..." : text;
}

function esc(text) {
    const el = document.createElement("span");
    el.textContent = text;
    return el.innerHTML;
}

// ── Event listeners ────────────────────────────────────────────

// Send button
btnSend.addEventListener("click", () => {
    const text = chatInput.value.trim();
    if (!text) return;
    chatInput.value = "";
    chatInput.style.height = "auto";
    sendInputResponse(text);
});

// Enter to send, Shift+Enter for newline, Up/Down for history
chatInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        btnSend.click();
    } else if (e.key === "ArrowUp" && inputHistory.length > 0) {
        e.preventDefault();
        if (historyIndex === -1) historyIndex = inputHistory.length;
        if (historyIndex > 0) {
            historyIndex--;
            chatInput.value = inputHistory[historyIndex];
        }
    } else if (e.key === "ArrowDown" && historyIndex >= 0) {
        e.preventDefault();
        historyIndex++;
        if (historyIndex >= inputHistory.length) {
            historyIndex = -1;
            chatInput.value = "";
        } else {
            chatInput.value = inputHistory[historyIndex];
        }
    }
});

// Auto-resize textarea
chatInput.addEventListener("input", () => {
    chatInput.style.height = "auto";
    chatInput.style.height = Math.min(chatInput.scrollHeight, 120) + "px";
});

// Abort button
btnAbort.addEventListener("click", () => {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ kind: "abort" }));
    }
});

// ── Session info ───────────────────────────────────────────────

async function loadSessionInfo() {
    try {
        const resp = await fetch("/api/session");
        if (resp.ok) {
            const info = await resp.json();
            const label = document.getElementById("session-label");
            const parts = [];
            if (info.project) parts.push(info.project);
            if (info.session_name) parts.push(info.session_name);
            else if (info.session_id) parts.push(info.session_id.slice(0, 12));
            if (info.model) parts.push(info.model);
            label.textContent = parts.join(" | ");
            document.title = `Alan Code — ${info.project || ""}`;
        }
    } catch (e) {
        // Ignore — session info is optional
    }
}

// ── Git Tree buttons ──────────────────────────────────────────
// All buttons operate on the selected node (click a node first).

const btnMoveTo = document.getElementById("btn-move-here");
const btnRevertTo = document.getElementById("btn-revert-to");
const btnConvRevertTo = document.getElementById("btn-conv-revert-to");
const btnAllRevertTo = document.getElementById("btn-all-revert-to");

if (btnMoveTo) {
    btnMoveTo.addEventListener("click", () => {
        if (!isAgentRunning && gtSelectedNode) {
            sendInputResponse("/move " + gtSelectedNode);
        }
    });
}

if (btnRevertTo) {
    btnRevertTo.addEventListener("click", () => {
        if (!isAgentRunning && gtSelectedNode) {
            if (confirm("Revert repo to selected commit? Commits after it will be destroyed.")) {
                sendInputResponse("/revert " + gtSelectedNode);
            }
        }
    });
}

if (btnConvRevertTo) {
    btnConvRevertTo.addEventListener("click", () => {
        if (!isAgentRunning && gtSelectedNode) {
            if (confirm("Revert conversation to selected commit? The agent will forget everything after it.")) {
                sendInputResponse("/convrevert " + gtSelectedNode);
            }
        }
    });
}

if (btnAllRevertTo) {
    btnAllRevertTo.addEventListener("click", () => {
        if (!isAgentRunning && gtSelectedNode) {
            if (confirm("Revert both repo and conversation to selected commit?")) {
                sendInputResponse("/allrevert " + gtSelectedNode);
            }
        }
    });
}

// Patch setAgentRunning to update GT buttons
const _origSetAgentRunning = setAgentRunning;
setAgentRunning = function(running) {
    _origSetAgentRunning(running);
    if (typeof _updateGitTreeButtons === "function") _updateGitTreeButtons();
};

// ── Start ──────────────────────────────────────────────────────

connect();
loadSessionInfo();
