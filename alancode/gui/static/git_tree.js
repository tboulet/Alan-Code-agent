/**
 * Git Tree SVG renderer for the AGT panel.
 *
 * Node positions are STABLE: cached in gtPositionCache.
 * Once assigned, a node's (x,y) never changes.
 */

// ── Constants ─────────────────────────────────────────────────────

const GT_NODE_RADIUS = 8;
const GT_SPACING_X = 90;
const GT_SPACING_Y = 56;
// Asymmetric horizontal padding. Commit-message labels use text-anchor:middle,
// so they extend ~half the truncated label width (≈ GT_MAX_LABEL_LEN * 3.5 px)
// to the left of the leftmost node. The right side additionally hosts branch
// tags drawn start-anchored next to the rightmost node.
const GT_PADDING_LEFT = 100;
const GT_PADDING_RIGHT = 140;
const GT_PADDING_Y = 40;
const GT_MAX_LABEL_LEN = 20;

// Runtime-adjustable parameters (not persisted).
let gtCurvature = 1.0;

// Colors are read once from CSS variables so the whole SVG shares a single
// source of truth with the rest of the app (and any theme extension has
// only one value to work with — no more stroke-vs-fill drift).
function _css(name, fallback) {
    const v = getComputedStyle(document.documentElement)
        .getPropertyValue(name).trim();
    return v || fallback;
}
const GT_BLUE   = _css("--blue",   "#89b4fa");
const GT_YELLOW = _css("--yellow", "#f9e2af");
const GT_GREEN  = _css("--green",  "#a6e3a1");
const GT_PINK   = _css("--mauve",  "#f5c2e7");
const GT_TEXT       = _css("--text",       "#cdd6f4");
const GT_TEXT_DIM   = _css("--text-dim",   "#6c7086");
const GT_TEXT_MUTED = _css("--text-muted", "#585b70");

// Semantic aliases — one role per colour, single hex each.
const GT_COLOR_PARENT_EDGE      = GT_TEXT_MUTED;
const GT_COLOR_ALAN             = GT_BLUE;
const GT_COLOR_EXTERNAL         = GT_TEXT_DIM;
const GT_COLOR_CURRENT          = GT_TEXT;
const GT_COLOR_CONV_PATH        = GT_BLUE;
const GT_COLOR_CONV_JUMP        = GT_BLUE;
const GT_COLOR_POST_COMPACTION  = GT_YELLOW;
const GT_COLOR_COMPACTION_RING  = GT_YELLOW;
const GT_COLOR_AGENT_RING       = GT_TEXT;
const GT_COLOR_BRANCH_TAG       = GT_GREEN;
const GT_COLOR_TEXT             = GT_TEXT_DIM;
const GT_COLOR_SELECTION        = GT_PINK;

// ── Stable position cache ─────────────────────────────────────────

const gtPositionCache = {};

// ── State ─────────────────────────────────────────────────────────

let gtSelectedNode = null;
let gtLastData = null;
let gtTooltipEl = null;

// ── Main render ───────────────────────────────────────────────────

function renderGitTree(data) {
    gtLastData = data;
    const svg = document.getElementById("git-tree-svg");
    const emptyMsg = document.getElementById("git-tree-empty");
    if (!svg) return;

    while (svg.firstChild) svg.removeChild(svg.firstChild);

    const nodes = data.nodes || [];
    const edges = data.edges || [];

    if (nodes.length === 0) {
        if (emptyMsg) emptyMsg.style.display = "block";
        svg.style.display = "none";
        return;
    }
    if (emptyMsg) emptyMsg.style.display = "none";
    svg.style.display = "block";

    // ── Cache merge (stability) ───────────────────────────────────
    for (const n of nodes) {
        if (n.sha === "__dirty__") {
            gtPositionCache[n.sha] = { x: n.x, y: n.y };
        } else if (!(n.sha in gtPositionCache)) {
            gtPositionCache[n.sha] = { x: n.x, y: n.y };
        }
    }

    // ── Compute pixel coordinates ─────────────────────────────────
    let minAbsX = 0, maxAbsX = 0, maxY = 0;
    for (const n of nodes) {
        const p = gtPositionCache[n.sha] || { x: n.x, y: n.y };
        if (p.x < minAbsX) minAbsX = p.x;
        if (p.x > maxAbsX) maxAbsX = p.x;
        if (p.y > maxY) maxY = p.y;
    }

    const rangeX = maxAbsX - minAbsX;
    const centerX = GT_PADDING_LEFT + (-minAbsX) * GT_SPACING_X;
    const svgWidth = GT_PADDING_LEFT + GT_PADDING_RIGHT + rangeX * GT_SPACING_X;
    const svgHeight = GT_PADDING_Y * 2 + maxY * GT_SPACING_Y;

    svg.setAttribute("width", Math.max(svgWidth, 250));
    svg.setAttribute("height", Math.max(svgHeight, 100));
    svg.setAttribute("viewBox",
        `0 0 ${Math.max(svgWidth, 250)} ${Math.max(svgHeight, 100)}`);

    const posMap = {};
    for (const n of nodes) {
        const c = gtPositionCache[n.sha] || { x: n.x, y: n.y };
        posMap[n.sha] = {
            px: centerX + c.x * GT_SPACING_X,
            py: GT_PADDING_Y + c.y * GT_SPACING_Y,
        };
    }

    // ── Define arrowhead marker for blue conv path ─────────────────
    const defs = document.createElementNS("http://www.w3.org/2000/svg", "defs");
    const marker = document.createElementNS("http://www.w3.org/2000/svg", "marker");
    marker.setAttribute("id", "gt-arrow");
    marker.setAttribute("viewBox", "0 0 10 10");
    marker.setAttribute("refX", "8");
    marker.setAttribute("refY", "5");
    marker.setAttribute("markerWidth", "3");
    marker.setAttribute("markerHeight", "3");
    marker.setAttribute("orient", "auto-start-reverse");
    const arrowPath = document.createElementNS("http://www.w3.org/2000/svg", "path");
    arrowPath.setAttribute("d", "M 0 1 L 8 5 L 0 9 Z");
    arrowPath.setAttribute("fill", GT_COLOR_CONV_PATH);
    marker.appendChild(arrowPath);
    defs.appendChild(marker);
    svg.appendChild(defs);

    // ── Layer 1: parent edges (grey) ──────────────────────────────
    for (const e of edges) {
        if (e.edge_type !== "parent") continue;
        const from = posMap[e.from_sha], to = posMap[e.to_sha];
        if (!from || !to) continue;
        _drawLine(svg, from, to, GT_COLOR_PARENT_EDGE, 1.5, "");
    }

    // ── Layer 2: conversation path (blue, thick, with arrows) ─────
    for (const e of edges) {
        if (e.edge_type !== "conv_path" && e.edge_type !== "conv_jump") continue;
        const from = posMap[e.from_sha], to = posMap[e.to_sha];
        if (!from || !to) continue;
        let el;
        if (e.edge_type === "conv_jump") {
            el = _drawArc(svg, from, to, GT_COLOR_CONV_PATH, 5, "");
        } else {
            el = _drawLine(svg, from, to, GT_COLOR_CONV_PATH, 5, "");
        }
        el.setAttribute("marker-end", "url(#gt-arrow)");
    }

    // ── Layer 3: post-compaction (yellow, thinner, on top of blue) ─
    for (const e of edges) {
        if (e.edge_type !== "post_compaction") continue;
        const from = posMap[e.from_sha], to = posMap[e.to_sha];
        if (!from || !to) continue;
        const isJump = edges.some(j =>
            j.edge_type === "conv_jump" &&
            j.from_sha === e.from_sha && j.to_sha === e.to_sha);
        if (isJump) {
            _drawArc(svg, from, to, GT_COLOR_POST_COMPACTION, 2, "");
        } else {
            _drawLine(svg, from, to, GT_COLOR_POST_COMPACTION, 2, "");
        }
    }

    // ── Draw nodes ────────────────────────────────────────────────
    for (const n of nodes) {
        const pos = posMap[n.sha];
        if (!pos) continue;

        // Session root / compaction marker: yellow ring
        if (n.is_compaction_marker || n.is_session_root) {
            _drawCircle(svg, pos.px, pos.py, GT_NODE_RADIUS + 5,
                "none", GT_COLOR_COMPACTION_RING, 2.5);
        }

        // Agent position: white ring
        if (n.is_agent_position) {
            _drawCircle(svg, pos.px, pos.py, GT_NODE_RADIUS + 8,
                "none", GT_COLOR_AGENT_RING, 2.5);
        }

        // Selection highlight: pink ring
        if (n.sha === gtSelectedNode) {
            _drawCircle(svg, pos.px, pos.py, GT_NODE_RADIUS + 11,
                "none", GT_COLOR_SELECTION, 2);
        }

        // Node circle
        let fill, stroke, strokeW, dash;
        if (n.node_type === "alan_commit") {
            fill = GT_COLOR_ALAN; stroke = GT_COLOR_ALAN; strokeW = 0; dash = "";
        } else if (n.node_type === "current") {
            fill = "none"; stroke = GT_COLOR_CURRENT; strokeW = 2; dash = "4,3";
        } else {
            fill = GT_COLOR_EXTERNAL; stroke = GT_COLOR_EXTERNAL; strokeW = 0; dash = "";
        }
        const circle = _drawCircle(svg, pos.px, pos.py, GT_NODE_RADIUS,
            fill, stroke, strokeW, dash);

        circle.style.cursor = "pointer";
        circle.addEventListener("click", () => _selectNode(n.sha));
        circle.addEventListener("mouseenter", (ev) => _showTooltip(ev, n));
        circle.addEventListener("mouseleave", _hideTooltip);

        // Commit message label (truncated)
        const label = _trunc(n.message || "???");
        _drawText(svg, pos.px, pos.py + GT_NODE_RADIUS + 16,
            label, GT_COLOR_TEXT, 10, "middle");

        // Branch tags
        if (n.branches && n.branches.length > 0) {
            _drawTag(svg, pos.px + GT_NODE_RADIUS + 10, pos.py - 4,
                n.branches.join(", "), GT_COLOR_BRANCH_TAG);
        }
    }

    // Scroll to show agent position (or bottom)
    const container = document.getElementById("git-tree-container");
    if (container) {
        // Find agent node pixel position
        const agentNode = nodes.find(n => n.is_agent_position);
        if (agentNode && posMap[agentNode.sha]) {
            const agentPy = posMap[agentNode.sha].py;
            const containerH = container.clientHeight;
            container.scrollTop = Math.max(0, agentPy - containerH / 2);
        } else {
            container.scrollTop = container.scrollHeight;
        }
    }

    // Re-enable buttons (slash commands like /convrevert don't trigger
    // setAgentRunning, so we must refresh button state after every render)
    _updateGitTreeButtons();
}

// ── Helpers ───────────────────────────────────────────────────────

function _trunc(text) {
    if (!text) return "";
    return text.length <= GT_MAX_LABEL_LEN ? text : text.slice(0, GT_MAX_LABEL_LEN - 1) + "\u2026";
}

function _drawLine(svg, from, to, color, width, dash) {
    const el = document.createElementNS("http://www.w3.org/2000/svg", "line");
    el.setAttribute("x1", from.px); el.setAttribute("y1", from.py);
    el.setAttribute("x2", to.px);   el.setAttribute("y2", to.py);
    el.setAttribute("stroke", color);
    el.setAttribute("stroke-width", width);
    el.setAttribute("stroke-linecap", "round");
    if (dash) el.setAttribute("stroke-dasharray", dash);
    svg.appendChild(el);
    return el;
}

function _drawArc(svg, from, to, color, width, dash) {
    const dx = to.px - from.px, dy = to.py - from.py;
    const dist = Math.sqrt(dx * dx + dy * dy);
    // Radii invert curvature: bigger radius = flatter arc.
    // gtCurvature ∈ [0.2 .. 2.0]; 1.0 is the current default.
    // Higher slider value = more curved → smaller effective radius.
    const scale = 1.0 / Math.max(gtCurvature, 0.05);
    const rx = Math.max(dist * 1.5 * scale, 80 * scale);
    const ry = Math.max(dist * 1.0 * scale, 50 * scale);
    const el = document.createElementNS("http://www.w3.org/2000/svg", "path");
    el.setAttribute("d", `M ${from.px} ${from.py} A ${rx} ${ry} 0 0 1 ${to.px} ${to.py}`);
    el.setAttribute("stroke", color);
    el.setAttribute("stroke-width", width);
    el.setAttribute("stroke-linecap", "round");
    el.setAttribute("fill", "none");
    if (dash) el.setAttribute("stroke-dasharray", dash);
    svg.appendChild(el);
    return el;
}

function _drawCircle(svg, cx, cy, r, fill, stroke, strokeW, dash) {
    const el = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    el.setAttribute("cx", cx); el.setAttribute("cy", cy); el.setAttribute("r", r);
    el.setAttribute("fill", fill || "none");
    if (stroke) el.setAttribute("stroke", stroke);
    if (strokeW) el.setAttribute("stroke-width", strokeW);
    if (dash) el.setAttribute("stroke-dasharray", dash);
    svg.appendChild(el);
    return el;
}

function _drawText(svg, x, y, text, color, size, anchor) {
    const el = document.createElementNS("http://www.w3.org/2000/svg", "text");
    el.setAttribute("x", x); el.setAttribute("y", y);
    el.setAttribute("fill", color);
    el.setAttribute("font-size", size);
    el.setAttribute("text-anchor", anchor || "middle");
    el.setAttribute("font-family", "monospace");
    el.textContent = text;
    svg.appendChild(el);
    return el;
}

function _drawTag(svg, x, y, text, color) {
    const el = _drawText(svg, x, y, text, color, 10, "start");
    el.setAttribute("font-weight", "bold");
    return el;
}

// ── Interaction ───────────────────────────────────────────────────

function _selectNode(sha) {
    gtSelectedNode = (gtSelectedNode === sha) ? null : sha;
    _updateGitTreeButtons();
    if (gtLastData) renderGitTree(gtLastData);
}

function _updateGitTreeButtons() {
    const hasSelection = !!gtSelectedNode;
    const btns = ["btn-move-here", "btn-revert-to", "btn-conv-revert-to", "btn-all-revert-to"];
    for (const id of btns) {
        const el = document.getElementById(id);
        if (el) el.disabled = !hasSelection || (typeof isAgentRunning !== "undefined" && isAgentRunning);
    }
}

function _showTooltip(event, node) {
    if (!gtTooltipEl) {
        gtTooltipEl = document.createElement("div");
        gtTooltipEl.className = "git-tree-tooltip";
        document.body.appendChild(gtTooltipEl);
    }
    const p = [];
    p.push(`[${node.short_sha}] ${node.message}`);
    if (node.author) p.push(`Author: ${node.author}`);
    if (node.timestamp) p.push(`Date: ${node.timestamp.slice(0, 19)}`);
    if (node.branches && node.branches.length) p.push(`Branch: ${node.branches.join(", ")}`);
    if (node.is_agent_position) p.push("[Agent is here]");
    if (node.is_compaction_marker) p.push("[Compaction point]");
    if (node.is_session_root) p.push("[Session start]");
    if (node.node_type === "alan_commit") p.push("[Alan commit]");
    if (node.node_type === "current") p.push("[Uncommitted changes]");

    gtTooltipEl.textContent = p.join("\n");
    gtTooltipEl.style.display = "block";
    gtTooltipEl.style.left = (event.pageX + 14) + "px";
    gtTooltipEl.style.top = (event.pageY - 10) + "px";
}

function _hideTooltip() {
    if (gtTooltipEl) gtTooltipEl.style.display = "none";
}

// ── Slider wiring ─────────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", () => {
    const slider = document.getElementById("gt-curvature");
    if (!slider) return;
    slider.addEventListener("input", () => {
        gtCurvature = parseFloat(slider.value);
        if (gtLastData) renderGitTree(gtLastData);
    });
});
