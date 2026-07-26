"""Site-specific configuration for each supported web assistant.

Adding a new assistant means adding a ``WebAssistant`` entry here (and, if the
site needs behavior the driver doesn't cover, subclassing it). The ``name``
doubles as the model string: ``--backend web --model chatgpt``.

Only ``chatgpt`` is verified end-to-end so far. The others carry their URL and
app-window class (both reliable - the class is the URL host for a Chrome
``--app`` window) plus best-effort keyboard shortcuts. ``focus_composer_keys``
and ``new_chat_keys`` differ per site and are UNVERIFIED for the non-ChatGPT
entries; confirm them (open the site as an app window and try the shortcut)
before relying on that assistant. ``verified`` records this state.
"""

from __future__ import annotations

from dataclasses import dataclass

# Default shortcuts, used until a site's real ones are confirmed. Escape then
# focus is a common pattern; many composers also auto-focus after a new chat.
DEFAULT_FOCUS_KEYS = "shift+Escape"
DEFAULT_NEW_CHAT_KEYS = "ctrl+shift+o"


@dataclass
class WebAssistant:
    """Everything the web driver needs to know about one assistant site.

    Attributes:
        name: Registry key, used as the ``model`` string.
        url: Page opened as a dedicated browser app window. App mode matters:
            it gives the window a URL-derived WM_CLASS that never changes,
            unlike titles, which follow the conversation name.
        window_classname: WM_CLASS fragment identifying that app window (the
            URL host, for a Chrome ``--app`` window).
        focus_composer_keys: xdotool key spec that focuses the message
            composer from anywhere in the page.
        new_chat_keys: xdotool key spec that opens a fresh conversation.
        page_load_wait_s: Delay after launching a fresh window before the
            composer is assumed ready.
        verified: True once the whole send/receive flow has been confirmed
            live against this site.
    """

    name: str
    url: str
    window_classname: str
    focus_composer_keys: str = DEFAULT_FOCUS_KEYS
    new_chat_keys: str = DEFAULT_NEW_CHAT_KEYS
    page_load_wait_s: float = 4.0
    verified: bool = False
    # Fixed UI strings that mark the start of trailing page chrome (footer,
    # composer placeholder, side panels). The tag-free reply fallback truncates
    # at the first of these so it cannot return interface text. Extend per site.
    reply_stop_markers: tuple[str, ...] = ()
    # CDP transport selectors (preferred over xdotool). cdp_composer_css is
    # focused and typed into; cdp_message_css matches assistant message nodes
    # (last = latest reply); cdp_new_chat_url is navigated to for a fresh chat.
    # Discover per site by inspecting the live DOM; empty = not yet configured.
    cdp_composer_css: str = ""
    cdp_message_css: str = ""
    cdp_new_chat_url: str = ""
    # How to submit: "" / "enter" presses Enter; anything else is a CSS selector
    # for a send button to click (some composers, e.g. Lexical, ignore Enter).
    cdp_submit: str = "enter"


_ALL = (
    WebAssistant(
        name="chatgpt",
        url="https://chatgpt.com",
        window_classname="chatgpt.com",
        focus_composer_keys="shift+Escape",
        new_chat_keys="ctrl+shift+o",
        verified=True,
        reply_stop_markers=(
            "ChatGPT can make mistakes",
            "Ask anything",
            "Set the style and tone",
            "Choose additional customizations",
            "Reference record history",
        ),
        cdp_composer_css="#prompt-textarea",
        cdp_message_css='[data-message-author-role="assistant"]',
        cdp_new_chat_url="https://chatgpt.com/",
    ),
    # --- Unverified: URL + window class are reliable; shortcuts need confirming.
    WebAssistant("deepseek", "https://chat.deepseek.com/", "chat.deepseek.com"),
    WebAssistant(
        "gemini", "https://gemini.google.com/", "gemini.google.com",
    ),
    WebAssistant(
        "claude", "https://claude.ai/new", "claude.ai__new",
        cdp_composer_css='[contenteditable="true"]',
        cdp_message_css="[data-is-streaming]",
        cdp_new_chat_url="https://claude.ai/new",
    ),
    WebAssistant("grok", "https://grok.com/", "grok.com"),
    WebAssistant("qwen", "https://chat.qwen.ai/", "chat.qwen.ai"),
    WebAssistant(
        "kimi", "https://www.kimi.com/", "www.kimi.com",
        cdp_composer_css='[contenteditable="true"]',
        cdp_message_css='[class*="markdown"]',
        cdp_new_chat_url="https://www.kimi.com/",
        cdp_submit=".send-button-container",
    ),
    WebAssistant(
        "copilot", "https://copilot.microsoft.com/", "copilot.microsoft.com",
        cdp_composer_css="#userInput",
        cdp_message_css='[data-testid*="message"]',
        cdp_new_chat_url="https://copilot.microsoft.com/",
    ),
    WebAssistant("perplexity", "https://www.perplexity.ai/", "www.perplexity.ai"),
    WebAssistant("mistral", "https://chat.mistral.ai/work", "chat.mistral.ai__work"),
)

ASSISTANTS: dict[str, WebAssistant] = {a.name: a for a in _ALL}
