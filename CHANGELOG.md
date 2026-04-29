# Changelog

- **2026-04-28** — Restructured system prompt for caching. `session_guidance`, `scratchpad`, and `environment` are now in the cacheable static block. Removed `git status` from the system prompt (it changed every turn and killed the cache).
- **2026-04-28** — Added `(estimate w/o cache)` indicator on the cost line when no cache tokens were reported by the provider, signalling possible cost overestimation.
- **2026-04-27** — Added `extra_tools` and `custom_system_prompt` parameters in the `AlanCodeAgent` constructor.
- **2026-04-18** — Cache token display improvements on the Anthropic provider.
- **2026-04-18** — LiteLLM caching: switched to `cache_control_injection_points` for cleaner, provider-agnostic caching.
- **2026-04-17** — Added prompt caching for both Anthropic and LiteLLM providers (up to 4 `cache_control` breakpoints per call).
- **2026-04-17** — Improved user-message rendering in the CLI: user input is now grey, with a blank line before the assistant response.
- **2026-04-17** — `litellm` is now the default provider (was `anthropic`). Default model is now `anthropic/claude-sonnet-4-6`.
- **2026-04-17** — Removed `force_supports_*` settings. Alan now assumes capabilities are supported and surfaces the real provider error when they aren't.
- **2026-04-17** — GUI chat panel now correctly displays assistant text on session resume and synthetic error messages (both were previously silently dropped).
- **2026-04-16** — Alan Code v1.0.0 — initial public release.
