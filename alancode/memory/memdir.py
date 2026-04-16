"""Memory directory management — ALAN.md, MEMORY.md, scratchpad."""

import logging
import os
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

MEMORY_MD = "MEMORY.md"
ALAN_MD = "ALAN.md"
MAX_ENTRYPOINT_LINES = 200
MAX_ENTRYPOINT_BYTES = 25_000

# Subdirectories inside .alan/memory/ (project-scoped)
PROJECT_MEMORY_SUBDIRS = ("project", "reference", "workflow")
# Subdirectories inside ~/.alan/memory/ (global, shared across projects)
GLOBAL_MEMORY_SUBDIRS = ("user", "feedback")
# All subdirectories (for backward compat)
MEMORY_SUBDIRS = PROJECT_MEMORY_SUBDIRS + GLOBAL_MEMORY_SUBDIRS


ALAN_MD_TEMPLATE = """\
# Project Instructions

<!-- This file is read by Alan Code at the start of every session. -->
<!-- Use it to give Alan context about your project, preferences, and conventions. -->
<!-- For long term and/or autonomous project memory, consider using the memory option instead. -->

## Project overview

<!-- Describe your project here. What does it do? What technologies does it use? -->

## Conventions

<!-- List coding conventions, naming patterns, or style preferences. -->
<!-- Example: "Use Google-style docstrings", "Prefer pathlib over os.path" -->

## Important files

<!-- Point Alan to key files or directories it should know about. -->
"""


def ensure_project_instructions(cwd: str) -> str:
    """Ensure ALAN.md exists in the project root. Creates it with a starter
    template if missing.

    Returns the absolute path to the file.
    """
    path = Path(cwd).resolve() / ALAN_MD
    if not path.exists():
        try:
            path.write_text(ALAN_MD_TEMPLATE, encoding="utf-8")
            logger.info("Created %s with starter template", path)
        except OSError as exc:
            logger.warning("Failed to create %s: %s", path, exc)
    return str(path)


def find_project_instructions(cwd: str) -> str | None:
    """Find ALAN.md in *cwd*.

    Returns the absolute path to ``ALAN.md`` if it exists in the working
    directory, otherwise ``None``.
    """
    candidate = Path(cwd).resolve() / ALAN_MD
    if candidate.is_file():
        return str(candidate)
    return None


def load_project_instructions(cwd: str) -> str | None:
    """Load ALAN.md content if it exists.

    Searches upward from *cwd* using :func:`find_project_instructions`,
    reads the file, and truncates it to safe limits.  Returns ``None`` if
    no ``ALAN.md`` is found.
    """
    path = find_project_instructions(cwd)
    if path is None:
        return None
    try:
        content = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        logger.warning("Failed to read %s: %s", path, exc)
        return None

    content = truncate_content(content)
    return (
        f"# Project instructions ({ALAN_MD})\n\n"
        f"The following is loaded from {path}:\n\n"
        f"{content}"
    )


def load_global_project_instructions() -> str | None:
    """Load global ALAN.md from ``~/.alan/ALAN.md``.

    Returns formatted section or None if no global instructions exist.
    Global instructions provide cross-project user preferences.
    """
    path = Path.home() / ".alan" / ALAN_MD
    if not path.is_file():
        return None

    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        logger.warning("Failed to read global %s: %s", path, exc)
        return None

    if not content.strip():
        return None

    content = truncate_content(content)
    return f"# Global user instructions (~/.alan/{ALAN_MD})\n\n" f"{content}"


def truncate_content(
    content: str,
    *,
    max_lines: int = MAX_ENTRYPOINT_LINES,
    max_bytes: int = MAX_ENTRYPOINT_BYTES,
) -> str:
    """Truncate *content* to line and byte limits.

    Line limiting is applied first, then byte limiting.  If truncation
    occurs, a trailing note is appended.
    """
    lines = content.splitlines(keepends=True)
    truncated = False

    if len(lines) > max_lines:
        lines = lines[:max_lines]
        truncated = True

    result = "".join(lines)

    encoded = result.encode("utf-8")
    if len(encoded) > max_bytes:
        # Decode back safely to avoid splitting a multi-byte character
        result = encoded[:max_bytes].decode("utf-8", errors="ignore")
        truncated = True

    if truncated:
        result += "\n\n[... truncated]"

    return result


# ── Memory directory ──────────────────────────────────────────────────────────


def get_memory_dir(cwd: str | None = None) -> Path:
    """Get the project memory directory path (``.alan/memory/``)."""
    effective_cwd = cwd or os.getcwd()
    return Path(effective_cwd) / ".alan" / "memory"


def get_global_memory_dir() -> Path:
    """Get the global memory directory path (``~/.alan/memory/``)."""
    return Path.home() / ".alan" / "memory"


def ensure_memory_structure(cwd: str) -> Path:
    """Create both project and global memory directory trees.

    Project memory (``.alan/memory/``): project, reference, workflow subdirs.
    Global memory (``~/.alan/memory/``): user, feedback subdirs.

    Returns the project memory root.
    """
    # Project memory
    mem_dir = get_memory_dir(cwd)
    mem_dir.mkdir(parents=True, exist_ok=True)
    for subdir in PROJECT_MEMORY_SUBDIRS:
        (mem_dir / subdir).mkdir(exist_ok=True)

    # Global memory
    global_dir = get_global_memory_dir()
    global_dir.mkdir(parents=True, exist_ok=True)
    for subdir in GLOBAL_MEMORY_SUBDIRS:
        (global_dir / subdir).mkdir(exist_ok=True)

    return mem_dir


# ── Scratchpad ────────────────────────────────────────────────────────────────


def get_scratchpad_dir(cwd: str, session_id: str) -> Path:
    """Get the scratchpad directory for a specific session.

    Returns ``.alan/sessions/<session_id>/scratchpad``.
    """
    return Path(cwd) / ".alan" / "sessions" / session_id / "scratchpad"


def cleanup_old_scratchpads(cwd: str, max_sessions: int = 5) -> None:
    """Remove oldest scratchpad directories if count exceeds *max_sessions*.

    Scans ``.alan/sessions/*/scratchpad/`` for session scratchpads.
    Directories are sorted by modification time; the oldest are removed first.
    Also cleans up the legacy ``.alan/scratchpad/`` directory.
    """
    # New layout: .alan/sessions/*/scratchpad/
    sessions_root = Path(cwd) / ".alan" / "sessions"
    if sessions_root.is_dir():
        scratch_dirs = []
        for session_dir in sessions_root.iterdir():
            if not session_dir.is_dir():
                continue
            scratch = session_dir / "scratchpad"
            if scratch.is_dir():
                scratch_dirs.append(scratch)

        if len(scratch_dirs) > max_sessions:
            scratch_dirs.sort(key=lambda d: d.stat().st_mtime)
            to_remove = scratch_dirs[: len(scratch_dirs) - max_sessions]
            for d in to_remove:
                # Safety: only delete paths within the sessions directory
                if not str(d.resolve()).startswith(str(sessions_root.resolve())):
                    logger.warning("Refusing to delete path outside sessions: %s", d)
                    continue
                try:
                    shutil.rmtree(d)
                    logger.debug("Removed old scratchpad: %s", d)
                except OSError as exc:
                    logger.warning("Failed to remove scratchpad %s: %s", d, exc)


# ── Memory index loading ─────────────────────────────────────────────────────


def load_memory_index(
    memory_path: str | None = None,
    cwd: str | None = None,
) -> str | None:
    """Load the MEMORY.md index as a formatted section.

    Looks for ``MEMORY.md`` inside the project memory directory
    (``.alan/memory/``) or at the explicitly provided *memory_path*.
    Truncates to :data:`MAX_ENTRYPOINT_LINES` / :data:`MAX_ENTRYPOINT_BYTES`.

    Returns ``"## Your memory index ({ENTRYPOINT_NAME})\\n\\n{content}"``
    or ``None`` if no memory file exists or it is empty.
    """
    if memory_path is not None:
        target = Path(memory_path)
    else:
        mem_dir = get_memory_dir(cwd)
        target = mem_dir / MEMORY_MD

    if not target.is_file():
        return None

    try:
        content = target.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        logger.warning("Failed to read memory file %s: %s", target, exc)
        return None

    if not content.strip():
        return None

    content = truncate_content(content)
    return f"## Your project memory index ({MEMORY_MD})\n\n{content}"


def load_global_memory_index() -> str | None:
    """Load the global MEMORY.md index from ``~/.alan/memory/``.

    Returns formatted section or None if no global memory exists.
    """
    target = get_global_memory_dir() / MEMORY_MD

    if not target.is_file():
        return None

    try:
        content = target.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        logger.warning("Failed to read global memory file %s: %s", target, exc)
        return None

    if not content.strip():
        return None

    content = truncate_content(content)
    return f"## Your global memory index (~/.alan/{MEMORY_MD})\n\n{content}"
