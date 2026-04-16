"""Memory snapshot management for AGT navigation.

Snapshots are stored in ``.alan/memory_snapshots/<commit_sha>/``
as full copies of the ``.alan/memory/`` directory.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def take_memory_snapshot(cwd: str, commit_sha: str) -> Path | None:
    """Copy ``.alan/memory/`` → ``.alan/memory_snapshots/<sha>/``.

    Returns the snapshot directory, or None if no memory dir exists.
    """
    src = Path(cwd) / ".alan" / "memory"
    if not src.exists():
        return None

    dst = Path(cwd) / ".alan" / "memory_snapshots" / commit_sha
    try:
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        logger.debug("Memory snapshot taken: %s", commit_sha[:7])
        return dst
    except OSError as e:
        logger.warning("Failed to take memory snapshot: %s", e)
        return None


def restore_memory_snapshot(cwd: str, target_sha: str) -> bool:
    """Restore memory from ``.alan/memory_snapshots/<sha>/``.

    If no snapshot exists for *target_sha*, walks git ancestors to find
    the nearest one.  Returns True if restored, False if no snapshot found.
    """
    dst = Path(cwd) / ".alan" / "memory"
    snap_dir = _find_snapshot(cwd, target_sha)
    if not snap_dir:
        logger.debug("No memory snapshot found for %s or ancestors", target_sha[:7])
        return False

    try:
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(snap_dir, dst)
        logger.debug("Memory restored from snapshot: %s", snap_dir.name[:7])
        return True
    except OSError as e:
        logger.warning("Failed to restore memory snapshot: %s", e)
        return False


def get_memory_diff(cwd: str, sha1: str, sha2: str) -> str:
    """Compute a text diff between two memory snapshots.

    Returns a human-readable diff, or empty string if no difference
    or snapshots are missing.
    """
    snap1 = _snapshot_path(cwd, sha1)
    snap2 = _snapshot_path(cwd, sha2)

    if not snap1.exists() and not snap2.exists():
        return ""

    lines: list[str] = []

    # Collect all files from both snapshots
    files1 = _list_files(snap1) if snap1.exists() else {}
    files2 = _list_files(snap2) if snap2.exists() else {}
    all_files = sorted(set(files1.keys()) | set(files2.keys()))

    for rel_path in all_files:
        content1 = files1.get(rel_path, "")
        content2 = files2.get(rel_path, "")
        if content1 == content2:
            continue
        if not content1:
            lines.append(f"+ {rel_path} (new file)")
        elif not content2:
            lines.append(f"- {rel_path} (deleted)")
        else:
            lines.append(f"~ {rel_path} (modified)")

    return "\n".join(lines)


def _snapshot_path(cwd: str, sha: str) -> Path:
    return Path(cwd) / ".alan" / "memory_snapshots" / sha


def _find_snapshot(cwd: str, sha: str, max_depth: int = 20) -> Path | None:
    """Find the nearest memory snapshot by walking ancestors."""
    current = sha
    for _ in range(max_depth):
        snap = _snapshot_path(cwd, current)
        if snap.exists():
            return snap
        # Walk to parent
        try:
            result = subprocess.run(
                ["git", "rev-parse", f"{current}~1"],
                cwd=cwd, capture_output=True, text=True, timeout=5,
                env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
            )
            if result.returncode != 0:
                break
            current = result.stdout.strip()
        except (subprocess.SubprocessError, FileNotFoundError):
            break
    return None


def _list_files(directory: Path) -> dict[str, str]:
    """List all files in a directory tree with their contents."""
    result: dict[str, str] = {}
    if not directory.exists():
        return result
    for root, _dirs, files in os.walk(directory):
        for f in files:
            full = Path(root) / f
            rel = str(full.relative_to(directory))
            try:
                result[rel] = full.read_text(errors="replace")
            except OSError:
                result[rel] = ""
    return result
