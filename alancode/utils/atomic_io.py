"""Atomic file writes.

Every place that writes JSON state (session, settings, transcripts) needs
to survive a crash or a concurrent write without corrupting the file.
The pattern is: write to a tempfile in the same directory, fsync, then
``os.replace`` onto the final path. ``os.replace`` is atomic on POSIX
and Windows.
"""

from __future__ import annotations

import json
import fcntl
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any


@contextmanager
def interprocess_lock(target: str | Path):
    """Serialize read-modify-write cycles associated with *target*.

    The lock lives beside the target and is released automatically by the
    kernel if a process exits. The small lock file itself is intentionally
    persistent so concurrent processes always rendezvous on the same inode.
    """
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    lock_path = target.with_name(f".{target.name}.lock")
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def atomic_write_text(path: str | Path, text: str) -> None:
    """Write *text* to *path* atomically.

    Creates a tempfile in the same directory (so ``os.replace`` stays a
    rename, not a cross-filesystem copy), writes, fsyncs, then replaces.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Tempfile must be in the same directory for os.replace to be atomic.
    fd, tmp_path = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        # Any failure — clean up the tempfile instead of leaving stragglers.
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def atomic_write_json(path: str | Path, data: Any, *, indent: int | None = 2) -> None:
    """Write *data* as JSON to *path* atomically."""
    if indent is None:
        text = json.dumps(data, default=str, separators=(",", ":"))
    else:
        text = json.dumps(data, default=str, indent=indent)
        text += "\n"
    atomic_write_text(path, text)
