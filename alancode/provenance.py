"""Where the running alancode actually came from.

An editable install keeps the dist-info written when it was first installed, so
``importlib.metadata.version("alancode")`` can report a version the code has not
been for months, while the import resolves to a live source tree. Anything
recording what produced a run - a benchmark harness, a bug report - needs the
source, not the metadata.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from alancode.__version__ import __version__

logger = logging.getLogger(__name__)

# A run should never stall on provenance; git here is a convenience, not a
# dependency.
GIT_TIMEOUT_S = 5


def _git(repo: Path, *args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True, text=True, timeout=GIT_TIMEOUT_S, check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("git %s failed: %s", args, exc)
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def get_provenance() -> dict[str, object]:
    """Describe the running alancode.

    Returns a dict with ``version`` and ``path`` always present, plus
    ``git_sha``, ``git_dirty`` and ``git_branch`` when the package is imported
    from a git checkout (``None`` when installed from a wheel, or when git is
    unavailable).

    ``git_dirty`` is the field that matters for reproducibility: a true value
    means the run used uncommitted edits and its ``git_sha`` alone does not
    identify the code.
    """
    package = Path(__file__).resolve().parent
    info: dict[str, object] = {
        "version": __version__,
        "path": str(package),
        "git_sha": None,
        "git_dirty": None,
        "git_branch": None,
    }

    if _git(package, "rev-parse", "--is-inside-work-tree") != "true":
        return info

    info["git_sha"] = _git(package, "rev-parse", "--short=12", "HEAD")
    branch = _git(package, "rev-parse", "--abbrev-ref", "HEAD")
    info["git_branch"] = None if branch == "HEAD" else branch

    status = _git(package, "status", "--porcelain")
    if status is not None:
        info["git_dirty"] = bool(status)

    return info


def provenance_string() -> str:
    """One-line form for logs and banners, e.g. ``1.3.14 (a2b0aff12345, dirty)``."""
    p = get_provenance()
    if not p["git_sha"]:
        return str(p["version"])
    suffix = ", dirty" if p["git_dirty"] else ""
    return f"{p['version']} ({p['git_sha']}{suffix})"
