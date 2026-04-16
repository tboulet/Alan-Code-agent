"""Environment detection utilities."""

import os
import platform
import shutil
import subprocess


def get_platform() -> str:
    """Return the current platform identifier: 'linux', 'darwin', or 'win32'."""
    system = platform.system().lower()
    if system == "linux":
        return "linux"
    elif system == "darwin":
        return "darwin"
    elif system == "windows":
        return "win32"
    return system


def get_shell() -> str:
    """Return the current shell name: 'bash', 'zsh', 'fish', or 'unknown'."""
    shell = os.environ.get("SHELL", "")
    if shell:
        basename = os.path.basename(shell)
        if basename in ("bash", "zsh", "fish", "sh", "dash", "ksh", "tcsh", "csh"):
            return basename
    # Fallback: check if common shells are available
    for candidate in ("bash", "zsh"):
        if shutil.which(candidate):
            return candidate
    return "unknown"


def get_os_version() -> str:
    """Return a human-readable OS version string, e.g. 'Linux 6.6.4' or 'Darwin 23.1.0'."""
    system = platform.system()
    release = platform.release()
    return f"{system} {release}"


def is_git_repo(cwd: str | None = None) -> bool:
    """Check whether the given directory (or the current directory) is inside a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=cwd or os.getcwd(),
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0 and result.stdout.strip() == "true"
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


def get_cwd() -> str:
    """Return the current working directory."""
    return os.getcwd()


MAX_GIT_STATUS_CHARS = 2000


def get_git_status(cwd: str | None = None) -> str | None:
    """Get a git status snapshot for the system prompt.

    Returns a formatted string with branch, status, recent commits, and
    git user — or None if not in a git repo.  Mirrors CC's approach:
    computed once at session start, included in the system prompt.
    """
    effective_cwd = cwd or os.getcwd()
    if not is_git_repo(effective_cwd):
        return None

    def _run(args: list[str]) -> str:
        try:
            r = subprocess.run(
                ["git"] + args,
                cwd=effective_cwd,
                capture_output=True,
                text=True,
                timeout=5,
            )
            return r.stdout.strip() if r.returncode == 0 else ""
        except (subprocess.SubprocessError, FileNotFoundError):
            return ""

    branch = _run(["rev-parse", "--abbrev-ref", "HEAD"])
    main_branch = _run(["rev-parse", "--verify", "--quiet", "main"]) and "main"
    if not main_branch:
        main_branch = _run(["rev-parse", "--verify", "--quiet", "master"]) and "master"
    if not main_branch:
        main_branch = branch  # fallback
    status = _run(["--no-optional-locks", "status", "--short"])
    log = _run(["--no-optional-locks", "log", "--oneline", "-n", "5"])
    user_name = _run(["config", "user.name"])

    # Truncate long status output
    if len(status) > MAX_GIT_STATUS_CHARS:
        status = (
            status[:MAX_GIT_STATUS_CHARS]
            + "\n... (truncated, run `git status` for full output)"
        )

    parts = [
        "This is the git status at the start of the conversation. "
        "Note that this status is a snapshot in time, and will not update "
        "during the conversation.",
        f"Current branch: {branch}",
        f"Main branch (you will usually use this for PRs): {main_branch}",
    ]
    if user_name:
        parts.append(f"Git user: {user_name}")
    parts.append(f"Status:\n{status or '(clean)'}")
    parts.append(f"Recent commits:\n{log or '(no commits)'}")

    return "\n\n".join(parts)
