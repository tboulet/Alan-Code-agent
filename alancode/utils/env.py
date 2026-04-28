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


