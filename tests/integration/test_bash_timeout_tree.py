import asyncio, os, subprocess, time
import pytest
from alancode.tools.builtin.bash import BashTool
from alancode.tools.base import ToolUseContext

MARKER = "alancode_tree_kill_test"


def _ancestors() -> set[int]:
    out, pid = set(), os.getpid()
    while pid and pid != 1:
        out.add(pid)
        try:
            pid = int(open(f"/proc/{pid}/stat").read().split()[3])
        except OSError:
            break
    return out


@pytest.mark.asyncio
async def test_timeout_kills_the_whole_process_tree_promptly(tmp_path):
    """A backgrounded child outlives a SIGKILL sent only to the shell, and
    keeps the inherited stdout pipe open, so reaping blocks on it. Measured
    on bench-04: a 300s timeout reported 87 minutes late while the runaway
    child grew to 240 GB.
    """
    mine = _ancestors()
    tool = BashTool()
    ctx = ToolUseContext(cwd=str(tmp_path), messages=[])
    command = f"python3 -c 'import time; time.sleep(120)  # {MARKER}' & sleep 120"

    started = time.monotonic()
    result = await tool.call({"command": command, "timeout": 2000}, ctx)
    elapsed = time.monotonic() - started

    assert result.is_error
    assert "timed out" in str(result.data)
    # The report must not wait on the surviving child.
    assert elapsed < 10, f"timeout took {elapsed:.1f}s to report"

    await asyncio.sleep(1.0)
    found = subprocess.run(
        ["pgrep", "-f", MARKER], capture_output=True, text=True
    ).stdout.split()
    survivors = []
    for pid in (int(p) for p in found):
        if pid in mine:   # this test's own wrapper carries MARKER in its cmdline
            continue
        try:
            if open(f"/proc/{pid}/stat").read().split()[2] != "Z":
                survivors.append(pid)
        except OSError:
            pass
    assert not survivors, f"process tree survived the timeout: {survivors}"
