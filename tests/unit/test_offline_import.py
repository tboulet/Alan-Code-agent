"""Package import must be deterministic on offline machines."""

import os
from pathlib import Path
import subprocess
import sys


def test_import_sets_litellm_offline_default():
    env = os.environ.copy()
    env.pop("LITELLM_LOCAL_MODEL_COST_MAP", None)
    repo_root = Path(__file__).resolve().parents[2]

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import os; import alancode; "
                "assert os.environ['LITELLM_LOCAL_MODEL_COST_MAP'] == 'True'"
            ),
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
