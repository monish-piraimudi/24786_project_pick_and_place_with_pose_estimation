"""Launch the default policy-evaluation command from an Emio Labs python button."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
PROJECT_DIR_STR = str(PROJECT_DIR)
if PROJECT_DIR_STR not in sys.path:
    sys.path.insert(0, PROJECT_DIR_STR)

from evaluate_il_policy import main


if __name__ == "__main__":
    sys.argv = [
        "evaluate_il_policy.py",
        "--mode",
        "policy",
        "--policy-path",
        "data/results/il_pick_place/bc_policy.pth",
        "--workspace-bounds-mm",
        "-35",
        "10",
        "-30",
        "20",
    ]
    main()
