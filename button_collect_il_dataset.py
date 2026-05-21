"""Launch the default dataset-collection command from an Emio Labs python button."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
PROJECT_DIR_STR = str(PROJECT_DIR)
if PROJECT_DIR_STR not in sys.path:
    sys.path.insert(0, PROJECT_DIR_STR)

from collect_il_dataset import main


if __name__ == "__main__":
    sys.argv = [
        "collect_il_dataset.py",
        "--episodes",
        "100",
        "--max-attempts",
        "140",
        "--workspace-bounds-mm",
        "-35",
        "10",
        "-30",
        "20",
        "--save-failed-episodes",
    ]
    main()
