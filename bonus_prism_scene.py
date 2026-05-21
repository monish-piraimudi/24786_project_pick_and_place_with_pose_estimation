"""Interactive bonus scene for a prism object geometry."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
PROJECT_DIR_STR = str(PROJECT_DIR)
if PROJECT_DIR_STR not in sys.path:
    sys.path.insert(0, PROJECT_DIR_STR)

from modules.pick_place_scene_entry import create_pick_place_scene


def createScene(rootnode):
    extra_argv = sys.argv[1:]
    return create_pick_place_scene(
        rootnode,
        argv=[
            "--no-camera-tracking",
            "--object-geometry",
            "prism",
            *extra_argv,
        ],
    )


if __name__ == "__main__":
    raise SystemExit("Run this file with runSofa to inspect the prism bonus scene in the GUI.")
