"""Interactive SOFA scene entrypoint for visualizing the trained policy without camera hardware."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
PROJECT_DIR_STR = str(PROJECT_DIR)
if PROJECT_DIR_STR not in sys.path:
    sys.path.insert(0, PROJECT_DIR_STR)

from modules.pick_place_scene_entry import create_pick_place_scene


def createScene(rootnode):
    return create_pick_place_scene(
        rootnode,
        argv=[
            "--mode",
            "policy_inspect",
            "--policy-path",
            "data/results/il_pick_place/bc_policy.pth",
            "--no-real-rgb-observation",
            "--no-camera-tracking",
        ],
    )


if __name__ == "__main__":
    raise SystemExit("Run this file with runSofa to inspect the trained policy in the GUI.")
