"""Canonical SOFA scene entrypoint for the Emio imitation-learning lab."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
PROJECT_DIR_STR = str(PROJECT_DIR)
if PROJECT_DIR_STR not in sys.path:
    sys.path.insert(0, PROJECT_DIR_STR)

from modules.pick_place_scene_entry import create_pick_place_scene


def createScene(rootnode):
    return create_pick_place_scene(rootnode, argv=["--no-connection", "--no-camera-tracking"])


if __name__ == "__main__":
    raise SystemExit(
        "Run this file with runSofa, or use collect_il_dataset.py / evaluate_il_policy.py for scripted rollouts."
    )
