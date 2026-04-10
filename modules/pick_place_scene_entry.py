"""Compatibility shim for older imports of the pick-and-place scene entry."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR_STR = str(PROJECT_DIR)
if PROJECT_DIR_STR not in sys.path:
    sys.path.insert(0, PROJECT_DIR_STR)

from modules.pick_place_il import createScene
from modules.pick_place_il_runtime import PickPlaceTaskConfig


def parse_scene_args(argv: list[str] | None = None, *, with_gui: bool = True) -> PickPlaceTaskConfig:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--connection", dest="connection", action="store_true")
    parser.add_argument("--no-connection", dest="connection", action="store_false")
    parser.add_argument("--real-rgb-observation", dest="real_rgb_observation", action="store_true")
    parser.add_argument("--no-real-rgb-observation", dest="real_rgb_observation", action="store_false")
    parser.add_argument("--camera-tracking", dest="camera_tracking", action="store_true")
    parser.add_argument("--no-camera-tracking", dest="camera_tracking", action="store_false")
    parser.add_argument("--camera-preview", dest="camera_preview", action="store_true")
    parser.add_argument("--no-camera-preview", dest="camera_preview", action="store_false")
    parser.add_argument("--camera-serial", dest="camera_serial", type=str, default=None)
    parser.add_argument(
        "--cube-marker-offset-mm",
        dest="cube_marker_offset_mm",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        default=(0.0, 0.0, 0.0),
    )
    parser.set_defaults(
        connection=True,
        real_rgb_observation=False,
        camera_tracking=False,
        camera_preview=False,
    )
    args, _unknown = parser.parse_known_args(argv)
    return PickPlaceTaskConfig(
        with_gui=with_gui,
        connection=bool(args.connection),
        real_rgb_observation=bool(args.real_rgb_observation),
        camera_tracking=bool(args.camera_tracking),
        camera_preview=bool(args.camera_preview),
        camera_serial=args.camera_serial,
        cube_marker_offset_mm=tuple(float(value) for value in args.cube_marker_offset_mm),
    )


def create_pick_place_scene(rootnode, argv: list[str] | None = None):
    if argv is None:
        return createScene(rootnode)

    argv_backup = list(sys.argv)
    sys.argv = [argv_backup[0], *argv]
    try:
        return createScene(rootnode)
    finally:
        sys.argv = argv_backup
