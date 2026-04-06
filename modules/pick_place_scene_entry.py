"""Shared scene-entry helpers for the Emio imitation-learning SOFA scenes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR_STR = str(PROJECT_DIR)
if PROJECT_DIR_STR not in sys.path:
    sys.path.insert(0, PROJECT_DIR_STR)

from modules.pick_place_il import PickPlaceTaskConfig, build_scene


def parse_scene_args(argv: list[str] | None = None, *, with_gui: bool = True) -> PickPlaceTaskConfig:
    """Parse SOFA scene arguments into the shared rollout config."""

    parser = argparse.ArgumentParser(description="SOFA scene for Emio imitation-learning pick and place")
    parser.add_argument(
        "--mode",
        choices=["expert", "collect", "policy", "dagger"],
        default="expert",
        help="Controller mode to run inside the scene",
    )
    parser.add_argument("--policy-path", type=str, default=None, help="Policy checkpoint path")
    parser.add_argument("--output-dir", type=str, default=None, help="Optional episode output directory")
    parser.add_argument("--episode-id", type=int, default=0, help="Episode identifier")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for cube and target placement")
    parser.add_argument("--max-steps", type=int, default=300, help="Maximum task control steps")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Disable GUI-specific scene accessories for batch or test execution",
    )
    parser.add_argument(
        "--save-failed-episodes",
        action="store_true",
        help="Persist failed trajectories when logging is enabled",
    )

    try:
        args, _unknown = parser.parse_known_args(argv)
    except SystemExit:
        args, _unknown = parser.parse_known_args([])

    return PickPlaceTaskConfig(
        mode=args.mode,
        policy_path=args.policy_path,
        output_dir=args.output_dir,
        episode_id=args.episode_id,
        seed=args.seed,
        max_steps=args.max_steps,
        with_gui=with_gui and not args.headless,
        log_episode=bool(args.output_dir),
        save_failed_episodes=args.save_failed_episodes,
    )


def create_pick_place_scene(rootnode, argv: list[str] | None = None):
    """Build the Emio pick-and-place scene from CLI-style arguments."""

    config = parse_scene_args(argv=argv, with_gui=True)
    build_scene(rootnode, config)
    return rootnode
