import argparse
import sys

from modules.pick_place_il import PickPlaceTaskConfig, build_scene


def _parse_args() -> PickPlaceTaskConfig:
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
        "--save-failed-episodes",
        action="store_true",
        help="Persist failed trajectories when logging is enabled",
    )
    try:
        args = parser.parse_args()
    except SystemExit:
        args = parser.parse_args([])

    return PickPlaceTaskConfig(
        mode=args.mode,
        policy_path=args.policy_path,
        output_dir=args.output_dir,
        episode_id=args.episode_id,
        seed=args.seed,
        max_steps=args.max_steps,
        with_gui=True,
        log_episode=bool(args.output_dir),
        save_failed_episodes=args.save_failed_episodes,
    )


def createScene(rootnode):
    config = _parse_args()
    build_scene(rootnode, config)
    return rootnode


if __name__ == "__main__":
    raise SystemExit(
        "Run this file with runSofa, or use collect_il_dataset.py / evaluate_il_policy.py for scripted rollouts."
    )
