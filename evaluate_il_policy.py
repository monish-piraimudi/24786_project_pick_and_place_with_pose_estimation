import argparse
import json
from dataclasses import replace
from pathlib import Path

from modules.imitation_data import aggregate_rollout_metrics, ensure_directory, write_manifest
from modules.pick_place_il_runtime import PickPlaceTaskConfig, run_single_episode


def main():
    parser = argparse.ArgumentParser(description="Evaluate expert or learned Emio pick-and-place controller")
    parser.add_argument(
        "--mode",
        choices=["expert", "policy"],
        default="policy",
        help="Rollout mode to evaluate",
    )
    parser.add_argument("--policy-path", type=str, default=None, help="Required for policy mode")
    parser.add_argument("--episodes", type=int, default=50, help="Number of rollout episodes")
    parser.add_argument("--start-seed", type=int, default=1000, help="Starting evaluation seed")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/results/il_pick_place/eval"),
        help="Directory for evaluation manifests and optional episodes",
    )
    parser.add_argument(
        "--save-rollouts",
        action="store_true",
        help="Save rollout trajectories as .npz files during evaluation",
    )
    parser.add_argument(
        "--connection",
        action="store_true",
        help="Enable the real robot connection components in the scene",
    )
    parser.add_argument(
        "--camera-tracking",
        action="store_true",
        help="Use Emio camera tracking to localize the cube marker during rollouts",
    )
    parser.add_argument(
        "--camera-preview",
        action="store_true",
        help="Show the Emio camera preview feed when camera tracking is enabled",
    )
    parser.add_argument(
        "--cube-marker-offset-mm",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=(0.0, 0.0, 0.0),
        help="Fixed XYZ offset from the tracked cube marker to the cube center in millimeters",
    )
    parser.add_argument(
        "--object-jitter-mm",
        type=float,
        default=15.0,
        help="Random X/Z jitter applied to the spawned block position",
    )
    parser.add_argument(
        "--place-jitter-mm",
        type=float,
        default=0.0,
        help="Random X/Z jitter applied to the place target position. Default keeps the place target fixed.",
    )
    args = parser.parse_args()

    if args.mode == "policy" and not args.policy_path:
        raise SystemExit("--policy-path is required when --mode policy is used")

    output_dir = ensure_directory(args.output_dir)
    base_config = PickPlaceTaskConfig(
        mode=args.mode,
        policy_path=args.policy_path,
        output_dir=str(output_dir / "episodes") if args.save_rollouts else None,
        log_episode=args.save_rollouts,
        save_failed_episodes=args.save_rollouts,
        connection=args.connection,
        camera_tracking=args.camera_tracking,
        camera_preview=args.camera_preview,
        cube_marker_offset_mm=tuple(float(value) for value in args.cube_marker_offset_mm),
        object_jitter_mm=args.object_jitter_mm,
        place_jitter_mm=args.place_jitter_mm,
    )

    entries = []
    for episode_idx in range(args.episodes):
        config = replace(
            base_config,
            episode_id=episode_idx,
            seed=args.start_seed + episode_idx,
        )
        summary = run_single_episode(config)
        entries.append(summary)
        print(
            f"episode={summary['episode_id']} seed={summary['seed']} "
            f"success={summary['total_success']} final_place_error_mm={summary['final_place_error_mm']:.3f}"
        )

    metrics = aggregate_rollout_metrics(entries)
    manifest_path = write_manifest(entries, output_dir / "rollout_manifest.json")
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print(f"Wrote rollout manifest to {manifest_path}")
    print(f"Wrote aggregate metrics to {metrics_path}")
    print(metrics)


if __name__ == "__main__":
    main()
