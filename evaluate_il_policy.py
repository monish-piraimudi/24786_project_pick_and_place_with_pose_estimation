import argparse
import json
from dataclasses import replace
from pathlib import Path

from modules.imitation_data import aggregate_rollout_metrics, ensure_directory, write_manifest
from modules.pick_place_il_runtime import DEFAULT_WORKSPACE_BOUNDS_MM, PickPlaceTaskConfig, run_single_episode


PROJECT_DIR = Path(__file__).resolve().parent


def _resolve_project_path(path: Path) -> Path:
    return path if path.is_absolute() else (PROJECT_DIR / path)


def main():
    parser = argparse.ArgumentParser(description="Evaluate expert or implicit learned Emio pick-and-place controller")
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
        default=PROJECT_DIR / "data/results/il_pick_place/eval",
        help="Directory for evaluation manifests and optional episodes. Relative paths are resolved from this lab folder.",
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
        "--camera-preview",
        action="store_true",
        help="Show the Emio camera preview feed when camera tracking is enabled",
    )
    parser.add_argument(
        "--real-rgb-observation",
        dest="real_rgb_observation",
        action="store_true",
        help="Use live Emio camera RGB frames as the policy observation source.",
    )
    parser.add_argument(
        "--no-real-rgb-observation",
        dest="real_rgb_observation",
        action="store_false",
        help="Disable real RGB camera observations and fall back to the synthetic render.",
    )
    parser.add_argument(
        "--camera-tracking",
        dest="camera_tracking",
        action="store_true",
        help="Enable marker-assisted cube tracking during rollouts.",
    )
    parser.add_argument(
        "--no-camera-tracking",
        dest="camera_tracking",
        action="store_false",
        help="Disable marker-assisted cube tracking during rollouts.",
    )
    parser.add_argument(
        "--camera-serial",
        type=str,
        default=None,
        help="Optional Emio camera serial to open explicitly.",
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
        "--workspace-bounds-mm",
        type=float,
        nargs=4,
        metavar=("X_MIN", "X_MAX", "Z_MIN", "Z_MAX"),
        default=DEFAULT_WORKSPACE_BOUNDS_MM,
        help="Continuous tray workspace bounds used to sample object X/Z positions.",
    )
    parser.add_argument(
        "--place-target-mm",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=None,
        help="Optional fixed place target override in millimeters.",
    )
    parser.set_defaults(real_rgb_observation=False, camera_tracking=False)
    args = parser.parse_args()

    if args.mode == "policy" and not args.policy_path:
        raise SystemExit("--policy-path is required when --mode policy is used")

    output_dir = ensure_directory(_resolve_project_path(Path(args.output_dir)))
    base_config = PickPlaceTaskConfig(
        mode=args.mode,
        policy_path=args.policy_path,
        output_dir=str(output_dir / "episodes") if args.save_rollouts else None,
        log_episode=args.save_rollouts,
        save_failed_episodes=args.save_rollouts,
        connection=args.connection,
        camera_tracking=args.camera_tracking,
        camera_preview=args.camera_preview,
        real_rgb_observation=args.real_rgb_observation,
        camera_serial=args.camera_serial,
        cube_marker_offset_mm=tuple(float(value) for value in args.cube_marker_offset_mm),
        object_workspace_bounds_mm=tuple(float(value) for value in args.workspace_bounds_mm),
        place_target_mm=None if args.place_target_mm is None else tuple(float(v) for v in args.place_target_mm),
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
