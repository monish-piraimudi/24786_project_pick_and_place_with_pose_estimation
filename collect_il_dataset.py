import argparse
from dataclasses import replace
from pathlib import Path

from modules.imitation_data import aggregate_rollout_metrics, ensure_directory, write_manifest
from modules.pick_place_il_runtime import DEFAULT_WORKSPACE_BOUNDS_MM, PickPlaceTaskConfig, run_single_episode


PROJECT_DIR = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="Collect implicit-BC episodes for Emio pick and place")
    parser.add_argument("--episodes", type=int, default=200, help="Number of successful episodes to save")
    parser.add_argument("--start-seed", type=int, default=0, help="Seed used for the first attempted episode")
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=400,
        help="Maximum attempted episodes before stopping collection",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "data/results/il_pick_place/episodes",
        help="Directory where episode .npz files will be saved. Relative paths are resolved from this lab folder.",
    )
    parser.add_argument(
        "--save-failed-episodes",
        action="store_true",
        help="Also persist failed rollouts instead of only successful ones",
    )
    parser.add_argument(
        "--connection",
        action="store_true",
        help="Enable the real robot connection components in the scene while collecting.",
    )
    parser.add_argument(
        "--camera-preview",
        action="store_true",
        help="Show the Emio camera preview while collecting.",
    )
    parser.add_argument(
        "--real-rgb-observation",
        dest="real_rgb_observation",
        action="store_true",
        help="Use live Emio camera RGB frames as the saved observation source.",
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
        help="Enable marker-assisted cube tracking during collection.",
    )
    parser.add_argument(
        "--no-camera-tracking",
        dest="camera_tracking",
        action="store_false",
        help="Disable marker-assisted cube tracking during collection.",
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
        help="Fixed XYZ offset from the tracked cube marker to the cube center in millimeters.",
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

    output_dir_arg = Path(args.output_dir)
    output_dir = ensure_directory(
        output_dir_arg if output_dir_arg.is_absolute() else (PROJECT_DIR / output_dir_arg)
    )
    manifest_path = output_dir / "manifest.json"

    base_config = PickPlaceTaskConfig(
        mode="collect",
        output_dir=str(output_dir),
        log_episode=True,
        save_failed_episodes=args.save_failed_episodes,
        connection=args.connection,
        camera_tracking=args.camera_tracking,
        camera_preview=args.camera_preview,
        real_rgb_observation=args.real_rgb_observation,
        camera_serial=args.camera_serial,
        object_workspace_bounds_mm=tuple(float(v) for v in args.workspace_bounds_mm),
        place_target_mm=None if args.place_target_mm is None else tuple(float(v) for v in args.place_target_mm),
        cube_marker_offset_mm=tuple(float(v) for v in args.cube_marker_offset_mm),
    )

    saved_entries = []
    attempted = 0
    successful = 0
    while attempted < args.max_attempts and successful < args.episodes:
        config = replace(
            base_config,
            episode_id=attempted,
            seed=args.start_seed + attempted,
        )
        summary = run_single_episode(config)
        attempted += 1
        print(
            f"episode={summary['episode_id']} seed={summary['seed']} "
            f"success={summary['total_success']} "
            f"pick={summary['pick_success']} place={summary['place_success']} "
            f"failure_phase={summary['failure_phase']} saved={summary['saved_path']}"
        )
        if summary["saved_path"] is None:
            continue
        saved_entries.append(summary)
        if summary["total_success"]:
            successful += 1

    write_manifest(saved_entries, manifest_path)
    metrics = aggregate_rollout_metrics(saved_entries)
    print(f"Saved {len(saved_entries)} episodes to {output_dir}")
    if not saved_entries:
        print(
            "No episode files were written. By default, collection only saves successful rollouts. "
            "Use --save-failed-episodes to persist failures too."
        )
    print(metrics)


if __name__ == "__main__":
    main()
