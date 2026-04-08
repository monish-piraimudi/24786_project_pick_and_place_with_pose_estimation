import argparse
from dataclasses import replace
from pathlib import Path

from modules.imitation_data import aggregate_rollout_metrics, ensure_directory, write_manifest
from modules.pick_place_il_runtime import PickPlaceTaskConfig, run_single_episode


PROJECT_DIR = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="Collect imitation-learning episodes for Emio pick and place")
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
        object_jitter_mm=args.object_jitter_mm,
        place_jitter_mm=args.place_jitter_mm,
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
