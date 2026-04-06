import argparse
from dataclasses import replace
from pathlib import Path

from modules.imitation_data import aggregate_rollout_metrics, ensure_directory, write_manifest
from modules.pick_place_il import PickPlaceTaskConfig, run_single_episode


def main():
    parser = argparse.ArgumentParser(description="Collect imitation-learning episodes for Emio pick and place")
    parser.add_argument(
        "--mode",
        choices=["collect", "dagger"],
        default="collect",
        help="Collection mode: expert collection or DAgger-style corrective logging",
    )
    parser.add_argument("--policy-path", type=str, default=None, help="Required for dagger mode")
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
        default=Path("data/results/il_pick_place/episodes"),
        help="Directory where episode .npz files will be saved",
    )
    parser.add_argument(
        "--save-failed-episodes",
        action="store_true",
        help="Also persist failed rollouts instead of only successful ones",
    )
    args = parser.parse_args()

    if args.mode == "dagger" and not args.policy_path:
        raise SystemExit("--policy-path is required when --mode dagger is used")

    output_dir = ensure_directory(args.output_dir)
    manifest_path = output_dir / "manifest.json"

    base_config = PickPlaceTaskConfig(
        mode=args.mode,
        policy_path=args.policy_path,
        output_dir=str(output_dir),
        log_episode=True,
        save_failed_episodes=args.save_failed_episodes,
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
        if summary["saved_path"] is None:
            continue
        saved_entries.append(summary)
        if summary["total_success"]:
            successful += 1
        print(
            f"episode={summary['episode_id']} seed={summary['seed']} "
            f"success={summary['total_success']} saved={summary['saved_path']}"
        )

    write_manifest(saved_entries, manifest_path)
    metrics = aggregate_rollout_metrics(saved_entries)
    print(f"Saved {len(saved_entries)} episodes to {output_dir}")
    print(metrics)


if __name__ == "__main__":
    main()
