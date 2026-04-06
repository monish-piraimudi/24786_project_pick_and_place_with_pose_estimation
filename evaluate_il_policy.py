import argparse
import json
from dataclasses import replace
from pathlib import Path

from modules.imitation_data import aggregate_rollout_metrics, ensure_directory, write_manifest
from modules.pick_place_il import PickPlaceTaskConfig, run_single_episode


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
