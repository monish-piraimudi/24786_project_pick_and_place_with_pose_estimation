"""Dataset utilities for imitation learning on the Emio pick-and-place task.

Episodes are first recorded as trajectories because that preserves rollout
order, success labels, and per-episode context. Training later flattens those
episodes into a large table of observation-action pairs for supervised
behavior-cloning updates.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def ensure_directory(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path."""

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


class EpisodeRecorder:
    """Collect one rollout worth of timestep data before saving it to disk."""

    def __init__(self, output_dir: str | Path, episode_id: int):
        self.output_dir = ensure_directory(output_dir)
        self.episode_id = int(episode_id)
        self._records: dict[str, list[np.ndarray | float | int | bool]] = {}

    def append(self, **values) -> None:
        """Append one timestep of logged values to the in-memory trajectory."""

        for key, value in values.items():
            self._records.setdefault(key, []).append(value)

    def save(self) -> Path:
        """Persist the recorded trajectory as a compressed NumPy episode file."""

        output_path = self.output_dir / f"episode_{self.episode_id:05d}.npz"
        arrays = {}
        for key, values in self._records.items():
            arrays[key] = np.asarray(values)
        np.savez_compressed(output_path, **arrays)
        return output_path


def load_episode_paths(dataset_dir: str | Path) -> list[Path]:
    """Return all saved episode files in a dataset directory."""

    dataset_dir = Path(dataset_dir)
    return sorted(dataset_dir.glob("episode_*.npz"))


def split_episode_paths(
    episode_paths: list[Path],
    seed: int,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> tuple[list[Path], list[Path], list[Path]]:
    """Split saved rollouts into train/validation/test sets by episode."""

    if not episode_paths:
        return [], [], []

    if not np.isclose(sum(ratios), 1.0):
        raise ValueError("Split ratios must sum to 1.0")

    rng = np.random.default_rng(seed)
    shuffled = list(episode_paths)
    rng.shuffle(shuffled)

    # We split whole episodes instead of individual rows so that near-identical
    # consecutive timesteps from one rollout do not leak across train/val/test.
    n_total = len(shuffled)
    n_train = max(1, int(round(n_total * ratios[0]))) if n_total >= 3 else max(1, n_total - 2)
    n_val = max(1, int(round(n_total * ratios[1]))) if n_total >= 3 else (1 if n_total > 1 else 0)
    if n_train + n_val >= n_total:
        n_val = max(0, n_total - n_train - 1)
    n_test = n_total - n_train - n_val
    if n_test == 0 and n_total >= 3:
        n_test = 1
        n_train = max(1, n_train - 1)

    train_paths = shuffled[:n_train]
    val_paths = shuffled[n_train : n_train + n_val]
    test_paths = shuffled[n_train + n_val :]
    return train_paths, val_paths, test_paths


def flatten_episode_dataset(
    episode_paths: list[Path],
    observation_key: str = "observation",
    action_key: str = "action",
) -> tuple[np.ndarray, np.ndarray]:
    """Flatten multiple trajectory files into step-wise training arrays."""

    observations = []
    actions = []
    for episode_path in episode_paths:
        with np.load(episode_path, allow_pickle=False) as episode:
            observations.append(np.asarray(episode[observation_key], dtype=np.float32))
            actions.append(np.asarray(episode[action_key], dtype=np.float32))

    if not observations:
        return (
            np.zeros((0, 0), dtype=np.float32),
            np.zeros((0, 0), dtype=np.float32),
        )

    return np.concatenate(observations, axis=0), np.concatenate(actions, axis=0)


def write_manifest(entries: list[dict], file_path: str | Path) -> Path:
    """Write rollout summaries to JSON for quick inspection and bookkeeping."""

    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as handle:
        json.dump(entries, handle, indent=2)
    return file_path


def aggregate_rollout_metrics(entries: list[dict]) -> dict:
    """Aggregate rollout-level metrics.

    These metrics answer "does the learned controller succeed when unrolled in
    the simulator?" which is different from the supervised training loss used
    while fitting the policy offline.
    """

    if not entries:
        return {
            "num_episodes": 0,
            "pick_success": 0.0,
            "place_success": 0.0,
            "total_success": 0.0,
            "final_place_error_mm": None,
            "dropped_object_rate": 0.0,
            "failure_phase_counts": {},
        }

    pick_success = np.mean([float(entry["pick_success"]) for entry in entries])
    place_success = np.mean([float(entry["place_success"]) for entry in entries])
    total_success = np.mean([float(entry["total_success"]) for entry in entries])
    dropped_rate = np.mean([float(entry["dropped_object"]) for entry in entries])
    final_errors = [float(entry["final_place_error_mm"]) for entry in entries]

    failure_counts: dict[str, int] = {}
    for entry in entries:
        phase = entry.get("failure_phase", "none")
        failure_counts[phase] = failure_counts.get(phase, 0) + 1

    return {
        "num_episodes": len(entries),
        "pick_success": float(pick_success),
        "place_success": float(place_success),
        "total_success": float(total_success),
        "final_place_error_mm": float(np.median(final_errors)),
        "dropped_object_rate": float(dropped_rate),
        "failure_phase_counts": failure_counts,
    }
