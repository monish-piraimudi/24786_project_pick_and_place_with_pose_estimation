"""Behavior-cloning policy helpers for the Emio pick-and-place task.

This module is the "learned policy" side of the imitation-learning pipeline:
we define a small neural network, save the normalization statistics it needs,
and load the checkpoint back for closed-loop rollouts inside SOFA.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


OBSERVATION_DIM = 29
ACTION_DIM = 4


class BehaviorCloningMLP(nn.Module):
    """Small MLP used for behavior cloning.

    In behavior cloning, the network is trained on logged expert
    state-action pairs and learns to predict the action that the expert would
    take from the current observation.
    """

    def __init__(self, input_dim: int = OBSERVATION_DIM, output_dim: int = ACTION_DIM):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def save_policy_checkpoint(
    file_path: str | Path,
    model: BehaviorCloningMLP,
    obs_mean: np.ndarray,
    obs_std: np.ndarray,
    metadata: dict | None = None,
) -> None:
    """Save the trained policy together with observation normalization stats."""

    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            # The policy was trained on normalized observations, so rollout code
            # must reuse the same mean/std at inference time.
            "obs_mean": np.asarray(obs_mean, dtype=np.float32),
            "obs_std": np.asarray(obs_std, dtype=np.float32),
            "metadata": metadata or {},
        },
        file_path,
    )


def load_policy_checkpoint(
    file_path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[BehaviorCloningMLP, np.ndarray, np.ndarray, dict]:
    """Load a trained policy and the statistics needed to preprocess inputs."""

    checkpoint = torch.load(file_path, map_location=device, weights_only=False)
    model = BehaviorCloningMLP()
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    obs_mean = np.asarray(checkpoint["obs_mean"], dtype=np.float32)
    obs_std = np.asarray(checkpoint["obs_std"], dtype=np.float32)
    metadata = dict(checkpoint.get("metadata", {}))
    return model, obs_mean, obs_std, metadata


class BehaviorCloningAgent:
    """Thin inference wrapper used by the SOFA controller during rollouts."""

    def __init__(
        self,
        model: BehaviorCloningMLP,
        obs_mean: np.ndarray,
        obs_std: np.ndarray,
        device: str | torch.device = "cpu",
    ):
        self.model = model
        self.obs_mean = np.asarray(obs_mean, dtype=np.float32)
        self.obs_std = np.asarray(obs_std, dtype=np.float32)
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def from_checkpoint(
        cls,
        file_path: str | Path,
        device: str | torch.device = "cpu",
    ) -> "BehaviorCloningAgent":
        """Construct an inference agent from a saved behavior-cloning checkpoint."""

        model, obs_mean, obs_std, _ = load_policy_checkpoint(file_path, device=device)
        return cls(model=model, obs_mean=obs_mean, obs_std=obs_std, device=device)

    def predict(self, observation: np.ndarray) -> np.ndarray:
        """Predict a single action for one observation vector.

        The observation mixes positions, motor states, and phase indicators
        that live on different numeric scales, so we normalize it exactly the
        same way it was normalized during training before passing it to the MLP.
        """

        observation = np.asarray(observation, dtype=np.float32)
        normalized = (observation - self.obs_mean) / self.obs_std
        with torch.inference_mode():
            tensor = torch.from_numpy(normalized).to(self.device).unsqueeze(0)
            action = self.model(tensor).cpu().numpy()[0]
        return action.astype(np.float32)
