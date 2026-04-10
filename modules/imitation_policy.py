"""Implicit behavior-cloning helpers for the image-based pick-and-place policy."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from modules.camera_observation import default_image_shape


ACTION_DIM = 4
DEFAULT_ACTION_BOUNDS = np.asarray(
    [
        [-5.0, 5.0],
        [-5.0, 5.0],
        [-5.0, 5.0],
        [0.0, 1.0],
    ],
    dtype=np.float32,
)
DEFAULT_SEARCH_CONFIG = {
    "num_samples": 192,
    "num_elites": 24,
    "num_iters": 4,
    "min_std": 0.05,
}


class ImplicitBCPolicy(nn.Module):
    """Energy model over `(observation, action)` pairs."""

    def __init__(self, input_channels: int = 3, action_dim: int = ACTION_DIM):
        super().__init__()
        self.image_encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.energy_head = nn.Sequential(
            nn.Linear(256 + 128, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def encode_observation(self, x: torch.Tensor) -> torch.Tensor:
        return self.image_encoder(x)

    def forward(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        obs_embed = self.encode_observation(x)
        action_embed = self.action_encoder(action)
        return self.energy_head(torch.cat([obs_embed, action_embed], dim=-1)).squeeze(-1)

    def forward_with_embedding(self, obs_embed: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        action_embed = self.action_encoder(action)
        return self.energy_head(torch.cat([obs_embed, action_embed], dim=-1)).squeeze(-1)


def _normalize_action_bounds(action_bounds) -> np.ndarray:
    if action_bounds is None:
        return DEFAULT_ACTION_BOUNDS.copy()
    array = np.asarray(action_bounds, dtype=np.float32).reshape(ACTION_DIM, 2)
    lower = np.minimum(array[:, 0], array[:, 1])
    upper = np.maximum(array[:, 0], array[:, 1])
    return np.stack([lower, upper], axis=1).astype(np.float32)


def save_policy_checkpoint(
    file_path: str | Path,
    model: nn.Module,
    image_mean: np.ndarray,
    image_std: np.ndarray,
    image_shape: tuple[int, int, int] | None = None,
    metadata: dict | None = None,
) -> None:
    """Save the trained implicit policy with image normalization stats."""

    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(metadata or {})
    metadata.setdefault("model_type", "implicit_bc")
    metadata.setdefault("action_bounds", DEFAULT_ACTION_BOUNDS.tolist())
    metadata.setdefault("search_config", dict(DEFAULT_SEARCH_CONFIG))
    torch.save(
        {
            "state_dict": model.state_dict(),
            "image_mean": np.asarray(image_mean, dtype=np.float32),
            "image_std": np.asarray(image_std, dtype=np.float32),
            "image_shape": tuple(image_shape or default_image_shape()),
            "metadata": metadata,
        },
        file_path,
    )


def load_policy_checkpoint(
    file_path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[ImplicitBCPolicy, np.ndarray, np.ndarray, tuple[int, int, int], dict]:
    """Load a trained implicit BC policy and its preprocessing metadata."""

    checkpoint = torch.load(file_path, map_location=device, weights_only=False)
    metadata = dict(checkpoint.get("metadata", {}))
    model_type = metadata.get("model_type", "implicit_bc")
    if model_type != "implicit_bc":
        raise ValueError(
            f"Checkpoint model_type={model_type!r} is not supported by the implicit BC runtime."
        )

    image_shape = tuple(checkpoint.get("image_shape") or default_image_shape())
    model = ImplicitBCPolicy(input_channels=int(image_shape[2]))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    image_mean = np.asarray(checkpoint["image_mean"], dtype=np.float32)
    image_std = np.asarray(checkpoint["image_std"], dtype=np.float32)
    return model, image_mean, image_std, image_shape, metadata


class ImplicitBCAgent:
    """Image-policy inference wrapper using CEM over bounded 4D actions."""

    def __init__(
        self,
        model: ImplicitBCPolicy,
        image_mean: np.ndarray,
        image_std: np.ndarray,
        image_shape: tuple[int, int, int] | None = None,
        action_bounds: np.ndarray | None = None,
        search_config: dict | None = None,
        device: str | torch.device = "cpu",
    ):
        self.model = model
        self.image_mean = np.asarray(image_mean, dtype=np.float32)
        self.image_std = np.asarray(image_std, dtype=np.float32)
        self.image_shape = tuple(image_shape or default_image_shape())
        self.action_bounds = _normalize_action_bounds(action_bounds)
        self.search_config = dict(DEFAULT_SEARCH_CONFIG)
        if search_config is not None:
            self.search_config.update(search_config)
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def from_checkpoint(
        cls,
        file_path: str | Path,
        device: str | torch.device = "cpu",
    ) -> "ImplicitBCAgent":
        model, image_mean, image_std, image_shape, metadata = load_policy_checkpoint(file_path, device=device)
        return cls(
            model=model,
            image_mean=image_mean,
            image_std=image_std,
            image_shape=image_shape,
            action_bounds=metadata.get("action_bounds"),
            search_config=metadata.get("search_config"),
            device=device,
        )

    def _prepare_image_tensor(self, observation: np.ndarray) -> torch.Tensor:
        image = np.asarray(observation, dtype=np.float32)
        if image.ndim != 3:
            raise ValueError(f"Expected HWC image observation, got shape {image.shape}")
        if tuple(image.shape) != self.image_shape:
            raise ValueError(f"Expected image shape {self.image_shape}, got {tuple(image.shape)}")
        if float(np.max(image)) > 1.5:
            image = image / 255.0
        image = (image - self.image_mean.reshape(1, 1, -1)) / self.image_std.reshape(1, 1, -1)
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        return torch.from_numpy(image).to(self.device).unsqueeze(0)

    def score_actions(self, observation: np.ndarray, actions: np.ndarray) -> np.ndarray:
        image_tensor = self._prepare_image_tensor(observation)
        actions = np.asarray(actions, dtype=np.float32).reshape(-1, ACTION_DIM)
        with torch.inference_mode():
            obs_embed = self.model.encode_observation(image_tensor)
            obs_embed = obs_embed.repeat(actions.shape[0], 1)
            action_tensor = torch.from_numpy(actions).to(self.device)
            energy = self.model.forward_with_embedding(obs_embed, action_tensor).cpu().numpy()
        return energy.astype(np.float32)

    def predict(self, observation: np.ndarray) -> np.ndarray:
        image_tensor = self._prepare_image_tensor(observation)
        lower = self.action_bounds[:, 0]
        upper = self.action_bounds[:, 1]
        num_samples = int(self.search_config["num_samples"])
        num_elites = int(self.search_config["num_elites"])
        num_iters = int(self.search_config["num_iters"])
        min_std = float(self.search_config["min_std"])

        mean = 0.5 * (lower + upper)
        std = 0.5 * (upper - lower)
        best_action = mean.copy()
        best_energy = np.inf

        with torch.inference_mode():
            obs_embed = self.model.encode_observation(image_tensor)
            for search_iter in range(num_iters):
                if search_iter == 0:
                    candidates = np.random.uniform(lower, upper, size=(num_samples, ACTION_DIM)).astype(np.float32)
                else:
                    candidates = np.random.normal(mean, std, size=(num_samples, ACTION_DIM)).astype(np.float32)
                    candidates = np.clip(candidates, lower, upper)

                obs_embed_batch = obs_embed.repeat(candidates.shape[0], 1)
                action_tensor = torch.from_numpy(candidates).to(self.device)
                energies = self.model.forward_with_embedding(obs_embed_batch, action_tensor).cpu().numpy()
                elite_idx = np.argsort(energies)[:num_elites]
                elites = candidates[elite_idx]
                mean = elites.mean(axis=0)
                std = np.maximum(elites.std(axis=0), min_std)

                if float(energies[elite_idx[0]]) < best_energy:
                    best_energy = float(energies[elite_idx[0]])
                    best_action = candidates[elite_idx[0]].copy()

        return np.clip(best_action, lower, upper).astype(np.float32)
