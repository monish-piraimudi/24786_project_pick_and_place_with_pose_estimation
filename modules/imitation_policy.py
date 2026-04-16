"""Implicit behavior-cloning helpers for the motor-angle pick-and-place policy."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


ACTION_DIM = 4
MODEL_TYPE = "implicit_bc_motor_state_v1"
DEFAULT_SEARCH_CONFIG = {
    "num_samples": 192,
    "num_elites": 24,
    "num_iters": 4,
    "min_std": 0.05,
}
DEFAULT_WARM_START = True
DEFAULT_WARM_START_STD_SCALE = 0.35
DEFAULT_ACTION_SMOOTHING_ALPHA = 0.35


class ImplicitBCPolicy(nn.Module):
    """State-only energy model over `(state, action)` pairs."""

    def __init__(self, state_dim: int, action_dim: int = ACTION_DIM):
        super().__init__()
        self.state_dim = int(state_dim)
        if self.state_dim <= 0:
            raise ValueError("Motor-angle implicit BC requires a positive state_dim.")
        self.state_encoder = nn.Sequential(
            nn.Linear(self.state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.energy_head = nn.Sequential(
            nn.Linear(64 + 128, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def encode_observation(self, state: torch.Tensor) -> torch.Tensor:
        if state is None:
            raise ValueError("Motor-angle implicit BC requires a state observation tensor.")
        return self.state_encoder(state)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        obs_embed = self.encode_observation(state)
        action_embed = self.action_encoder(action)
        return self.energy_head(torch.cat([obs_embed, action_embed], dim=-1)).squeeze(-1)

    def forward_with_embedding(self, obs_embed: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        action_embed = self.action_encoder(action)
        return self.energy_head(torch.cat([obs_embed, action_embed], dim=-1)).squeeze(-1)


def _normalize_action_bounds(action_bounds) -> np.ndarray:
    array = np.asarray(action_bounds, dtype=np.float32).reshape(ACTION_DIM, 2)
    lower = np.minimum(array[:, 0], array[:, 1])
    upper = np.maximum(array[:, 0], array[:, 1])
    return np.stack([lower, upper], axis=1).astype(np.float32)


def save_policy_checkpoint(
    file_path: str | Path,
    model: nn.Module,
    metadata: dict | None = None,
) -> None:
    """Save the trained motor-angle implicit policy."""

    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(metadata or {})
    metadata.setdefault("model_type", MODEL_TYPE)
    metadata.setdefault("action_bounds", [[-np.pi, np.pi]] * ACTION_DIM)
    metadata.setdefault("search_config", dict(DEFAULT_SEARCH_CONFIG))
    metadata.setdefault("warm_start", DEFAULT_WARM_START)
    metadata.setdefault("warm_start_std_scale", DEFAULT_WARM_START_STD_SCALE)
    metadata.setdefault("action_smoothing_alpha", DEFAULT_ACTION_SMOOTHING_ALPHA)
    metadata.setdefault("state_dim", int(getattr(model, "state_dim", 0)))
    metadata.setdefault("state_feature_names", [])
    metadata.setdefault("state_mean", [])
    metadata.setdefault("state_std", [])
    torch.save(
        {
            "state_dict": model.state_dict(),
            "metadata": metadata,
        },
        file_path,
    )


def load_policy_checkpoint(
    file_path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[ImplicitBCPolicy, dict]:
    """Load a trained motor-angle implicit BC policy."""

    checkpoint = torch.load(file_path, map_location=device, weights_only=False)
    metadata = dict(checkpoint.get("metadata", {}))
    model_type = metadata.get("model_type")
    if model_type != MODEL_TYPE:
        raise ValueError(
            f"Checkpoint model_type={model_type!r} is not supported. Expected {MODEL_TYPE!r}."
        )

    state_dim = int(metadata.get("state_dim", 0))
    model = ImplicitBCPolicy(state_dim=state_dim)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, metadata


class ImplicitBCAgent:
    """State-only motor-angle implicit BC inference wrapper using CEM."""

    def __init__(
        self,
        model: ImplicitBCPolicy,
        action_bounds: np.ndarray,
        search_config: dict | None = None,
        state_mean: np.ndarray | None = None,
        state_std: np.ndarray | None = None,
        state_feature_names: list[str] | tuple[str, ...] | None = None,
        warm_start: bool = DEFAULT_WARM_START,
        warm_start_std_scale: float = DEFAULT_WARM_START_STD_SCALE,
        action_smoothing_alpha: float = DEFAULT_ACTION_SMOOTHING_ALPHA,
        device: str | torch.device = "cpu",
    ):
        self.model = model
        self.action_bounds = _normalize_action_bounds(action_bounds)
        self.search_config = dict(DEFAULT_SEARCH_CONFIG)
        if search_config is not None:
            self.search_config.update(search_config)
        self.state_dim = int(getattr(model, "state_dim", 0))
        self.state_feature_names = list(state_feature_names or [])
        self.state_mean = np.asarray(state_mean, dtype=np.float32).reshape(self.state_dim)
        self.state_std = np.asarray(state_std, dtype=np.float32).reshape(self.state_dim)
        self.warm_start = bool(warm_start)
        self.warm_start_std_scale = float(warm_start_std_scale)
        self.action_smoothing_alpha = float(np.clip(action_smoothing_alpha, 0.0, 1.0))
        self.previous_action: np.ndarray | None = None
        self.model_type = MODEL_TYPE
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def from_checkpoint(
        cls,
        file_path: str | Path,
        device: str | torch.device = "cpu",
    ) -> "ImplicitBCAgent":
        model, metadata = load_policy_checkpoint(file_path, device=device)
        return cls(
            model=model,
            action_bounds=metadata.get("action_bounds"),
            search_config=metadata.get("search_config"),
            state_mean=metadata.get("state_mean"),
            state_std=metadata.get("state_std"),
            state_feature_names=metadata.get("state_feature_names"),
            warm_start=metadata.get("warm_start", DEFAULT_WARM_START),
            warm_start_std_scale=metadata.get("warm_start_std_scale", DEFAULT_WARM_START_STD_SCALE),
            action_smoothing_alpha=metadata.get("action_smoothing_alpha", DEFAULT_ACTION_SMOOTHING_ALPHA),
            device=device,
        )

    def _clip_action(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32).reshape(ACTION_DIM)
        return np.clip(action, self.action_bounds[:, 0], self.action_bounds[:, 1]).astype(np.float32)

    def reset_rollout(self) -> None:
        self.previous_action = None

    def set_previous_action(self, executed_action: np.ndarray | None) -> None:
        if executed_action is None:
            self.previous_action = None
            return
        self.previous_action = self._clip_action(executed_action)

    def smooth_action(self, raw_action: np.ndarray) -> np.ndarray:
        raw_action = self._clip_action(raw_action)
        if self.previous_action is None:
            return raw_action
        alpha = self.action_smoothing_alpha
        smoothed = alpha * raw_action + (1.0 - alpha) * self.previous_action
        return self._clip_action(smoothed)

    def _initial_search_distribution(self, warm_start_action: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        lower = self.action_bounds[:, 0]
        upper = self.action_bounds[:, 1]
        mean = 0.5 * (lower + upper)
        std = 0.5 * (upper - lower)

        if self.warm_start:
            if warm_start_action is None:
                warm_start_action = self.previous_action
            if warm_start_action is not None:
                mean = self._clip_action(warm_start_action)
                std = std * self.warm_start_std_scale

        min_std = float(self.search_config["min_std"])
        std = np.maximum(std, min_std).astype(np.float32)
        return mean.astype(np.float32), std

    def _prepare_state_tensor(self, state_observation: np.ndarray) -> torch.Tensor:
        state = np.asarray(state_observation, dtype=np.float32).reshape(-1)
        if state.shape[0] != self.state_dim:
            raise ValueError(f"Expected state dim {self.state_dim}, got {state.shape[0]}")
        state = (state - self.state_mean) / self.state_std
        return torch.from_numpy(state.astype(np.float32)).to(self.device).unsqueeze(0)

    def score_actions(
        self,
        state_observation: np.ndarray,
        actions: np.ndarray,
    ) -> np.ndarray:
        state_tensor = self._prepare_state_tensor(state_observation)
        actions = np.asarray(actions, dtype=np.float32).reshape(-1, ACTION_DIM)
        with torch.inference_mode():
            obs_embed = self.model.encode_observation(state_tensor)
            obs_embed = obs_embed.repeat(actions.shape[0], 1)
            action_tensor = torch.from_numpy(actions).to(self.device)
            energy = self.model.forward_with_embedding(obs_embed, action_tensor).cpu().numpy()
        return energy.astype(np.float32)

    def predict(
        self,
        state_observation: np.ndarray,
        warm_start_action: np.ndarray | None = None,
    ) -> np.ndarray:
        state_tensor = self._prepare_state_tensor(state_observation)
        lower = self.action_bounds[:, 0]
        upper = self.action_bounds[:, 1]
        num_samples = int(self.search_config["num_samples"])
        num_elites = int(self.search_config["num_elites"])
        num_iters = int(self.search_config["num_iters"])
        min_std = float(self.search_config["min_std"])

        mean, std = self._initial_search_distribution(warm_start_action=warm_start_action)
        best_action = mean.copy()
        best_energy = np.inf

        with torch.inference_mode():
            obs_embed = self.model.encode_observation(state_tensor)
            for _ in range(num_iters):
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

        return self._clip_action(best_action)
