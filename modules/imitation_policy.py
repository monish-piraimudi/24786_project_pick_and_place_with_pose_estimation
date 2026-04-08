"""Behavior-cloning helpers for the image-based pick-and-place policy."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from modules.camera_observation import default_image_shape


ACTION_DIM = 4


class BehaviorCloningCNN(nn.Module):
    """Compact CNN used for image-based behavior cloning."""

    def __init__(self, input_channels: int = 3, output_dim: int = ACTION_DIM):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def save_policy_checkpoint(
    file_path: str | Path,
    model: BehaviorCloningCNN,
    image_mean: np.ndarray,
    image_std: np.ndarray,
    image_shape: tuple[int, int, int] | None = None,
    metadata: dict | None = None,
) -> None:
    """Save the trained CNN together with image normalization stats."""

    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "image_mean": np.asarray(image_mean, dtype=np.float32),
            "image_std": np.asarray(image_std, dtype=np.float32),
            "image_shape": tuple(image_shape or default_image_shape()),
            "metadata": metadata or {},
        },
        file_path,
    )


def load_policy_checkpoint(
    file_path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[BehaviorCloningCNN, np.ndarray, np.ndarray, tuple[int, int, int], dict]:
    """Load a trained image policy and its preprocessing metadata."""

    checkpoint = torch.load(file_path, map_location=device, weights_only=False)
    image_shape = tuple(checkpoint.get("image_shape") or default_image_shape())
    model = BehaviorCloningCNN(input_channels=int(image_shape[2]))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    image_mean = np.asarray(checkpoint["image_mean"], dtype=np.float32)
    image_std = np.asarray(checkpoint["image_std"], dtype=np.float32)
    metadata = dict(checkpoint.get("metadata", {}))
    return model, image_mean, image_std, image_shape, metadata


class BehaviorCloningAgent:
    """Thin image-policy inference wrapper used by the SOFA controller."""

    def __init__(
        self,
        model: BehaviorCloningCNN,
        image_mean: np.ndarray,
        image_std: np.ndarray,
        image_shape: tuple[int, int, int] | None = None,
        device: str | torch.device = "cpu",
    ):
        self.model = model
        self.image_mean = np.asarray(image_mean, dtype=np.float32)
        self.image_std = np.asarray(image_std, dtype=np.float32)
        self.image_shape = tuple(image_shape or default_image_shape())
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def from_checkpoint(
        cls,
        file_path: str | Path,
        device: str | torch.device = "cpu",
    ) -> "BehaviorCloningAgent":
        model, image_mean, image_std, image_shape, _metadata = load_policy_checkpoint(file_path, device=device)
        return cls(
            model=model,
            image_mean=image_mean,
            image_std=image_std,
            image_shape=image_shape,
            device=device,
        )

    def predict(self, observation: np.ndarray) -> np.ndarray:
        """Predict a single action from one RGB image observation."""

        image = np.asarray(observation, dtype=np.float32)
        if image.ndim != 3:
            raise ValueError(f"Expected HWC image observation, got shape {image.shape}")
        if tuple(image.shape) != self.image_shape:
            raise ValueError(f"Expected image shape {self.image_shape}, got {tuple(image.shape)}")
        if float(np.max(image)) > 1.5:
            image = image / 255.0
        image = (image - self.image_mean.reshape(1, 1, -1)) / self.image_std.reshape(1, 1, -1)
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        with torch.inference_mode():
            tensor = torch.from_numpy(image).to(self.device).unsqueeze(0)
            action = self.model(tensor).cpu().numpy()[0]
        return action.astype(np.float32)
