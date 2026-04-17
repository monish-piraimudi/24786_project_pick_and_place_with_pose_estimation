"""SOFA-safe real RGB observation helpers built on the Emio Python API."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from modules.camera_observation import default_image_shape

try:
    from emioapi import EmioCamera
except ImportError:  # pragma: no cover - exercised only in environments without emioapi
    EmioCamera = None


def _resize_frame_nearest(frame: np.ndarray, image_shape: tuple[int, int, int]) -> np.ndarray:
    frame = np.asarray(frame)
    if frame.ndim != 3:
        raise ValueError(f"Expected HWC frame, got shape {frame.shape}")

    out_h, out_w, out_c = image_shape
    if frame.shape[2] == 1 and out_c == 3:
        frame = np.repeat(frame, 3, axis=2)
    if frame.shape[2] != out_c:
        raise ValueError(f"Expected {out_c} channels, got frame shape {frame.shape}")

    in_h, in_w = frame.shape[:2]
    if (in_h, in_w) == (out_h, out_w):
        return frame.astype(np.uint8, copy=False)

    y_idx = np.clip(np.round(np.linspace(0, in_h - 1, out_h)).astype(np.int32), 0, in_h - 1)
    x_idx = np.clip(np.round(np.linspace(0, in_w - 1, out_w)).astype(np.int32), 0, in_w - 1)
    resized = frame[y_idx][:, x_idx]
    return resized.astype(np.uint8, copy=False)


def _as_tracker_array(trackers_pos) -> np.ndarray:
    if trackers_pos is None:
        return np.zeros((0, 3), dtype=np.float32)
    values = np.asarray(trackers_pos, dtype=np.float32)
    if values.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    return values.reshape(-1, 3).astype(np.float32)


@dataclass
class EmioCameraConfig:
    image_shape: tuple[int, int, int] = default_image_shape()
    show: bool = False
    track_markers: bool = True
    compute_point_cloud: bool = False
    configuration: str = "extended"
    camera_serial: str | None = None


class EmioCameraObservationSource:
    """Wrapper around `EmioCamera` for RGB observations and marker tracking."""

    def __init__(self, config: EmioCameraConfig | None = None):
        self.config = config or EmioCameraConfig()
        self._camera = None
        self._opened = False

    @property
    def is_open(self) -> bool:
        return bool(self._opened)

    def open(self) -> bool:
        if self._opened:
            return True
        if EmioCamera is None:
            raise RuntimeError(
                "emioapi is required for real RGB observations. "
                "This SOFA-safe pipeline expects EmioCamera from emioapi."
            )

        self._camera = EmioCamera(
            show=self.config.show,
            track_markers=self.config.track_markers,
            compute_point_cloud=self.config.compute_point_cloud,
            configuration=self.config.configuration,
        )
        self._opened = bool(self._camera.open(camera_serial=self.config.camera_serial))
        return self._opened

    def close(self) -> None:
        if self._camera is not None:
            try:
                self._camera.close()
            finally:
                self._opened = False
                self._camera = None

    def update(self) -> tuple[np.ndarray, np.ndarray]:
        if not self._opened or self._camera is None:
            raise RuntimeError("EmioCameraObservationSource.update() called before open().")

        self._camera.update()
        frame = getattr(self._camera, "frame", None)
        if frame is None:
            raise RuntimeError("EmioCamera did not provide an RGB frame.")
        rgb = _resize_frame_nearest(np.asarray(frame, dtype=np.uint8), self.config.image_shape)
        trackers = _as_tracker_array(getattr(self._camera, "trackers_pos", None))
        return rgb, trackers
