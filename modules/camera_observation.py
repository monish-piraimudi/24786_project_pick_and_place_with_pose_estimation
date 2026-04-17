"""Helpers for synthetic image observations used by the IL pipeline."""

from __future__ import annotations

import math

import numpy as np


DEFAULT_IMAGE_HEIGHT = 96
DEFAULT_IMAGE_WIDTH = 96
DEFAULT_IMAGE_CHANNELS = 3
DEFAULT_WORKSPACE_BOUNDS_MM = (-95.0, 95.0, -95.0, 95.0)


def default_image_shape() -> tuple[int, int, int]:
    return (DEFAULT_IMAGE_HEIGHT, DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_CHANNELS)


def _world_to_pixel(
    x_mm: float,
    z_mm: float,
    *,
    width: int,
    height: int,
    workspace_bounds_mm: tuple[float, float, float, float],
) -> tuple[float, float]:
    x_min, x_max, z_min, z_max = workspace_bounds_mm
    x_norm = (float(x_mm) - x_min) / max(1e-6, x_max - x_min)
    z_norm = (float(z_mm) - z_min) / max(1e-6, z_max - z_min)
    px = x_norm * (width - 1)
    py = (1.0 - z_norm) * (height - 1)
    return px, py


def _draw_disc(image: np.ndarray, center_xy: tuple[float, float], radius_px: float, color: tuple[int, int, int]) -> None:
    height, width = image.shape[:2]
    cx, cy = center_xy
    yy, xx = np.ogrid[:height, :width]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius_px**2
    image[mask] = color


def _draw_ring(
    image: np.ndarray,
    center_xy: tuple[float, float],
    radius_px: float,
    thickness_px: float,
    color: tuple[int, int, int],
) -> None:
    height, width = image.shape[:2]
    cx, cy = center_xy
    yy, xx = np.ogrid[:height, :width]
    dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
    inner = max(0.0, radius_px - thickness_px)
    mask = (dist2 <= radius_px**2) & (dist2 >= inner**2)
    image[mask] = color


def _draw_rotated_square(
    image: np.ndarray,
    center_xy: tuple[float, float],
    half_extent_px: float,
    yaw_deg: float,
    color: tuple[int, int, int],
) -> None:
    height, width = image.shape[:2]
    cx, cy = center_xy
    radians = math.radians(float(yaw_deg))
    cos_yaw = math.cos(radians)
    sin_yaw = math.sin(radians)

    yy, xx = np.ogrid[:height, :width]
    x = xx - cx
    y = yy - cy
    local_x = x * cos_yaw + y * sin_yaw
    local_y = -x * sin_yaw + y * cos_yaw
    mask = (np.abs(local_x) <= half_extent_px) & (np.abs(local_y) <= half_extent_px)
    image[mask] = color


def _draw_grid(image: np.ndarray, spacing_px: int = 12) -> None:
    image[::spacing_px, :, :] = np.maximum(image[::spacing_px, :, :], np.array([36, 42, 48], dtype=np.uint8))
    image[:, ::spacing_px, :] = np.maximum(image[:, ::spacing_px, :], np.array([36, 42, 48], dtype=np.uint8))


def _draw_phase_bar(image: np.ndarray, phase_index: int, num_phases: int) -> None:
    height, width = image.shape[:2]
    bar_height = max(6, height // 10)
    strip = image[height - bar_height : height]
    strip[:] = np.maximum(strip, np.array([18, 20, 24], dtype=np.uint8))

    gap = 2
    cell_width = max(4, (width - gap * (num_phases + 1)) // num_phases)
    for index in range(num_phases):
        x0 = gap + index * (cell_width + gap)
        x1 = min(width - gap, x0 + cell_width)
        color = np.array([235, 140, 58], dtype=np.uint8) if index == phase_index else np.array([78, 84, 92], dtype=np.uint8)
        strip[1:-1, x0:x1, :] = color


def render_pick_place_image(
    *,
    tcp_position: np.ndarray,
    cube_position: np.ndarray,
    target_position: np.ndarray,
    tip_positions: np.ndarray | None,
    cube_yaw_deg: float,
    phase_index: int,
    num_phases: int,
    image_shape: tuple[int, int, int] = default_image_shape(),
    workspace_bounds_mm: tuple[float, float, float, float] = DEFAULT_WORKSPACE_BOUNDS_MM,
) -> np.ndarray:
    """Render a compact top-down RGB observation for the policy."""

    height, width, channels = image_shape
    if channels != 3:
        raise ValueError(f"Expected 3 channels for RGB policy input, got {channels}")

    image = np.zeros((height, width, channels), dtype=np.uint8)
    image[:] = np.array([12, 16, 21], dtype=np.uint8)
    _draw_grid(image)

    target_xy = _world_to_pixel(
        target_position[0],
        target_position[2],
        width=width,
        height=height,
        workspace_bounds_mm=workspace_bounds_mm,
    )
    cube_xy = _world_to_pixel(
        cube_position[0],
        cube_position[2],
        width=width,
        height=height,
        workspace_bounds_mm=workspace_bounds_mm,
    )
    tcp_xy = _world_to_pixel(
        tcp_position[0],
        tcp_position[2],
        width=width,
        height=height,
        workspace_bounds_mm=workspace_bounds_mm,
    )

    _draw_ring(image, target_xy, radius_px=max(6.0, width * 0.08), thickness_px=2.5, color=(74, 201, 112))
    _draw_rotated_square(
        image,
        cube_xy,
        half_extent_px=max(5.0, width * 0.055),
        yaw_deg=cube_yaw_deg,
        color=(230, 232, 236),
    )

    if tip_positions is not None:
        for tip_position in np.asarray(tip_positions, dtype=np.float32):
            tip_xy = _world_to_pixel(
                tip_position[0],
                tip_position[2],
                width=width,
                height=height,
                workspace_bounds_mm=workspace_bounds_mm,
            )
            _draw_disc(image, tip_xy, radius_px=2.5, color=(247, 208, 91))

    _draw_disc(image, tcp_xy, radius_px=3.0, color=(224, 72, 72))
    _draw_phase_bar(image, phase_index=phase_index, num_phases=num_phases)
    return image
