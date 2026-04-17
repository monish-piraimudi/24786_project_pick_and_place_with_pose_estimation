"""Simulation-only RGB observation helpers using an Emio-like camera view."""

from __future__ import annotations

import importlib.util
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from modules.camera_observation import default_image_shape

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

try:
    import pygame
except ImportError:  # pragma: no cover - exercised only when pygame is unavailable
    pygame = None

try:
    from OpenGL import GL, GLU
except ImportError:  # pragma: no cover - exercised only when PyOpenGL is unavailable
    GL = None
    GLU = None


ASSETS_DIR = Path(__file__).resolve().parents[3]
PARAMETERS_PATH = ASSETS_DIR / "parameters.py"
DEFAULT_CAMERA_TRANSLATION = np.array([147.0, 5.0], dtype=np.float32)
DEFAULT_CAMERA_FORWARD_AXIS = np.array([1.0, 0.0, 0.0], dtype=np.float32)
DEFAULT_CAMERA_UP_AXIS = np.array([0.0, 0.0, -1.0], dtype=np.float32)
CUBE_HALF_EXTENT_MM = 5.0
TCP_MARKER_SIZE_MM = 3.0
TIP_MARKER_SIZE_MM = 2.5
TARGET_RING_INNER_MM = 7.0
TARGET_RING_OUTER_MM = 10.0


def _load_camera_translation() -> np.ndarray:
    if not PARAMETERS_PATH.is_file():
        return DEFAULT_CAMERA_TRANSLATION.copy()
    spec = importlib.util.spec_from_file_location("_emio_parameters", PARAMETERS_PATH)
    if spec is None or spec.loader is None:
        return DEFAULT_CAMERA_TRANSLATION.copy()
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    values = getattr(module, "cameraTranslation", DEFAULT_CAMERA_TRANSLATION)
    return np.asarray(values, dtype=np.float32).reshape(2)


def _resize_frame_nearest(frame: np.ndarray, image_shape: tuple[int, int, int]) -> np.ndarray:
    frame = np.asarray(frame)
    if frame.ndim != 3:
        raise ValueError(f"Expected HWC frame, got shape {frame.shape}")

    out_h, out_w, out_c = image_shape
    if frame.shape[2] != out_c:
        raise ValueError(f"Expected {out_c} channels, got frame shape {frame.shape}")

    in_h, in_w = frame.shape[:2]
    if (in_h, in_w) == (out_h, out_w):
        return frame.astype(np.uint8, copy=False)

    y_idx = np.clip(np.round(np.linspace(0, in_h - 1, out_h)).astype(np.int32), 0, in_h - 1)
    x_idx = np.clip(np.round(np.linspace(0, in_w - 1, out_w)).astype(np.int32), 0, in_w - 1)
    resized = frame[y_idx][:, x_idx]
    return resized.astype(np.uint8, copy=False)


def _crop_frame(frame: np.ndarray, crop_norm_xywh: tuple[float, float, float, float]) -> np.ndarray:
    frame = np.asarray(frame, dtype=np.uint8)
    height, width = frame.shape[:2]
    x0, y0, w_norm, h_norm = [float(value) for value in crop_norm_xywh]

    x0_px = int(np.clip(np.floor(x0 * width), 0, width - 1))
    y0_px = int(np.clip(np.floor(y0 * height), 0, height - 1))
    x1_px = int(np.clip(np.ceil((x0 + w_norm) * width), x0_px + 1, width))
    y1_px = int(np.clip(np.ceil((y0 + h_norm) * height), y0_px + 1, height))
    return frame[y0_px:y1_px, x0_px:x1_px]


def _as_array(value) -> np.ndarray:
    if hasattr(value, "value"):
        value = value.value
    return np.asarray(value, dtype=np.float32)


def _rigid_xyz(mechanical_object) -> np.ndarray:
    return _as_array(mechanical_object.position).reshape(-1)[:3].astype(np.float32)


def _rigid_pose(mechanical_object) -> np.ndarray:
    return _as_array(mechanical_object.position).reshape(-1).astype(np.float32)


def _gripper_tips(gripper_state) -> np.ndarray:
    return _as_array(gripper_state.position).reshape(-1, 3).astype(np.float32)


def _axis_angle_to_quaternion(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float32)
    axis = axis / max(1e-6, float(np.linalg.norm(axis)))
    half_angle = 0.5 * float(angle_rad)
    sin_half = math.sin(half_angle)
    return np.asarray(
        [axis[0] * sin_half, axis[1] * sin_half, axis[2] * sin_half, math.cos(half_angle)],
        dtype=np.float32,
    )


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = np.asarray(q1, dtype=np.float32)
    x2, y2, z2, w2 = np.asarray(q2, dtype=np.float32)
    return np.asarray(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=np.float32,
    )


def quaternion_conjugate(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float32)
    return np.asarray([-quaternion[0], -quaternion[1], -quaternion[2], quaternion[3]], dtype=np.float32)


def rotate_vector(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float32)
    v = np.asarray(vector, dtype=np.float32)
    qv = np.asarray([v[0], v[1], v[2], 0.0], dtype=np.float32)
    return quaternion_multiply(quaternion_multiply(q, qv), quaternion_conjugate(q))[:3]


def quaternion_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(quaternion, dtype=np.float32)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.asarray(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float32,
    )


def _euler_offset_quaternion(rotation_offset_deg: tuple[float, float, float]) -> np.ndarray:
    rx, ry, rz = [math.radians(float(value)) for value in rotation_offset_deg]
    q = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    for axis, angle in (((1.0, 0.0, 0.0), rx), ((0.0, 1.0, 0.0), ry), ((0.0, 0.0, 1.0), rz)):
        if abs(angle) <= 1e-8:
            continue
        q = quaternion_multiply(_axis_angle_to_quaternion(np.asarray(axis, dtype=np.float32), angle), q)
    return q


def compute_emio_camera_pose(
    *,
    extended: bool = True,
    translation_offset_mm: tuple[float, float, float] = (0.0, 0.0, 0.0),
    rotation_offset_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[np.ndarray, np.ndarray]:
    camera_translation = _load_camera_translation()
    xz_distance = float(camera_translation[0])
    y_offset = float(camera_translation[1])
    diagonal = math.cos(math.pi / 4.0) * xz_distance
    position = np.asarray(
        [
            -diagonal,
            -y_offset if extended else y_offset,
            -diagonal,
        ],
        dtype=np.float32,
    )
    position = position + np.asarray(translation_offset_mm, dtype=np.float32).reshape(3)

    base_orientation = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    rotations = (
        _axis_angle_to_quaternion(np.asarray([0.0, 1.0, 0.0], dtype=np.float32), -math.pi / 4.0),
        _axis_angle_to_quaternion(
            np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
            math.pi / 4.0 if extended else 3.0 * math.pi / 4.0,
        ),
        _axis_angle_to_quaternion(np.asarray([1.0, 0.0, 0.0], dtype=np.float32), math.pi / 2.0),
    )
    for rotation in rotations:
        base_orientation = quaternion_multiply(rotation, base_orientation)

    orientation = quaternion_multiply(_euler_offset_quaternion(rotation_offset_deg), base_orientation)
    orientation = orientation / max(1e-6, float(np.linalg.norm(orientation)))
    return position.astype(np.float32), orientation.astype(np.float32)


def camera_forward_up_vectors(orientation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    forward = rotate_vector(orientation, DEFAULT_CAMERA_FORWARD_AXIS)
    up = rotate_vector(orientation, DEFAULT_CAMERA_UP_AXIS)
    forward = forward / max(1e-6, float(np.linalg.norm(forward)))
    up = up / max(1e-6, float(np.linalg.norm(up)))
    return forward.astype(np.float32), up.astype(np.float32)


@dataclass
class SimEmioCameraConfig:
    image_shape: tuple[int, int, int] = default_image_shape()
    render_shape: tuple[int, int, int] = (256, 256, 3)
    extended: bool = True
    fov_deg: float = 45.0
    translation_offset_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation_offset_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    crop_norm_xywh: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)


class SimEmioCameraObservationSource:
    """Offscreen simulation-only RGB renderer aligned to the Emio camera pose."""

    def __init__(self, handles: dict, config: SimEmioCameraConfig | None = None):
        self.handles = handles
        self.config = config or SimEmioCameraConfig()
        self._opened = False

    @property
    def is_open(self) -> bool:
        return bool(self._opened)

    def open(self) -> bool:
        render_h, render_w, render_c = self.config.render_shape
        if render_c != 3:
            raise ValueError(f"SimEmioCamera render_shape must have 3 channels, got {self.config.render_shape}")
        _ensure_offscreen_gl_context(render_w, render_h)
        self._opened = True
        return True

    def close(self) -> None:
        self._opened = False

    def update(self) -> np.ndarray:
        if not self._opened:
            raise RuntimeError("SimEmioCameraObservationSource.update() called before open().")

        render_h, render_w, _render_c = self.config.render_shape
        _ensure_offscreen_gl_context(render_w, render_h)
        camera_position, camera_orientation = compute_emio_camera_pose(
            extended=self.config.extended,
            translation_offset_mm=self.config.translation_offset_mm,
            rotation_offset_deg=self.config.rotation_offset_deg,
        )
        forward, up = camera_forward_up_vectors(camera_orientation)
        look_at = camera_position + forward

        _begin_frame(render_w, render_h, self.config.fov_deg, camera_position, look_at, up)
        self._draw_scene()
        frame = _read_framebuffer(render_w, render_h)
        cropped = _crop_frame(frame, self.config.crop_norm_xywh)
        return _resize_frame_nearest(cropped, self.config.image_shape)

    def _draw_scene(self) -> None:
        tuning = self.handles["tuning"]
        block_pose = _rigid_pose(self.handles["block_mo"])
        cube_center = block_pose[:3]
        cube_orientation = block_pose[3:7] if block_pose.shape[0] >= 7 else np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        target_position = np.asarray(tuning["place_position"], dtype=np.float32)
        tcp_position = _rigid_xyz(self.handles["tcp_mo"])
        tip_positions = _gripper_tips(self.handles["gripper_state"])

        workspace_y = float(np.asarray(tuning["object_position"], dtype=np.float32)[1] - 2.0)
        _draw_workspace_plane(workspace_y)
        _draw_target_ring(target_position, workspace_y + 0.5)
        _draw_box(
            cube_center,
            np.asarray([2.0 * CUBE_HALF_EXTENT_MM, 2.0 * CUBE_HALF_EXTENT_MM, 2.0 * CUBE_HALF_EXTENT_MM], dtype=np.float32),
            cube_orientation,
            color=(0.90, 0.92, 0.95),
        )
        _draw_box(
            tcp_position,
            np.asarray([2.0 * TCP_MARKER_SIZE_MM] * 3, dtype=np.float32),
            np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            color=(0.88, 0.28, 0.24),
        )
        for tip_position in tip_positions:
            _draw_box(
                tip_position,
                np.asarray([2.0 * TIP_MARKER_SIZE_MM] * 3, dtype=np.float32),
                np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
                color=(0.96, 0.80, 0.36),
            )


_CONTEXT_SIZE: tuple[int, int] | None = None


def _ensure_offscreen_gl_context(width: int, height: int) -> None:
    global _CONTEXT_SIZE

    if pygame is None or GL is None or GLU is None:
        raise RuntimeError(
            "Synthetic Emio-camera observations require pygame and PyOpenGL with GL support."
        )

    if _CONTEXT_SIZE == (width, height):
        return

    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    if "SDL_VIDEODRIVER" not in os.environ and not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        os.environ["SDL_VIDEODRIVER"] = "offscreen"

    pygame.display.quit()
    if not pygame.get_init():
        pygame.init()

    last_error = None
    drivers = [os.environ.get("SDL_VIDEODRIVER")] if os.environ.get("SDL_VIDEODRIVER") else ["offscreen", "x11", "wayland"]
    for driver in drivers:
        try:
            if driver:
                os.environ["SDL_VIDEODRIVER"] = driver
            pygame.display.quit()
            pygame.display.init()
            pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 2)
            pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 1)
            pygame.display.set_mode((width, height), pygame.OPENGL | pygame.HIDDEN)
            _CONTEXT_SIZE = (width, height)
            return
        except Exception as exc:  # pragma: no cover - driver fallback
            last_error = exc

    raise RuntimeError(
        "Synthetic Emio-camera observations require an offscreen OpenGL-capable SDL driver."
    ) from last_error


def _begin_frame(
    width: int,
    height: int,
    fov_deg: float,
    camera_position: np.ndarray,
    look_at: np.ndarray,
    up: np.ndarray,
) -> None:
    GL.glViewport(0, 0, int(width), int(height))
    GL.glEnable(GL.GL_DEPTH_TEST)
    GL.glDisable(GL.GL_CULL_FACE)
    GL.glClearColor(0.11, 0.12, 0.14, 1.0)
    GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

    GL.glMatrixMode(GL.GL_PROJECTION)
    GL.glLoadIdentity()
    GLU.gluPerspective(float(fov_deg), float(width) / max(1.0, float(height)), 5.0, 1000.0)

    GL.glMatrixMode(GL.GL_MODELVIEW)
    GL.glLoadIdentity()
    GLU.gluLookAt(
        float(camera_position[0]),
        float(camera_position[1]),
        float(camera_position[2]),
        float(look_at[0]),
        float(look_at[1]),
        float(look_at[2]),
        float(up[0]),
        float(up[1]),
        float(up[2]),
    )


def _read_framebuffer(width: int, height: int) -> np.ndarray:
    GL.glFinish()
    pixels = GL.glReadPixels(0, 0, int(width), int(height), GL.GL_RGB, GL.GL_UNSIGNED_BYTE)
    image = np.frombuffer(pixels, dtype=np.uint8).reshape(int(height), int(width), 3)
    return np.flipud(image).copy()


def _draw_workspace_plane(y_level: float) -> None:
    extent = 120.0
    GL.glColor3f(0.22, 0.25, 0.28)
    GL.glBegin(GL.GL_QUADS)
    GL.glVertex3f(-extent, y_level, -extent)
    GL.glVertex3f(extent, y_level, -extent)
    GL.glVertex3f(extent, y_level, extent)
    GL.glVertex3f(-extent, y_level, extent)
    GL.glEnd()

    GL.glColor3f(0.30, 0.34, 0.38)
    GL.glBegin(GL.GL_LINES)
    for value in range(-100, 101, 20):
        GL.glVertex3f(float(value), y_level + 0.1, -extent)
        GL.glVertex3f(float(value), y_level + 0.1, extent)
        GL.glVertex3f(-extent, y_level + 0.1, float(value))
        GL.glVertex3f(extent, y_level + 0.1, float(value))
    GL.glEnd()


def _draw_target_ring(center: np.ndarray, y_level: float) -> None:
    center = np.asarray(center, dtype=np.float32)
    segments = 40
    GL.glColor3f(0.29, 0.79, 0.44)
    GL.glBegin(GL.GL_TRIANGLE_STRIP)
    for index in range(segments + 1):
        theta = 2.0 * math.pi * float(index) / float(segments)
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        GL.glVertex3f(
            float(center[0] + TARGET_RING_OUTER_MM * cos_t),
            float(y_level),
            float(center[2] + TARGET_RING_OUTER_MM * sin_t),
        )
        GL.glVertex3f(
            float(center[0] + TARGET_RING_INNER_MM * cos_t),
            float(y_level),
            float(center[2] + TARGET_RING_INNER_MM * sin_t),
        )
    GL.glEnd()


def _draw_box(center: np.ndarray, size_xyz: np.ndarray, quaternion: np.ndarray, *, color: tuple[float, float, float]) -> None:
    center = np.asarray(center, dtype=np.float32)
    half_sizes = 0.5 * np.asarray(size_xyz, dtype=np.float32)
    rotation = quaternion_to_matrix(quaternion)
    corners = np.asarray(
        [
            [-half_sizes[0], -half_sizes[1], -half_sizes[2]],
            [half_sizes[0], -half_sizes[1], -half_sizes[2]],
            [half_sizes[0], half_sizes[1], -half_sizes[2]],
            [-half_sizes[0], half_sizes[1], -half_sizes[2]],
            [-half_sizes[0], -half_sizes[1], half_sizes[2]],
            [half_sizes[0], -half_sizes[1], half_sizes[2]],
            [half_sizes[0], half_sizes[1], half_sizes[2]],
            [-half_sizes[0], half_sizes[1], half_sizes[2]],
        ],
        dtype=np.float32,
    )
    world_corners = corners @ rotation.T + center.reshape(1, 3)
    faces = (
        (0, 1, 2, 3),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (2, 3, 7, 6),
        (1, 2, 6, 5),
        (0, 3, 7, 4),
    )
    GL.glColor3f(*color)
    GL.glBegin(GL.GL_QUADS)
    for face in faces:
        for index in face:
            vertex = world_corners[index]
            GL.glVertex3f(float(vertex[0]), float(vertex[1]), float(vertex[2]))
    GL.glEnd()
