"""Shared helpers for scripted and interactive policy rollouts."""

from __future__ import annotations

import numpy as np

from modules.camera_observation import render_pick_place_image


PHASE_NAMES = [
    "approach_pick",
    "descend_pick",
    "close_gripper",
    "lift",
    "approach_place",
    "descend_place",
    "open_gripper",
    "retreat",
]
PHASE_INDEX = {name: index for index, name in enumerate(PHASE_NAMES)}

ACTION_DIM = 4
ACTION_LIMIT_MM = 5.0
POSE_TOLERANCE_MM = 8.0
CLOSE_GRIPPER_TIMEOUT_STEPS = 18
OPEN_GRIPPER_TIMEOUT_STEPS = 10
MAX_EPISODE_STEPS = 360


def as_array(value) -> np.ndarray:
    if hasattr(value, "value"):
        value = value.value
    return np.asarray(value, dtype=np.float32)


def rigid_xyz(mechanical_object) -> np.ndarray:
    return as_array(mechanical_object.position).reshape(-1)[:3].astype(np.float32)


def rigid_pose(mechanical_object) -> np.ndarray:
    return as_array(mechanical_object.position).reshape(-1).astype(np.float32)


def gripper_tips(gripper_state) -> np.ndarray:
    return as_array(gripper_state.position).reshape(-1, 3).astype(np.float32)


def quaternion_to_yaw_degrees(quaternion: np.ndarray) -> float:
    quaternion = np.asarray(quaternion, dtype=np.float32)
    sin_half = float(np.clip(quaternion[1], -1.0, 1.0))
    return float(np.degrees(2.0 * np.arcsin(sin_half)))


def clip_delta(delta: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(delta, dtype=np.float32), -ACTION_LIMIT_MM, ACTION_LIMIT_MM)


def phase_target(demo, phase: str) -> np.ndarray:
    pick_position = np.asarray(demo.pick_position, dtype=np.float32)
    place_position = np.asarray(demo.place_position, dtype=np.float32)

    lookup = {
        "approach_pick": pick_position + np.array([0.0, demo.hover_lift_height, 0.0], dtype=np.float32),
        "descend_pick": pick_position + np.array([0.0, demo.pick_height_offset, 0.0], dtype=np.float32),
        "close_gripper": pick_position + np.array([0.0, demo.pick_height_offset, 0.0], dtype=np.float32),
        "lift": pick_position + np.array([0.0, demo.hover_lift_height, 0.0], dtype=np.float32),
        "approach_place": place_position + np.array([0.0, demo.hover_lift_height, 0.0], dtype=np.float32),
        "descend_place": place_position + np.array([0.0, demo.place_height_offset, 0.0], dtype=np.float32),
        "open_gripper": place_position + np.array([0.0, demo.place_height_offset, 0.0], dtype=np.float32),
        "retreat": place_position + np.array([0.0, demo.hover_lift_height, 0.0], dtype=np.float32),
    }
    return lookup[phase]


def phase_opening(demo, phase: str) -> float:
    if phase in {"close_gripper", "lift", "approach_place", "descend_place"}:
        return float(demo.opening_closed)
    return float(demo.opening_open)


def phase_from_elapsed(demo, elapsed: float) -> str | None:
    if elapsed < 2.0:
        return "approach_pick"
    if elapsed < 4.0:
        return "descend_pick"
    if elapsed < 6.0:
        return "close_gripper"
    if elapsed < 8.0:
        return "lift"
    if elapsed < 10.0:
        return "approach_place"
    if elapsed < 12.0:
        return "descend_place"
    if elapsed < 14.0:
        return "open_gripper"
    if elapsed < float(demo.demo_duration):
        return "retreat"
    return None


def opening_to_command(demo, opening_mm: float) -> float:
    opening_open = float(demo.opening_open)
    opening_closed = float(demo.opening_closed)
    opening_mm = float(np.clip(opening_mm, min(opening_open, opening_closed), max(opening_open, opening_closed)))
    if abs(opening_closed - opening_open) < 1e-6:
        return 0.0
    return float(np.clip((opening_mm - opening_open) / (opening_closed - opening_open), 0.0, 1.0))


def command_to_opening(demo, command: float) -> float:
    command = float(np.clip(command, 0.0, 1.0))
    return float(demo.opening_open + command * (demo.opening_closed - demo.opening_open))


def at_waypoint(handles: dict, phase: str) -> bool:
    tcp_position = rigid_xyz(handles["tcp_mo"])
    return float(np.linalg.norm(phase_target(handles["demo"], phase) - tcp_position)) <= POSE_TOLERANCE_MM


def advance_phase(handles: dict, phase: str, phase_step: int, close_counter: int) -> tuple[str, int, int, bool]:
    evaluator = handles["evaluator"]
    next_phase = phase
    next_phase_step = phase_step
    next_close_counter = close_counter
    done = False

    if phase == "approach_pick" and at_waypoint(handles, phase):
        next_phase = "descend_pick"
    elif phase == "descend_pick" and at_waypoint(handles, phase):
        next_phase = "close_gripper"
    elif phase == "close_gripper":
        next_close_counter += 1
        if evaluator.is_attached or next_close_counter >= CLOSE_GRIPPER_TIMEOUT_STEPS:
            next_phase = "lift"
    elif phase == "lift" and (evaluator.lifted or at_waypoint(handles, phase)):
        next_phase = "approach_place"
    elif phase == "approach_place" and at_waypoint(handles, phase):
        next_phase = "descend_place"
    elif phase == "descend_place" and at_waypoint(handles, phase):
        next_phase = "open_gripper"
    elif phase == "open_gripper" and (
        (not evaluator.is_attached) or phase_step >= OPEN_GRIPPER_TIMEOUT_STEPS
    ):
        next_phase = "retreat"
    elif phase == "retreat" and at_waypoint(handles, phase) and (
        evaluator.placed or phase_step >= OPEN_GRIPPER_TIMEOUT_STEPS
    ):
        done = True

    if next_phase != phase:
        next_phase_step = 0
        if next_phase == "close_gripper":
            next_close_counter = 0

    if evaluator.placed and next_phase == "retreat":
        done = True

    return next_phase, next_phase_step, next_close_counter, done


def render_observation(
    handles: dict,
    phase: str,
    image_shape: tuple[int, int, int],
    cube_position: np.ndarray | None = None,
    cube_yaw_deg: float | None = None,
) -> np.ndarray:
    block_pose = rigid_pose(handles["block_mo"])
    if cube_position is None:
        cube_position = block_pose[:3]
    if cube_yaw_deg is None:
        cube_yaw_deg = quaternion_to_yaw_degrees(block_pose[3:7])
    return render_pick_place_image(
        tcp_position=rigid_xyz(handles["tcp_mo"]),
        cube_position=np.asarray(cube_position, dtype=np.float32),
        target_position=np.asarray(handles["tuning"]["place_position"], dtype=np.float32),
        tip_positions=gripper_tips(handles["gripper_state"]),
        cube_yaw_deg=float(cube_yaw_deg),
        phase_index=PHASE_INDEX[phase],
        num_phases=len(PHASE_NAMES),
        image_shape=image_shape,
    )


def expert_action(handles: dict, phase: str) -> np.ndarray:
    tcp_position = rigid_xyz(handles["tcp_mo"])
    delta = clip_delta(phase_target(handles["demo"], phase) - tcp_position)
    command = opening_to_command(handles["demo"], phase_opening(handles["demo"], phase))
    return np.concatenate([delta, np.array([command], dtype=np.float32)]).astype(np.float32)


def apply_expert_target(handles: dict, phase: str) -> np.ndarray:
    demo = handles["demo"]
    target = phase_target(demo, phase)
    opening = phase_opening(demo, phase)
    demo._set_target_position(target)
    demo._set_gripper_opening(opening)
    return expert_action(handles, phase)


def apply_policy_action(handles: dict, action: np.ndarray) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if action.shape[0] != ACTION_DIM:
        raise ValueError(f"Expected action dim {ACTION_DIM}, got {action.shape[0]}")

    target_mo = handles["target_mo"]
    demo = handles["demo"]
    clipped_delta = clip_delta(action[:3])
    gripper_command = float(np.clip(action[3], 0.0, 1.0))
    current_target = as_array(target_mo.position).reshape(-1)[:3].astype(np.float32)
    next_target = current_target + clipped_delta
    demo._set_target_position(next_target)
    demo._set_gripper_opening(command_to_opening(demo, gripper_command))
    return np.concatenate([clipped_delta, np.array([gripper_command], dtype=np.float32)]).astype(np.float32)
