from __future__ import annotations

import sys
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from modules.camera_observation import default_image_shape
from modules.imitation_data import EpisodeRecorder
from modules.pick_place_policy_shared import (
    ACTION_DIM,
    MAX_EPISODE_STEPS,
    PHASE_INDEX,
    PHASE_NAMES,
    advance_phase,
    apply_policy_action,
    expert_action,
    phase_from_elapsed,
    render_observation,
    rigid_pose,
    rigid_xyz,
)
from modules.sofa_bootstrap import bootstrap_and_validate_sofa


bootstrap_and_validate_sofa()

import Sofa
import SofaRuntime

from modules import pick_place_il as scene

PROJECT_DIR = Path(__file__).resolve().parent.parent

CAMERA_MISSING_TIMEOUT_STEPS = 5


def _resolve_project_path(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    path = Path(path)
    return path if path.is_absolute() else (PROJECT_DIR / path)


@dataclass
class PickPlaceTaskConfig:
    mode: str = "expert"
    policy_path: str | None = None
    output_dir: str | None = None
    episode_id: int = 0
    seed: int = 0
    max_steps: int = MAX_EPISODE_STEPS
    control_dt: float = 0.05
    with_gui: bool = False
    log_episode: bool = False
    save_failed_episodes: bool = False
    connection: bool = False
    camera_tracking: bool = False
    camera_preview: bool = False
    image_shape: tuple[int, int, int] = field(default_factory=default_image_shape)
    task_tuning: dict | None = None
    object_jitter_mm: float = 15.0
    place_jitter_mm: float = 0.0
    cube_marker_offset_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)


def _resolve_task_tuning(overrides: dict | None) -> dict[str, float | np.ndarray]:
    tuning: dict[str, float | np.ndarray] = {}
    for key, value in scene._default_task_tuning().items():
        if isinstance(value, (list, tuple, np.ndarray)):
            tuning[key] = np.asarray(value, dtype=np.float32)
        else:
            tuning[key] = float(value)

    if overrides is None:
        return tuning

    for key, value in overrides.items():
        if isinstance(value, (list, tuple, np.ndarray)):
            tuning[key] = np.asarray(value, dtype=np.float32)
        else:
            tuning[key] = float(value)
    return tuning


def _sample_scene_tuning(config: PickPlaceTaskConfig) -> dict[str, float | np.ndarray]:
    tuning = _resolve_task_tuning(config.task_tuning)
    rng = np.random.default_rng(config.seed)

    object_position = np.asarray(tuning["object_position"], dtype=np.float32).copy()
    place_position = np.asarray(tuning["place_position"], dtype=np.float32).copy()
    pick_offset = np.asarray(tuning["pick_position"], dtype=np.float32) - np.asarray(
        tuning["object_position"], dtype=np.float32
    )

    object_position[[0, 2]] += rng.uniform(
        -float(config.object_jitter_mm),
        float(config.object_jitter_mm),
        size=2,
    ).astype(np.float32)
    place_position[[0, 2]] += rng.uniform(
        -float(config.place_jitter_mm),
        float(config.place_jitter_mm),
        size=2,
    ).astype(np.float32)

    tuning["object_position"] = object_position
    tuning["place_position"] = place_position
    tuning["pick_position"] = object_position + pick_offset
    return tuning


def _scene_tuning_payload(tuning: dict[str, float | np.ndarray]) -> dict[str, float | list[float]]:
    payload = {}
    for key, value in tuning.items():
        if isinstance(value, np.ndarray):
            payload[key] = [float(item) for item in value]
        else:
            payload[key] = float(value)
    return payload


def _camera_cube_state(handles: dict) -> tuple[np.ndarray | None, bool, str]:
    tracker = handles.get("camera_tracker")
    if tracker is None or getattr(tracker, "mo", None) is None:
        return None, False, "camera tracker unavailable"

    tracker_points = np.asarray(tracker.mo.position.value, dtype=np.float32).reshape(-1, 3)
    if tracker_points.shape[0] < 1:
        return None, False, "camera tracker returned no points"

    marker_position = tracker_points[0]
    if not np.all(np.isfinite(marker_position)) or float(np.linalg.norm(marker_position)) <= 1e-3:
        return None, False, "cube marker not visible"

    cube_position = marker_position + np.asarray(handles["cube_marker_offset_mm"], dtype=np.float32)
    return cube_position.astype(np.float32), True, "camera tracking active"


def _resolve_handles(root, tuning: dict[str, float | np.ndarray]) -> dict:
    handles = {
        "root": root,
        "demo": None,
        "evaluator": None,
        "target_mo": None,
        "tcp_mo": None,
        "block_mo": None,
        "gripper_state": None,
        "gripper_opening": None,
        "camera_tracker": None,
        "tuning": tuning,
    }

    if handles["target_mo"] is None:
        handles["target_mo"] = root.Modelling.Target.getMechanicalState()
    if handles["tcp_mo"] is None:
        handles["tcp_mo"] = root.Modelling.TCP.getMechanicalState()
    if handles["block_mo"] is None:
        handles["block_mo"] = root.Modelling.Block.getMechanicalState()
    if handles["demo"] is None:
        handles["demo"] = root.getObject("AutoPickAndPlaceDemo")
    if handles["evaluator"] is None:
        handles["evaluator"] = root.getObject("PickAndPlaceEvaluator")
    if handles["gripper_state"] is None:
        handles["gripper_state"] = root.Simulation.Emio.centerpart.Effector.getMechanicalState()
    if handles["gripper_opening"] is None:
        handles["gripper_opening"] = root.Simulation.Emio.centerpart.Effector.Distance.DistanceMapping.restLengths
    if handles["camera_tracker"] is None:
        handles["camera_tracker"] = root.getObject("DotTracker")

    return handles


def _build_scene(config: PickPlaceTaskConfig) -> tuple[object, dict]:
    tuning = _sample_scene_tuning(config)
    root = Sofa.Core.Node("root")

    SofaRuntime.importPlugin("Sofa.Component")
    SofaRuntime.importPlugin("Sofa.GL.Component")

    argv_backup = list(sys.argv)
    sys.argv = ["pick_place_il.py"]
    if config.connection:
        sys.argv.append("--connection")
    else:
        sys.argv.append("--no-connection")
    if config.camera_tracking:
        sys.argv.append("--camera-tracking")
    else:
        sys.argv.append("--no-camera-tracking")
    if config.camera_preview:
        sys.argv.append("--camera-preview")
    else:
        sys.argv.append("--no-camera-preview")
    sys.argv.extend(
        [
            "--cube-marker-offset-mm",
            str(float(config.cube_marker_offset_mm[0])),
            str(float(config.cube_marker_offset_mm[1])),
            str(float(config.cube_marker_offset_mm[2])),
        ]
    )

    scene.set_runtime_task_tuning(_scene_tuning_payload(tuning))
    try:
        scene.createScene(root)
    finally:
        scene.set_runtime_task_tuning(None)
        sys.argv = argv_backup

    Sofa.Simulation.init(root)
    handles = _resolve_handles(root, tuning)
    handles["cube_marker_offset_mm"] = np.asarray(config.cube_marker_offset_mm, dtype=np.float32)
    if config.mode == "policy":
        handles["demo"].is_active = False
        root.autoDemoActive.value = False
    else:
        handles["demo"].is_active = True
        root.autoDemoActive.value = True
    return root, handles


def run_single_episode(config: PickPlaceTaskConfig) -> dict:
    root, handles = _build_scene(config)
    demo = handles["demo"]
    evaluator = handles["evaluator"]
    camera_tracking_requested = bool(config.camera_tracking)
    camera_tracking_available = False
    camera_status_message = (
        "camera tracking requires --connection"
        if camera_tracking_requested and not config.connection
        else ("camera tracking disabled" if not camera_tracking_requested else "camera tracker unavailable")
    )
    camera_missing_steps = 0
    last_camera_cube_position = None

    policy_agent = None
    if config.mode == "policy":
        if not config.policy_path:
            raise ValueError("policy mode requires a policy checkpoint")
        from modules.imitation_policy import BehaviorCloningAgent

        policy_path = _resolve_project_path(config.policy_path)
        if policy_path is None or not policy_path.is_file():
            raise FileNotFoundError(
                f"Policy checkpoint not found: {policy_path}. "
                "Relative --policy-path values are resolved from this lab folder."
            )
        policy_agent = BehaviorCloningAgent.from_checkpoint(policy_path)

    recorder = None
    if config.log_episode and config.output_dir:
        recorder = EpisodeRecorder(config.output_dir, config.episode_id)

    done = False
    task_step = 0
    dropped_object = False
    saved_path = None
    failure_phase = "not_started"
    initial_object_y = float(np.asarray(handles["tuning"]["object_position"], dtype=np.float32)[1])

    while task_step < config.max_steps and not done:
        if config.mode in {"expert", "collect"}:
            elapsed = float(root.time.value)
            phase = phase_from_elapsed(demo, elapsed)
            if phase is None:
                break
        else:
            if task_step == 0:
                phase = PHASE_NAMES[0]
                phase_step = 0
                close_counter = 0
            phase, phase_step, close_counter, done = advance_phase(
                handles, phase, phase_step, close_counter
            )
            if done:
                break

        camera_cube_position = None
        camera_cube_yaw_deg = None
        if camera_tracking_requested:
            camera_cube_position, camera_step_available, camera_step_status = _camera_cube_state(handles)
            camera_status_message = camera_step_status
            if camera_step_available:
                camera_tracking_available = True
                camera_missing_steps = 0
                last_camera_cube_position = np.asarray(camera_cube_position, dtype=np.float32)
                camera_cube_yaw_deg = 0.0
            else:
                camera_missing_steps += 1
                if (
                    task_step == 0
                    or last_camera_cube_position is None
                    or camera_missing_steps > CAMERA_MISSING_TIMEOUT_STEPS
                ):
                    failure_phase = phase
                    done = True
                    break
                camera_cube_position = np.asarray(last_camera_cube_position, dtype=np.float32)
                camera_cube_yaw_deg = 0.0
                camera_status_message = "cube marker temporarily lost"

        observation = render_observation(
            handles,
            phase,
            config.image_shape,
            cube_position=camera_cube_position,
            cube_yaw_deg=camera_cube_yaw_deg,
        )

        if config.mode in {"expert", "collect"}:
            action = expert_action(handles, phase)
            executed_action = action
        else:
            action = np.asarray(policy_agent.predict(observation), dtype=np.float32)
            executed_action = apply_policy_action(handles, action)

        failure_phase = phase

        if recorder is not None:
            recorder.append(
                observation=observation.astype(np.uint8),
                action=action.astype(np.float32),
                executed_action=executed_action.astype(np.float32),
                phase_index=np.int32(PHASE_INDEX[phase]),
                episode_step=np.int32(task_step),
                cube_pose=rigid_pose(handles["block_mo"]).astype(np.float32),
                object_position=np.asarray(handles["tuning"]["object_position"], dtype=np.float32),
                pick_position=np.asarray(handles["tuning"]["pick_position"], dtype=np.float32),
                target_position=np.asarray(handles["tuning"]["place_position"], dtype=np.float32),
                effector_pose=rigid_pose(handles["tcp_mo"]).astype(np.float32),
                gripper_opening=np.float32(root.commandedOpening.value),
                held_flag=np.int32(int(evaluator.is_attached or evaluator.lifted)),
                task_score=np.float32(root.taskScore.value),
                task_lifted=np.int32(int(root.taskLifted.value)),
                task_placed=np.int32(int(root.taskPlaced.value)),
                pick_success=np.int32(int(root.taskLifted.value)),
                place_success=np.int32(int(root.taskPlaced.value)),
                total_success=np.int32(int(root.taskLifted.value and root.taskPlaced.value)),
            )

        Sofa.Simulation.animate(root, config.control_dt)
        task_step += 1
        if config.mode == "policy":
            phase_step += 1
        elif not demo.is_active:
            done = True

        block_position = rigid_xyz(handles["block_mo"])
        if root.taskLifted.value and not root.taskPlaced.value:
            if (not evaluator.is_attached) and (not evaluator.is_falling):
                if float(block_position[1]) <= initial_object_y + 1.0 and phase in {
                    "approach_place",
                    "descend_place",
                    "open_gripper",
                    "retreat",
                }:
                    dropped_object = True

    pick_success = bool(root.taskLifted.value)
    place_success = bool(root.taskPlaced.value)
    total_success = bool(pick_success and place_success)
    if recorder is not None and (config.save_failed_episodes or total_success):
        saved_path = recorder.save()

    final_place_error = float(
        np.linalg.norm(
            rigid_xyz(handles["block_mo"]) - np.asarray(handles["tuning"]["place_position"], dtype=np.float32)
        )
    )

    return {
        "episode_id": config.episode_id,
        "seed": config.seed,
        "mode": config.mode,
        "pick_success": pick_success,
        "place_success": place_success,
        "total_success": total_success,
        "task_score": float(root.taskScore.value),
        "final_place_error_mm": final_place_error,
        "dropped_object": bool(dropped_object and not total_success),
        "num_steps": task_step,
        "failure_phase": "completed" if total_success else failure_phase,
        "saved_path": str(saved_path) if saved_path else None,
        "connection_requested": bool(config.connection),
        "camera_tracking_requested": camera_tracking_requested,
        "camera_tracking_available": bool(camera_tracking_available),
        "camera_status_message": camera_status_message,
    }


def run_episode_batch(
    base_config: PickPlaceTaskConfig,
    num_episodes: int,
    start_seed: int = 0,
    successful_only: bool = False,
    max_attempts: int | None = None,
) -> list[dict]:
    summaries = []
    attempts = 0
    successes = 0
    max_attempts = max_attempts or num_episodes

    while attempts < max_attempts and (successes < num_episodes if successful_only else attempts < num_episodes):
        config = replace(base_config, seed=start_seed + attempts, episode_id=attempts)
        summary = run_single_episode(config)
        summaries.append(summary)
        if summary["total_success"]:
            successes += 1
        attempts += 1
        if not successful_only and attempts >= num_episodes:
            break

    return summaries
