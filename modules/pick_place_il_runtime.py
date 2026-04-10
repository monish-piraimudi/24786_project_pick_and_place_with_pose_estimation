from __future__ import annotations

import sys
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from modules.camera_observation import default_image_shape
from modules.emio_camera_observation import EmioCameraConfig, EmioCameraObservationSource
from modules.imitation_data import EpisodeRecorder
from modules.imitation_policy import ImplicitBCAgent
from modules.pick_place_policy_shared import (
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
DEFAULT_WORKSPACE_BOUNDS_MM = (-35.0, 10.0, -30.0, 20.0)


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
    camera_tracking: bool = True
    camera_preview: bool = False
    real_rgb_observation: bool = False
    camera_serial: str | None = None
    image_shape: tuple[int, int, int] = field(default_factory=default_image_shape)
    task_tuning: dict | None = None
    object_workspace_bounds_mm: tuple[float, float, float, float] = DEFAULT_WORKSPACE_BOUNDS_MM
    place_target_mm: tuple[float, float, float] | None = None
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
    base_pick_position = np.asarray(tuning["pick_position"], dtype=np.float32).copy()
    pick_height_offset = float(base_pick_position[1] - object_position[1])

    x_min, x_max, z_min, z_max = (float(v) for v in config.object_workspace_bounds_mm)
    object_position[0] = float(rng.uniform(x_min, x_max))
    object_position[2] = float(rng.uniform(z_min, z_max))

    if config.place_target_mm is not None:
        place_position = np.asarray(config.place_target_mm, dtype=np.float32).copy()

    tuning["object_position"] = object_position
    tuning["place_position"] = place_position
    pick_position = object_position.copy()
    pick_position[1] = object_position[1] + pick_height_offset
    tuning["pick_position"] = pick_position
    return tuning


def _scene_tuning_payload(tuning: dict[str, float | np.ndarray]) -> dict[str, float | list[float]]:
    payload = {}
    for key, value in tuning.items():
        if isinstance(value, np.ndarray):
            payload[key] = [float(item) for item in value]
        else:
            payload[key] = float(value)
    return payload


def _tracker_cube_state(trackers: np.ndarray, cube_marker_offset_mm: np.ndarray) -> tuple[np.ndarray | None, bool, str]:
    trackers = np.asarray(trackers, dtype=np.float32).reshape(-1, 3)
    if trackers.shape[0] < 1:
        return None, False, "camera tracker returned no points"

    marker_position = trackers[0]
    if not np.all(np.isfinite(marker_position)) or float(np.linalg.norm(marker_position)) <= 1e-3:
        return None, False, "cube marker not visible"

    cube_position = marker_position + np.asarray(cube_marker_offset_mm, dtype=np.float32)
    return cube_position.astype(np.float32), True, "camera tracking active"


def _resolve_handles(root, tuning: dict[str, float | np.ndarray]) -> dict:
    handles = {
        "root": root,
        "demo": root.getObject("AutoPickAndPlaceDemo"),
        "evaluator": root.getObject("PickAndPlaceEvaluator"),
        "target_mo": root.Modelling.Target.getMechanicalState(),
        "tcp_mo": root.Modelling.TCP.getMechanicalState(),
        "block_mo": root.Modelling.Block.getMechanicalState(),
        "pick_marker_mo": root.Modelling.PickMarker.getMechanicalState(),
        "gripper_state": root.Simulation.Emio.centerpart.Effector.getMechanicalState(),
        "gripper_opening": root.Simulation.Emio.centerpart.Effector.Distance.DistanceMapping.restLengths,
        "camera_source": None,
        "tuning": tuning,
        "pick_base_y_offset": float(
            np.asarray(tuning["pick_position"], dtype=np.float32)[1]
            - np.asarray(tuning["object_position"], dtype=np.float32)[1]
        ),
    }
    return handles


def _set_block_pose(handles: dict, xyz: np.ndarray) -> None:
    block_mo = handles["block_mo"]
    values = np.asarray(block_mo.position.value[0], dtype=np.float32)
    if values.size >= 7:
        q = values[3:7]
    else:
        q = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    block_mo.position.value = [[float(xyz[0]), float(xyz[1]), float(xyz[2]), float(q[0]), float(q[1]), float(q[2]), float(q[3])]]


def _sync_task_to_cube(handles: dict, cube_position: np.ndarray) -> None:
    cube_position = np.asarray(cube_position, dtype=np.float32).copy()
    pick_position = cube_position.copy()
    pick_position[1] = cube_position[1] + float(handles["pick_base_y_offset"])
    handles["tuning"]["object_position"] = cube_position
    handles["tuning"]["pick_position"] = pick_position
    handles["demo"].configure_positions(pick_position, np.asarray(handles["tuning"]["place_position"], dtype=np.float32))
    handles["evaluator"].reset_task(cube_position, pick_position)
    handles["pick_marker_mo"].position.value = [[float(pick_position[0]), float(pick_position[1]), float(pick_position[2])]]
    _set_block_pose(handles, cube_position)


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
    sys.argv.append("--no-camera-tracking")
    if config.camera_preview:
        sys.argv.append("--camera-preview")
    else:
        sys.argv.append("--no-camera-preview")

    scene.set_runtime_task_tuning(_scene_tuning_payload(tuning))
    try:
        scene.createScene(root)
    finally:
        scene.set_runtime_task_tuning(None)
        sys.argv = argv_backup

    Sofa.Simulation.init(root)
    handles = _resolve_handles(root, tuning)
    if config.mode == "policy":
        handles["demo"].is_active = False
        root.autoDemoActive.value = False
    else:
        handles["demo"].is_active = True
        root.autoDemoActive.value = True
    return root, handles


def _open_camera_source(config: PickPlaceTaskConfig) -> EmioCameraObservationSource | None:
    if not config.real_rgb_observation and not config.camera_tracking:
        return None

    camera_source = EmioCameraObservationSource(
        EmioCameraConfig(
            image_shape=config.image_shape,
            show=config.camera_preview,
            track_markers=config.camera_tracking,
            compute_point_cloud=False,
            configuration="extended",
            camera_serial=config.camera_serial,
        )
    )
    if not camera_source.open():
        raise RuntimeError(
            "Failed to open EmioCamera for RGB observation/tracking. "
            "Enable the camera path only when a camera device is connected, or disable it with "
            "--no-real-rgb-observation --no-camera-tracking."
        )
    return camera_source


def run_single_episode(config: PickPlaceTaskConfig) -> dict:
    root, handles = _build_scene(config)
    demo = handles["demo"]
    evaluator = handles["evaluator"]
    camera_tracking_requested = bool(config.camera_tracking)
    camera_tracking_available = False
    camera_status_message = (
        "camera tracking disabled" if not camera_tracking_requested else "camera tracker unavailable"
    )
    camera_missing_steps = 0
    last_camera_cube_position = None
    initial_object_y = float(np.asarray(handles["tuning"]["object_position"], dtype=np.float32)[1])

    policy_agent = None
    if config.mode == "policy":
        if not config.policy_path:
            raise ValueError("policy mode requires a policy checkpoint")
        policy_path = _resolve_project_path(config.policy_path)
        if policy_path is None or not policy_path.is_file():
            raise FileNotFoundError(
                f"Policy checkpoint not found: {policy_path}. "
                "Relative --policy-path values are resolved from this lab folder."
            )
        policy_agent = ImplicitBCAgent.from_checkpoint(policy_path)

    recorder = None
    if config.log_episode and config.output_dir:
        recorder = EpisodeRecorder(config.output_dir, config.episode_id)

    camera_source = None
    done = False
    task_step = 0
    dropped_object = False
    saved_path = None
    failure_phase = "not_started"

    try:
        camera_source = _open_camera_source(config)
        handles["camera_source"] = camera_source

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

            observation = None
            camera_cube_position = None
            trackers = np.zeros((0, 3), dtype=np.float32)
            if camera_source is not None:
                observation, trackers = camera_source.update()

            if camera_tracking_requested:
                camera_cube_position, camera_step_available, camera_step_status = _tracker_cube_state(
                    trackers,
                    np.asarray(config.cube_marker_offset_mm, dtype=np.float32),
                )
                camera_status_message = camera_step_status
                if camera_step_available:
                    camera_tracking_available = True
                    camera_missing_steps = 0
                    last_camera_cube_position = np.asarray(camera_cube_position, dtype=np.float32)
                    _sync_task_to_cube(handles, camera_cube_position)
                    root.cameraCubeX.value = float(camera_cube_position[0])
                    root.cameraCubeY.value = float(camera_cube_position[1])
                    root.cameraCubeZ.value = float(camera_cube_position[2])
                    root.cameraTrackingAvailable.value = True
                else:
                    root.cameraTrackingAvailable.value = False
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
                    _sync_task_to_cube(handles, camera_cube_position)
                    camera_status_message = "cube marker temporarily lost"

            if observation is None:
                observation = render_observation(
                    handles,
                    phase,
                    config.image_shape,
                    cube_position=camera_cube_position,
                    cube_yaw_deg=0.0 if camera_cube_position is not None else None,
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
                    tracked_cube_position=(
                        np.asarray(camera_cube_position, dtype=np.float32)
                        if camera_cube_position is not None
                        else np.full((3,), np.nan, dtype=np.float32)
                    ),
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
    finally:
        if camera_source is not None:
            camera_source.close()

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
        "real_rgb_observation": bool(config.real_rgb_observation),
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
