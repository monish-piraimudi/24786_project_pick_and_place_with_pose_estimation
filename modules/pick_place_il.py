import argparse
import glob
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import Sofa

try:
    import Sofa.ImGui as MyGui
except ImportError:
    MyGui = None

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../")

from modules.camera_observation import default_image_shape
from modules.emio_camera_observation import EmioCameraConfig, EmioCameraObservationSource
from modules.imitation_policy import ImplicitBCAgent
from modules.pick_place_policy_shared import (
    MAX_EPISODE_STEPS,
    PHASE_NAMES,
    advance_phase,
    apply_policy_motor_action,
    build_state_observation,
    phase_opening,
    phase_target,
)


_RUNTIME_TASK_TUNING = None
PROJECT_DIR = Path(__file__).resolve().parent.parent


def set_runtime_task_tuning(tuning):
    global _RUNTIME_TASK_TUNING
    if tuning is None:
        _RUNTIME_TASK_TUNING = None
        return

    runtime_tuning = {}
    for key, value in tuning.items():
        if isinstance(value, (list, tuple, np.ndarray)):
            runtime_tuning[key] = list(np.asarray(value, dtype=float))
        else:
            runtime_tuning[key] = float(value)
    _RUNTIME_TASK_TUNING = runtime_tuning


def _parse_scene_args(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--mode",
        choices=["expert", "policy_inspect"],
        default="expert",
    )
    parser.add_argument(
        "--policy-path",
        type=str,
        default="data/results/il_pick_place/bc_policy.pth",
    )
    parser.add_argument("--cube-x-mm", dest="cube_x_mm", type=float, default=None)
    parser.add_argument("--cube-z-mm", dest="cube_z_mm", type=float, default=None)
    parser.add_argument("--connection", dest="connection", action="store_true")
    parser.add_argument("--no-connection", dest="connection", action="store_false")
    parser.add_argument("--real-rgb-observation", dest="real_rgb_observation", action="store_true")
    parser.add_argument("--no-real-rgb-observation", dest="real_rgb_observation", action="store_false")
    parser.add_argument("--camera-tracking", dest="camera_tracking", action="store_true")
    parser.add_argument("--no-camera-tracking", dest="camera_tracking", action="store_false")
    parser.add_argument("--camera-preview", dest="camera_preview", action="store_true")
    parser.add_argument("--no-camera-preview", dest="camera_preview", action="store_false")
    parser.add_argument("--camera-serial", dest="camera_serial", type=str, default=None)
    parser.add_argument(
        "--cube-marker-offset-mm",
        dest="cube_marker_offset_mm",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        default=(0.0, 0.0, 0.0),
    )
    parser.set_defaults(
        connection=True,
        real_rgb_observation=False,
        camera_tracking=False,
        camera_preview=False,
    )
    args, _unknown = parser.parse_known_args(argv)
    return args


def _resolve_tray_mesh_path():
    """Resolve tray mesh across local lab copies and emio-labs installs."""
    scene_dir = os.path.dirname(os.path.realpath(__file__))

    direct_candidates = [
        os.path.normpath(os.path.join(scene_dir, "../../data/meshes/tray.stl")),
        os.path.normpath(os.path.join(scene_dir, "../data/meshes/tray.stl")),
        os.path.normpath(os.path.join(scene_dir, "data/meshes/tray.stl")),
    ]

    # Try parent roots to support running from copied lab folders.
    probe = scene_dir
    for _ in range(8):
        direct_candidates.append(os.path.join(probe, "data/meshes/tray.stl"))
        direct_candidates.append(os.path.join(probe, "assets/data/meshes/tray.stl"))
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        probe = parent

    # Also try emio-labs versioned installs under the user home folder.
    emio_candidates = sorted(
        glob.glob(os.path.expanduser("~/emio-labs/*/assets/data/meshes/tray.stl"))
    )
    direct_candidates.extend(emio_candidates)

    for path in direct_candidates:
        if os.path.isfile(path):
            return path

    return None


def _vec3_distance(a, b):
    return float(np.linalg.norm(np.array(a) - np.array(b)))


def _is_valid_tracker_xyz(xyz):
    xyz = np.asarray(xyz, dtype=float).reshape(-1)
    if xyz.size < 3 or not np.all(np.isfinite(xyz[:3])):
        return False
    return float(np.linalg.norm(xyz[:3])) > 1e-3


def _default_task_tuning():
    """Return one selected parameter set for the task."""

    success_params = {
        "gripper_opening_open": 40.0,
        "gripper_opening_closed": 12.0,
        "gripper_opening_min": 5.0,
        "gripper_opening_max": 40.0,
        "pick_position": [-3.0, -165.0, 12.0],
        "place_position": [-24.0, -165.0, -20.0],
        "object_position": [-3.0, -170.0, 12.0],
        "hover_lift_height": 45.0,
        "pick_height_offset": 9.0,
        "place_height_offset": 11.0,
        "lift_success_delta": 30.0,
    }

    fail_params = {
        "gripper_opening_open": 30.0,
        "gripper_opening_closed": 12.0,
        "gripper_opening_min": 5.0,
        "gripper_opening_max": 35.0,
        "pick_position": [5.0, -165.0, 20.0],
        "place_position": [-24.0, -165.0, -12.0],
        "object_position": [-3.0, -170.0, 12.0],
        "hover_lift_height": 42.0,
        "pick_height_offset": 9.0,
        "place_height_offset": 11.0,
        "lift_success_delta": 30.0,
    }
    # Select the most specific non-empty parameter set.
    if success_params:
        return success_params
    return fail_params


def _pick_position_for_object(object_position, reference_pick_position, reference_object_position):
    object_position = np.asarray(object_position, dtype=float).copy()
    reference_pick_position = np.asarray(reference_pick_position, dtype=float)
    reference_object_position = np.asarray(reference_object_position, dtype=float)
    pick_position = object_position.copy()
    pick_position[1] = float(
        object_position[1] + (reference_pick_position[1] - reference_object_position[1])
    )
    return pick_position


class PickAndPlaceEvaluator(Sofa.Core.Controller):

    def __init__(
        self,
        root,
        block_mo,
        tcp_mo,
        gripper_mo,
        gripper_opening,
        object_position,
        pick_position,
        place_position,
        gripper_opening_closed,
        lift_success_delta,
    ):
        Sofa.Core.Controller.__init__(self)
        self.name = "PickAndPlaceEvaluator"

        self.root = root
        self.block_mo = block_mo
        self.tcp_mo = tcp_mo
        self.gripper_mo = gripper_mo
        self.gripper_opening = gripper_opening
        self.object_position = np.array(object_position)
        self.pick_position = np.array(pick_position)
        self.place_position = np.array(place_position)
        self.gripper_opening_closed = float(gripper_opening_closed)
        self.tray_height = float(object_position[1])
        self.lift_success_delta = float(lift_success_delta)
        self.attach_distance_threshold = 11.0
        self.attach_opening_threshold = self.gripper_opening_closed + 0.5
        self.attach_offset = np.array([0.0, -10.0, 0.0], dtype=float)
        self.is_attached = False
        self.lifted = False
        self.placed = False
        self.is_falling = False
        self._runtime_error_logged = False
        self._frame_count = 0
        print("[P3][Evaluator] Initialized")

    def reset_task(self, object_position, pick_position):
        self.object_position = np.asarray(object_position, dtype=float)
        self.pick_position = np.asarray(pick_position, dtype=float)
        self.tray_height = float(self.object_position[1])
        self.is_attached = False
        self.lifted = False
        self.placed = False
        self.is_falling = False
        self._runtime_error_logged = False
        self._frame_count = 0

    def _to_flat_array(self, data):
        if hasattr(data, "value"):
            data = data.value
        return np.array(data, dtype=float).reshape(-1)

    def _tcp_position(self):
        values = self._to_flat_array(self.tcp_mo.position)
        if values.size < 3:
            return None
        return values[0:3]

    def _block_position(self):
        values = self._to_flat_array(self.block_mo.position)
        if values.size < 3:
            return None
        return values[0:3]

    def _set_block_position(self, xyz):
        values = self._to_flat_array(self.block_mo.position)
        if values.size >= 7:
            q = values[3:7]
        else:
            q = np.array([0.0, 0.0, 0.0, 1.0])
        self.block_mo.position.value = [list(np.array(xyz, dtype=float)) + list(q)]

    def _get_gripper_opening(self):
        opening_data = self.gripper_opening
        if hasattr(opening_data, "value"):
            opening_data = opening_data.value
        opening_array = np.array(opening_data, dtype=float).reshape(-1)
        if opening_array.size == 0:
            return 0.0
        # restLengths can be multi-valued; median is more stable than index 0.
        return float(np.median(opening_array))

    def _gripper_end_positions(self):
        values = np.array(
            self.gripper_mo.position.value,
            dtype=float,
        ).reshape(-1, 3)
        if values.shape[0] < 2:
            return None, None
        return values[0], values[1]

    def _grasp_point(self):
        end_a, end_b = self._gripper_end_positions()
        if end_a is None or end_b is None:
            return None, None
        return 0.5 * (end_a + end_b), _vec3_distance(end_a, end_b)

    def _attach_point(self, tcp_pos, grasp_point):
        """Prefer TCP-based pickup anchor; fall back to grasp midpoint."""
        if tcp_pos is not None:
            return np.array(tcp_pos, dtype=float) + self.attach_offset
        return np.array(grasp_point, dtype=float)

    def onAnimateBeginEvent(self, e):
        _ = e
        try:
            self._frame_count += 1
            elapsed = float(self.root.time.value)
            self.root.taskElapsed.value = elapsed

            tcp_pos = self._tcp_position()
            block_pos = self._block_position()
            if tcp_pos is None or block_pos is None:
                print(
                    "[P3][Evaluator] Invalid positions",
                    "frame=",
                    self._frame_count,
                    "tcp=",
                    tcp_pos,
                    "block=",
                    block_pos,
                )
                return

            opening = self._get_gripper_opening()
            grasp_point, end_distance = self._grasp_point()
            if grasp_point is None:
                print("[P3][Evaluator] Missing gripper end positions")
                return
            attach_point = self._attach_point(tcp_pos, grasp_point)
            dist_mid = _vec3_distance(grasp_point, block_pos)
            dist_attach = _vec3_distance(attach_point, block_pos)
            self.root.tcpX.value = float(tcp_pos[0])
            self.root.tcpY.value = float(tcp_pos[1])
            self.root.tcpZ.value = float(tcp_pos[2])
            self.root.blockX.value = float(block_pos[0])
            self.root.blockY.value = float(block_pos[1])
            self.root.blockZ.value = float(block_pos[2])
            self.root.graspMidX.value = float(grasp_point[0])
            self.root.graspMidY.value = float(grasp_point[1])
            self.root.graspMidZ.value = float(grasp_point[2])
            self.root.distToMidpoint.value = float(dist_mid)
            self.root.distToAttach.value = float(dist_attach)
            self.root.gripperOpeningForGrasp.value = float(opening)

            if (not self.is_attached) and (not self.placed):
                close_enough = dist_attach <= self.attach_distance_threshold
                closed_enough = opening <= self.attach_opening_threshold

                if close_enough and closed_enough:
                    self.is_attached = True
                    self.is_falling = False
                    print(
                        "[P3][Evaluator] ATTACH frame",
                        self._frame_count,
                        "opening=",
                        round(opening, 3),
                        "dist_attach=",
                        round(dist_attach, 3),
                    )

            if self.is_attached:
                self._set_block_position(attach_point)
                block_pos = self._block_position()
                if block_pos is None:
                    return
                if block_pos[1] > self.object_position[1] + self.lift_success_delta:
                    self.lifted = True
                    print("[P3][Evaluator] LIFTED frame", self._frame_count)

                if opening > self.gripper_opening_closed:
                    self.is_attached = False
                    self.is_falling = True
                    print("[P3][Evaluator] PLACED frame", self._frame_count)

            elif self.is_falling:
                fall_y = max(self.tray_height, block_pos[1] - 4.0)
                self._set_block_position([block_pos[0], fall_y, block_pos[2]])
                if fall_y <= self.tray_height:
                    self.is_falling = False
                    self.placed = True

            score = 0.0
            if self.lifted:
                score += 50.0
            if self.placed:
                score += 50.0

            if self.placed:
                score += max(0.0, 20.0 - elapsed)

            self.root.taskScore.value = float(score)
            self.root.taskLifted.value = bool(self.lifted)
            self.root.taskPlaced.value = bool(self.placed)
        except Exception as exc:
            if not self._runtime_error_logged:
                Sofa.msg_error(
                    "PickAndPlaceEvaluator",
                    "Runtime error in evaluator: " + str(exc),
                )
                print("[P3][Evaluator] EXCEPTION", repr(exc))
                traceback.print_exc()
                self._runtime_error_logged = True


class AutoPickAndPlaceDemo(Sofa.Core.Controller):

    def __init__(
        self,
        root,
        target_mo,
        block_mo,
        gripper_opening,
        pick_position,
        place_position,
        gripper_opening_open,
        gripper_opening_closed,
        gripper_opening_min,
        gripper_opening_max,
        hover_lift_height,
        pick_height_offset,
        place_height_offset,
    ):
        Sofa.Core.Controller.__init__(self)
        self.name = "AutoPickAndPlaceDemo"
        self.root = root
        self.target_mo = target_mo
        self.block_mo = block_mo
        self.gripper_opening = gripper_opening
        self.pick_position = np.array(pick_position, dtype=float)
        self.place_position = np.array(place_position, dtype=float)
        self.opening_open = float(gripper_opening_open)
        self.opening_closed = float(gripper_opening_closed)
        self.opening_min = float(gripper_opening_min)
        self.opening_max = float(gripper_opening_max)
        self.hover_lift_height = float(hover_lift_height)
        self.pick_height_offset = float(pick_height_offset)
        self.place_height_offset = float(place_height_offset)
        # mm/s limit for stable beam deformation while opening/closing
        self.opening_rate = 45.0
        self.demo_duration = 16.0
        self.is_active = True
        self.start_time = None
        self.last_phase = None
        print("[P3][Demo] Initialized")

    def configure_positions(self, pick_position, place_position):
        self.pick_position = np.asarray(pick_position, dtype=float)
        self.place_position = np.asarray(place_position, dtype=float)

    def reset_demo_state(self, current_time: float | None = None):
        self.start_time = current_time
        self.last_phase = None

    def _set_target_position(self, xyz):
        self.target_mo.position.value = [
            [
                float(xyz[0]),
                float(xyz[1]),
                float(xyz[2]),
                0.0,
                0.0,
                0.0,
                1.0,
            ]
        ]
        self.root.targetX.value = float(xyz[0])
        self.root.targetY.value = float(xyz[1])
        self.root.targetZ.value = float(xyz[2])

    def _set_gripper_opening(self, value):
        target = float(np.clip(value, self.opening_min, self.opening_max))
        self.root.commandedOpening.value = target

        opening_data = self.gripper_opening

        try:
            if hasattr(opening_data, "value"):
                opening_data.value = target
            elif isinstance(opening_data, (list, tuple)):
                # Array-like but immutable, cannot set
                print(
                    "[P3][Demo] WARNING: gripper_opening is immutable "
                    f"(type={type(opening_data).__name__})"
                )
                return
            else:
                # Try slice assignment for array-like objects
                opening_data[:] = target
        except (TypeError, AttributeError, IndexError) as e:
            # Silent fail - gripper_opening may be in an inconsistent state
            print(
                f"[P3][Demo] WARNING: failed to set gripper: {e} "
                f"(type={type(opening_data).__name__})"
            )

    def _set_block_position(self, xyz):
        values = self.block_mo.position.value[0]
        if len(values) >= 7:
            q = values[3:7]
        else:
            q = [0.0, 0.0, 0.0, 1.0]
        self.block_mo.position.value = [
            [
                float(xyz[0]),
                float(xyz[1]),
                float(xyz[2]),
                q[0],
                q[1],
                q[2],
                q[3],
            ]
        ]

    def onAnimateBeginEvent(self, e):
        _ = e
        if not self.is_active:
            return

        current_time = float(self.root.time.value)
        if self.start_time is None:
            self.start_time = current_time

        elapsed = current_time - self.start_time

        approach_pick = self.pick_position + np.array(
            [0.0, self.hover_lift_height, 0.0]
        )
        at_pick = self.pick_position + np.array([0.0, self.pick_height_offset, 0.0])
        lift_pick = self.pick_position + np.array([0.0, self.hover_lift_height, 0.0])
        approach_place = self.place_position + np.array(
            [0.0, self.hover_lift_height, 0.0]
        )
        at_place = self.place_position + np.array([0.0, self.place_height_offset, 0.0])

        if elapsed < 2.0:
            phase = 0
            self._set_target_position(approach_pick)
            self._set_gripper_opening(self.opening_open)
        elif elapsed < 4.0:
            phase = 1
            self._set_target_position(at_pick)
            self._set_gripper_opening(self.opening_open)
        elif elapsed < 6.0:
            phase = 2
            self._set_target_position(at_pick)
            self._set_gripper_opening(self.opening_closed)
        elif elapsed < 8.0:
            phase = 3
            self._set_target_position(lift_pick)
            self._set_gripper_opening(self.opening_closed)
        elif elapsed < 10.0:
            phase = 4
            self._set_target_position(approach_place)
            self._set_gripper_opening(self.opening_closed)
        elif elapsed < 12.0:
            phase = 5
            self._set_target_position(at_place)
            self._set_gripper_opening(self.opening_closed)
        elif elapsed < 14.0:
            phase = 6
            self._set_target_position(at_place)
            self._set_gripper_opening(self.opening_open)
        elif elapsed < self.demo_duration:
            phase = 7
            self._set_target_position(approach_place)
            self._set_gripper_opening(self.opening_open)
        else:
            self.is_active = False
            self.root.autoDemoActive.value = False
            print("[P3][Demo] timeline ended; manual GUI control restored")
            return

        if phase != self.last_phase:
            self.last_phase = phase



class PolicyInspectController(Sofa.Core.Controller):

    def __init__(
        self,
        root,
        target_mo,
        tcp_mo,
        block_mo,
        gripper_state,
        demo,
        evaluator,
        pick_marker_mo,
        tuning,
        policy_path,
        image_shape,
        real_rgb_observation,
        camera_tracking,
        camera_preview,
        camera_serial,
        cube_marker_offset_mm,
    ):
        Sofa.Core.Controller.__init__(self)
        self.name = "PolicyInspectController"
        self.root = root
        self.target_mo = target_mo
        self.tcp_mo = tcp_mo
        self.block_mo = block_mo
        self.gripper_state = gripper_state
        self.demo = demo
        self.evaluator = evaluator
        self.pick_marker_mo = pick_marker_mo
        self.tuning = tuning
        self.image_shape = tuple(image_shape)
        self.real_rgb_observation = bool(real_rgb_observation)
        self.camera_tracking = bool(camera_tracking)
        self.camera_preview = bool(camera_preview)
        self.camera_serial = camera_serial
        self.cube_marker_offset_mm = np.asarray(cube_marker_offset_mm, dtype=np.float32)
        self.max_steps = MAX_EPISODE_STEPS
        self.reference_object_position = np.asarray(tuning["object_position"], dtype=float).copy()
        self.reference_pick_position = np.asarray(tuning["pick_position"], dtype=float).copy()
        self.place_position = np.asarray(tuning["place_position"], dtype=float)
        self.policy_path = self._resolve_policy_path(policy_path)
        self.policy_agent = None
        self.camera_source = None
        self.handles = {
            "root": root,
            "demo": demo,
            "evaluator": evaluator,
            "target_mo": target_mo,
            "tcp_mo": tcp_mo,
            "block_mo": block_mo,
            "gripper_state": gripper_state,
            "motor_actuators": [root.Simulation.Emio.getChild(f"Motor{i}").JointActuator for i in range(4)],
            "camera_source": None,
            "tuning": tuning,
        }
        self.is_running = False
        self.phase = PHASE_NAMES[0]
        self.phase_step = 0
        self.close_counter = 0
        self.step_count = 0
        self._start_requested_last = False
        self._load_error_logged = False
        self._auto_start_pending = True
        self._last_cube_selector = None
        self._camera_init_attempted = False
        self._camera_ready = False
        root.policyCheckpointPath.value = str(self.policy_path)
        self._load_policy()
        self._sync_scene_to_selector(update_block=True)

    def _resolve_policy_path(self, policy_path):
        path = Path(policy_path)
        return path if path.is_absolute() else (PROJECT_DIR / path)

    def _load_policy(self):
        try:
            if not self.policy_path.is_file():
                raise FileNotFoundError(
                    f"Policy checkpoint not found: {self.policy_path}. "
                    "Use --policy-path to point at a trained checkpoint."
                )
            self.policy_agent = ImplicitBCAgent.from_checkpoint(self.policy_path)
        except Exception as exc:
            self.policy_agent = None
            if not self._load_error_logged:
                Sofa.msg_error(
                    "PolicyInspectController",
                    "Failed to load policy checkpoint: " + str(exc),
                )
                print("[P3][PolicyInspect] EXCEPTION", repr(exc))
                traceback.print_exc()
                self._load_error_logged = True

    def _open_camera_source(self):
        try:
            if self.real_rgb_observation or self.camera_tracking:
                self.camera_source = EmioCameraObservationSource(
                    EmioCameraConfig(
                        image_shape=self.image_shape,
                        show=self.camera_preview,
                        track_markers=self.camera_tracking,
                        compute_point_cloud=False,
                        configuration="extended",
                        camera_serial=self.camera_serial,
                    )
                )
                if not self.camera_source.open():
                    raise RuntimeError("Failed to open EmioCamera for policy inspection.")
                self.handles["camera_source"] = self.camera_source

        except Exception as exc:
            if self.camera_source is not None:
                try:
                    self.camera_source.close()
                except Exception:
                    pass
            self.camera_source = None
            self.handles["camera_source"] = None
            if not self._load_error_logged:
                Sofa.msg_error(
                    "PolicyInspectController",
                    "Failed to open policy-inspection camera source: " + str(exc),
                )
                print("[P3][PolicyInspect] EXCEPTION", repr(exc))
                traceback.print_exc()
                self._load_error_logged = True
            return False
        return True

    def cleanup(self):
        if self.camera_source is not None:
            self.camera_source.close()
            self.camera_source = None
            self.handles["camera_source"] = None
        self._camera_ready = False

    def _ensure_camera_source(self) -> bool:
        if self._camera_ready:
            return True
        if self._camera_init_attempted:
            return False
        self._camera_init_attempted = True
        self._camera_ready = bool(self._open_camera_source())
        return self._camera_ready

    def _selected_object_position(self):
        base = np.asarray(self.tuning["object_position"], dtype=float).copy()
        base[0] = float(self.root.cubeStartX.value)
        base[2] = float(self.root.cubeStartZ.value)
        return base

    def _selected_cube_key(self):
        return (
            round(float(self.root.cubeStartX.value), 4),
            round(float(self.root.cubeStartZ.value), 4),
        )

    def _set_block_pose(self, xyz):
        values = np.asarray(self.block_mo.position.value[0], dtype=float)
        if values.size >= 7:
            q = values[3:7]
        else:
            q = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
        self.block_mo.position.value = [
            [
                float(xyz[0]),
                float(xyz[1]),
                float(xyz[2]),
                float(q[0]),
                float(q[1]),
                float(q[2]),
                float(q[3]),
            ]
        ]

    def _update_camera_cube(self, trackers):
        trackers = np.asarray(trackers, dtype=np.float32).reshape(-1, 3)
        if trackers.shape[0] < 1:
            self.root.cameraTrackingAvailable.value = False
            return None
        marker_position = trackers[0]
        if not np.all(np.isfinite(marker_position)) or float(np.linalg.norm(marker_position)) <= 1e-3:
            self.root.cameraTrackingAvailable.value = False
            return None
        cube_position = marker_position + self.cube_marker_offset_mm
        self.root.cubeStartX.value = float(cube_position[0])
        self.root.cubeStartZ.value = float(cube_position[2])
        self.root.cameraCubeX.value = float(cube_position[0])
        self.root.cameraCubeY.value = float(cube_position[1])
        self.root.cameraCubeZ.value = float(cube_position[2])
        self.root.cameraTrackingAvailable.value = True
        self._sync_task_to_camera_cube(cube_position)
        return cube_position.astype(np.float32)

    def _sync_scene_to_selector(self, update_block):
        object_position = self._selected_object_position()
        pick_position = _pick_position_for_object(
            object_position,
            self.reference_pick_position,
            self.reference_object_position,
        )
        self.tuning["object_position"] = object_position.astype(np.float32)
        self.tuning["pick_position"] = pick_position.astype(np.float32)
        self.demo.configure_positions(pick_position, self.place_position)
        self.evaluator.reset_task(object_position, pick_position)
        if self.pick_marker_mo is not None:
            self.pick_marker_mo.position.value = [list(np.asarray(pick_position, dtype=float))]
        if update_block:
            self._set_block_pose(object_position)
        return object_position, pick_position

    def _sync_task_to_camera_cube(self, cube_position):
        cube_position = np.asarray(cube_position, dtype=np.float32).copy()
        pick_position = _pick_position_for_object(
            cube_position,
            self.reference_pick_position,
            self.reference_object_position,
        ).astype(np.float32)
        self.tuning["object_position"] = cube_position
        self.tuning["pick_position"] = pick_position
        self.demo.configure_positions(pick_position, self.place_position)
        self.evaluator.reset_task(cube_position, pick_position)
        if self.pick_marker_mo is not None:
            self.pick_marker_mo.position.value = [[float(pick_position[0]), float(pick_position[1]), float(pick_position[2])]]
        self._set_block_pose(cube_position)

    def _reset_rollout(self):
        self._sync_scene_to_selector(update_block=True)
        self.root.taskScore.value = 0.0
        self.root.taskLifted.value = False
        self.root.taskPlaced.value = False
        self.root.autoDemoActive.value = False
        self.phase = PHASE_NAMES[0]
        self.phase_step = 0
        self.close_counter = 0
        self.step_count = 0
        self.demo.is_active = False
        self.demo.reset_demo_state(current_time=float(self.root.time.value))
        self.demo._set_target_position(phase_target(self.demo, self.phase))
        self.demo._set_gripper_opening(phase_opening(self.demo, self.phase))
        if self.policy_agent is not None:
            self.policy_agent.reset_rollout()
        self.is_running = True
        self.root.policyInspectActive.value = True
        self._auto_start_pending = False
        self._last_cube_selector = self._selected_cube_key()
        print(
            "[P3][PolicyInspect] start",
            "cube_x=",
            round(float(self.root.cubeStartX.value), 3),
            "cube_z=",
            round(float(self.root.cubeStartZ.value), 3),
        )

    def _stop_rollout(self, reason):
        if self.is_running:
            print("[P3][PolicyInspect] stop", reason)
        self.is_running = False
        self.root.policyInspectActive.value = False

    def onAnimateBeginEvent(self, _event):
        if self.policy_agent is None:
            self.root.policyInspectActive.value = False
            return

        if not self._ensure_camera_source():
            self._stop_rollout("camera_init_failed")
            return

        if self.camera_source is not None:
            try:
                _camera_frame, trackers = self.camera_source.update()
                if self.camera_tracking:
                    self._update_camera_cube(trackers)
            except Exception:
                pass

        selected_cube = self._selected_cube_key()
        slider_changed = self._last_cube_selector is not None and selected_cube != self._last_cube_selector
        camera_pose_moved = (
            self.camera_tracking
            and self._last_cube_selector is not None
            and np.linalg.norm(np.asarray(selected_cube, dtype=float) - np.asarray(self._last_cube_selector, dtype=float)) >= 8.0
        )

        if self._auto_start_pending and not self.is_running:
            self._reset_rollout()
            return

        if not self.is_running:
            if (slider_changed and not self.camera_tracking) or camera_pose_moved:
                self._reset_rollout()
            else:
                self._last_cube_selector = selected_cube
        else:
            if self.step_count > 0:
                self.phase, self.phase_step, self.close_counter, done = advance_phase(
                    self.handles,
                    self.phase,
                    self.phase_step,
                    self.close_counter,
                )
                if done:
                    self._stop_rollout("completed")
                    return
            if self.step_count >= self.max_steps:
                self._stop_rollout("max_steps_reached")
                return

            state_observation = build_state_observation(
                self.handles,
                self.phase,
            )
            action = np.asarray(self.policy_agent.predict(state_observation), dtype=np.float32)
            smoothed_action = self.policy_agent.smooth_action(action)
            self.demo._set_target_position(phase_target(self.demo, self.phase))
            self.demo._set_gripper_opening(phase_opening(self.demo, self.phase))
            executed_action = apply_policy_motor_action(self.handles, smoothed_action)
            self.policy_agent.set_previous_action(executed_action)
            self.phase_step += 1
            self.step_count += 1

    def __del__(self):  # pragma: no cover - best effort cleanup during SOFA shutdown
        try:
            self.cleanup()
        except Exception:
            pass


class EmioCameraMonitor(Sofa.Core.Controller):

    def __init__(self, root, image_shape, camera_preview, camera_serial, cube_marker_offset_mm):
        Sofa.Core.Controller.__init__(self)
        self.name = "EmioCameraMonitor"
        self.root = root
        self.cube_marker_offset_mm = np.asarray(cube_marker_offset_mm, dtype=np.float32)
        self.camera_source = None
        self._error_logged = False
        try:
            self.camera_source = EmioCameraObservationSource(
                EmioCameraConfig(
                    image_shape=image_shape,
                    show=bool(camera_preview),
                    track_markers=True,
                    compute_point_cloud=False,
                    configuration="extended",
                    camera_serial=camera_serial,
                )
            )
            if not self.camera_source.open():
                raise RuntimeError("Failed to open EmioCamera monitor.")
        except Exception as exc:
            self.camera_source = None
            self._log_error(exc)

    def _log_error(self, exc):
        if self._error_logged:
            return
        Sofa.msg_error("EmioCameraMonitor", "Failed to open/update EmioCamera: " + str(exc))
        print("[P3][SceneCamera] EXCEPTION", repr(exc))
        traceback.print_exc()
        self._error_logged = True

    def onAnimateBeginEvent(self, _event):
        if self.camera_source is None:
            self.root.cameraTrackingAvailable.value = False
            return
        try:
            _frame, trackers = self.camera_source.update()
        except Exception as exc:
            self.root.cameraTrackingAvailable.value = False
            self._log_error(exc)
            return

        trackers = np.asarray(trackers, dtype=np.float32).reshape(-1, 3)
        if trackers.shape[0] < 1 or not _is_valid_tracker_xyz(trackers[0]):
            self.root.cameraTrackingAvailable.value = False
            return

        cube_xyz = trackers[0] + self.cube_marker_offset_mm
        self.root.cameraCubeX.value = float(cube_xyz[0])
        self.root.cameraCubeY.value = float(cube_xyz[1])
        self.root.cameraCubeZ.value = float(cube_xyz[2])
        self.root.cameraTrackingAvailable.value = True

    def __del__(self):  # pragma: no cover - best effort cleanup
        try:
            if self.camera_source is not None:
                self.camera_source.close()
        except Exception:
            pass


def createScene(rootnode):
    from parts.controllers.assemblycontroller import AssemblyController
    from parts.emio import Emio
    from parts.gripper import Gripper
    from utils.header import addHeader, addSolvers

    print("[P3][Scene] createScene start")
    args = _parse_scene_args()
    tuning = _default_task_tuning()
    if _RUNTIME_TASK_TUNING:
        tuning.update(_RUNTIME_TASK_TUNING)

    settings, modelling, simulation = addHeader(
        rootnode,
        inverse=True,
        withCollision=True,
    )
    print("[P3][Scene] header added")
    settings.addObject(
        "RequiredPlugin",
        name="collision_geometry",
        pluginName=["Sofa.Component.Collision.Geometry"],
    )
    addSolvers(simulation)
    print("[P3][Scene] solvers added")

    rootnode.dt = 0.05
    rootnode.gravity = [0.0, -9810.0, 0.0]
    rootnode.VisualStyle.displayFlags.value = ["hideBehavior", "hideWireframe"]

    rootnode.addData(name="taskScore", type="float", value=0.0)
    rootnode.addData(name="taskElapsed", type="float", value=0.0)
    rootnode.addData(name="taskLifted", type="bool", value=False)
    rootnode.addData(name="taskPlaced", type="bool", value=False)
    rootnode.addData(name="autoDemoActive", type="bool", value=True)
    rootnode.addData(name="targetX", type="float", value=0.0)
    rootnode.addData(name="targetY", type="float", value=0.0)
    rootnode.addData(name="targetZ", type="float", value=0.0)
    rootnode.addData(name="tcpX", type="float", value=0.0)
    rootnode.addData(name="tcpY", type="float", value=0.0)
    rootnode.addData(name="tcpZ", type="float", value=0.0)
    rootnode.addData(name="blockX", type="float", value=0.0)
    rootnode.addData(name="blockY", type="float", value=0.0)
    rootnode.addData(name="blockZ", type="float", value=0.0)
    rootnode.addData(name="graspMidX", type="float", value=0.0)
    rootnode.addData(name="graspMidY", type="float", value=0.0)
    rootnode.addData(name="graspMidZ", type="float", value=0.0)
    rootnode.addData(name="distToMidpoint", type="float", value=0.0)
    rootnode.addData(name="distToAttach", type="float", value=0.0)
    rootnode.addData(name="gripperOpeningForGrasp", type="float", value=0.0)
    rootnode.addData(name="commandedOpening", type="float", value=0.0)
    rootnode.addData(name="cameraCubeX", type="float", value=0.0)
    rootnode.addData(name="cameraCubeY", type="float", value=0.0)
    rootnode.addData(name="cameraCubeZ", type="float", value=0.0)
    rootnode.addData(name="cameraTrackingAvailable", type="bool", value=False)
    rootnode.addData(name="cubeStartX", type="float", value=0.0)
    rootnode.addData(name="cubeStartZ", type="float", value=0.0)
    rootnode.addData(name="policyInspectActive", type="bool", value=False)
    rootnode.addData(name="policyCheckpointPath", type="string", value="")
    print("[P3][Scene] task data added")

    emio = Emio(
        name="Emio",
        legsName=["blueleg"],
        legsModel=["beam"],
        legsPositionOnMotor=[
            "counterclockwisedown",
            "clockwisedown",
            "counterclockwisedown",
            "clockwisedown",
        ],
        centerPartName="whitepart",
        centerPartType="deformable",
        centerPartModel="beam",
        centerPartClass=Gripper,
        platformLevel=2,
        extended=True,
    )
    if not emio.isValid():
        print("[P3][Scene] emio invalid")
        return
    print("[P3][Scene] emio valid")

    simulation.addChild(emio)
    emio.attachCenterPartToLegs()
    emio.addObject(AssemblyController(emio))
    print("[P3][Scene] emio inserted")

    if (args.camera_tracking or args.real_rgb_observation) and args.mode != "policy_inspect":
        try:
            rootnode.addObject(
                EmioCameraMonitor(
                    root=rootnode,
                    image_shape=default_image_shape(),
                    camera_preview=args.camera_preview,
                    camera_serial=args.camera_serial,
                    cube_marker_offset_mm=args.cube_marker_offset_mm,
                )
            )
            print("[P3][Scene] emio camera monitor added")
        except Exception as exc:
            Sofa.msg_error(__file__, "Camera not detected: " + str(exc))

    # Workaround: skip gripper collision setup here because the current
    # gripper collision helper builds a filename with a DataString + str,
    # which crashes scene loading in this environment.

    tray_mesh = _resolve_tray_mesh_path()
    if tray_mesh is not None:
        tray = modelling.addChild("Tray")
        tray.addObject(
            "MeshSTLLoader",
            filename=tray_mesh,
            translation=[0, 20, 0],
        )
        tray.addObject(
            "OglModel",
            src=tray.MeshSTLLoader.linkpath,
            color=[0.3, 0.3, 0.3, 0.2],
        )
        print("[P3][Scene] tray added from", tray_mesh)
    else:
        print("[P3][Scene] WARNING: tray.stl not found; skipping tray visual")

    emio.effector.addObject(
        "MechanicalObject",
        template="Rigid3",
        position=[0, 0, 0, 0, 0, 0, 1] * 4,
    )
    emio.effector.addObject("RigidMapping", rigidIndexPerPoint=[0, 1, 2, 3])
    print("[P3][Scene] effector added")

    effector_target = modelling.addChild("Target")
    effector_target.addObject("EulerImplicitSolver", firstOrder=True)
    effector_target.addObject(
        "CGLinearSolver",
        iterations=50,
        tolerance=1e-10,
        threshold=1e-10,
    )
    target_mo = effector_target.addObject(
        "MechanicalObject",
        template="Rigid3",
        position=[0, -150, 0, 0, 0, 0, 1],
        showObject=False,
        showObjectScale=20,
    )
    print("[P3][Scene] target added")

    emio.addInverseComponentAndGUI(
        effector_target.getMechanicalState().position.linkpath,
        withGUI=MyGui is not None,
        barycentric=True,
    )
    tcp = modelling.addChild("TCP")
    tcp_mo = tcp.addObject(
        "MechanicalObject",
        template="Rigid3",
        position=emio.effector.EffectorCoord.barycenter.linkpath,
    )
    if MyGui is not None:
        MyGui.setIPController(
            rootnode.Modelling.Target,
            tcp,
            rootnode.ConstraintSolver,
        )
    print("[P3][Scene] inverse + TCP added")

    opening_data = emio.centerpart.Effector.Distance.DistanceMapping.restLengths
    gripper_state = emio.centerpart.Effector.getMechanicalState()
    print("[P3][Scene] opening data bound")
    if MyGui is not None:
        MyGui.MoveWindow.addAccessory(
            "Gripper's opening (mm)",
            opening_data,
            tuning["gripper_opening_min"],
            tuning["gripper_opening_max"],
        )
        MyGui.ProgramWindow.addGripper(
            opening_data,
            tuning["gripper_opening_min"],
            tuning["gripper_opening_max"],
        )
        MyGui.ProgramWindow.importProgram(
            os.path.dirname(__file__) + "/mypickandplace.crprog"
        )
        MyGui.IOWindow.addSubscribableData("/Gripper", opening_data)
    print("[P3][Scene] GUI controls added")

    # User-tuned waypoints and object spawn pose.
    reference_pick_position = np.asarray(tuning["pick_position"], dtype=float).copy()
    place_position = np.asarray(tuning["place_position"], dtype=float).copy()
    reference_object_position = np.asarray(tuning["object_position"], dtype=float).copy()
    object_position = reference_object_position.copy()
    if args.cube_x_mm is not None:
        object_position[0] = float(args.cube_x_mm)
    if args.cube_z_mm is not None:
        object_position[2] = float(args.cube_z_mm)
    pick_position = _pick_position_for_object(
        object_position,
        reference_pick_position,
        reference_object_position,
    )
    tuning["object_position"] = object_position.tolist()
    tuning["pick_position"] = pick_position.tolist()
    tuning["place_position"] = place_position.tolist()
    rootnode.cubeStartX.value = float(object_position[0])
    rootnode.cubeStartZ.value = float(object_position[2])
    rootnode.policyCheckpointPath.value = str(
        Path(args.policy_path) if Path(args.policy_path).is_absolute() else (PROJECT_DIR / args.policy_path)
    )

    # Keep the block kinematic for robust task evaluation.
    block = modelling.addChild("Block")
    block_mo = block.addObject(
        "MechanicalObject",
        template="Rigid3",
        position=[
            object_position[0],
            object_position[1],
            object_position[2],
            0,
            0,
            0,
            1,
        ],
        showObject=False,
    )

    block_visual = block.addChild("Visual")
    block_visual.addObject(
        "MeshOBJLoader",
        name="loader",
        filename=os.path.join(os.path.dirname(__file__), "cube.obj"),
        translation=[-5.0, -32.0, 5.0],
    )
    block_visual.addObject("MeshTopology", src="@loader")
    block_visual.addObject("OglModel", src="@loader", color=[0.82, 0.62, 0.25, 1.0])
    block_visual.addObject("RigidMapping", index=0)

    print("[P3][Scene] kinematic block added")

    pick_marker = modelling.addChild("PickMarker")
    pick_marker_mo = pick_marker.addObject(
        "MechanicalObject",
        template="Vec3",
        position=[pick_position.tolist()],
        showObject=False,
        showObjectScale=8,
        showColor=[1.0, 0.3, 0.3, 1.0],
    )
    place_marker = modelling.addChild("PlaceMarker")
    place_marker.addObject(
        "MechanicalObject",
        template="Vec3",
        position=[place_position],
        showObject=False,
        showObjectScale=8,
        showColor=[0.3, 1.0, 0.3, 1.0],
    )
    print("[P3][Scene] markers added")

    evaluator = PickAndPlaceEvaluator(
        root=rootnode,
        block_mo=block_mo,
        tcp_mo=tcp_mo,
        gripper_mo=gripper_state,
        gripper_opening=opening_data,
        object_position=object_position,
        pick_position=pick_position,
        place_position=place_position,
        gripper_opening_closed=tuning["gripper_opening_closed"],
        lift_success_delta=tuning["lift_success_delta"],
    )
    rootnode.addObject(evaluator)
    print("[P3][Scene] evaluator added")

    demo = AutoPickAndPlaceDemo(
        root=rootnode,
        target_mo=target_mo,
        block_mo=block_mo,
        gripper_opening=opening_data,
        pick_position=pick_position,
        place_position=place_position,
        gripper_opening_open=tuning["gripper_opening_open"],
        gripper_opening_closed=tuning["gripper_opening_closed"],
        gripper_opening_min=tuning["gripper_opening_min"],
        gripper_opening_max=tuning["gripper_opening_max"],
        hover_lift_height=tuning["hover_lift_height"],
        pick_height_offset=tuning["pick_height_offset"],
        place_height_offset=tuning["place_height_offset"],
    )
    rootnode.addObject(demo)
    print("[P3][Scene] demo path added")

    if args.mode == "policy_inspect":
        demo.is_active = False
        rootnode.autoDemoActive.value = False
        rootnode.addObject(
            PolicyInspectController(
                root=rootnode,
                target_mo=target_mo,
                tcp_mo=tcp_mo,
                block_mo=block_mo,
                gripper_state=gripper_state,
                demo=demo,
                evaluator=evaluator,
                pick_marker_mo=pick_marker_mo,
                tuning=tuning,
                policy_path=args.policy_path,
                image_shape=default_image_shape(),
                real_rgb_observation=args.real_rgb_observation,
                camera_tracking=args.camera_tracking,
                camera_preview=args.camera_preview,
                camera_serial=args.camera_serial,
                cube_marker_offset_mm=args.cube_marker_offset_mm,
            )
        )
        print("[P3][Scene] policy inspect controller added")

    if MyGui is not None:
        MyGui.MyRobotWindow.addInformation("Task score", rootnode.taskScore)
        MyGui.MyRobotWindow.addInformation("Task lifted", rootnode.taskLifted)
        MyGui.MyRobotWindow.addInformation("Task placed", rootnode.taskPlaced)
        MyGui.MyRobotWindow.addInformation(
            "Auto demo active",
            rootnode.autoDemoActive,
        )
        MyGui.MyRobotWindow.addInformation(
            "Task elapsed (s)",
            rootnode.taskElapsed,
        )
        MyGui.PlottingWindow.addData("tcp x", rootnode.tcpX)
        MyGui.PlottingWindow.addData("tcp y", rootnode.tcpY)
        MyGui.PlottingWindow.addData("tcp z", rootnode.tcpZ)
        MyGui.PlottingWindow.addData("block x", rootnode.blockX)
        MyGui.PlottingWindow.addData("block y", rootnode.blockY)
        MyGui.PlottingWindow.addData("block z", rootnode.blockZ)
        MyGui.PlottingWindow.addData("grasp midpoint x", rootnode.graspMidX)
        MyGui.PlottingWindow.addData("grasp midpoint y", rootnode.graspMidY)
        MyGui.PlottingWindow.addData("grasp midpoint z", rootnode.graspMidZ)
        MyGui.PlottingWindow.addData(
            "diag distance block-midpoint",
            rootnode.distToMidpoint,
        )
        MyGui.PlottingWindow.addData(
            "diag distance block-attach",
            rootnode.distToAttach,
        )
        MyGui.PlottingWindow.addData(
            "diag gripper opening (GUI)",
            rootnode.gripperOpeningForGrasp,
        )
        MyGui.PlottingWindow.addData(
            "diag gripper opening command",
            rootnode.commandedOpening,
        )
        MyGui.MyRobotWindow.addInformation(
            "Camera tracking available",
            rootnode.cameraTrackingAvailable,
        )
        MyGui.PlottingWindow.addData("camera cube x", rootnode.cameraCubeX)
        MyGui.PlottingWindow.addData("camera cube y", rootnode.cameraCubeY)
        MyGui.PlottingWindow.addData("camera cube z", rootnode.cameraCubeZ)
        if args.mode == "policy_inspect":
            MyGui.MoveWindow.addAccessory("Cube start X (mm)", rootnode.cubeStartX, -95.0, 95.0)
            MyGui.MoveWindow.addAccessory("Cube start Z (mm)", rootnode.cubeStartZ, -95.0, 95.0)
            MyGui.MyRobotWindow.addInformation(
                "Policy inspect active",
                rootnode.policyInspectActive,
            )
            MyGui.IOWindow.addSubscribableData("/PolicyInspect/Active", rootnode.policyInspectActive)
            MyGui.IOWindow.addSubscribableData("/PolicyInspect/CubeStartX", rootnode.cubeStartX)
            MyGui.IOWindow.addSubscribableData("/PolicyInspect/CubeStartZ", rootnode.cubeStartZ)
    print("[P3][Scene] info fields added")

    if args.connection:
        emio.addConnectionComponents()
        print("[P3][Scene] connection components added")

    print("[P3][Scene] createScene complete")
    return rootnode
