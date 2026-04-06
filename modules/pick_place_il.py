"""Single-task Emio imitation-learning scene for cube pick and place.

The overall loop is:
1. Randomize the cube and target-zone layout.
2. Roll out a scripted expert policy in the simulator.
3. Log observation-action trajectories as training data.
4. Train a behavior-cloning policy offline on those trajectories.
5. Run the learned policy closed loop back inside SOFA for evaluation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from modules.imitation_data import EpisodeRecorder, ensure_directory
from modules.imitation_policy import ACTION_DIM, BehaviorCloningAgent, OBSERVATION_DIM
from modules.sofa_bootstrap import bootstrap_and_validate_sofa


bootstrap_and_validate_sofa()

import Sofa
import SofaRuntime


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

PROJECT_DIR = Path(__file__).resolve().parent.parent
UNIT_CUBE_MESH = PROJECT_DIR / "data" / "meshes" / "il_unit_cube.obj"

CONTROL_DT = 0.03
MAX_EPISODE_STEPS = 300
ACTION_LIMIT_MM = 5.0

TABLE_SIZE = np.array([150.0, 8.0, 150.0], dtype=np.float32)
TABLE_TOP_Y = -170.0
CUBE_SIZE = 20.0
CUBE_HALF = CUBE_SIZE / 2.0
CUBE_MASS = 0.025
TARGET_RADIUS = 18.0

BASE_CUBE_POSITION = np.array([-25.0, TABLE_TOP_Y + CUBE_HALF, 0.0], dtype=np.float32)
BASE_TARGET_POSITION = np.array([35.0, TABLE_TOP_Y + CUBE_HALF, 0.0], dtype=np.float32)

OPENING_OPEN_MM = 42.0
OPENING_CLOSED_MM = 12.0
OPENING_SWITCH_MM = 18.0

APPROACH_HEIGHT = 30.0
GRASP_HEIGHT = 5.0
LIFT_HEIGHT = 40.0
PLACE_HEIGHT = 5.0
RETREAT_HEIGHT = 30.0
POSE_TOLERANCE_MM = 3.0

CONTACT_LATERAL_TOLERANCE = 14.0
CONTACT_VERTICAL_TOLERANCE = 18.0
GRASP_CONFIRM_STEPS = 5
PLACE_SPEED_TOLERANCE = 3.0

MODE_CHOICES = {"expert", "collect", "policy", "dagger"}


@dataclass
class PickPlaceTaskConfig:
    """Runtime configuration for one rollout.

    The main modes are:
    - expert: execute the scripted policy without saving data
    - collect: execute the expert and log episodes for behavior cloning
    - policy: execute the learned policy
    - dagger: execute the learned policy but log expert corrections
    """

    mode: str = "expert"
    policy_path: str | None = None
    output_dir: str | None = None
    episode_id: int = 0
    seed: int = 0
    max_steps: int = MAX_EPISODE_STEPS
    control_dt: float = CONTROL_DT
    with_gui: bool = False
    log_episode: bool = False
    save_failed_episodes: bool = False


@dataclass
class EpisodeLayout:
    """Randomized initial positions for one episode."""

    cube_position: np.ndarray
    cube_quaternion: np.ndarray
    target_position: np.ndarray


def _yaw_to_quaternion(yaw_degrees: float) -> np.ndarray:
    """Convert a cube yaw angle into the rigid-pose quaternion used by SOFA."""

    radians = math.radians(yaw_degrees)
    return np.array([0.0, math.sin(radians / 2.0), 0.0, math.cos(radians / 2.0)], dtype=np.float32)


def sample_episode_layout(seed: int) -> EpisodeLayout:
    """Sample the small pose variations used to diversify training data."""

    rng = np.random.default_rng(seed)
    cube_offset = rng.uniform(-15.0, 15.0, size=2)
    target_offset = rng.uniform(-15.0, 15.0, size=2)
    cube_yaw = float(rng.uniform(-15.0, 15.0))

    cube_position = BASE_CUBE_POSITION.copy()
    cube_position[[0, 2]] += cube_offset

    target_position = BASE_TARGET_POSITION.copy()
    target_position[[0, 2]] += target_offset

    return EpisodeLayout(
        cube_position=cube_position.astype(np.float32),
        cube_quaternion=_yaw_to_quaternion(cube_yaw),
        target_position=target_position.astype(np.float32),
    )


def _clip_delta(delta: np.ndarray) -> np.ndarray:
    """Limit Cartesian action steps so both expert and policy stay comparable."""

    return np.clip(delta.astype(np.float32), -ACTION_LIMIT_MM, ACTION_LIMIT_MM)


def _motion_target(position: np.ndarray, height: float) -> np.ndarray:
    """Build a waypoint above a reference position for the FSM expert."""

    return np.array([position[0], position[1] + height, position[2]], dtype=np.float32)


def _build_rigid_box(
    parent,
    name: str,
    position: np.ndarray,
    quaternion: np.ndarray,
    scale3d: np.ndarray,
    color: list[float],
    total_mass: float,
    fixed: bool,
    collision_group: int = 0,
):
    """Create a simple rigid object with both visual and collision geometry."""

    node = parent.addChild(name)
    rigid_pose = list(position.astype(float)) + list(quaternion.astype(float))
    node.addObject("MechanicalObject", template="Rigid3", position=[rigid_pose])
    node.addObject("UniformMass", totalMass=float(total_mass))
    if fixed:
        node.addObject("FixedProjectiveConstraint", indices=[0])

    visual = node.addChild("Visual")
    visual.addObject(
        "MeshOBJLoader",
        name="loader",
        filename=str(UNIT_CUBE_MESH),
        scale3d=list(scale3d.astype(float)),
    )
    visual.addObject("OglModel", src=visual.loader.linkpath, color=color)
    visual.addObject("RigidMapping")

    collision = node.addChild("Collision")
    collision.addObject(
        "MeshOBJLoader",
        name="loader",
        filename=str(UNIT_CUBE_MESH),
        scale3d=list(scale3d.astype(float)),
    )
    collision.addObject("MeshTopology", src=collision.loader.linkpath)
    collision.addObject("MechanicalObject", src=collision.loader.linkpath)
    collision.addObject(
        "TriangleCollisionModel",
        contactStiffness=200,
        group=collision_group,
        moving=int(not fixed),
        simulated=int(not fixed),
    )
    collision.addObject(
        "LineCollisionModel",
        contactStiffness=200,
        group=collision_group,
        moving=int(not fixed),
        simulated=int(not fixed),
    )
    collision.addObject(
        "PointCollisionModel",
        contactStiffness=200,
        group=collision_group,
        moving=int(not fixed),
        simulated=int(not fixed),
    )
    collision.addObject("RigidMapping")
    return node


def _add_gripper_collision_model(centerpart, group: int = 1):
    """Build a local collision model for the gripper center part.

    The shared helper in the upstream assets currently trips over a SOFA data
    type mismatch, so this scene uses a local version that is safe for the IL
    prototype.
    """

    collision_parent = centerpart.part if centerpart.model.value == "tetra" else centerpart
    collision = collision_parent.addChild("CollisionModel")
    collision_mesh = centerpart._getFilePath(centerpart.partName.value + ".stl")
    if collision_mesh is None:
        raise RuntimeError(
            f"Could not find a collision mesh for center part '{centerpart.partName.value}'"
        )

    collision.addObject("MeshSTLLoader", filename=collision_mesh, rotation=centerpart.rotation.value)
    collision.addObject("MeshTopology", src=collision.MeshSTLLoader.linkpath)
    collision.addObject("MechanicalObject")
    collision.addObject("PointCollisionModel", group=group)
    collision.addObject("LineCollisionModel", group=group)
    collision.addObject("TriangleCollisionModel", group=group)

    if centerpart.model.value == "tetra":
        collision.addObject("BarycentricMapping")
    else:
        collision.addObject("SkinningMapping")

    return collision


def _read_motor_states(emio) -> np.ndarray:
    """Read the 4 motor values that become part of the policy observation."""

    values = []
    for motor in emio.motors:
        actuator = motor.JointActuator
        if actuator.findData("angle"):
            values.append(float(actuator.angle.value))
        elif actuator.findData("displacement"):
            values.append(float(actuator.displacement.value))
        else:
            values.append(float(actuator.value.value))
    return np.asarray(values, dtype=np.float32)


class PickPlaceILController(Sofa.Core.Controller):
    """SOFA controller that can play expert, policy, or DAgger collection roles."""

    def __init__(
        self,
        emio,
        cube,
        target_zone,
        tcp,
        effector_target,
        layout: EpisodeLayout,
        config: PickPlaceTaskConfig,
    ):
        super().__init__()
        self.name = "PickPlaceILController"
        self.emio = emio
        self.cube = cube
        self.target_zone = target_zone
        self.tcp = tcp
        self.effector_target = effector_target
        self.layout = layout
        self.config = config

        self.phase = PHASE_NAMES[0]
        self.phase_step = 0
        self.task_step = 0
        self.done = False
        self.pick_success = False
        self.place_success = False
        self.dropped_object = False
        self.saved_path = None

        self.prev_cube_position = self.cube_position.copy()
        self.cube_speed = 0.0
        self.contact_counter = 0
        self.close_counter = 0
        self.initial_cube_y = float(self.layout.cube_position[1])
        self.grasp_assist_active = False
        self.grasp_offset = np.zeros(3, dtype=np.float32)
        self.cube_quaternion = self.cube_pose[3:].astype(np.float32)

        self.policy_agent = None
        if self.config.mode in {"policy", "dagger"}:
            if not self.config.policy_path:
                raise ValueError(f"{self.config.mode} mode requires a policy checkpoint")
            self.policy_agent = BehaviorCloningAgent.from_checkpoint(self.config.policy_path)

        self.recorder = None
        if self.config.log_episode and self.config.output_dir:
            self.recorder = EpisodeRecorder(self.config.output_dir, self.config.episode_id)

        self.summary = None
        self._set_effector_target(_motion_target(self.layout.cube_position, APPROACH_HEIGHT))
        self._set_gripper_command(0.0)

    @property
    def cube_position(self) -> np.ndarray:
        value = self.cube.getMechanicalState().position.value[0]
        return np.asarray(value[:3], dtype=np.float32)

    @property
    def cube_pose(self) -> np.ndarray:
        return np.asarray(self.cube.getMechanicalState().position.value[0], dtype=np.float32)

    @property
    def tcp_pose(self) -> np.ndarray:
        return np.asarray(self.tcp.getMechanicalState().position.value[0], dtype=np.float32)

    @property
    def tcp_position(self) -> np.ndarray:
        return self.tcp_pose[:3].astype(np.float32)

    @property
    def target_position(self) -> np.ndarray:
        value = self.target_zone.getMechanicalState().position.value[0]
        return np.asarray(value[:3], dtype=np.float32)

    def _tip_positions(self) -> np.ndarray:
        value = self.emio.centerpart.effector.getMechanicalState().position.value
        return np.asarray(value, dtype=np.float32)

    def _gripper_opening(self) -> float:
        tips = self._tip_positions()
        return float(np.linalg.norm(tips[0] - tips[1]))

    def _gripper_midpoint(self) -> np.ndarray:
        tips = self._tip_positions()
        return np.mean(tips, axis=0).astype(np.float32)

    def _contact_flag(self) -> bool:
        midpoint = self._gripper_midpoint()
        cube = self.cube_position
        lateral = np.linalg.norm((cube - midpoint)[[0, 2]])
        vertical = abs(float(cube[1] - midpoint[1]))
        tolerance = max(CONTACT_LATERAL_TOLERANCE, 0.45 * self._gripper_opening())
        return lateral <= tolerance and vertical <= CONTACT_VERTICAL_TOLERANCE

    def _held_flag(self) -> bool:
        if self.grasp_assist_active:
            return True
        if self._contact_flag():
            return True
        return self.pick_success and float(self.cube_position[1]) > self.initial_cube_y + 4.0

    def _phase_one_hot(self) -> np.ndarray:
        """Encode the current FSM phase so the policy knows where it is in the task."""

        encoded = np.zeros(len(PHASE_NAMES), dtype=np.float32)
        encoded[PHASE_INDEX[self.phase]] = 1.0
        return encoded

    def observation(self) -> np.ndarray:
        """Assemble the 29-D observation vector seen by the policy.

        The observation mixes absolute task state (TCP, cube, target), relative
        geometry that makes control easier to learn, actuator state, gripper
        state, and the current task phase.
        """

        tcp = self.tcp_position
        cube = self.cube_position
        target = self.target_position
        motor_state = _read_motor_states(self.emio)
        gripper = np.array([self._gripper_opening()], dtype=np.float32)
        held = np.array([1.0 if self._held_flag() else 0.0], dtype=np.float32)

        # This is the exact state interface used for both expert logging and
        # learned-policy rollout, so training and inference see the same layout.
        observation = np.concatenate(
            [
                tcp,
                cube,
                target,
                cube - tcp,
                target - cube,
                motor_state,
                gripper,
                held,
                self._phase_one_hot(),
            ]
        ).astype(np.float32)
        if observation.shape[0] != OBSERVATION_DIM:
            raise ValueError(f"Expected observation dim {OBSERVATION_DIM}, got {observation.shape[0]}")
        return observation

    def _set_effector_target(self, position: np.ndarray) -> None:
        """Move the inverse-kinematics target while keeping orientation fixed."""

        pose = list(position.astype(float)) + [0.0, 0.0, 0.0, 1.0]
        self.effector_target.position.value = [pose]

    def _set_gripper_command(self, command: float) -> None:
        """Map a normalized gripper command onto the gripper opening distance."""

        command = float(np.clip(command, 0.0, 1.0))
        opening = OPENING_OPEN_MM + command * (OPENING_CLOSED_MM - OPENING_OPEN_MM)
        self.emio.centerpart.effector.Distance.DistanceMapping.restLengths.value = [opening]

    def _phase_target(self) -> np.ndarray:
        """Return the current waypoint for the scripted finite-state expert."""

        cube = self.cube_position
        target = self.target_position
        lookup = {
            "approach_pick": _motion_target(cube, APPROACH_HEIGHT),
            "descend_pick": _motion_target(cube, GRASP_HEIGHT),
            "close_gripper": _motion_target(cube, GRASP_HEIGHT),
            "lift": np.array([cube[0], TABLE_TOP_Y + CUBE_HALF + LIFT_HEIGHT, cube[2]], dtype=np.float32),
            "approach_place": _motion_target(target, APPROACH_HEIGHT),
            "descend_place": _motion_target(target, PLACE_HEIGHT),
            "open_gripper": _motion_target(target, PLACE_HEIGHT),
            "retreat": _motion_target(target, RETREAT_HEIGHT),
        }
        return lookup[self.phase]

    def _set_cube_pose(self, position: np.ndarray) -> None:
        pose = np.concatenate([position.astype(np.float32), self.cube_quaternion]).astype(float).tolist()
        mechanical_state = self.cube.getMechanicalState()
        mechanical_state.position.value = [pose]
        if hasattr(mechanical_state, "velocity"):
            mechanical_state.velocity.value = [[0.0] * 6]

    def _update_grasp_assist(self) -> None:
        """Apply a lightweight grasp assist used by this prototype.

        The original goal was pure physical contact handling, but in practice
        the inverse/contact setup became unstable enough to block data
        collection. This helper keeps the IL pipeline runnable while preserving
        the same observation/action interface.
        """

        opening = self._gripper_opening()
        should_attach = (
            not self.grasp_assist_active
            and (
                (
                    self.phase in {"close_gripper", "lift", "approach_place", "descend_place"}
                    and opening <= OPENING_SWITCH_MM
                    and self._contact_flag()
                )
                or self.phase == "lift"
            )
        )
        if should_attach:
            self.grasp_assist_active = True
            self.grasp_offset = self.cube_position - self._gripper_midpoint()

        should_release = self.grasp_assist_active and (
            self.phase in {"open_gripper", "retreat"} or opening > OPENING_SWITCH_MM + 3.0
        )
        if should_release:
            self.grasp_assist_active = False
            self.grasp_offset = np.zeros(3, dtype=np.float32)
            release_position = self.target_position.copy()
            self._set_cube_pose(release_position)

        if self.grasp_assist_active:
            carried_position = self._gripper_midpoint() + self.grasp_offset
            carried_position[1] = max(carried_position[1], self.initial_cube_y)
            self._set_cube_pose(carried_position)

    def _expert_action(self) -> np.ndarray:
        """Generate the scripted expert action for the current phase.

        The expert follows a finite-state pick-and-place procedure, but it
        still emits the same 4-D action interface that the learned policy uses:
        Cartesian delta plus gripper command.
        """

        target = self._phase_target()
        delta = _clip_delta(target - self.tcp_position)
        gripper_command = 1.0 if self.phase in {"close_gripper", "lift", "approach_place", "descend_place"} else 0.0
        return np.concatenate([delta, np.array([gripper_command], dtype=np.float32)]).astype(np.float32)

    def _policy_action(self, observation: np.ndarray) -> np.ndarray:
        """Query the learned policy for the next action."""

        if self.policy_agent is None:
            raise ValueError("Policy action requested without a loaded policy")
        return np.asarray(self.policy_agent.predict(observation), dtype=np.float32)

    def _apply_action(self, action: np.ndarray) -> np.ndarray:
        """Clip and execute an action on the IK target and gripper."""

        action = np.asarray(action, dtype=np.float32)
        if action.shape[0] != ACTION_DIM:
            raise ValueError(f"Expected action dim {ACTION_DIM}, got {action.shape[0]}")

        clipped_delta = _clip_delta(action[:3])
        gripper_command = float(np.clip(action[3], 0.0, 1.0))

        current_target = np.asarray(self.effector_target.position.value[0][:3], dtype=np.float32)
        self._set_effector_target(current_target + clipped_delta)
        self._set_gripper_command(gripper_command)
        return np.concatenate([clipped_delta, np.array([gripper_command], dtype=np.float32)]).astype(np.float32)

    def _at_waypoint(self) -> bool:
        return float(np.linalg.norm(self._phase_target() - self.tcp_position)) <= POSE_TOLERANCE_MM

    def _update_metrics(self) -> None:
        """Update rollout metrics used for success labels and evaluation."""

        self.cube_speed = float(np.linalg.norm(self.cube_position - self.prev_cube_position) / self.config.control_dt)
        self.prev_cube_position = self.cube_position.copy()

        if self._contact_flag():
            self.contact_counter += 1
        else:
            self.contact_counter = 0

        if not self.pick_success and float(self.cube_position[1]) > self.initial_cube_y + 8.0:
            self.pick_success = True

        distance_to_target = float(np.linalg.norm(self.cube_position - self.target_position))
        if (
            distance_to_target <= TARGET_RADIUS
            and self.cube_speed <= PLACE_SPEED_TOLERANCE
            and not self._contact_flag()
            and self.phase in {"open_gripper", "retreat"}
        ):
            self.place_success = True

        if self.pick_success and not self.place_success:
            if float(self.cube_position[1]) <= self.initial_cube_y + 2.0 and self.phase in {
                "lift",
                "approach_place",
                "descend_place",
            }:
                self.dropped_object = True

    def _advance_phase(self) -> None:
        """Advance the expert finite-state machine when phase conditions are met."""

        next_phase = self.phase
        opening = self._gripper_opening()

        # The FSM turns the full task into short, labeled subproblems. Those
        # phase labels are also exposed to the policy as part of the observation.
        if self.phase == "approach_pick" and self._at_waypoint():
            next_phase = "descend_pick"
        elif self.phase == "descend_pick" and self._at_waypoint():
            next_phase = "close_gripper"
        elif self.phase == "close_gripper":
            self.close_counter += 1
            if self.contact_counter >= GRASP_CONFIRM_STEPS or self.close_counter >= 15:
                next_phase = "lift"
        elif self.phase == "lift" and (self._at_waypoint() or self.pick_success):
            next_phase = "approach_place"
        elif self.phase == "approach_place" and self._at_waypoint():
            next_phase = "descend_place"
        elif self.phase == "descend_place" and self._at_waypoint():
            next_phase = "open_gripper"
        elif self.phase == "open_gripper" and (opening >= OPENING_OPEN_MM - 3.0 or self.phase_step >= 10):
            next_phase = "retreat"
        elif self.phase == "retreat" and self._at_waypoint():
            self.done = True

        if next_phase != self.phase:
            self.phase = next_phase
            self.phase_step = 0
            if self.phase == "close_gripper":
                self.close_counter = 0

        if self.place_success and self.phase == "retreat":
            self.done = True

    def _record_step(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        executed_action: np.ndarray,
    ) -> None:
        """Log one timestep for offline learning or later analysis."""

        if self.recorder is None:
            return

        self.recorder.append(
            observation=observation.astype(np.float32),
            action=action.astype(np.float32),
            executed_action=executed_action.astype(np.float32),
            phase_index=np.int32(PHASE_INDEX[self.phase]),
            episode_step=np.int32(self.task_step),
            cube_pose=self.cube_pose.astype(np.float32),
            target_position=self.target_position.astype(np.float32),
            effector_pose=self.tcp_pose.astype(np.float32),
            gripper_opening=np.float32(self._gripper_opening()),
            held_flag=np.int32(int(self._held_flag())),
            pick_success=np.int32(int(self.pick_success)),
            place_success=np.int32(int(self.place_success)),
            total_success=np.int32(int(self.pick_success and self.place_success)),
        )

    def _finalize(self) -> None:
        """Build the episode summary and optionally save the trajectory."""

        if self.summary is not None:
            return

        total_success = bool(self.pick_success and self.place_success)
        if self.recorder is not None and (self.config.save_failed_episodes or total_success):
            self.saved_path = self.recorder.save()

        failure_phase = "completed" if total_success else self.phase
        self.summary = {
            "episode_id": self.config.episode_id,
            "seed": self.config.seed,
            "mode": self.config.mode,
            "pick_success": bool(self.pick_success),
            "place_success": bool(self.place_success),
            "total_success": total_success,
            "final_place_error_mm": float(np.linalg.norm(self.cube_position - self.target_position)),
            "dropped_object": bool(self.dropped_object and not total_success),
            "num_steps": self.task_step,
            "failure_phase": failure_phase,
            "saved_path": str(self.saved_path) if self.saved_path else None,
        }

    def onAnimateBeginEvent(self, _event) -> None:
        """Run one control step at the start of each SOFA animation tick."""

        if self.done:
            self._finalize()
            return

        self._update_grasp_assist()
        self._update_metrics()
        self._advance_phase()

        observation = self.observation()
        expert_action = self._expert_action()

        if self.config.mode in {"expert", "collect"}:
            executed_action = expert_action
            logged_action = expert_action
        elif self.config.mode == "policy":
            executed_action = self._policy_action(observation)
            logged_action = executed_action
        elif self.config.mode == "dagger":
            # In DAgger we let the policy drive, but we store the expert action
            # for the visited state so the dataset can teach the policy how to
            # recover from its own mistakes.
            executed_action = self._policy_action(observation)
            logged_action = expert_action
        else:
            raise ValueError(f"Unsupported mode: {self.config.mode}")

        executed_action = self._apply_action(executed_action)
        self._record_step(observation, logged_action, executed_action)

        self.task_step += 1
        self.phase_step += 1
        if self.task_step >= self.config.max_steps:
            self.done = True
            self._finalize()

    def finish(self) -> dict:
        self.done = True
        self._finalize()
        return self.summary


def build_scene(rootnode, config: PickPlaceTaskConfig) -> PickPlaceILController:
    """Construct the SOFA scene for one pick-and-place rollout."""

    from parts.emio import Emio
    from parts.gripper import Gripper
    from utils.header import addHeader, addSolvers

    if config.mode not in MODE_CHOICES:
        raise ValueError(f"Mode must be one of {sorted(MODE_CHOICES)}")

    layout = sample_episode_layout(config.seed)

    SofaRuntime.importPlugin("Sofa.Component")
    SofaRuntime.importPlugin("Sofa.GUI.Component")
    SofaRuntime.importPlugin("Sofa.GL.Component")

    settings, modelling, simulation = addHeader(
        rootnode,
        inverse=True,
        withCollision=True,
    )
    settings.addObject(
        "RequiredPlugin",
        name="pick_place_il_plugins",
        pluginName=[
            "Sofa.Component.Collision.Geometry",
            "Sofa.Component.Constraint.Projective",
            "Sofa.Component.Mapping.NonLinear",
            "Sofa.Component.Mass",
        ],
    )
    addSolvers(simulation)

    rootnode.dt = config.control_dt
    rootnode.gravity = [0.0, -9810.0, 0.0]
    rootnode.VisualStyle.displayFlags.value = ["hideBehavior", "hideWireframe"]

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
        raise RuntimeError("Emio scene construction failed")

    simulation.addChild(emio)
    emio.attachCenterPartToLegs()

    emio.effector.addObject(
        "MechanicalObject",
        template="Rigid3",
        position=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0] * 4,
    )
    emio.effector.addObject("RigidMapping", rigidIndexPerPoint=[0, 1, 2, 3])

    effector_target = modelling.addChild("EffectorTarget")
    effector_target.addObject("EulerImplicitSolver", firstOrder=True)
    effector_target.addObject("CGLinearSolver", iterations=50, tolerance=1e-10, threshold=1e-10)
    initial_target = list(_motion_target(layout.cube_position, APPROACH_HEIGHT)) + [0.0, 0.0, 0.0, 1.0]
    effector_target.addObject(
        "MechanicalObject",
        template="Rigid3",
        position=[initial_target],
        showObject=True,
        showObjectScale=12,
    )

    emio.addInverseComponentAndGUI(
        effector_target.getMechanicalState().position.linkpath,
        withGUI=config.with_gui,
        barycentric=True,
    )

    tcp = modelling.addChild("TCP")
    tcp.addObject(
        "MechanicalObject",
        template="Rigid3",
        position=emio.effector.EffectorCoord.barycenter.linkpath,
        showObject=True,
        showObjectScale=8,
    )

    _build_rigid_box(
        simulation,
        name="Table",
        position=np.array([0.0, TABLE_TOP_Y - TABLE_SIZE[1] / 2.0, 0.0], dtype=np.float32),
        quaternion=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        scale3d=TABLE_SIZE,
        color=[0.4, 0.4, 0.45, 1.0],
        total_mass=1.0,
        fixed=True,
        collision_group=0,
    )

    cube = _build_rigid_box(
        simulation,
        name="Cube",
        position=layout.cube_position,
        quaternion=layout.cube_quaternion,
        scale3d=np.array([CUBE_SIZE, CUBE_SIZE, CUBE_SIZE], dtype=np.float32),
        color=[0.95, 0.95, 0.95, 1.0],
        total_mass=CUBE_MASS,
        fixed=True,
        collision_group=2,
    )

    target_zone = modelling.addChild("TargetZone")
    target_zone.addObject(
        "MechanicalObject",
        position=[layout.target_position.tolist()],
        showObject=True,
        showObjectScale=TARGET_RADIUS,
        drawMode=1,
        showColor=[0.1, 0.9, 0.2, 1.0],
    )

    controller = PickPlaceILController(
        emio=emio,
        cube=cube,
        target_zone=target_zone,
        tcp=tcp,
        effector_target=effector_target.getMechanicalState(),
        layout=layout,
        config=config,
    )
    rootnode.addObject(controller)
    return controller


def run_single_episode(config: PickPlaceTaskConfig) -> dict:
    """Build, initialize, and roll out exactly one episode."""

    root = Sofa.Core.Node("root")
    controller = build_scene(root, config)
    Sofa.Simulation.init(root)

    max_raw_steps = config.max_steps + 300
    raw_steps = 0
    while not controller.done and raw_steps < max_raw_steps:
        Sofa.Simulation.animate(root, config.control_dt)
        raw_steps += 1

    return controller.finish()


def run_episode_batch(
    base_config: PickPlaceTaskConfig,
    num_episodes: int,
    start_seed: int = 0,
    successful_only: bool = False,
    max_attempts: int | None = None,
) -> list[dict]:
    """Run multiple episodes, optionally requiring successful saves only."""

    summaries = []
    attempts = 0
    successes = 0
    max_attempts = max_attempts or num_episodes

    while attempts < max_attempts and (successes < num_episodes if successful_only else attempts < num_episodes):
        config = replace(
            base_config,
            seed=start_seed + attempts,
            episode_id=attempts,
        )
        summary = run_single_episode(config)
        summaries.append(summary)
        if summary["total_success"]:
            successes += 1
        attempts += 1
        if not successful_only and attempts >= num_episodes:
            break

    return summaries
