import unittest

import numpy as np

from modules.pick_place_policy_shared import (
    PHASE_NAMES,
    STATE_DELTA_SCALE_MM,
    STATE_FEATURE_NAMES,
    STATE_POSITION_CENTER_MM,
    STATE_POSITION_SCALE_MM,
    advance_phase,
    apply_policy_motor_action,
    build_state_observation,
    motor_action_bounds,
    render_observation,
)


class _ValueWrapper:
    def __init__(self, value):
        self.value = value


class _MockMechanicalObject:
    def __init__(self, position):
        self.position = _ValueWrapper([position])


class _MockDemo:
    def __init__(self):
        self.opening_open = 30.0
        self.opening_closed = 12.0
        self.hover_lift_height = 42.0
        self.pick_height_offset = 9.0
        self.place_height_offset = 11.0
        self.pick_position = np.array([1.0, -160.0, 3.0], dtype=np.float32)
        self.place_position = np.array([4.0, -160.0, -5.0], dtype=np.float32)
        self.commanded_target = None
        self.commanded_opening = None

    def _set_target_position(self, xyz):
        self.commanded_target = np.asarray(xyz, dtype=np.float32)

    def _set_gripper_opening(self, value):
        self.commanded_opening = float(value)


class _MockRoot:
    def __init__(self, opening=30.0):
        self.commandedOpening = _ValueWrapper(float(opening))


class _MockEvaluator:
    def __init__(self, *, attached=False, lifted=False, placed=False):
        self.is_attached = bool(attached)
        self.lifted = bool(lifted)
        self.placed = bool(placed)


class _MockActuator:
    def __init__(self, value=0.0, angle=0.0, min_angle=-1.0, max_angle=1.0):
        self.value = float(value)
        self.angle = _ValueWrapper(float(angle))
        self.minAngle = _ValueWrapper(float(min_angle))
        self.maxAngle = _ValueWrapper(float(max_angle))


def _make_handles(*, tcp_xyz):
    return {
        "root": _MockRoot(),
        "tcp_mo": _MockMechanicalObject([*tcp_xyz, 0.0, 0.0, 0.0, 1.0]),
        "block_mo": _MockMechanicalObject([5.0, -170.0, 7.0, 0.0, 0.0, 0.0, 1.0]),
        "demo": _MockDemo(),
        "evaluator": _MockEvaluator(),
        "motor_actuators": [
            _MockActuator(value=-0.1, angle=-0.2, min_angle=-1.0, max_angle=1.0),
            _MockActuator(value=0.2, angle=0.1, min_angle=-1.5, max_angle=1.5),
            _MockActuator(value=0.3, angle=0.25, min_angle=-2.0, max_angle=2.0),
            _MockActuator(value=-0.4, angle=-0.35, min_angle=-2.5, max_angle=2.5),
        ],
        "tuning": {
            "object_position": np.array([5.0, -170.0, 7.0], dtype=np.float32),
            "pick_position": np.array([5.0, -161.0, 7.0], dtype=np.float32),
            "place_position": np.array([0.0, -160.0, -10.0], dtype=np.float32),
            "gripper_opening_min": 5.0,
            "gripper_opening_max": 35.0,
        },
    }


class StateObservationTests(unittest.TestCase):
    def test_build_state_observation_uses_expected_feature_order(self):
        handles = _make_handles(tcp_xyz=[2.0, -149.0, 3.0])
        handles["root"].commandedOpening.value = 20.0
        handles["evaluator"] = _MockEvaluator(attached=True)

        state = build_state_observation(handles, "close_gripper")

        tcp = np.array([2.0, -149.0, 3.0], dtype=np.float32)
        cube = np.array([5.0, -170.0, 7.0], dtype=np.float32)
        goal = np.array([0.0, -160.0, -10.0], dtype=np.float32)
        expected = np.array(
            [
                *((tcp - STATE_POSITION_CENTER_MM) / STATE_POSITION_SCALE_MM),
                *((cube - STATE_POSITION_CENTER_MM) / STATE_POSITION_SCALE_MM),
                *((goal - STATE_POSITION_CENTER_MM) / STATE_POSITION_SCALE_MM),
                *((cube - tcp) / STATE_DELTA_SCALE_MM),
                *((goal - cube) / STATE_DELTA_SCALE_MM),
                0.5,
                1.0,
            ],
            dtype=np.float32,
        )

        self.assertEqual(state.shape, (len(STATE_FEATURE_NAMES),))
        np.testing.assert_allclose(state, expected)

    def test_render_observation_uses_sim_camera_source(self):
        handles = _make_handles(tcp_xyz=[2.0, -149.0, 3.0])

        class _DummySimCameraSource:
            def update(self):
                return np.full((8, 8, 3), 77, dtype=np.uint8)

        handles["sim_camera_source"] = _DummySimCameraSource()

        observation = render_observation(handles, "approach_pick", (8, 8, 3))

        self.assertEqual(observation.shape, (8, 8, 3))
        self.assertEqual(observation.dtype, np.uint8)
        self.assertTrue(np.all(observation == 77))


class MotorActionTests(unittest.TestCase):
    def test_motor_action_bounds_use_actuator_limits(self):
        handles = _make_handles(tcp_xyz=[0.0, 0.0, 0.0])

        bounds = motor_action_bounds(handles)

        np.testing.assert_allclose(
            bounds,
            np.array(
                [
                    [-1.0, 1.0],
                    [-1.5, 1.5],
                    [-2.0, 2.0],
                    [-2.5, 2.5],
                ],
                dtype=np.float32,
            ),
        )

    def test_apply_policy_motor_action_clips_and_sets_actuators(self):
        handles = _make_handles(tcp_xyz=[0.0, 0.0, 0.0])

        executed = apply_policy_motor_action(handles, np.array([2.0, -2.0, 0.5, -4.0], dtype=np.float32))

        np.testing.assert_allclose(executed, np.array([1.0, -1.5, 0.5, -2.5], dtype=np.float32))
        self.assertEqual(handles["motor_actuators"][0].value, 1.0)
        self.assertEqual(handles["motor_actuators"][1].value, -1.5)
        self.assertEqual(handles["motor_actuators"][2].value, 0.5)
        self.assertEqual(handles["motor_actuators"][3].value, -2.5)


class AdvancePhaseTests(unittest.TestCase):
    def test_approach_pick_advances_when_tcp_reaches_waypoint(self):
        handles = _make_handles(tcp_xyz=[1.0, -118.0, 3.0])

        phase, phase_step, close_counter, done = advance_phase(handles, "approach_pick", 0, 0)

        self.assertEqual(phase, "descend_pick")
        self.assertEqual(phase_step, 0)
        self.assertEqual(close_counter, 0)
        self.assertFalse(done)

    def test_close_gripper_advances_to_lift_when_attached(self):
        handles = _make_handles(tcp_xyz=[1.0, -151.0, 3.0])
        handles["evaluator"] = _MockEvaluator(attached=True)

        phase, phase_step, close_counter, done = advance_phase(handles, "close_gripper", 5, 3)

        self.assertEqual(phase, "lift")
        self.assertEqual(phase_step, 0)
        self.assertEqual(close_counter, 4)
        self.assertFalse(done)

    def test_open_gripper_advances_to_retreat_when_object_released(self):
        handles = _make_handles(tcp_xyz=[4.0, -149.0, -5.0])
        handles["evaluator"] = _MockEvaluator(attached=False)

        phase, phase_step, close_counter, done = advance_phase(handles, "open_gripper", 4, 0)

        self.assertEqual(phase, "retreat")
        self.assertEqual(phase_step, 0)
        self.assertFalse(done)


if __name__ == "__main__":
    unittest.main()
