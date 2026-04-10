import math
import unittest

import numpy as np

from modules.sim_emio_camera_observation import (
    SimEmioCameraConfig,
    SimEmioCameraObservationSource,
    _crop_frame,
    _resize_frame_nearest,
    camera_forward_up_vectors,
    compute_emio_camera_pose,
)


class _ValueWrapper:
    def __init__(self, value):
        self.value = value


class _MockMechanicalObject:
    def __init__(self, position):
        self.position = _ValueWrapper([position])


class _MockGripperState:
    def __init__(self, positions):
        self.position = _ValueWrapper(positions)


def _make_handles():
    return {
        "block_mo": _MockMechanicalObject([5.0, -170.0, 7.0, 0.0, 0.0, 0.0, 1.0]),
        "tcp_mo": _MockMechanicalObject([2.0, -149.0, 3.0, 0.0, 0.0, 0.0, 1.0]),
        "gripper_state": _MockGripperState([[0.0, -151.0, 0.0], [4.0, -151.0, 0.0]]),
        "tuning": {
            "object_position": np.array([5.0, -170.0, 7.0], dtype=np.float32),
            "place_position": np.array([0.0, -160.0, -10.0], dtype=np.float32),
        },
    }


class CameraPoseTests(unittest.TestCase):
    def test_compute_emio_camera_pose_matches_extended_prefab_math(self):
        position, orientation = compute_emio_camera_pose(extended=True)
        expected_diagonal = math.cos(math.pi / 4.0) * 147.0
        np.testing.assert_allclose(position, np.array([-expected_diagonal, -5.0, -expected_diagonal], dtype=np.float32))

        forward, up = camera_forward_up_vectors(orientation)
        np.testing.assert_allclose(forward, np.array([0.5, -math.sqrt(0.5), 0.5], dtype=np.float32), atol=1e-5)
        np.testing.assert_allclose(up, np.array([0.5, math.sqrt(0.5), 0.5], dtype=np.float32), atol=1e-5)

    def test_compute_emio_camera_pose_applies_offsets(self):
        position, orientation = compute_emio_camera_pose(
            extended=True,
            translation_offset_mm=(1.0, 2.0, 3.0),
            rotation_offset_deg=(0.0, 0.0, 90.0),
        )
        base_position, base_orientation = compute_emio_camera_pose(extended=True)

        np.testing.assert_allclose(position, base_position + np.array([1.0, 2.0, 3.0], dtype=np.float32))
        self.assertFalse(np.allclose(orientation, base_orientation))


class ImageProcessingTests(unittest.TestCase):
    def test_crop_and_resize_preserve_rgb_shape(self):
        frame = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)

        cropped = _crop_frame(frame, (0.25, 0.25, 0.5, 0.5))
        resized = _resize_frame_nearest(cropped, (6, 5, 3))

        self.assertEqual(cropped.shape, (4, 4, 3))
        self.assertEqual(resized.shape, (6, 5, 3))
        self.assertEqual(resized.dtype, np.uint8)


class SourceSmokeTests(unittest.TestCase):
    def test_update_returns_uint8_rgb(self):
        source = SimEmioCameraObservationSource(
            _make_handles(),
            SimEmioCameraConfig(image_shape=(32, 32, 3), render_shape=(64, 64, 3)),
        )

        source.open()
        try:
            frame = source.update()
        finally:
            source.close()

        self.assertEqual(frame.shape, (32, 32, 3))
        self.assertEqual(frame.dtype, np.uint8)
        self.assertGreater(int(frame.max()) - int(frame.min()), 0)


if __name__ == "__main__":
    unittest.main()
