import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from modules.imitation_data import flatten_hybrid_episode_dataset
from modules.imitation_policy import (
    HYBRID_MODEL_TYPE,
    ImplicitBCAgent,
    ImplicitBCPolicy,
    load_policy_checkpoint,
    save_policy_checkpoint,
)
from modules.pick_place_policy_shared import STATE_FEATURE_NAMES


class HybridDatasetTests(unittest.TestCase):
    def test_flatten_hybrid_episode_dataset_preserves_alignment(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "episode_00000.npz"
            np.savez_compressed(
                path,
                observation=np.zeros((2, 96, 96, 3), dtype=np.uint8),
                state_observation=np.asarray([[0.1, 0.2, 0.0, 0.3, 0.4], [0.5, 0.6, 1.0, 0.7, 0.8]], dtype=np.float32),
                action=np.asarray([[1.0, 2.0, 3.0, 0.0], [4.0, 5.0, 6.0, 1.0]], dtype=np.float32),
            )

            obs, state, action = flatten_hybrid_episode_dataset([path])

        self.assertEqual(obs.shape, (2, 96, 96, 3))
        self.assertEqual(state.shape, (2, len(STATE_FEATURE_NAMES)))
        self.assertEqual(action.shape, (2, 4))
        np.testing.assert_allclose(state[1], np.array([0.5, 0.6, 1.0, 0.7, 0.8], dtype=np.float32))
        np.testing.assert_allclose(action[1], np.array([4.0, 5.0, 6.0, 1.0], dtype=np.float32))


class HybridPolicyTests(unittest.TestCase):
    def test_hybrid_checkpoint_roundtrip_and_predict(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "hybrid_policy.pth"
            model = ImplicitBCPolicy(input_channels=3, state_dim=len(STATE_FEATURE_NAMES))
            image_mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
            image_std = np.array([0.25, 0.25, 0.25], dtype=np.float32)
            metadata = {
                "model_type": HYBRID_MODEL_TYPE,
                "state_dim": len(STATE_FEATURE_NAMES),
                "state_feature_names": list(STATE_FEATURE_NAMES),
                "state_mean": [0.0] * len(STATE_FEATURE_NAMES),
                "state_std": [1.0] * len(STATE_FEATURE_NAMES),
            }
            save_policy_checkpoint(
                checkpoint_path,
                model,
                image_mean=image_mean,
                image_std=image_std,
                image_shape=(96, 96, 3),
                metadata=metadata,
            )

            loaded_model, loaded_image_mean, loaded_image_std, image_shape, loaded_metadata = load_policy_checkpoint(
                checkpoint_path
            )
            agent = ImplicitBCAgent.from_checkpoint(checkpoint_path)

        self.assertEqual(image_shape, (96, 96, 3))
        self.assertEqual(loaded_model.state_dim, len(STATE_FEATURE_NAMES))
        np.testing.assert_allclose(loaded_image_mean, image_mean)
        np.testing.assert_allclose(loaded_image_std, image_std)
        self.assertEqual(loaded_metadata["model_type"], HYBRID_MODEL_TYPE)

        observation = np.zeros((96, 96, 3), dtype=np.uint8)
        state_observation = np.zeros((len(STATE_FEATURE_NAMES),), dtype=np.float32)
        scores = agent.score_actions(
            observation,
            np.asarray([[0.0, 0.0, 0.0, 0.0], [1.0, -1.0, 0.5, 1.0]], dtype=np.float32),
            state_observation=state_observation,
        )
        action = agent.predict(observation, state_observation=state_observation)

        self.assertEqual(scores.shape, (2,))
        self.assertEqual(action.shape, (4,))

    def test_image_only_checkpoints_remain_loadable(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "image_only_policy.pth"
            model = ImplicitBCPolicy(input_channels=3, state_dim=0)
            save_policy_checkpoint(
                checkpoint_path,
                model,
                image_mean=np.array([0.5, 0.5, 0.5], dtype=np.float32),
                image_std=np.array([0.25, 0.25, 0.25], dtype=np.float32),
                image_shape=(96, 96, 3),
                metadata={"model_type": "implicit_bc"},
            )

            agent = ImplicitBCAgent.from_checkpoint(checkpoint_path)

        self.assertEqual(agent.state_dim, 0)
        action = agent.predict(np.zeros((96, 96, 3), dtype=np.uint8))
        self.assertEqual(action.shape, (4,))


if __name__ == "__main__":
    unittest.main()
