import tempfile
import unittest
from pathlib import Path

import numpy as np

from modules.imitation_data import flatten_state_episode_dataset
from modules.imitation_policy import (
    MODEL_TYPE,
    ImplicitBCAgent,
    ImplicitBCPolicy,
    load_policy_checkpoint,
    save_policy_checkpoint,
)
from modules.pick_place_policy_shared import STATE_FEATURE_NAMES


class StateDatasetTests(unittest.TestCase):
    def test_flatten_state_episode_dataset_preserves_alignment(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "episode_00000.npz"
            first_state = np.linspace(0.1, 1.7, len(STATE_FEATURE_NAMES), dtype=np.float32)
            second_state = np.linspace(1.8, 3.4, len(STATE_FEATURE_NAMES), dtype=np.float32)
            np.savez_compressed(
                path,
                observation=np.zeros((2, 96, 96, 3), dtype=np.uint8),
                state_observation=np.stack([first_state, second_state], axis=0),
                action=np.asarray([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]], dtype=np.float32),
            )

            state, action = flatten_state_episode_dataset([path])

        self.assertEqual(state.shape, (2, len(STATE_FEATURE_NAMES)))
        self.assertEqual(action.shape, (2, 4))
        np.testing.assert_allclose(state[1], second_state)
        np.testing.assert_allclose(action[1], np.array([0.5, 0.6, 0.7, 0.8], dtype=np.float32))


class PolicyCheckpointTests(unittest.TestCase):
    def test_motor_state_checkpoint_roundtrip_and_predict(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "motor_state_policy.pth"
            action_bounds = np.asarray(
                [[-1.0, 1.0], [-1.5, 1.5], [-2.0, 2.0], [-2.5, 2.5]],
                dtype=np.float32,
            )
            model = ImplicitBCPolicy(state_dim=len(STATE_FEATURE_NAMES))
            metadata = {
                "model_type": MODEL_TYPE,
                "action_bounds": action_bounds.tolist(),
                "state_dim": len(STATE_FEATURE_NAMES),
                "state_feature_names": list(STATE_FEATURE_NAMES),
                "state_mean": [0.0] * len(STATE_FEATURE_NAMES),
                "state_std": [1.0] * len(STATE_FEATURE_NAMES),
                "warm_start": True,
                "warm_start_std_scale": 0.2,
                "action_smoothing_alpha": 0.4,
            }
            save_policy_checkpoint(checkpoint_path, model, metadata=metadata)

            loaded_model, loaded_metadata = load_policy_checkpoint(checkpoint_path)
            agent = ImplicitBCAgent.from_checkpoint(checkpoint_path)

        self.assertEqual(loaded_model.state_dim, len(STATE_FEATURE_NAMES))
        self.assertEqual(loaded_metadata["model_type"], MODEL_TYPE)
        np.testing.assert_allclose(agent.action_bounds, action_bounds)
        state_observation = np.zeros((len(STATE_FEATURE_NAMES),), dtype=np.float32)
        scores = agent.score_actions(
            state_observation,
            np.asarray([[0.0, 0.0, 0.0, 0.0], [0.5, -0.5, 0.7, 1.0]], dtype=np.float32),
        )
        action = agent.predict(state_observation)
        self.assertEqual(scores.shape, (2,))
        self.assertEqual(action.shape, (4,))
        self.assertTrue(np.all(action >= action_bounds[:, 0]))
        self.assertTrue(np.all(action <= action_bounds[:, 1]))

    def test_non_motor_checkpoint_type_fails_clearly(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "legacy_policy.pth"
            model = ImplicitBCPolicy(state_dim=len(STATE_FEATURE_NAMES))
            save_policy_checkpoint(
                checkpoint_path,
                model,
                metadata={
                    "model_type": "implicit_bc_hybrid_v2",
                    "action_bounds": [[-1.0, 1.0]] * 4,
                    "state_dim": len(STATE_FEATURE_NAMES),
                    "state_feature_names": list(STATE_FEATURE_NAMES),
                    "state_mean": [0.0] * len(STATE_FEATURE_NAMES),
                    "state_std": [1.0] * len(STATE_FEATURE_NAMES),
                },
            )

            with self.assertRaisesRegex(ValueError, "not supported"):
                load_policy_checkpoint(checkpoint_path)

    def test_smooth_action_uses_previous_executed_action(self):
        model = ImplicitBCPolicy(state_dim=len(STATE_FEATURE_NAMES))
        agent = ImplicitBCAgent(
            model=model,
            action_bounds=np.asarray([[-1.0, 1.0], [-1.5, 1.5], [-2.0, 2.0], [-2.5, 2.5]], dtype=np.float32),
            state_mean=np.zeros((len(STATE_FEATURE_NAMES),), dtype=np.float32),
            state_std=np.ones((len(STATE_FEATURE_NAMES),), dtype=np.float32),
            state_feature_names=list(STATE_FEATURE_NAMES),
            action_smoothing_alpha=0.25,
        )

        raw_action = np.array([0.7, -1.0, 1.5, -1.5], dtype=np.float32)
        np.testing.assert_allclose(agent.smooth_action(raw_action), raw_action)

        previous_action = np.array([0.2, 0.4, -0.6, 1.0], dtype=np.float32)
        agent.set_previous_action(previous_action)
        expected = 0.25 * raw_action + 0.75 * previous_action
        np.testing.assert_allclose(agent.smooth_action(raw_action), expected.astype(np.float32))

    def test_initial_search_distribution_warm_starts_from_previous_action(self):
        model = ImplicitBCPolicy(state_dim=len(STATE_FEATURE_NAMES))
        bounds = np.asarray([[-1.0, 1.0], [-1.5, 1.5], [-2.0, 2.0], [-2.5, 2.5]], dtype=np.float32)
        agent = ImplicitBCAgent(
            model=model,
            action_bounds=bounds,
            state_mean=np.zeros((len(STATE_FEATURE_NAMES),), dtype=np.float32),
            state_std=np.ones((len(STATE_FEATURE_NAMES),), dtype=np.float32),
            state_feature_names=list(STATE_FEATURE_NAMES),
            warm_start=True,
            warm_start_std_scale=0.5,
        )

        mean, std = agent._initial_search_distribution()
        np.testing.assert_allclose(mean, 0.5 * (bounds[:, 0] + bounds[:, 1]))
        np.testing.assert_allclose(std, np.maximum(0.5 * (bounds[:, 1] - bounds[:, 0]), agent.search_config["min_std"]))

        previous_action = np.array([0.2, 0.4, -0.6, 1.0], dtype=np.float32)
        agent.set_previous_action(previous_action)
        mean, std = agent._initial_search_distribution()
        np.testing.assert_allclose(mean, previous_action)
        np.testing.assert_allclose(
            std,
            np.maximum(0.5 * (bounds[:, 1] - bounds[:, 0]) * 0.5, agent.search_config["min_std"]),
        )


if __name__ == "__main__":
    unittest.main()
