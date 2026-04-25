import importlib
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class WrapperEntrypointTests(unittest.TestCase):
    def _import_wrapper_with_stub(self, module_name):
        calls = []
        stub_scene_entry = types.ModuleType("modules.pick_place_scene_entry")

        def fake_create_pick_place_scene(rootnode, argv=None):
            calls.append({"rootnode": rootnode, "argv": list(argv) if argv is not None else None})
            return {"rootnode": rootnode, "argv": argv}

        stub_scene_entry.create_pick_place_scene = fake_create_pick_place_scene

        with mock.patch.dict(sys.modules, {"modules.pick_place_scene_entry": stub_scene_entry}):
            sys.modules.pop(module_name, None)
            module = importlib.import_module(module_name)

        return module, calls

    def test_imitation_lab_scene_entrypoint_keeps_connection_available(self):
        module, calls = self._import_wrapper_with_stub("imitation_lab")

        rootnode = object()
        result = module.createScene(rootnode)

        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0]["rootnode"], rootnode)
        self.assertEqual(calls[0]["argv"], ["--no-camera-tracking"])
        self.assertEqual(result["argv"], ["--no-camera-tracking"])

    def test_policy_inspect_wrapper_does_not_force_no_connection(self):
        module, calls = self._import_wrapper_with_stub("policy_inspect_scene")

        module.createScene(object())

        self.assertEqual(
            calls[0]["argv"],
            [
                "--mode",
                "policy_inspect",
                "--policy-path",
                "data/results/il_pick_place/bc_policy.pth",
                "--no-real-rgb-observation",
                "--no-camera-tracking",
            ],
        )
        self.assertNotIn("--no-connection", calls[0]["argv"])

    def test_policy_inspect_camera_wrapper_does_not_force_no_connection(self):
        module, calls = self._import_wrapper_with_stub("policy_inspect_camera_scene")

        module.createScene(object())

        self.assertEqual(
            calls[0]["argv"],
            [
                "--mode",
                "policy_inspect",
                "--policy-path",
                "data/results/il_pick_place/bc_policy.pth",
                "--no-real-rgb-observation",
                "--camera-tracking",
                "--camera-preview",
            ],
        )
        self.assertNotIn("--no-connection", calls[0]["argv"])


class PickPlaceParserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = cls._load_pick_place_il_module()

    @staticmethod
    def _load_pick_place_il_module():
        sofa_module = types.ModuleType("Sofa")
        sofa_module.msg_error = lambda *args, **kwargs: None
        sofa_module.Core = types.SimpleNamespace(Controller=type("Controller", (), {}))

        stub_imitation_policy = types.ModuleType("modules.imitation_policy")

        class _StubImplicitBCAgent:
            @classmethod
            def from_checkpoint(cls, _path):
                return cls()

            def predict(self, _state_observation):
                return [0.0, 0.0, 0.0, 0.0]

            def smooth_action(self, action):
                return action

            def set_previous_action(self, _action):
                return None

            def reset_rollout(self):
                return None

        stub_imitation_policy.ImplicitBCAgent = _StubImplicitBCAgent

        with mock.patch.dict(
            sys.modules,
            {
                "Sofa": sofa_module,
                "Sofa.ImGui": None,
                "modules.imitation_policy": stub_imitation_policy,
            },
        ):
            module = sys.modules.get("modules.pick_place_il")
            if module is None:
                module = importlib.import_module("modules.pick_place_il")
            return module

    def test_parse_scene_args_defaults_connection_to_true(self):
        args = self.module._parse_scene_args([])

        self.assertTrue(args.connection)
        self.assertFalse(args.camera_tracking)
        self.assertFalse(args.real_rgb_observation)

    def test_parse_scene_args_honors_no_connection_flag(self):
        args = self.module._parse_scene_args(["--no-connection", "--camera-tracking"])

        self.assertFalse(args.connection)
        self.assertTrue(args.camera_tracking)


if __name__ == "__main__":
    unittest.main()
