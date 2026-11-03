from __future__ import annotations

import unittest

import numpy as np

from utils.training_utils.robomimic_image_wrapper import RobomimicImageWrapper


class _FakeRobomimicEnv:
    def __init__(self) -> None:
        self.reset_calls = 0
        self.reset_to_payloads: list[dict[str, object]] = []

    def get_observation(self) -> dict[str, np.ndarray]:
        return {
            "robot0_eef_pos": np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
            "robot0_eef_quat": np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            "robot0_gripper_qpos": np.asarray([0.0, 0.0], dtype=np.float32),
            "object": np.arange(5, dtype=np.float32),
        }

    def get_state(self) -> dict[str, object]:
        return {"states": np.asarray([0.5, 1.5, 2.5], dtype=np.float32), "model": "<xml/>"}

    def reset(self):
        self.reset_calls += 1
        return self.get_observation()

    def reset_to(self, payload: dict[str, object]):
        self.reset_to_payloads.append(payload)
        return self.get_observation()

    def render(self, mode: str, height: int, width: int, camera_name: str) -> np.ndarray:
        assert mode == "rgb_array"
        assert camera_name
        return np.zeros((height, width, 3), dtype=np.uint8)

    def close(self) -> None:
        return None


class TestRobomimicImageWrapper(unittest.TestCase):
    def test_seed_cache_preserves_model_and_observation_layout(self) -> None:
        env = _FakeRobomimicEnv()
        wrapper = RobomimicImageWrapper(
            env=env,
            state_obs_keys=["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"],
            render_camera_names=["agentview", "robot0_eye_in_hand"],
            render_hw=(84, 84),
        )
        wrapper.seed(5)
        obs = wrapper.reset()
        self.assertEqual(env.reset_calls, 1)
        self.assertEqual(len(env.reset_to_payloads), 0)
        self.assertEqual(obs["states"].shape, (9,))
        self.assertEqual(obs["rgb"].shape, (6, 84, 84))
        self.assertNotIn("aux_states", obs)

        wrapper.seed(5)
        _ = wrapper.reset()
        self.assertEqual(len(env.reset_to_payloads), 1)
        self.assertEqual(env.reset_to_payloads[0]["model"], "<xml/>")

    def test_aux_obs_keys_add_aux_states(self) -> None:
        wrapper = RobomimicImageWrapper(
            env=_FakeRobomimicEnv(),
            state_obs_keys=["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"],
            render_camera_names=["agentview"],
            aux_obs_keys=["object", "robot0_eef_pos"],
        )
        obs = wrapper.reset()
        self.assertEqual(obs["aux_states"].shape, (8,))
        np.testing.assert_allclose(obs["aux_states"][:5], np.arange(5, dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
