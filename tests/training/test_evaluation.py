from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation

from envs.image_parallel.factory import ensure_robosuite_env_registered
from utils.common.rotation import rotvec_to_rotation_6d
from utils.training_utils.evaluation import (
    _build_distributed_episode_specs,
    _undo_absolute_action,
    run_image_policy_eval,
)


def _eval_cfg(n_obs_steps: int = 1):
    cfg = OmegaConf.create(
        {
            "task": {
                "benchmark": "mimicgen_core",
                "task_name": "square_d2",
                "evaluation": {"kind": "robomimic_image", "dataset_path": "/tmp/fake_dataset.hdf5"},
            },
            "action": {"action_pose_repr": "absolute", "rotation_rep": "rotation_6d"},
            "model": {"rollout": {"n_obs_steps": n_obs_steps, "n_latency_steps": 0}},
        }
    )
    OmegaConf.set_struct(cfg, True)
    return cfg


class _FakeImageEnv:
    """Succeeds on even seeds at step 3, times out otherwise."""

    def __init__(self) -> None:
        self._seed = 0
        self._t = 0

    def seed(self, seed: int) -> None:
        self._seed = int(seed)

    @staticmethod
    def _obs() -> dict[str, np.ndarray]:
        return {"states": np.zeros(9, np.float32), "rgb": np.zeros((6, 84, 84), np.float32)}

    def reset(self) -> dict[str, np.ndarray]:
        self._t = 0
        return self._obs()

    def step(self, action: np.ndarray):
        assert action.shape == (7,), action.shape   # rot6d undone -> rotvec
        self._t += 1
        reward = 1.0 if (self._seed % 2 == 0 and self._t >= 3) else 0.0
        return self._obs(), reward, reward > 0, {}

    def close(self) -> None:
        return None


class _FakePolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))

    def reset(self) -> None:
        return None

    def predict_action(self, obs):
        b = obs["rgb"].shape[0]
        act = torch.zeros(b, 4, 10)
        act[..., 3] = 1.0
        act[..., 7] = 1.0   # identity rotation in rot6d
        return {"action": act, "action_pred": act}


class TestEvaluation(unittest.TestCase):
    def test_undo_absolute_action_handles_multi_step_actions(self) -> None:
        rotvec = np.asarray([[0.1, -0.2, 0.3], [0.0, 0.5, 0.0]], dtype=np.float64)
        rot6d = rotvec_to_rotation_6d(rotvec)
        action = np.concatenate(
            [np.arange(6, dtype=np.float32).reshape(2, 3), rot6d.astype(np.float32), np.ones((2, 1), np.float32)],
            axis=-1,
        )
        undone = _undo_absolute_action(action)
        self.assertEqual(undone.shape, (2, 7))
        delta = Rotation.from_rotvec(rotvec) * Rotation.from_rotvec(undone[:, 3:6].astype(np.float64)).inv()
        self.assertLess(float(np.max(delta.magnitude())), 1e-5)
        np.testing.assert_allclose(undone[:, :3], action[:, :3])

    def test_build_distributed_episode_specs_partitions_episodes_by_rank(self) -> None:
        specs = [
            _build_distributed_episode_specs(num_episodes=5, test_start_seed=100, rank=r, world_size=2)
            for r in range(2)
        ]
        self.assertEqual([s["episode_index"] for s in specs[0]], [0, 2, 4])
        self.assertEqual([s["episode_index"] for s in specs[1]], [1, 3])
        self.assertEqual(specs[1][0]["episode_seed"], 101)

    def test_ensure_robosuite_env_registered_loads_mimicgen_envs(self) -> None:
        ensure_robosuite_env_registered("Threading_D2")
        with self.assertRaises(AssertionError):
            ensure_robosuite_env_registered("NoSuchEnv_D9")

    def test_run_image_policy_eval_single_env_writes_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("utils.training_utils.evaluation.create_image_env", return_value=_FakeImageEnv()):
                summary = run_image_policy_eval(
                    cfg=_eval_cfg(),
                    policy=_FakePolicy(),
                    output_dir=Path(tmpdir),
                    num_episodes=4,
                    test_start_seed=10,
                    max_steps=6,
                    num_envs_per_rank=1,
                    checkpoint_path="/tmp/ckpt.ckpt",
                    policy_state_key="ema_state",
                )
            self.assertEqual(summary["num_episodes"], 4)
            self.assertAlmostEqual(summary["success_rate"], 0.5)
            self.assertEqual([e["seed"] for e in summary["episodes"]], [10, 11, 12, 13])
            self.assertEqual([e["success"] for e in summary["episodes"]], [1.0, 0.0, 1.0, 0.0])
            self.assertEqual([e["steps"] for e in summary["episodes"]], [3, 6, 3, 6])
            written = json.loads((Path(tmpdir) / "summary.json").read_text())
            self.assertEqual(written["success_rate"], summary["success_rate"])

    def test_run_image_policy_eval_uses_parallel_runner_when_enabled(self) -> None:
        rows = [
            {"episode_index": 1, "seed": 11, "success": 0.0, "max_reward": 0.0, "total_reward": 0.0, "steps": 6},
            {"episode_index": 0, "seed": 10, "success": 1.0, "max_reward": 1.0, "total_reward": 1.0, "steps": 3},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch("utils.training_utils.evaluation._run_parallel_image_episodes", return_value=rows) as runner,
                patch("utils.training_utils.evaluation.create_image_env") as create_env,
            ):
                summary = run_image_policy_eval(
                    cfg=_eval_cfg(),
                    policy=_FakePolicy(),
                    output_dir=Path(tmpdir),
                    num_episodes=2,
                    test_start_seed=10,
                    max_steps=6,
                    num_envs_per_rank=2,
                )
            runner.assert_called_once()
            create_env.assert_not_called()
            self.assertEqual([e["episode_index"] for e in summary["episodes"]], [0, 1])
            self.assertAlmostEqual(summary["success_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
