from __future__ import annotations

import os
import unittest
from pathlib import Path

from hydra import compose, initialize_config_dir

from envs.task_registry import get_task_entry, list_task_names
from utils.training_utils.factories import _validate_image_task_registry_contract
from utils.training_utils.runtime import register_training_resolvers

REPO_ROOT = Path(__file__).resolve().parents[2]


class TestTaskRegistryContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ.setdefault("TRAINING_ROOT", "/tmp/robot_policy_overfitting_study_training")
        os.environ.setdefault("MIMICGEN_DATA_ROOT", "/tmp/robot_policy_overfitting_study_mimicgen")
        register_training_resolvers()

    def test_image_task_configs_match_registry(self) -> None:
        with initialize_config_dir(version_base=None, config_dir=str(REPO_ROOT / "configs")):
            for task_name in list_task_names("mimicgen"):
                cfg = compose(
                    config_name="diffusion_image",
                    overrides=[f"task=mimicgen/{task_name}_image", "action=absolute_rotation6d"],
                )
                entry = get_task_entry("mimicgen", task_name)
                self.assertEqual(
                    int(cfg.task.observation.shape_meta.obs.states.shape[0]),
                    int(entry.image_lowdim_obs_dim),
                )
                self.assertEqual(
                    int(cfg.task.observation.shape_meta.obs.rgb.shape[0]),
                    int(3 * len(entry.rgb_obs_keys)),
                )
                self.assertEqual(
                    tuple(int(x) for x in cfg.task.observation.shape_meta.obs.rgb.shape[1:]),
                    tuple(entry.image_size),
                )
                _validate_image_task_registry_contract(cfg)

    def test_lowdim_task_configs_carry_the_asymmetric_critic_state_contract(self) -> None:
        """dice_rl's asymmetric critic reads <task>_lowdim.yaml for the simulator state keys."""
        from omegaconf import OmegaConf

        for task_name in list_task_names("mimicgen"):
            tcfg = OmegaConf.to_container(
                OmegaConf.load(REPO_ROOT / "configs" / "task" / "mimicgen" / f"{task_name}_lowdim.yaml"),
                resolve=False,
            )
            entry = get_task_entry("mimicgen", task_name)
            self.assertEqual(tuple(str(k) for k in tcfg["observation"]["state_keys"]), entry.lowdim_obs_keys)
            self.assertEqual(int(tcfg["observation"]["obs_dim"]), int(entry.lowdim_obs_dim))
            self.assertIn("${oc.env:MIMICGEN_DATA_ROOT}", str(tcfg["data"]["zarr_path"]))
            self.assertIn("${action.zarr_suffix}", str(tcfg["data"]["zarr_path"]))


if __name__ == "__main__":
    unittest.main()
