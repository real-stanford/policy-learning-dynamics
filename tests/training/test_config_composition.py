from __future__ import annotations

import os
import unittest
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from envs.task_registry import get_task_entry, list_task_names
from utils.training_utils.runtime import register_training_resolvers

REPO_ROOT = Path(__file__).resolve().parents[2]


class TestConfigComposition(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ.setdefault("TRAINING_ROOT", "/tmp/robot_policy_overfitting_study_training")
        os.environ.setdefault("MIMICGEN_DATA_ROOT", "/tmp/robot_policy_overfitting_study_mimicgen")
        register_training_resolvers()

    def _compose(self, config_name: str, overrides: list[str]):
        with initialize_config_dir(version_base=None, config_dir=str(REPO_ROOT / "configs")):
            return compose(config_name=config_name, overrides=overrides)

    def test_every_paper_task_composes_with_absolute_rotation6d_paths(self) -> None:
        for task_name in list_task_names("mimicgen"):
            cfg = self._compose(
                "diffusion_image",
                [f"task=mimicgen/{task_name}_image", "action=absolute_rotation6d"],
            )
            self.assertEqual(str(cfg.task.name), f"{task_name}_image")
            self.assertEqual(
                str(cfg.task.data.zarr_path),
                os.path.join(os.environ["MIMICGEN_DATA_ROOT"], "core", f"{task_name}_image_abs.zarr"),
            )
            self.assertEqual(int(cfg.task.action.model_dim), 10)
            self.assertEqual(str(cfg.dataset.processing.action_pose_repr), "absolute")
            entry = get_task_entry("mimicgen", task_name)
            self.assertEqual(int(cfg.task.evaluation.max_steps), entry.horizon)
            self.assertEqual(str(cfg.task.evaluation.kind), "robomimic_image")
            self.assertEqual(int(cfg.task.evaluation.test_start_seed), 100000)

    def test_training_defaults_match_the_paper_runs(self) -> None:
        cfg = self._compose("diffusion_image", ["task=mimicgen/threading_d2_image"])
        OmegaConf.resolve(cfg)
        self.assertEqual(int(cfg.training.training.num_epochs), 800)
        self.assertEqual(int(cfg.training.training.save_every), 50)
        self.assertEqual(int(cfg.training.dataloader.batch_size), 512)
        self.assertEqual(int(cfg.dataset.processing.max_train_episodes), 200)
        self.assertAlmostEqual(float(cfg.dataset.processing.val_ratio), 0.02)
        self.assertEqual(int(cfg.dataset.processing.seed), 42)
        self.assertTrue(bool(cfg.training.training.use_ema))
        self.assertEqual(str(cfg.training.training.lr_scheduler), "cosine")
        self.assertEqual(int(cfg.model.scheduler.num_train_timesteps), 100)
        self.assertEqual(str(cfg.model.scheduler.type), "ddim")
        self.assertEqual(list(cfg.model.unet.down_dims), [512, 1024, 2048])
        self.assertFalse(bool(cfg.training.jointaug_w_oa_joint_model.enabled))
        self.assertFalse(bool(cfg.training.compile.enabled))

    def test_generator_config_shares_the_policy_split(self) -> None:
        cfg = self._compose("joint_generator_image", ["task=mimicgen/mug_cleanup_d1_image"])
        self.assertEqual(int(cfg.dataset.processing.max_train_episodes), 200)
        self.assertAlmostEqual(float(cfg.dataset.processing.val_ratio), 0.02)
        self.assertEqual(int(cfg.dataset.processing.seed), 42)
        self.assertEqual(str(cfg.model.scheduler.prediction_type), "v_prediction")
        self.assertEqual(int(cfg.model.scheduler.num_train_timesteps), 1000)


if __name__ == "__main__":
    unittest.main()
