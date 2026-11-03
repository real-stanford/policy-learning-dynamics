from __future__ import annotations

import copy
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation

from tests.models._tiny_image_policy import build_tiny_image_policy
from utils.common.rotation import rotation_6d_to_rotvec, rotvec_to_rotation_6d
from utils.training_utils.checkpointing import CheckpointManager
from utils.training_utils.compile import (
    compile_policy_modules,
    load_compiled_state_dict,
    sanitize_compiled_state_dict,
)
from utils.training_utils.ema import EMAConfig, EMAHandler
from utils.training_utils.runtime import (
    apply_smoke_overrides,
    consume_training_cli_args,
    format_run_name,
    sanitize_exp_suffix,
)


class TestRuntimeAndCheckpointing(unittest.TestCase):
    def _maybe_compile_policy(self, policy, *, compiled: bool):
        if not compiled:
            return policy
        compile_cfg = OmegaConf.create(
            {"enabled": True, "mode": "default", "fullgraph": False, "dynamic": False, "cudagraphs": False}
        )
        return compile_policy_modules(policy, compile_cfg, "test_policy")

    def _assign_distinct_state(self, policy) -> None:
        with torch.no_grad():
            for index, parameter in enumerate(policy.parameters(), start=1):
                parameter.fill_(float(index) / 10.0)
            for index, buffer in enumerate(policy.buffers(), start=1):
                if buffer.is_floating_point():
                    buffer.fill_(float(index) / 100.0)
                    continue
                buffer.fill_(index)

    def _assert_policy_states_equal(self, left, right) -> None:
        left_state = sanitize_compiled_state_dict(left.state_dict())
        right_state = sanitize_compiled_state_dict(right.state_dict())
        self.assertEqual(set(left_state.keys()), set(right_state.keys()))
        for key in sorted(left_state.keys()):
            self.assertTrue(torch.equal(left_state[key], right_state[key]), msg=f"State mismatch for key '{key}'.")

    def test_format_run_name_appends_sanitized_suffix(self) -> None:
        self.assertEqual(format_run_name("2026.03.18_12.00.00", None), "2026.03.18_12.00.00")
        self.assertEqual(
            format_run_name("2026.03.18_12.00.00", "transport debug"), "2026.03.18_12.00.00_transport_debug"
        )
        self.assertEqual(sanitize_exp_suffix("a/b c"), "a_b_c")

    def test_apply_smoke_overrides_updates_training(self) -> None:
        cfg = OmegaConf.create(
            {
                "smoke": {"enabled": True, "num_epochs": 2, "max_train_steps": 3, "max_val_steps": 4},
                "training": {
                    "training": {
                        "num_epochs": 100,
                        "save_every": 50,
                        "val_every": 5,
                        "sample_every": 5,
                        "max_train_steps": None,
                        "max_val_steps": None,
                    }
                },
            }
        )
        apply_smoke_overrides(cfg)
        self.assertEqual(int(cfg.training.training.num_epochs), 2)
        self.assertEqual(int(cfg.training.training.save_every), 1)
        self.assertEqual(int(cfg.training.training.max_train_steps), 3)
        self.assertEqual(int(cfg.training.training.max_val_steps), 4)

    def test_consume_training_cli_args_accepts_smoke_test_alias(self) -> None:
        original_argv = list(sys.argv)
        try:
            sys.argv[:] = ["pretrain_image.py", "--smoke-test", "task=mimicgen/square_d2_image", "--exp_suffix", "rel"]
            consume_training_cli_args()
            self.assertEqual(
                sys.argv,
                ["pretrain_image.py", "smoke.enabled=true", "task=mimicgen/square_d2_image", "runtime.exp_suffix=rel"],
            )
        finally:
            sys.argv[:] = original_argv

    def test_checkpoint_manager_tracks_last_and_best_without_writing_best(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = OmegaConf.create(
                {"topk": {"monitor_key": "val_loss", "mode": "min", "k": 1, "format_str": "unused"}, "save_last_ckpt": True}
            )
            manager = CheckpointManager(output_dir=tmpdir, cfg=cfg)
            payload = {"tensor": torch.tensor([1.0])}
            first = manager.save(payload=payload, metrics={"epoch": 0, "val_loss": 3.0}, save_last=True)
            second = manager.save(payload=None, metrics={"epoch": 1, "val_loss": 4.0}, save_last=False)
            third = manager.save(payload=payload, metrics={"epoch": 2, "val_loss": 2.0}, save_last=True)
            self.assertTrue(Path(first["last_path"]).exists())
            self.assertIsNone(first["best_path"])   # best.ckpt is never written
            self.assertTrue(bool(first["best_updated"]))
            self.assertFalse(bool(second["best_updated"]))
            self.assertTrue(bool(third["best_updated"]))
            self.assertEqual(manager.best_epoch, 2)
            self.assertAlmostEqual(float(manager.best_score), 2.0)

    def test_checkpoint_roundtrip_loads_between_compiled_and_uncompiled_policies(self) -> None:
        for source_compiled, target_compiled in ((False, False), (False, True), (True, False), (True, True)):
            source_policy = self._maybe_compile_policy(build_tiny_image_policy(seed=0), compiled=source_compiled)
            self._assign_distinct_state(source_policy)
            checkpoint_fd, checkpoint_name = tempfile.mkstemp(suffix=".pt")
            os.close(checkpoint_fd)
            checkpoint_path = Path(checkpoint_name)
            try:
                torch.save({"model_state": sanitize_compiled_state_dict(source_policy.state_dict())}, checkpoint_path)
                loaded_payload = torch.load(checkpoint_path, map_location="cpu")
                target_policy = self._maybe_compile_policy(build_tiny_image_policy(seed=1), compiled=target_compiled)
                load_compiled_state_dict(target_policy, loaded_payload["model_state"])
                self._assert_policy_states_equal(source_policy, target_policy)
            finally:
                checkpoint_path.unlink(missing_ok=True)

    def test_ema_handler_accepts_compiled_child_modules(self) -> None:
        source_policy = build_tiny_image_policy(seed=0)
        compiled_policy = self._maybe_compile_policy(copy.deepcopy(source_policy), compiled=True)
        compiled_root = compiled_policy._orig_mod
        first_param_name, first_param = next(iter(compiled_root.model.named_parameters()))
        with torch.no_grad():
            first_param.add_(1.0)
        handler = EMAHandler(
            source_policy, EMAConfig(update_after_step=0, inv_gamma=1.0, power=0.75, min_value=0.0, max_value=0.9999)
        )
        handler.step(compiled_policy)
        ema_param = dict(handler.ema_model.model.named_parameters())[first_param_name]
        source_param = dict(compiled_root.model.named_parameters())[first_param_name]
        self.assertTrue(torch.allclose(ema_param, source_param))

    def test_rotation_6d_round_trip_matches_rotvec(self) -> None:
        rotvec = np.asarray([[0.1, -0.2, 0.3], [-0.4, 0.5, -0.6]], dtype=np.float64)
        recovered = rotation_6d_to_rotvec(rotvec_to_rotation_6d(rotvec))
        delta = Rotation.from_rotvec(rotvec) * Rotation.from_rotvec(recovered).inv()
        self.assertLess(float(np.max(delta.magnitude())), 1.0e-5)


if __name__ == "__main__":
    unittest.main()
