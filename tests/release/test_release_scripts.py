"""Synthetic checks of the release tooling: checkpoint stripping / config rewrite and the download patterns."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from scripts.release import manifest as M
from scripts.release.download_release import ckpt_patterns
from scripts.release.stage_release import KEEP_KEYS, LEGACY_OA_KEY, OA_KEY, merge_manifest, release_cfg, strip_checkpoint, write_fragment


def fake_cfg(task: str = "hammer_cleanup_d1"):
    data = str(M.RESEARCH_MIMICGEN)
    return OmegaConf.create({
        "task": {
            "name": f"{task}_image", "task_name": task, "modality": "image",
            "data": {"hdf5_path": f"{data}/core/{task}.hdf5", "zarr_path": f"{data}/core/{task}_image_abs.zarr"},
            "evaluation": {"dataset_path": f"{data}/core/{task}.hdf5", "max_steps": 500},
        },
        "dataset": {"source": {"hdf5_path": f"{data}/core/{task}.hdf5", "zarr_path": f"{data}/core/{task}_image_abs.zarr"},
                    "processing": {"max_train_episodes": 200, "seed": 42}},
        "training": {LEGACY_OA_KEY: {"enabled": False, "oa_model_dir": "outputs/training/x/checkpoints", "ckpt_indices": [200, 100]},
                     "optimizer": {"lr": 1e-4}},
        "runtime": {"training_root": str(M.RESEARCH_TRAINING),
                    "output_dir": f"{M.RESEARCH_TRAINING}/{task}_image/2026.01.01_00.00.00_x"},
    })


class ReleaseCfgTest(unittest.TestCase):
    def test_rewrite_only_touches_paths(self):
        src = fake_cfg()
        dst = release_cfg(src, "hammer_cleanup_d1")
        raw_src = OmegaConf.to_container(src, resolve=False)
        raw_dst = OmegaConf.to_container(dst, resolve=False)
        self.assertEqual(raw_dst["task"]["data"]["hdf5_path"], "${oc.env:MIMICGEN_DATA_ROOT}/core/hammer_cleanup_d1.hdf5")
        self.assertEqual(raw_dst["dataset"]["source"]["zarr_path"], "${oc.env:MIMICGEN_DATA_ROOT}/core/hammer_cleanup_d1_image_abs.zarr")
        self.assertEqual(raw_dst["task"]["evaluation"]["dataset_path"], "${oc.env:MIMICGEN_DATA_ROOT}/core/hammer_cleanup_d1.hdf5")
        self.assertEqual(raw_dst["runtime"]["training_root"], "${oc.env:TRAINING_ROOT}")
        self.assertEqual(raw_dst["training"][OA_KEY]["oa_model_dir"], "pretrained_ckpts/joint_generator_ckpts/hammer_cleanup_d1_image/checkpoints")
        # untouched parts identical
        self.assertEqual(raw_dst["runtime"]["output_dir"], raw_src["runtime"]["output_dir"])
        self.assertEqual(raw_dst["dataset"]["processing"], raw_src["dataset"]["processing"])
        self.assertEqual(raw_dst["training"][OA_KEY]["ckpt_indices"], [200, 100])
        self.assertNotIn(LEGACY_OA_KEY, raw_dst["training"])
        self.assertIn(LEGACY_OA_KEY, raw_src["training"])
        self.assertEqual(raw_dst["task"]["evaluation"]["max_steps"], 500)
        # resolves against the user's environment
        os.environ["MIMICGEN_DATA_ROOT"] = "/somewhere/mimicgen"
        self.assertEqual(str(dst.task.evaluation.dataset_path), "/somewhere/mimicgen/core/hammer_cleanup_d1.hdf5")
        self.assertEqual(str(dst.dataset.source.zarr_path), "/somewhere/mimicgen/core/hammer_cleanup_d1_image_abs.zarr")

    def test_generator_cfg_without_mixer_block(self):
        src = fake_cfg()
        with OmegaConf.open_dict(src) if hasattr(OmegaConf, "open_dict") else _nullcontext():
            del src.training[LEGACY_OA_KEY]
        dst = release_cfg(src, "mug_cleanup_d1")
        self.assertNotIn(OA_KEY, dst.training)
        self.assertNotIn(LEGACY_OA_KEY, dst.training)

    def test_strip_checkpoint_roundtrip(self):
        payload = {
            "cfg": fake_cfg(),
            "model_state": {"w": torch.randn(3, 4), "b": torch.zeros(4, dtype=torch.float64)},
            "ema_state": {"w": torch.randn(3, 4), "b": torch.ones(4)},
            "optimizer_state": {"state": {0: {"exp_avg": torch.randn(3, 4)}}},
            "scheduler_state": {"last_epoch": 3},
            "normalizer_state": {"action": {"min": np.arange(4.0), "max": np.arange(4.0) + 1}},
            "epoch": 3,
            "global_step": 123,
        }
        with tempfile.TemporaryDirectory() as d:
            src, dst = Path(d) / "src.ckpt", Path(d) / "dst.ckpt"
            torch.save(payload, src)
            strip_checkpoint(src, dst, "hammer_cleanup_d1")
            self.assertFalse(dst.with_name(dst.name + ".tmp").exists())
            out = torch.load(dst, map_location="cpu", weights_only=False)
            self.assertEqual(tuple(out), KEEP_KEYS)
            for k in ("model_state", "ema_state"):
                for name, t in payload[k].items():
                    self.assertTrue(torch.equal(t, out[k][name]))
                    self.assertEqual(t.dtype, out[k][name].dtype)
            np.testing.assert_array_equal(out["normalizer_state"]["action"]["min"], np.arange(4.0))
            self.assertEqual(out["epoch"], 3)
            self.assertEqual(out["global_step"], 123)
            self.assertEqual(str(OmegaConf.to_container(out["cfg"], resolve=False)["task"]["data"]["hdf5_path"]),
                             "${oc.env:MIMICGEN_DATA_ROOT}/core/hammer_cleanup_d1.hdf5")

    def test_strip_rejects_unknown_payload(self):
        with tempfile.TemporaryDirectory() as d:
            src, dst = Path(d) / "src.ckpt", Path(d) / "dst.ckpt"
            torch.save({"cfg": fake_cfg(), "model_state": {}}, src)
            with self.assertRaises(AssertionError):
                strip_checkpoint(src, dst, "hammer_cleanup_d1")
            self.assertFalse(dst.exists())


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


class ManifestTest(unittest.TestCase):
    def test_fragments_merge(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            write_fragment(root, "bc_pretraining_ckpts", "square_d2_image", {"epochs": [50]})
            write_fragment(root, "bc_pretraining_data", "square_d2", {"x": 1})
            write_fragment(root, "bc_pretraining_ckpts", "square_d2_image", {"epochs": [50, 100]})   # rerun refreshes
            m = merge_manifest(root)
            self.assertEqual(m, {"bc_pretraining_ckpts": {"square_d2_image": {"epochs": [50, 100]}},
                                 "bc_pretraining_data": {"square_d2": {"x": 1}}})
            self.assertEqual(json.load(open(root / "release_manifest.json")), m)
            self.assertFalse(list(root.glob("*.tmp")))

    def test_manifest_consistency(self):
        for task in M.TASKS:
            self.assertEqual(M.GEN_RUNS[task].epochs, tuple(sorted(M.AUG_GEN_INDICES[task])))
            self.assertEqual(M.BC_RUNS[task].epochs, tuple(range(50, 801, 50)))
        self.assertEqual(M.AUG_RUNS["three_piece_assembly_d1"].epochs, tuple(range(50, 751, 50)))
        self.assertIsNotNone(M.AUG_RUNS["three_piece_assembly_d1"].checkpoints_run)
        self.assertEqual(M.AUG_RUNS["hammer_cleanup_d1"].eval_subdir, "eval_ddim_num_steps_10_eta_0.0_pass_10")
        self.assertEqual(M.task_dir("square_d2"), "square_d2_image")
        self.assertEqual(M.ckpt_name(50), "epoch_0050.ckpt")

    def test_download_patterns(self):
        pats = ckpt_patterns(M.GROUP_BC_CKPTS, "threading_d2", None)
        self.assertIn("bc_pretraining_ckpts/threading_d2_image/.hydra/*", pats)
        self.assertEqual(sum(p.endswith(".ckpt") for p in pats), 16)
        pats = ckpt_patterns(M.GROUP_AUG_CKPTS, "hammer_cleanup_d1", [600])
        self.assertEqual([p for p in pats if p.endswith(".ckpt")], ["joint_aug_ckpts/hammer_cleanup_d1_image/checkpoints/epoch_0600.ckpt"])
        with self.assertRaises(AssertionError):
            ckpt_patterns(M.GROUP_GEN_CKPTS, "square_d2", [300])   # square shipped generator epoch 200 only


if __name__ == "__main__":
    unittest.main()
