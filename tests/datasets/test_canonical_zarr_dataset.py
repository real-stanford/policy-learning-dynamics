from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import zarr

from datasets.canonical_zarr_dataset import CanonicalImageDataset


def _write_dataset(root_path: Path, n_episodes: int = 2) -> None:
    root = zarr.open(str(root_path), mode="w")
    data = root.require_group("data")
    meta = root.require_group("meta")
    for episode_idx in range(n_episodes):
        length = 3 if episode_idx % 2 == 0 else 2
        base = float(episode_idx)
        episode = data.require_group(f"episode_{episode_idx:06d}")
        episode["states"] = np.asarray(
            [[base + 2 * t, base + 2 * t + 1] for t in range(length)], dtype=np.float32
        )
        episode["actions"] = np.asarray(
            [[base / 10.0 + 0.1 * (t + 1)] for t in range(length)], dtype=np.float32
        )
        episode["rgb"] = np.full((length, 6, 4, 5), fill_value=episode_idx * 20, dtype=np.uint8)
    meta["schema_version"] = np.asarray("1.0.0", dtype="<U16")
    meta["task_name"] = np.asarray("square_d2", dtype="<U32")
    meta["observation_type"] = np.asarray("rgb+low_dim", dtype="<U32")
    meta["action_pose_repr"] = np.asarray("delta", dtype="<U16")
    meta["action_mode"] = np.asarray("delta", dtype="<U16")
    meta["rotation_rep"] = np.asarray("axis_angle", dtype="<U32")
    meta["obs_dim"] = np.asarray(2, dtype=np.int64)
    meta["action_dim"] = np.asarray(1, dtype=np.int64)
    meta["num_episodes"] = np.asarray(n_episodes, dtype=np.int64)
    meta["camera_names"] = np.asarray(["cam0", "cam1"], dtype="<U16")
    meta["rgb_layout"] = np.asarray("camera_concat_chw", dtype="<U32")


class TestCanonicalImageDataset(unittest.TestCase):
    def test_image_dataset_shapes_padding_and_float_rgb(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = Path(tmpdir) / "dataset.zarr"
            _write_dataset(zarr_path)
            dataset = CanonicalImageDataset(
                zarr_path, horizon=4, pad_before=1, pad_after=1, n_obs_steps=2
            )
            sample = dataset[0]
            self.assertEqual(tuple(sample["obs"]["states"].shape), (2, 2))
            self.assertEqual(tuple(sample["obs"]["rgb"].shape), (2, 6, 4, 5))
            self.assertEqual(tuple(sample["action"].shape), (4, 1))
            self.assertEqual(sample["obs"]["rgb"].dtype, torch.float32)
            self.assertGreaterEqual(float(sample["obs"]["rgb"].min()), 0.0)
            self.assertLessEqual(float(sample["obs"]["rgb"].max()), 1.0)
            # pad_before=1: the first window repeats frame 0
            torch.testing.assert_close(sample["obs"]["states"][0], sample["obs"]["states"][1])
            torch.testing.assert_close(sample["action"][0], sample["action"][1])

    def test_in_memory_matches_lazy_and_detaches_from_disk(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = Path(tmpdir) / "dataset.zarr"
            _write_dataset(zarr_path)
            kwargs = dict(horizon=3, pad_before=1, pad_after=1, n_obs_steps=2)

            lazy = CanonicalImageDataset(zarr_path, **kwargs)
            in_mem = CanonicalImageDataset(zarr_path, in_memory=True, **kwargs)
            self.assertFalse(lazy.in_memory)
            self.assertTrue(in_mem.in_memory)
            self.assertEqual(len(lazy), len(in_mem))
            self.assertEqual(lazy.episode_names, in_mem.episode_names)
            np.testing.assert_array_equal(lazy.train_mask, in_mem.train_mask)

            lazy_samples = [lazy[i] for i in range(len(lazy))]
            lazy_norm = lazy.get_normalizer()
            lazy_val = lazy.get_validation_dataset()

            # The in-memory copy must be self-contained: removing the on-disk store
            # after init must not affect sampling, normalizer, or validation split.
            shutil.rmtree(zarr_path)
            for i, expected in enumerate(lazy_samples):
                got = in_mem[i]
                torch.testing.assert_close(got["obs"]["states"], expected["obs"]["states"])
                torch.testing.assert_close(got["obs"]["rgb"], expected["obs"]["rgb"])
                torch.testing.assert_close(got["action"], expected["action"])
            actions = in_mem.get_all_actions().numpy()
            np.testing.assert_allclose(
                in_mem.get_normalizer()["action"].normalize(actions),
                lazy_norm["action"].normalize(actions),
            )
            self.assertEqual(len(in_mem.get_validation_dataset()), len(lazy_val))

    def test_normalizer_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = Path(tmpdir) / "dataset.zarr"
            _write_dataset(zarr_path)
            dataset = CanonicalImageDataset(zarr_path, horizon=3)
            normalizer = dataset.get_normalizer()
            self.assertEqual(set(normalizer.keys()), {"states", "action", "rgb"})
            actions = dataset.get_all_actions().numpy()
            restored = normalizer["action"].unnormalize(normalizer["action"].normalize(actions))
            self.assertTrue(np.allclose(actions, restored))

    def test_seeded_split_is_deterministic_and_validation_is_fixed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = Path(tmpdir) / "dataset.zarr"
            _write_dataset(zarr_path, n_episodes=10)
            full = CanonicalImageDataset(zarr_path, horizon=2, seed=7, val_ratio=0.2, ratio=1.0)
            again = CanonicalImageDataset(zarr_path, horizon=2, seed=7, val_ratio=0.2, ratio=1.0)
            sub = CanonicalImageDataset(
                zarr_path, horizon=2, seed=7, val_ratio=0.2, max_train_episodes=4
            )
            np.testing.assert_array_equal(full.train_mask, again.train_mask)
            np.testing.assert_array_equal(full.val_mask, sub.val_mask)
            self.assertEqual(int(full.val_mask.sum()), 2)
            self.assertEqual(int(full.train_mask.sum()), 8)
            self.assertEqual(int(sub.train_mask.sum()), 4)
            self.assertTrue(np.all(sub.train_mask <= full.train_mask))
            self.assertFalse(np.any(sub.train_mask & sub.val_mask))

    def test_episode_include_path_selects_exactly_the_listed_episodes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = Path(tmpdir) / "dataset.zarr"
            _write_dataset(zarr_path, n_episodes=10)
            include = Path(tmpdir) / "train.json"
            include.write_text(json.dumps([1, 4, 6]))
            dataset = CanonicalImageDataset(
                zarr_path, horizon=2, val_ratio=0.0, episode_include_path=str(include)
            )
            np.testing.assert_array_equal(np.flatnonzero(dataset.train_mask), [1, 4, 6])
            with self.assertRaises(AssertionError):
                CanonicalImageDataset(
                    zarr_path,
                    horizon=2,
                    val_ratio=0.0,
                    max_train_episodes=2,
                    episode_include_path=str(include),
                )

    def test_absolute_actions_cannot_be_derived_from_a_delta_store(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = Path(tmpdir) / "dataset.zarr"
            _write_dataset(zarr_path)
            with self.assertRaisesRegex(AssertionError, "Unsupported action representation"):
                CanonicalImageDataset(zarr_path, horizon=3, action_pose_repr="absolute")


if __name__ == "__main__":
    unittest.main()
