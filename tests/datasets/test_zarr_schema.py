from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from datasets.zarr_schema import validate_canonical_zarr


@unittest.skipUnless(
    importlib.util.find_spec("zarr") is not None, "zarr is not installed"
)
class TestCanonicalZarrSchema(unittest.TestCase):
    def test_validator_accepts_minimal_lowdim_dataset(self) -> None:
        import zarr

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "dataset.zarr"
            root = zarr.open(str(path), mode="w")
            root.require_group("data")
            root.require_group("meta")
            root["meta"]["schema_version"] = np.asarray("1.0.0", dtype="<U16")
            root["meta"]["task_name"] = np.asarray("transport", dtype="<U32")
            root["meta"]["observation_type"] = np.asarray("low_dim", dtype="<U16")
            root["meta"]["action_pose_repr"] = np.asarray("delta", dtype="<U16")
            root["meta"]["action_mode"] = np.asarray("delta", dtype="<U16")
            root["meta"]["rotation_rep"] = np.asarray("axis_angle", dtype="<U16")
            root["meta"]["obs_dim"] = np.asarray(59, dtype=np.int64)
            root["meta"]["action_dim"] = np.asarray(14, dtype=np.int64)
            root["meta"]["num_episodes"] = np.asarray(1, dtype=np.int64)

            episode = root["data"].require_group("episode_000000")
            episode["states"] = np.zeros((4, 59), dtype=np.float32)
            episode["actions"] = np.zeros((4, 14), dtype=np.float32)
            episode["episode_success"] = np.asarray([1.0], dtype=np.float32)

            self.assertEqual(validate_canonical_zarr(path), [])
