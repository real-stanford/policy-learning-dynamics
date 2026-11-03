from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import h5py
import numpy as np

from datasets.mimicgen_conversion import (
    convert_mimicgen_hdf5_to_zarr,
    resolve_mimicgen_jobs,
    stack_hdf5_rgb_observations,
)


class TestMimicgenConversionHelpers(unittest.TestCase):
    def _write_minimal_mimicgen_hdf5(
        self,
        path: Path,
        *,
        with_model_file: bool,
    ) -> None:
        with h5py.File(path, "w") as handle:
            demo = handle.create_group("data").create_group("demo_0")
            demo.create_dataset("states", data=np.zeros((3, 9), dtype=np.float32))
            demo.create_dataset("actions", data=np.zeros((3, 7), dtype=np.float32))
            demo.create_dataset("rewards", data=np.asarray([0.0, 0.0, 1.0]))
            if with_model_file:
                demo.attrs["model_file"] = "<xml/>"
            obs = demo.create_group("obs")
            obs.create_dataset(
                "object",
                data=np.zeros((3, 17), dtype=np.float32),
            )
            obs.create_dataset(
                "robot0_eef_pos",
                data=np.zeros((3, 3), dtype=np.float32),
            )
            obs.create_dataset(
                "robot0_eef_quat",
                data=np.zeros((3, 4), dtype=np.float32),
            )
            obs.create_dataset(
                "robot0_gripper_qpos",
                data=np.zeros((3, 2), dtype=np.float32),
            )

    def test_stack_hdf5_rgb_observations_concatenates_chw(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            hdf5_path = Path(tmpdir) / "demo.hdf5"
            with h5py.File(hdf5_path, "w") as handle:
                demo = handle.create_group("data").create_group("demo_0")
                obs = demo.create_group("obs")
                obs.create_dataset(
                    "agentview_image",
                    data=np.ones((3, 4, 5, 3), dtype=np.uint8),
                )
                obs.create_dataset(
                    "robot0_eye_in_hand_image",
                    data=np.full((3, 4, 5, 3), fill_value=2, dtype=np.uint8),
                )
                stacked = stack_hdf5_rgb_observations(
                    demo,
                    ("agentview_image", "robot0_eye_in_hand_image"),
                )
                self.assertEqual(stacked.shape, (3, 6, 4, 5))
                self.assertEqual(int(stacked[0, 0, 0, 0]), 1)
                self.assertEqual(int(stacked[0, 3, 0, 0]), 2)

    def test_absolute_conversion_forwards_model_file_to_action_converter(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "square_d2.hdf5"
            output_path = Path(tmpdir) / "square_d2_lowdim_abs.zarr"
            self._write_minimal_mimicgen_hdf5(input_path, with_model_file=True)

            converter_instance = Mock()
            converter_instance.close.return_value = None
            mocked_actions = np.zeros((3, 10), dtype=np.float32)
            with (
                patch(
                    "datasets.mimicgen_conversion.RobomimicAbsoluteActionConverter",
                    return_value=converter_instance,
                ),
                patch(
                    "datasets.mimicgen_conversion.convert_action_sequence",
                    return_value=mocked_actions,
                ) as mock_convert_action_sequence,
            ):
                convert_mimicgen_hdf5_to_zarr(
                    input_path,
                    output_path,
                    task_name="square_d2",
                    modality="lowdim",
                    action_pose_repr="absolute",
                    rotation_rep="rotation_6d",
                    jobs=1,
                )

            self.assertTrue(output_path.exists())
            self.assertEqual(mock_convert_action_sequence.call_count, 1)
            call_kwargs = mock_convert_action_sequence.call_args.kwargs
            self.assertEqual(call_kwargs["action_pose_repr"], "absolute")
            self.assertEqual(call_kwargs["rotation_rep"], "rotation_6d")
            self.assertEqual(call_kwargs["model_xml"], "<xml/>")

    def test_absolute_conversion_requires_model_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "square_d2.hdf5"
            output_path = Path(tmpdir) / "square_d2_lowdim_abs.zarr"
            self._write_minimal_mimicgen_hdf5(input_path, with_model_file=False)

            converter_instance = Mock()
            converter_instance.close.return_value = None
            with patch(
                "datasets.mimicgen_conversion.RobomimicAbsoluteActionConverter",
                return_value=converter_instance,
            ):
                with self.assertRaisesRegex(
                    AssertionError, "requires per-episode 'model_file'"
                ):
                    convert_mimicgen_hdf5_to_zarr(
                        input_path,
                        output_path,
                        task_name="square_d2",
                        modality="lowdim",
                        action_pose_repr="absolute",
                        rotation_rep="rotation_6d",
                        jobs=1,
                    )

    def test_conversion_rejects_nonpositive_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "square_d2.hdf5"
            output_path = Path(tmpdir) / "square_d2_lowdim.zarr"
            self._write_minimal_mimicgen_hdf5(input_path, with_model_file=False)

            with self.assertRaisesRegex(ValueError, "Expected jobs >= 1"):
                convert_mimicgen_hdf5_to_zarr(
                    input_path,
                    output_path,
                    task_name="square_d2",
                    modality="lowdim",
                    action_pose_repr="delta",
                    rotation_rep="axis_angle",
                    jobs=0,
                )

    def test_resolve_mimicgen_jobs_respects_explicit_value(self) -> None:
        self.assertEqual(resolve_mimicgen_jobs(3), 3)

    def test_resolve_mimicgen_jobs_autotunes_from_cpu_count(self) -> None:
        with patch("datasets.mimicgen_conversion.os.cpu_count", return_value=12):
            self.assertEqual(resolve_mimicgen_jobs(None), 8)
        with patch("datasets.mimicgen_conversion.os.cpu_count", return_value=4):
            self.assertEqual(resolve_mimicgen_jobs(None), 4)

    def test_resolve_mimicgen_jobs_requires_positive_cpu_count(self) -> None:
        with patch("datasets.mimicgen_conversion.os.cpu_count", return_value=None):
            with self.assertRaisesRegex(
                AssertionError, "Expected a positive os.cpu_count"
            ):
                resolve_mimicgen_jobs(None)
