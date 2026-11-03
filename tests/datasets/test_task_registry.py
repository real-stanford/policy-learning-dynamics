from __future__ import annotations

import unittest

from envs.task_registry import (
    MIMICGEN_TASK_REGISTRY,
    get_task_entry,
    infer_task_name_from_path,
    list_task_names,
)

PAPER_TASKS = (
    "hammer_cleanup_d1",
    "mug_cleanup_d1",
    "square_d2",
    "threading_d2",
    "three_piece_assembly_d1",
)


class TestTaskRegistry(unittest.TestCase):
    def test_paper_tasks_are_registered_with_shared_image_contract(self) -> None:
        self.assertEqual(list_task_names("mimicgen"), PAPER_TASKS)
        for task_name in PAPER_TASKS:
            entry = get_task_entry("mimicgen", task_name)
            self.assertEqual(entry.rgb_obs_keys, ("agentview_image", "robot0_eye_in_hand_image"))
            self.assertEqual(entry.image_size, (84, 84))
            self.assertEqual(entry.image_lowdim_obs_dim, 9)
            self.assertEqual(entry.delta_action_dim, 7)
            self.assertEqual(entry.absolute_rotation6d_action_dim, 10)
            self.assertEqual(entry.upstream_action_mode, "delta")

    def test_horizons(self) -> None:
        self.assertEqual(MIMICGEN_TASK_REGISTRY["square_d2"].horizon, 400)
        self.assertEqual(MIMICGEN_TASK_REGISTRY["threading_d2"].horizon, 400)
        self.assertEqual(MIMICGEN_TASK_REGISTRY["mug_cleanup_d1"].horizon, 500)
        self.assertEqual(MIMICGEN_TASK_REGISTRY["hammer_cleanup_d1"].horizon, 500)
        self.assertEqual(MIMICGEN_TASK_REGISTRY["three_piece_assembly_d1"].horizon, 500)

    def test_path_inference_matches_registry(self) -> None:
        self.assertEqual(
            infer_task_name_from_path("mimicgen", "/tmp/data/mimicgen/core/square_d2.hdf5"),
            "square_d2",
        )
        self.assertIsNone(infer_task_name_from_path("mimicgen", "/tmp/data/mimicgen/core/x.hdf5"))

    def test_unknown_task_raises(self) -> None:
        with self.assertRaises(KeyError):
            get_task_entry("mimicgen", "stack_d1")


if __name__ == "__main__":
    unittest.main()
