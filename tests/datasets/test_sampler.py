from __future__ import annotations

import unittest

import numpy as np

from datasets.sampler import create_sequence_indices, sample_padded_sequence


class TestSampler(unittest.TestCase):
    def test_create_sequence_indices_includes_padding_windows(self) -> None:
        indices = create_sequence_indices(
            np.asarray([3], dtype=np.int64),
            sequence_length=4,
            pad_before=1,
            pad_after=1,
        )
        self.assertEqual(indices.shape, (2, 5))
        self.assertTrue(np.array_equal(indices[0], np.asarray([0, 0, 3, 1, 4])))
        self.assertTrue(np.array_equal(indices[1], np.asarray([0, 0, 3, 0, 3])))

    def test_sample_padded_sequence_repeats_boundaries(self) -> None:
        array = np.asarray([[1], [2], [3]], dtype=np.float32)
        sample = sample_padded_sequence(
            array,
            buffer_start_idx=0,
            buffer_end_idx=3,
            sample_start_idx=1,
            sample_end_idx=4,
            sequence_length=4,
        )
        self.assertTrue(
            np.allclose(sample[:, 0], np.asarray([1, 1, 2, 3], dtype=np.float32))
        )
