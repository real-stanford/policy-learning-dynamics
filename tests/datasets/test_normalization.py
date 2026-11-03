from __future__ import annotations

import unittest

import numpy as np

from utils.common.normalization import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
    array_to_stats,
    get_identity_normalizer_from_stat,
    get_range_normalizer_from_stat,
)


class TestNormalization(unittest.TestCase):
    def test_range_normalizer_round_trip(self) -> None:
        data = np.asarray([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
        normalizer = SingleFieldLinearNormalizer.create_fit(data)
        restored = normalizer.unnormalize(normalizer.normalize(data))
        self.assertTrue(np.allclose(restored, data))

    def test_identity_normalizer_is_passthrough(self) -> None:
        data = np.asarray([[0.1], [0.2], [0.3]], dtype=np.float32)
        normalizer = get_identity_normalizer_from_stat(array_to_stats(data))
        self.assertTrue(np.allclose(normalizer.normalize(data), data))

    def test_linear_normalizer_handles_dicts(self) -> None:
        normalizer = LinearNormalizer()
        obs = np.asarray([[0.0], [1.0]], dtype=np.float32)
        action = np.asarray([[2.0], [4.0]], dtype=np.float32)
        normalizer["obs"] = get_range_normalizer_from_stat(array_to_stats(obs))
        normalizer["action"] = get_range_normalizer_from_stat(array_to_stats(action))
        restored = normalizer.unnormalize(
            normalizer.normalize({"obs": obs, "action": action})
        )
        self.assertTrue(np.allclose(restored["obs"], obs))
        self.assertTrue(np.allclose(restored["action"], action))

    def test_linear_normalizer_state_dict_round_trip(self) -> None:
        source = LinearNormalizer()
        data = np.asarray([[0.0], [1.0]], dtype=np.float32)
        source["obs"] = get_range_normalizer_from_stat(array_to_stats(data))

        target = LinearNormalizer()
        target.load_state_dict(source.state_dict())
        normalized = target["obs"].normalize(data)
        restored = target["obs"].unnormalize(normalized)
        self.assertTrue(np.allclose(restored, data))
