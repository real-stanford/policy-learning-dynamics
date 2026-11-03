from __future__ import annotations

import copy
import unittest

import torch

from utils.oa_joint_image_mixer import OAJointImageMixer


class _FakeGenerator:
    """Stands in for a frozen JointImageOADiTModel: deterministic constant samples."""

    def __init__(self, value: float) -> None:
        self.value = float(value)
        self.device_moves = 0

    def to(self, device) -> "_FakeGenerator":
        self.device_moves += 1
        return self

    def sample(self, n: int, device=None, num_inference_steps: int | None = None) -> dict:
        return {
            "obs": {
                "rgb": torch.full((n, 1, 6, 84, 84), self.value, device=device),
                "states": torch.full((n, 1, 9), self.value, device=device),
            },
            "action": torch.full((n, 8, 10), self.value, device=device),
        }


def _batch(batch_size: int = 10) -> dict:
    return {
        "obs": {"rgb": torch.zeros(batch_size, 1, 6, 84, 84), "states": torch.zeros(batch_size, 1, 9)},
        "action": torch.zeros(batch_size, 8, 10),
    }


class TestOAJointImageMixer(unittest.TestCase):
    def test_ramp(self) -> None:
        m = OAJointImageMixer([_FakeGenerator(0.5)], max_noisy_fraction=0.5, start_epochs=10, ramp_epochs=20)
        m.set_epoch(0)
        self.assertEqual(m.noisy_fraction(), 0.0)
        m.set_epoch(10)
        self.assertEqual(m.noisy_fraction(), 0.0)
        m.set_epoch(15)
        self.assertAlmostEqual(m.noisy_fraction(), 0.25)
        m.set_epoch(30)
        self.assertEqual(m.noisy_fraction(), 0.5)

    def test_counts_split_equally(self) -> None:
        m = OAJointImageMixer([_FakeGenerator(v) for v in (0.1, 0.2, 0.3, 0.4)])
        self.assertEqual(m._counts(61), [16, 15, 15, 15])

    def test_mix_batch_replaces_trailing_rows_out_of_place(self) -> None:
        m = OAJointImageMixer([_FakeGenerator(0.25), _FakeGenerator(0.75)], max_noisy_fraction=0.4, ramp_epochs=10)
        m.set_epoch(10)
        batch = _batch(10)
        mixed = m.mix_batch(batch)
        # 4 rows replaced: 2 from each generator, in generator order
        self.assertTrue(torch.all(mixed["action"][:6] == 0))
        self.assertTrue(torch.all(mixed["action"][6:8] == 0.25))
        self.assertTrue(torch.all(mixed["action"][8:10] == 0.75))
        self.assertTrue(torch.all(mixed["obs"]["rgb"][6:8] == 0.25))
        self.assertTrue(torch.all(mixed["obs"]["states"][8:10] == 0.75))
        self.assertTrue(torch.all(batch["action"] == 0))  # input untouched

    def test_zero_fraction_is_noop(self) -> None:
        m = OAJointImageMixer([_FakeGenerator(1.0)], max_noisy_fraction=0.5, start_epochs=5, ramp_epochs=10)
        m.set_epoch(0)
        batch = _batch(4)
        self.assertIs(m.mix_batch(batch), batch)

    def test_periodic_pool_reuse(self) -> None:
        m = OAJointImageMixer(
            [_FakeGenerator(0.5)], max_noisy_fraction=0.5, ramp_epochs=1,
            periodic_resampling=True, refresh_interval=2,
        )
        m.set_steps_per_epoch(3)
        m.set_epoch(1)
        mixed = m.mix_batch(_batch(4))
        self.assertEqual(int(m._pool_rgb.shape[0]), 6)  # ceil(0.5 * 3 * 4)
        self.assertEqual(m._pool_window, 0)
        self.assertTrue(torch.all(mixed["action"][2:] == 0.5))
        self.assertTrue(torch.allclose(mixed["obs"]["rgb"][2:], torch.full((2, 1, 6, 84, 84), 128 / 255)))
        m.set_epoch(2)
        m.mix_batch(_batch(4))
        self.assertEqual(m._pool_window, 1)  # rebuilt at the refresh boundary

    def test_deepcopy_is_inactive_shell(self) -> None:
        m = OAJointImageMixer([_FakeGenerator(0.5)])
        clone = copy.deepcopy(m)
        self.assertFalse(clone.active)
        self.assertIs(clone.models, m.models)


if __name__ == "__main__":
    unittest.main()
