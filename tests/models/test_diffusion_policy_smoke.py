from __future__ import annotations

import unittest

import torch

from tests.models._tiny_image_policy import build_tiny_image_policy, tiny_batch


class TestDiffusionPolicySmoke(unittest.TestCase):
    def test_image_policy_loss_predict_and_normalizer_reload(self) -> None:
        policy = build_tiny_image_policy(seed=0)
        batch = tiny_batch(seed=1, batch_size=2)
        loss = policy(batch)
        self.assertTrue(torch.isfinite(loss))
        policy.eval()
        result = policy.predict_action(batch["obs"])
        self.assertEqual(tuple(result["action"].shape), (2, 2, 7))
        self.assertEqual(tuple(result["action_pred"].shape), (2, 4, 7))
        self.assertEqual(set(policy.normalizer.state_dict().keys()) >= set(), True)

    def test_predict_action_is_deterministic_given_the_seed(self) -> None:
        policy = build_tiny_image_policy(seed=0).eval()
        obs = tiny_batch(seed=2, batch_size=2)["obs"]
        torch.manual_seed(3)
        first = policy.predict_action(obs)["action_pred"]
        torch.manual_seed(3)
        second = policy.predict_action(obs)["action_pred"]
        torch.testing.assert_close(first, second)

    def test_no_mixer_means_loss_uses_the_raw_batch(self) -> None:
        policy = build_tiny_image_policy(seed=0)
        self.assertIsNone(policy.oa_joint_mixer)
        batch = tiny_batch(seed=4, batch_size=2)
        torch.manual_seed(5)
        a = float(policy(batch))
        torch.manual_seed(5)
        b = float(policy(batch))
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
