from __future__ import annotations

import unittest

import numpy as np
import torch
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from models.generative.joint_image_oa_dit import JointImageOADiTModel
from utils.common.normalization import (
    LinearNormalizer,
    array_to_stats,
    get_image_identity_normalizer,
    get_range_normalizer_from_stat,
)


def tiny_generator(seed: int = 0) -> JointImageOADiTModel:
    torch.manual_seed(seed)
    scheduler = DDIMScheduler(
        num_train_timesteps=20,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="v_prediction",
        set_alpha_to_one=True,
        steps_offset=0,
    )
    model = JointImageOADiTModel(
        rgb_shape=(6, 84, 84),
        state_dim=9,
        action_dim=10,
        n_obs_steps=1,
        horizon=8,
        noise_scheduler=scheduler,
        d_model=16,
        num_layers=1,
        num_heads=2,
        time_embed_dim=16,
        unet_channels=(4, 8, 16),
        num_res_blocks=1,
        num_attn_heads=2,
        dropout=0.0,
        num_inference_steps=2,
    )
    normalizer = LinearNormalizer()
    normalizer["states"] = get_range_normalizer_from_stat(
        array_to_stats(np.random.randn(16, 9).astype(np.float32))
    )
    normalizer["action"] = get_range_normalizer_from_stat(
        array_to_stats(np.random.randn(16, 10).astype(np.float32))
    )
    normalizer["rgb"] = get_image_identity_normalizer()
    model.set_normalizer(normalizer)
    return model


class TestJointGenerator(unittest.TestCase):
    def test_loss_and_components(self) -> None:
        model = tiny_generator()
        batch = {
            "obs": {"rgb": torch.rand(2, 1, 6, 84, 84), "states": torch.randn(2, 1, 9)},
            "action": torch.randn(2, 8, 10),
        }
        loss = model(batch)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(set(model._loss_components), {"loss_img", "loss_state", "loss_act"})

    def test_rgb_normalizer_override(self) -> None:
        model = tiny_generator()
        x = torch.tensor([0.0, 0.5, 1.0])
        self.assertTrue(torch.allclose(model.normalizer["rgb"].normalize(x), torch.tensor([-1.0, 0.0, 1.0])))

    def test_sample_shapes_and_determinism(self) -> None:
        model = tiny_generator().eval()
        g1 = torch.Generator().manual_seed(3)
        g2 = torch.Generator().manual_seed(3)
        a = model.sample(3, device=torch.device("cpu"), generator=g1, num_inference_steps=2)
        b = model.sample(3, device=torch.device("cpu"), generator=g2, num_inference_steps=2)
        self.assertEqual(tuple(a["obs"]["rgb"].shape), (3, 1, 6, 84, 84))
        self.assertEqual(tuple(a["obs"]["states"].shape), (3, 1, 9))
        self.assertEqual(tuple(a["action"].shape), (3, 8, 10))
        self.assertGreaterEqual(float(a["obs"]["rgb"].min()), 0.0)
        self.assertLessEqual(float(a["obs"]["rgb"].max()), 1.0)
        self.assertTrue(torch.equal(a["action"], b["action"]))


if __name__ == "__main__":
    unittest.main()
