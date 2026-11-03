"""A tiny but structurally faithful DiffusionUnetImagePolicy for tests.

Mirrors the threading baseline where it matters for the training loss: T=100
squaredcos_cap_v2 forward process, epsilon prediction, obs_as_global_cond (so the
condition mask is identically False), DDIMScheduler object. Everything else is shrunk.
"""
from __future__ import annotations

import numpy as np
import torch
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from models.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
from models.vision.model_getter import get_resnet
from models.vision.multi_image_obs_encoder import MultiImageObsEncoder
from utils.common.normalization import (
    LinearNormalizer,
    array_to_stats,
    get_identity_normalizer_from_stat,
    get_image_identity_normalizer,
)

SHAPE_META = {
    "obs": {
        "states": {"shape": [9], "type": "low_dim"},
        "rgb": {"shape": [6, 84, 84], "type": "rgb"},
    },
    "action": {"shape": [7]},
}


def threading_like_scheduler(num_train_timesteps: int = 100) -> DDIMScheduler:
    return DDIMScheduler(
        num_train_timesteps=num_train_timesteps,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
        set_alpha_to_one=True,
        steps_offset=0,
    )


def build_tiny_image_policy(seed: int) -> DiffusionUnetImagePolicy:
    torch.manual_seed(seed)
    np.random.seed(seed)
    obs_encoder = MultiImageObsEncoder(
        shape_meta=SHAPE_META,
        rgb_model=get_resnet("resnet18", weights=None),
        crop_shape=(76, 76),
        random_crop=False,
        use_group_norm=False,
        share_rgb_model=False,
        imagenet_norm=False,
    )
    policy = DiffusionUnetImagePolicy(
        shape_meta=SHAPE_META,
        noise_scheduler=threading_like_scheduler(),
        obs_encoder=obs_encoder,
        horizon=4,
        n_action_steps=2,
        n_obs_steps=1,
        num_inference_steps=2,
        obs_as_global_cond=True,
        diffusion_step_embed_dim=32,
        down_dims=(32, 64),
        kernel_size=3,
        n_groups=4,
        cond_predict_scale=True,
    )
    normalizer = LinearNormalizer()
    normalizer["states"] = get_identity_normalizer_from_stat(
        array_to_stats(np.random.randn(8, 9).astype(np.float32)))
    normalizer["rgb"] = get_image_identity_normalizer()
    normalizer["action"] = get_identity_normalizer_from_stat(
        array_to_stats(np.random.randn(8, 7).astype(np.float32)))
    policy.set_normalizer(normalizer)
    return policy


def tiny_batch(seed: int, batch_size: int = 16) -> dict:
    g = torch.Generator().manual_seed(seed)
    return {
        "obs": {
            "states": torch.randn(batch_size, 4, 9, generator=g),
            "rgb": torch.rand(batch_size, 4, 6, 84, 84, generator=g),
        },
        "action": torch.randn(batch_size, 4, 7, generator=g),
    }
