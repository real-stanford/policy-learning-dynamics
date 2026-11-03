from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from models.diffusion.conditional_unet1d import ConditionalUnet1D
from models.diffusion.mask_generator import LowdimMaskGenerator
from models.policy.base_image_policy import BaseImagePolicy
from models.vision.multi_image_obs_encoder import MultiImageObsEncoder
from utils.common.normalization import LinearNormalizer
from utils.common.pytorch_util import dict_apply


class DiffusionUnetImagePolicy(BaseImagePolicy):
    """Image-conditioned diffusion policy (Diffusion Policy, Chi et al.).

    The observation encoder maps ``{rgb, states}`` to a feature vector that conditions
    the 1-D U-Net globally; the U-Net denoises an action chunk of length ``horizon``.
    Training uses the epsilon-prediction MSE at a uniformly drawn timestep; inference
    runs the attached diffusers scheduler for ``num_inference_steps`` steps.
    """

    def __init__(
        self,
        shape_meta: dict,
        noise_scheduler: DDPMScheduler,
        obs_encoder: MultiImageObsEncoder,
        horizon: int,
        n_action_steps: int,
        n_obs_steps: int,
        num_inference_steps: int | None = None,
        obs_as_global_cond: bool = True,
        diffusion_step_embed_dim: int = 256,
        down_dims=(256, 512, 1024),
        kernel_size: int = 5,
        n_groups: int = 8,
        cond_predict_scale: bool = True,
        **scheduler_step_kwargs,
    ) -> None:
        super().__init__()
        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1
        action_dim = int(action_shape[0])
        obs_feature_dim = int(obs_encoder.output_shape()[0])

        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            global_cond_dim = obs_feature_dim * int(n_obs_steps)

        self.obs_encoder = obs_encoder
        self.model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
        )
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False,
        )
        self.normalizer = LinearNormalizer()
        self.horizon = int(horizon)
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = int(n_action_steps)
        self.n_obs_steps = int(n_obs_steps)
        self.obs_as_global_cond = bool(obs_as_global_cond)
        self.scheduler_step_kwargs = dict(scheduler_step_kwargs)
        self.num_inference_steps = (
            int(num_inference_steps)
            if num_inference_steps is not None
            else int(noise_scheduler.config.num_train_timesteps)
        )

        # Optional joint (image, state, action) generator mixer (a plain, non-Module
        # object attached by the factory when training.jointaug_w_oa_joint_model
        # is enabled). When set, a ramping fraction of each TRAINING batch's rows is
        # replaced by GENERATED (rgb, states, action). None / inactive -> unchanged.
        self.oa_joint_mixer = None

    def conditional_sample(
        self,
        condition_data: torch.Tensor,
        condition_mask: torch.Tensor,
        local_cond: torch.Tensor | None = None,
        global_cond: torch.Tensor | None = None,
        generator=None,
    ) -> torch.Tensor:
        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator,
        )
        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        for timestep in self.noise_scheduler.timesteps:
            trajectory[condition_mask] = condition_data[condition_mask]
            model_output = self.model(
                trajectory,
                timestep,
                local_cond=local_cond,
                global_cond=global_cond,
            )
            trajectory = self.noise_scheduler.step(
                model_output,
                timestep,
                trajectory,
                generator=generator,
                **self.scheduler_step_kwargs,
            ).prev_sample
        trajectory[condition_mask] = condition_data[condition_mask]
        return trajectory

    def predict_action(
        self, obs_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        assert "past_action" not in obs_dict
        normalized_obs = self.normalizer.normalize(obs_dict)
        first_value = next(iter(normalized_obs.values()))
        batch_size = int(first_value.shape[0])

        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            encoder_input = dict_apply(
                normalized_obs,
                lambda x: x[:, : self.n_obs_steps, ...].reshape(-1, *x.shape[2:]),
            )
            encoded_obs = self.obs_encoder(encoder_input)
            global_cond = encoded_obs.reshape(batch_size, -1)
            condition_data = torch.zeros(
                (batch_size, self.horizon, self.action_dim),
                device=self.device,
                dtype=self.dtype,
            )
            condition_mask = torch.zeros_like(condition_data, dtype=torch.bool)
        else:
            encoder_input = dict_apply(
                normalized_obs,
                lambda x: x[:, : self.n_obs_steps, ...].reshape(-1, *x.shape[2:]),
            )
            encoded_obs = self.obs_encoder(encoder_input).reshape(
                batch_size, self.n_obs_steps, -1
            )
            condition_data = torch.zeros(
                (batch_size, self.horizon, self.action_dim + self.obs_feature_dim),
                device=self.device,
                dtype=self.dtype,
            )
            condition_mask = torch.zeros_like(condition_data, dtype=torch.bool)
            condition_data[:, : self.n_obs_steps, self.action_dim :] = encoded_obs
            condition_mask[:, : self.n_obs_steps, self.action_dim :] = True

        sampled = self.conditional_sample(
            condition_data=condition_data,
            condition_mask=condition_mask,
            local_cond=local_cond,
            global_cond=global_cond,
        )
        normalized_action_pred = sampled[..., : self.action_dim]
        action_pred = self.normalizer["action"].unnormalize(normalized_action_pred)
        start = self.n_obs_steps - 1
        end = start + self.n_action_steps
        return {
            "action": action_pred[:, start:end],
            "action_pred": action_pred,
        }

    def set_normalizer(self, normalizer: LinearNormalizer) -> None:
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Joint (image, state, action) generator mixing (train-only): replace a ramping
        # fraction of the batch's rows with GENERATED data. Generated rgb stays RAW [0,1]
        # and states/action stay RAW; THIS policy's normalizer normalizes them below, so
        # generated and real rows train identically. Inactive on the EMA model (eval
        # mode), so validation is unaffected. No-op when no mixer is attached.
        mx = self.oa_joint_mixer
        if mx is not None and mx.active and self.training:
            batch = mx.mix_batch(batch)

        normalized_obs = self.normalizer.normalize(batch["obs"])
        normalized_action = self.normalizer["action"].normalize(batch["action"])
        batch_size = normalized_action.shape[0]
        horizon = normalized_action.shape[1]

        local_cond = None
        global_cond = None
        trajectory = normalized_action
        cond_data = trajectory
        if self.obs_as_global_cond:
            encoder_input = dict_apply(
                normalized_obs,
                lambda x: x[:, : self.n_obs_steps, ...].reshape(-1, *x.shape[2:]),
            )
            encoded_obs = self.obs_encoder(encoder_input)
            global_cond = encoded_obs.reshape(batch_size, -1)
        else:
            encoder_input = dict_apply(
                normalized_obs,
                lambda x: x.reshape(-1, *x.shape[2:]),
            )
            encoded_obs = self.obs_encoder(encoder_input).reshape(
                batch_size, horizon, -1
            )
            cond_data = torch.cat((normalized_action, encoded_obs), dim=-1)
            trajectory = cond_data.detach()

        condition_mask = self.mask_generator(trajectory.shape)
        noise = torch.randn_like(trajectory)
        timesteps = torch.randint(
            low=0,
            high=int(self.noise_scheduler.config.num_train_timesteps),
            size=(trajectory.shape[0],),
            device=trajectory.device,
        ).long()
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)
        noisy_trajectory[condition_mask] = cond_data[condition_mask]
        pred = self.model(
            noisy_trajectory,
            timesteps,
            local_cond=local_cond,
            global_cond=global_cond,
        )

        prediction_type = self.noise_scheduler.config.prediction_type
        if prediction_type == "epsilon":
            target = noise
        elif prediction_type == "sample":
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type '{prediction_type}'.")

        loss = F.mse_loss(pred, target, reduction="none")
        loss = loss * (~condition_mask).to(loss.dtype)
        loss = loss.reshape(loss.shape[0], -1).mean(dim=1)
        return loss.mean()
