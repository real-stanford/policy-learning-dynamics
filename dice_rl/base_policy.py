from __future__ import annotations

import contextlib
from pathlib import Path

import torch
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from omegaconf import open_dict

from utils.training_utils.checkpointing import load_checkpoint_payload
from utils.training_utils.compile import load_compiled_state_dict
from utils.training_utils.factories import build_policy


def _autocast_context(device: torch.device, dtype_name: str | None):
    if dtype_name is None:
        return contextlib.nullcontext()
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[dtype_name]
    return torch.autocast(device_type=device.type, dtype=dtype)


class FrozenDiffusionBasePolicy:
    def __init__(
        self,
        *,
        checkpoint_path: str | Path,
        device: torch.device | str,
        num_inference_steps: int,
        ddim_eta: float,
        state_key: str,
        autocast: str | None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        assert self.checkpoint_path.exists(), f"Missing checkpoint {self.checkpoint_path}"
        self.device = torch.device(device)
        assert state_key in {"ema_state", "model_state"}, state_key
        assert autocast in {None, "bf16", "fp16"}, autocast
        self.autocast = autocast
        payload = load_checkpoint_payload(
            self.checkpoint_path, map_location="cpu", mmap=True
        )
        self.cfg = payload["cfg"]
        assert str(self.cfg.task.modality) == "image", (
            "dice_rl finetunes image checkpoints only; got modality "
            f"'{self.cfg.task.modality}'."
        )
        assert str(self.cfg.model.name) == "diffusion_unet_image", (
            f"Unsupported model '{self.cfg.model.name}' (expected diffusion_unet_image)."
        )
        assert state_key in payload, (
            f"Checkpoint has no '{state_key}' (keys: {sorted(payload)})."
        )
        self.state_key = state_key
        with open_dict(self.cfg):
            # The generated-sample mixer is a training-only hook of the BC stage; a frozen
            # base policy must never attach it (nor require its generator checkpoints).
            if "jointaug_w_oa_joint_model" in self.cfg.training:
                self.cfg.training.jointaug_w_oa_joint_model.enabled = False
        policy = build_policy(self.cfg)
        load_compiled_state_dict(policy, payload[state_key])
        policy.normalizer.load_state_dict(payload["normalizer_state"])
        policy.to(self.device)
        policy.eval()
        for param in policy.parameters():
            param.requires_grad_(False)
        self.policy = policy
        assert bool(policy.obs_as_global_cond), "dice_rl requires obs_conditioning=global."

        self.horizon = int(policy.horizon)
        self.n_obs_steps = int(policy.n_obs_steps)
        self.n_action_steps = int(policy.n_action_steps)
        self.action_dim = int(policy.action_dim)
        self.feature_dim = int(policy.obs_feature_dim)
        self.cond_dim = self.feature_dim * self.n_obs_steps
        chunk_start = self.n_obs_steps - 1  # DiffusionUnetImagePolicy.predict_action
        assert chunk_start == 0 and self.n_action_steps == self.horizon, (
            "dice_rl treats the executed chunk as the full predicted horizon; the "
            f"checkpoint has n_obs_steps={self.n_obs_steps}, n_action_steps="
            f"{self.n_action_steps}, horizon={self.horizon}."
        )

        self.num_inference_steps = int(num_inference_steps)
        assert self.num_inference_steps >= 1
        self.ddim_eta = float(ddim_eta)
        self.scheduler = self._build_ddim_scheduler()
        self.scheduler.set_timesteps(self.num_inference_steps)
        self.timesteps = [int(t) for t in self.scheduler.timesteps]

        self.action_pose_repr = str(self.cfg.action.action_pose_repr)
        self.rotation_rep = str(self.cfg.action.rotation_rep)
        self.rgb_shape = tuple(
            int(v) for v in self.cfg.task.observation.shape_meta.obs.rgb.shape
        )
        self.state_dim = int(self.cfg.task.observation.shape_meta.obs.states.shape[0])

        self.aux_features_enabled = False
        self.state_features_dim = 0

    def _build_ddim_scheduler(self) -> DDIMScheduler:
        """Always sample with DDIM using the checkpoint's training noise schedule
        (a DDPM-scheduled checkpoint samples identically under DDIM eta=0 given
        the same betas; cf. ``scripts/eval_checkpoints.py::_make_scheduler``)."""
        scheduler_cfg = self.cfg.model.scheduler
        assert str(scheduler_cfg.prediction_type) == "epsilon", (
            f"Expected epsilon prediction, got {scheduler_cfg.prediction_type}."
        )
        return DDIMScheduler(
            num_train_timesteps=int(scheduler_cfg.num_train_timesteps),
            beta_start=float(scheduler_cfg.beta_start),
            beta_end=float(scheduler_cfg.beta_end),
            beta_schedule=str(scheduler_cfg.beta_schedule),
            clip_sample=bool(scheduler_cfg.clip_sample),
            prediction_type=str(scheduler_cfg.prediction_type),
            set_alpha_to_one=bool(getattr(scheduler_cfg, "set_alpha_to_one", True)),
            steps_offset=int(getattr(scheduler_cfg, "steps_offset", 0)),
        )

    # ---------------------------------------------------------------- encoder
    @torch.no_grad()
    def encode_frames(self, states: torch.Tensor, rgb: torch.Tensor) -> torch.Tensor:
        """Per-frame conditioning features.

        states: (N, S) raw proprio, rgb: (N, C, H, W) float in [0, 1] (or uint8).
        Returns (N, feature_dim) = [encoder(rgb) | normalized states] exactly as
        the policy's ``global_cond`` for one obs step.
        """
        assert states.ndim == 2 and states.shape[1] == self.state_dim, tuple(states.shape)
        assert rgb.ndim == 4 and tuple(rgb.shape[1:]) == self.rgb_shape, tuple(rgb.shape)
        if rgb.dtype == torch.uint8:
            rgb = rgb.to(self.device, dtype=torch.float32) / 255.0
        obs = {
            "states": states.to(self.device, dtype=torch.float32),
            "rgb": rgb.to(self.device, dtype=torch.float32),
        }
        normalized = self.policy.normalizer.normalize(obs)
        with _autocast_context(self.device, self.autocast):
            features = self.policy.obs_encoder(normalized)
        return features.to(torch.float32)

    def enable_state_features(self, state_dim: int) -> None:
        """Asymmetric critic: frames become ``[policy features | lowdim state]``.
        ``encode_frames`` is UNCHANGED -- the env runner and the expert builder
        append the state themselves; this only rebinds the dim bookkeeping so
        buffers/cond carry augmented frames while the policy keeps consuming its
        own slice (``policy_cond``). Idempotent."""
        if self.state_features_dim:
            assert int(state_dim) == self.state_features_dim
            return
        assert int(state_dim) > 0
        self.state_features_dim = int(state_dim)
        self.policy_feature_dim = int(self.feature_dim)
        self.policy_cond_dim = int(self.cond_dim)
        self.feature_dim = self.policy_feature_dim + self.state_features_dim
        self.cond_dim = self.feature_dim * self.n_obs_steps
        self.aux_features_enabled = True

    def policy_cond(self, cond: torch.Tensor) -> torch.Tensor:
        """Slice the policy's own conditioning out of aux-augmented cond."""
        if not self.aux_features_enabled:
            return cond
        batch = cond.shape[0]
        frames = cond.reshape(batch, self.n_obs_steps, self.feature_dim)
        return frames[:, :, : self.policy_feature_dim].reshape(batch, -1)

    # ---------------------------------------------------------------- sampler
    @torch.no_grad()
    def sample_actions(self, cond: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Deterministic DDIM denoising from ``noise`` (B, H, A) under ``cond``
        (B, cond_dim). Returns the NORMALIZED action chunk (B, H, A)."""
        batch = noise.shape[0]
        expected_cond = self.policy_cond_dim if self.aux_features_enabled else self.cond_dim
        assert cond.shape == (batch, expected_cond), (
            f"cond must be ({batch}, {expected_cond}), got {tuple(cond.shape)}"
        )
        assert noise.shape == (batch, self.horizon, self.action_dim), tuple(noise.shape)
        trajectory = noise.to(self.device, dtype=torch.float32).clone()
        cond = cond.to(self.device, dtype=torch.float32)
        scheduler = self.scheduler
        scheduler.set_timesteps(self.num_inference_steps)
        with _autocast_context(self.device, self.autocast):
            for timestep in scheduler.timesteps:
                model_output = self.policy.model(
                    trajectory, timestep, local_cond=None, global_cond=cond
                )
                trajectory = scheduler.step(
                    model_output.to(torch.float32),
                    timestep,
                    trajectory,
                    eta=self.ddim_eta,
                ).prev_sample
        return trajectory

    # ------------------------------------------------------------- normalizer
    def unnormalize(self, normalized_actions: torch.Tensor) -> torch.Tensor:
        return self.policy.normalizer["action"].unnormalize(normalized_actions)

    def normalize(self, raw_actions: torch.Tensor) -> torch.Tensor:
        return self.policy.normalizer["action"].normalize(raw_actions)

    def summary(self) -> dict:
        return {
            "checkpoint": str(self.checkpoint_path),
            "state_key": self.state_key,
            "task": str(self.cfg.task.name),
            "horizon": self.horizon,
            "n_obs_steps": self.n_obs_steps,
            "action_dim": self.action_dim,
            "feature_dim": self.feature_dim,
            "cond_dim": self.cond_dim,
            "num_inference_steps": self.num_inference_steps,
            "ddim_eta": self.ddim_eta,
            "timesteps": self.timesteps,
            "autocast": self.autocast,
            "action_pose_repr": self.action_pose_repr,
            "rotation_rep": self.rotation_rep,
        }
