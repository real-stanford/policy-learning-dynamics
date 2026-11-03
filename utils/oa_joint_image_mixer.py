"""Mix GENERATED (image-obs, state, action) rows from trained joint generators into image
policy training (the ``jointaug_w_oa_joint_model`` experiment).

After a warmup ramp, each training batch has a fraction of its rows REPLACED by samples
drawn (equally) from a set of joint-generator checkpoints; clean and generated rows are both
trained with the policy's normal diffusion BC loss. The fraction ramps linearly from 0 to
``max_noisy_fraction`` between ``start_epochs`` and ``ramp_epochs``.

Normalization contract:
  - rgb: the generator samples in its own [-1,1] space and unnormalizes to RAW [0,1] inside
    ``sample()``; the image policy's rgb normalizer is identity ([0,1] passthrough) and the
    ImageNet normalization happens inside the obs encoder, so generated rgb drops into the
    batch exactly like dataset rgb.
  - states / action: the generator UNNORMALIZES with ITS OWN normalizer inside ``sample()``
    -> RAW; the policy's ``compute_loss`` then RE-NORMALIZES with the POLICY's normalizer.
RAW is the only thing that crosses the boundary. The generator's sampler and diffusion
hyperparameters come from the generator's OWN saved config; only ``num_inference_steps`` is
overridden here.

This is a PLAIN object (not nn.Module) attached as ``policy.oa_joint_mixer``, so the frozen
generators never enter the policy state_dict / EMA / torch.compile graph. ``mix_batch`` is
``@torch._dynamo.disable``'d (eager) so a compiled policy compiles only its UNet.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from models.generative.joint_image_oa_dit import (
    JointImageOADiTModel,
    build_joint_image_model_from_cfg,
)
from utils.training_utils.checkpointing import load_checkpoint_payload


class OAJointImageMixer:
    """Holds K frozen JointImageOADiTModel checkpoints and replaces a ramping fraction of each
    image training batch's rows with generated (rgb, states, action). Inactive copies (the
    EMA model) never mix.

    Two generation modes:
      * periodic_resampling=False: regenerate FRESH rows every step.
      * periodic_resampling=True: generate a POOL once per ``refresh_interval`` epochs (sized
        to one epoch's worth of generated rows for THIS rank, so reuse == refresh_interval)
        and draw rows from it each batch. The pool size is derived at runtime from
        steps_per_epoch x batch_size x max_noisy_fraction, so it is agnostic to #ranks, batch
        size and dataset size.
    """

    def __init__(
        self,
        models: List[JointImageOADiTModel],
        max_noisy_fraction: float = 0.5,
        start_epochs: int = 0,
        ramp_epochs: int = 100,
        num_inference_steps: int = 16,
        periodic_resampling: bool = False,
        refresh_interval: int = 20,
    ) -> None:
        assert len(models) > 0, "OAJointImageMixer requires at least one generator checkpoint."
        self.models = models  # frozen generators (eval, requires_grad=False)
        self.max_noisy_fraction = float(max_noisy_fraction)
        self.start_epochs = int(start_epochs)
        self.ramp_epochs = int(ramp_epochs)
        self.num_inference_steps = int(num_inference_steps)
        assert 0.0 <= self.max_noisy_fraction <= 1.0, (
            f"max_noisy_fraction must be in [0, 1], got {self.max_noisy_fraction}."
        )
        assert self.start_epochs >= 0, f"start_epochs must be >= 0, got {self.start_epochs}."
        assert self.ramp_epochs >= self.start_epochs, (
            f"ramp_epochs ({self.ramp_epochs}) must be >= start_epochs "
            f"({self.start_epochs}); the ramp runs from start_epochs to ramp_epochs."
        )
        self.active = True
        self.current_epoch = 0
        self._device = None
        self.periodic_resampling = bool(periodic_resampling)
        self.refresh_interval = max(1, int(refresh_interval))
        self._steps_per_epoch: int | None = None  # set by the trainer (rows/epoch = this * B)
        self._pool_rgb = None      # (P, n_obs, 6, 84, 84) uint8 on CPU ([0,255])
        self._pool_state = None    # (P, n_obs, state_dim) float32 on CPU (RAW)
        self._pool_act = None      # (P, horizon, action_dim) float32 on CPU (RAW)
        self._pool_window: int | None = None  # which refresh window the pool was built for
        self._perm = None          # shuffled draw order over the current pool
        self._cursor = 0           # position in self._perm
        self._last_batch_size = 0  # last per-rank batch size, to size the pool on (re)build

    # ------------------------------------------------------------------ ramp
    def set_epoch(self, epoch: int) -> None:
        """Set by the trainer at each epoch start; drives the linear ramp."""
        self.current_epoch = int(epoch)

    def set_steps_per_epoch(self, steps: int) -> None:
        """Called by the trainer each epoch with len(train_dataloader) (per-rank batches).
        Used only by periodic_resampling to size the pool."""
        self._steps_per_epoch = int(steps)

    def noisy_fraction(self) -> float:
        """Delayed linear ramp:
          epoch <= start_epochs              -> 0.0 (clean data only)
          start_epochs < e < ramp_epochs     -> linear 0 -> max_noisy_fraction
          epoch >= ramp_epochs               -> max_noisy_fraction (flat)"""
        if self.current_epoch <= self.start_epochs:
            return 0.0
        duration = self.ramp_epochs - self.start_epochs
        if duration <= 0:  # start_epochs == ramp_epochs -> step straight to max
            return self.max_noisy_fraction
        progress = (self.current_epoch - self.start_epochs) / duration
        return min(self.max_noisy_fraction, self.max_noisy_fraction * progress)

    def _counts(self, n: int) -> List[int]:
        """Split n generated rows equally across the K checkpoints (remainder to the first
        few): 61 over 4 -> [16, 15, 15, 15]."""
        k = len(self.models)
        base, rem = divmod(n, k)
        return [base + 1 if i < rem else base for i in range(k)]

    # ------------------------------------------------------------------ generation
    def _ensure_models_on(self, device) -> None:
        """Move the frozen generators to the batch device lazily (once)."""
        if self._device != device:
            for model in self.models:
                model.to(device)
            self._device = device

    def _generate(self, n: int, device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate n rows fresh, split equally across the K generators (each unnormalizes
        with its OWN normalizer inside sample()). Returns (rgb [0,1], states RAW, action RAW)
        on ``device``."""
        rgb_parts, state_parts, act_parts = [], [], []
        for model, count in zip(self.models, self._counts(n)):
            if count <= 0:
                continue
            gen = model.sample(count, device=device, num_inference_steps=self.num_inference_steps)
            rgb_parts.append(gen["obs"]["rgb"])       # (count, n_obs, 6, 84, 84) in [0,1]
            state_parts.append(gen["obs"]["states"])  # (count, n_obs, state_dim) RAW
            act_parts.append(gen["action"])           # (count, horizon, action_dim) RAW
        return (
            torch.cat(rgb_parts, dim=0),
            torch.cat(state_parts, dim=0),
            torch.cat(act_parts, dim=0),
        )

    def _build_pool(self, device, batch_size: int) -> None:
        """Generate one epoch's worth of rows for THIS rank into a CPU pool (rgb as uint8,
        lossless w.r.t. the 8-bit training images and 4x smaller). Sized as
        ceil(max_noisy_fraction * steps_per_epoch * batch_size)."""
        assert self._steps_per_epoch is not None, (
            "periodic_resampling needs steps_per_epoch; the trainer must call "
            "set_steps_per_epoch(len(train_dataloader))."
        )
        rows_per_epoch = self._steps_per_epoch * int(batch_size)
        pool_size = max(1, math.ceil(self.max_noisy_fraction * rows_per_epoch))
        chunk = 256  # bound per-generation GPU memory; concatenated on CPU
        rgb_parts, state_parts, act_parts = [], [], []
        for model, count in zip(self.models, self._counts(pool_size)):
            remaining = int(count)
            while remaining > 0:
                c = min(remaining, chunk)
                gen = model.sample(c, device=device, num_inference_steps=self.num_inference_steps)
                rgb_u8 = (gen["obs"]["rgb"].clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
                rgb_parts.append(rgb_u8.cpu())
                state_parts.append(gen["obs"]["states"].float().cpu())
                act_parts.append(gen["action"].float().cpu())
                remaining -= c
        self._pool_rgb = torch.cat(rgb_parts, dim=0)
        self._pool_state = torch.cat(state_parts, dim=0)
        self._pool_act = torch.cat(act_parts, dim=0)

    def _draw_from_pool(self, n: int, device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(Re)build the pool at each refresh-window boundary, then draw the next ``n`` rows
        from a per-pool shuffled index buffer (without replacement; reshuffle when a pass is
        exhausted). Returns (rgb [0,1], states RAW, action RAW) on ``device``."""
        window = self.current_epoch // self.refresh_interval
        if self._pool_rgb is None or self._pool_window != window:
            self._build_pool(device, self._last_batch_size)
            self._pool_window = window
            self._perm = torch.randperm(self._pool_rgb.shape[0])
            self._cursor = 0
        pool_size = self._pool_rgb.shape[0]
        n = min(n, pool_size)
        if self._cursor + n > pool_size:  # pass exhausted -> reshuffle for the next pass
            self._perm = torch.randperm(pool_size)
            self._cursor = 0
        idx = self._perm[self._cursor : self._cursor + n]
        self._cursor += n
        rgb = self._pool_rgb[idx].to(device).float().div_(255.0)  # uint8 [0,255] -> [0,1]
        state = self._pool_state[idx].to(device)
        act = self._pool_act[idx].to(device)
        return rgb, state, act

    @torch._dynamo.disable
    @torch.no_grad()
    def mix_batch(self, batch: Dict) -> Dict:
        """Return a batch with the last ``round(noisy_fraction * B)`` rows replaced by generated
        (rgb [0,1], states RAW, action RAW). The batch is already shuffled, so replacing the
        trailing rows is an unbiased random subset. No-op when the fraction rounds to 0."""
        action = batch["action"]
        batch_size = int(action.shape[0])
        n = int(round(self.noisy_fraction() * batch_size))
        if n <= 0:
            return batch
        n = min(n, batch_size)
        device = action.device
        self._ensure_models_on(device)

        # Force fp32 generation: these are TRAINING DATA, so their precision must not depend
        # on the policy's autocast setting.
        with torch.autocast(device_type=device.type, enabled=False):
            if self.periodic_resampling:
                self._last_batch_size = batch_size  # needed to size the pool on (re)build
                gen_rgb, gen_state, gen_act = self._draw_from_pool(n, device)
            else:
                gen_rgb, gen_state, gen_act = self._generate(n, device)

        # Bind n to the ACTUAL number of generated rows (the pool path caps at pool_size).
        n = gen_rgb.shape[0]

        # Replace the trailing n rows out-of-place (do not mutate the dataloader batch).
        obs = batch["obs"]
        rgb = obs["rgb"].clone()
        states = obs["states"].clone()
        act = action.clone()
        rgb[batch_size - n :] = gen_rgb.to(rgb.dtype)
        states[batch_size - n :] = gen_state.to(states.dtype)
        act[batch_size - n :] = gen_act.to(act.dtype)
        mixed = dict(batch)
        mixed["obs"] = {**obs, "rgb": rgb, "states": states}
        mixed["action"] = act
        return mixed

    def __deepcopy__(self, memo):
        """Return an INACTIVE shell that SHARES the frozen generators (the EMA copy, used in
        eval, must never mix). The pool is not copied."""
        clone = OAJointImageMixer.__new__(OAJointImageMixer)
        clone.models = self.models  # shared references, not deep-copied
        clone.max_noisy_fraction = self.max_noisy_fraction
        clone.start_epochs = self.start_epochs
        clone.ramp_epochs = self.ramp_epochs
        clone.num_inference_steps = self.num_inference_steps
        clone.current_epoch = self.current_epoch
        clone._device = self._device
        clone.periodic_resampling = self.periodic_resampling
        clone.refresh_interval = self.refresh_interval
        clone._steps_per_epoch = self._steps_per_epoch
        clone._pool_rgb = None
        clone._pool_state = None
        clone._pool_act = None
        clone._pool_window = None
        clone._perm = None
        clone._cursor = 0
        clone._last_batch_size = 0
        clone.active = False  # the EMA copy must never mix
        return clone


def load_generator_checkpoint(ckpt: Path, map_location="cpu") -> Tuple[JointImageOADiTModel, object]:
    """Rebuild a frozen generator from ITS OWN pickled config (incl. its scheduler) and load
    its EMA weights (raw weights when no EMA was kept) and normalizer. Returns (model, cfg).
    mmap=True reads only the tensors that are touched (every rank loads all K checkpoints)."""
    # Lazy import: factories imports this module indirectly (circular otherwise).
    from utils.training_utils.factories import build_noise_scheduler

    assert ckpt.exists(), f"generator checkpoint not found: {ckpt}"
    payload = load_checkpoint_payload(str(ckpt), map_location=map_location, mmap=True)
    gcfg = payload["cfg"]
    model = build_joint_image_model_from_cfg(gcfg, build_noise_scheduler(gcfg))
    state_key = "ema_state" if payload.get("ema_state") is not None else "model_state"
    state = {k.replace("_orig_mod.", ""): v for k, v in payload[state_key].items()}
    model.load_state_dict(state)
    model.normalizer.load_state_dict(payload["normalizer_state"])  # generator's own
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    return model, gcfg


def load_oa_joint_image_mixer(
    oa_cfg,
    expected_rgb_shape,
    expected_state_dim: int,
    expected_action_dim: int,
    expected_n_obs_steps: int,
    expected_horizon: int,
    map_location="cpu",
) -> OAJointImageMixer:
    """Load the generator checkpoint mixture named by ``oa_cfg`` and validate that each
    generator's (rgb_shape, state_dim, action_dim, n_obs_steps, horizon) matches the image
    policy's."""
    ckpt_dir = Path(str(oa_cfg.oa_model_dir)).expanduser()
    indices = [int(i) for i in list(oa_cfg.ckpt_indices)]
    assert ckpt_dir.exists(), f"oa_model_dir does not exist: {ckpt_dir}"
    assert indices, "jointaug_w_oa_joint_model.ckpt_indices must be non-empty."
    expected_rgb_shape = tuple(int(x) for x in expected_rgb_shape)

    models: List[JointImageOADiTModel] = []
    for idx in indices:
        ckpt = ckpt_dir / f"epoch_{idx:04d}.ckpt"
        model, _gcfg = load_generator_checkpoint(ckpt, map_location=map_location)
        assert (
            tuple(model.rgb_shape) == expected_rgb_shape
            and model.state_dim == expected_state_dim
            and model.action_dim == expected_action_dim
            and model.n_obs_steps == expected_n_obs_steps
            and model.horizon == expected_horizon
        ), (
            f"generator {ckpt.name} layout (rgb={tuple(model.rgb_shape)}, "
            f"state={model.state_dim}, action={model.action_dim}, n_obs={model.n_obs_steps}, "
            f"horizon={model.horizon}) does not match the image policy "
            f"(rgb={expected_rgb_shape}, state={expected_state_dim}, "
            f"action={expected_action_dim}, n_obs={expected_n_obs_steps}, "
            f"horizon={expected_horizon}). Same task layout is required."
        )
        models.append(model)

    return OAJointImageMixer(
        models=models,
        max_noisy_fraction=float(oa_cfg.max_noisy_fraction),
        start_epochs=int(oa_cfg.start_epochs),
        ramp_epochs=int(oa_cfg.ramp_epochs),
        num_inference_steps=int(oa_cfg.num_inference_steps),
        periodic_resampling=bool(oa_cfg.periodic_resampling),
        refresh_interval=int(oa_cfg.refresh_interval),
    )
