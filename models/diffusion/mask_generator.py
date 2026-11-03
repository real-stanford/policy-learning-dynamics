from __future__ import annotations

import torch

from models.common.module_attr_mixin import ModuleAttrMixin


class LowdimMaskGenerator(ModuleAttrMixin):
    def __init__(
        self,
        action_dim: int,
        obs_dim: int,
        max_n_obs_steps: int = 2,
        fix_obs_steps: bool = True,
        action_visible: bool = False,
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.obs_dim = int(obs_dim)
        self.max_n_obs_steps = int(max_n_obs_steps)
        self.fix_obs_steps = bool(fix_obs_steps)
        self.action_visible = bool(action_visible)

    @torch.no_grad()
    def forward(self, shape, seed: int | None = None):
        batch_size, horizon, feature_dim = shape
        expected_dim = self.action_dim + self.obs_dim
        assert feature_dim == expected_dim, (
            f"Expected feature dim {expected_dim}, got {feature_dim}."
        )

        mask = torch.zeros(shape, dtype=torch.bool, device=self.device)
        is_action_dim = torch.zeros_like(mask)
        is_action_dim[..., : self.action_dim] = True
        is_obs_dim = ~is_action_dim

        if self.fix_obs_steps:
            obs_steps = torch.full(
                (batch_size,),
                fill_value=self.max_n_obs_steps,
                dtype=torch.long,
                device=self.device,
            )
        else:
            rng = None
            if seed is not None:
                rng = torch.Generator(device=self.device)
                rng.manual_seed(seed)
            obs_steps = torch.randint(
                low=1,
                high=self.max_n_obs_steps + 1,
                size=(batch_size,),
                generator=rng,
                device=self.device,
            )

        steps = torch.arange(horizon, device=self.device).reshape(1, horizon)
        obs_mask = (steps < obs_steps[:, None]).reshape(batch_size, horizon, 1)
        obs_mask = obs_mask.expand(batch_size, horizon, feature_dim) & is_obs_dim
        mask |= obs_mask

        if self.action_visible:
            action_steps = torch.clamp(obs_steps - 1, min=0)
            action_mask = (steps < action_steps[:, None]).reshape(
                batch_size, horizon, 1
            )
            action_mask = action_mask.expand(batch_size, horizon, feature_dim)
            mask |= action_mask & is_action_dim

        return mask
