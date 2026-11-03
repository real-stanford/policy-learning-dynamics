from __future__ import annotations

import logging
from typing import Sequence

import torch
import torch.nn as nn

from models.diffusion.conv1d_components import Conv1dBlock, Downsample1d, Upsample1d
from models.diffusion.positional_embedding import SinusoidalPosEmb

logger = logging.getLogger(__name__)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int = 3,
        n_groups: int = 8,
        cond_predict_scale: bool = False,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
                Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
            ]
        )
        cond_channels = out_channels * 2 if cond_predict_scale else out_channels
        self.cond_predict_scale = bool(cond_predict_scale)
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
        )
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond).unsqueeze(-1)
        if self.cond_predict_scale:
            embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
            scale = embed[:, 0]
            bias = embed[:, 1]
            out = scale * out + bias
        else:
            out = out + embed
        out = self.blocks[1](out)
        return out + self.residual_conv(x)


class ConditionalUnet1D(nn.Module):
    """Diffusion Policy's 1-D conditional U-Net over an action chunk (B, T, D)."""

    def __init__(
        self,
        input_dim: int,
        local_cond_dim: int | None = None,
        global_cond_dim: int | None = None,
        diffusion_step_embed_dim: int = 256,
        down_dims: Sequence[int] = (256, 512, 1024),
        kernel_size: int = 3,
        n_groups: int = 8,
        cond_predict_scale: bool = False,
    ) -> None:
        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = int(down_dims[0])

        dsed = int(diffusion_step_embed_dim)
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed
        if global_cond_dim is not None:
            cond_dim += int(global_cond_dim)

        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        self.local_cond_encoder = None
        if local_cond_dim is not None:
            _, dim_out = in_out[0]
            dim_in = int(local_cond_dim)
            self.local_cond_encoder = nn.ModuleList(
                [
                    ConditionalResidualBlock1D(
                        dim_in,
                        dim_out,
                        cond_dim=cond_dim,
                        kernel_size=kernel_size,
                        n_groups=n_groups,
                        cond_predict_scale=cond_predict_scale,
                    ),
                    ConditionalResidualBlock1D(
                        dim_in,
                        dim_out,
                        cond_dim=cond_dim,
                        kernel_size=kernel_size,
                        n_groups=n_groups,
                        cond_predict_scale=cond_predict_scale,
                    ),
                ]
            )

        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList(
            [
                ConditionalResidualBlock1D(
                    mid_dim,
                    mid_dim,
                    cond_dim=cond_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale,
                ),
                ConditionalResidualBlock1D(
                    mid_dim,
                    mid_dim,
                    cond_dim=cond_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale,
                ),
            ]
        )

        self.down_modules = nn.ModuleList()
        for index, (dim_in, dim_out) in enumerate(in_out):
            is_last = index >= (len(in_out) - 1)
            self.down_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            dim_in,
                            dim_out,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                            cond_predict_scale=cond_predict_scale,
                        ),
                        ConditionalResidualBlock1D(
                            dim_out,
                            dim_out,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                            cond_predict_scale=cond_predict_scale,
                        ),
                        Downsample1d(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        self.up_modules = nn.ModuleList()
        reversed_in_out = list(reversed(in_out[1:]))
        for index, (dim_in, dim_out) in enumerate(reversed_in_out):
            is_last = index >= (len(in_out) - 1)
            self.up_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            dim_out * 2,
                            dim_in,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                            cond_predict_scale=cond_predict_scale,
                        ),
                        ConditionalResidualBlock1D(
                            dim_in,
                            dim_in,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                            cond_predict_scale=cond_predict_scale,
                        ),
                        Upsample1d(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        self.final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, kernel_size=1),
        )
        logger.info(
            "ConditionalUnet1D parameters: %e",
            sum(param.numel() for param in self.parameters()),
        )

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor | float | int,
        local_cond: torch.Tensor | None = None,
        global_cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = sample.transpose(1, 2)

        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=x.device)
        elif timesteps.ndim == 0:
            timesteps = timesteps[None].to(x.device)
        else:
            timesteps = timesteps.to(x.device)
        timesteps = timesteps.expand(x.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)
        if global_cond is not None:
            global_feature = torch.cat((global_feature, global_cond), dim=-1)

        local_features = []
        if local_cond is not None:
            local_x = local_cond.transpose(1, 2)
            assert self.local_cond_encoder is not None
            first_block, second_block = self.local_cond_encoder
            local_features.append(first_block(local_x, global_feature))
            local_features.append(second_block(local_x, global_feature))

        skip_connections = []
        for index, (resnet1, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet1(x, global_feature)
            if index == 0 and local_features:
                x = x + local_features[0]
            x = resnet2(x, global_feature)
            skip_connections.append(x)
            x = downsample(x)

        for module in self.mid_modules:
            x = module(x, global_feature)

        for index, (resnet1, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, skip_connections.pop()), dim=1)
            x = resnet1(x, global_feature)
            if index == len(self.up_modules) and local_features:
                x = x + local_features[1]
            x = resnet2(x, global_feature)
            x = upsample(x)

        return self.final_conv(x).transpose(1, 2)
