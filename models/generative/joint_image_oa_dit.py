from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.common.module_attr_mixin import ModuleAttrMixin
from models.diffusion.positional_embedding import SinusoidalPosEmb
from models.generative.conv2d_components import (
    Downsample2d,
    ResBlock2d,
    SelfAttention2d,
    Upsample2d,
)
from utils.common.normalization import LinearNormalizer, SingleFieldLinearNormalizer


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    # x: (B, N, C); shift/scale: (B, C) broadcast across the token axis.
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class _MMStream(nn.Module):
    """Per-modality weights for one MM-DiT block: norm / adaLN / QKV / out-proj / MLP. The
    cross-modal attention itself is performed by the parent MMDiTBlock; this just holds one
    modality's dedicated parameters. adaLN is zero-init -> identity at start."""

    def __init__(self, d_model: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )
        self.ada_ln = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 6 * d_model))
        nn.init.zeros_(self.ada_ln[1].weight)
        nn.init.zeros_(self.ada_ln[1].bias)


class MMDiTBlock(nn.Module):
    """Dual-stream Multimodal Diffusion Transformer block (SD3-style). Each modality (image
    tokens, lowdim state+action tokens) has its OWN norm/adaLN/QKV/MLP weights; the two
    streams are joined ONLY inside the attention operation (Q,K,V concatenated across
    modalities -> one softmax over all tokens -> split back)."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads."
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.img = _MMStream(d_model, dropout)
        self.low = _MMStream(d_model, dropout)

    def _mha(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        b, nq, d = q.shape
        nk = k.shape[1]
        q = q.reshape(b, nq, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(b, nk, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(b, nk, self.num_heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v)  # (b, heads, nq, head_dim)
        return out.transpose(1, 2).reshape(b, nq, d)

    def forward(
        self, img: torch.Tensor, low: torch.Tensor, cond: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        i_sa, i_ca, i_ga, i_sm, i_cm, i_gm = self.img.ada_ln(cond).chunk(6, dim=-1)
        l_sa, l_ca, l_ga, l_sm, l_cm, l_gm = self.low.ada_ln(cond).chunk(6, dim=-1)
        n_img = img.shape[1]
        # joint attention: per-modality QKV, concatenate across tokens, one attention, split.
        iq, ik, iv = self.img.qkv(_modulate(self.img.norm1(img), i_sa, i_ca)).chunk(3, dim=-1)
        lq, lk, lv = self.low.qkv(_modulate(self.low.norm1(low), l_sa, l_ca)).chunk(3, dim=-1)
        attn = self._mha(
            torch.cat([iq, lq], dim=1),
            torch.cat([ik, lk], dim=1),
            torch.cat([iv, lv], dim=1),
        )
        img = img + i_ga.unsqueeze(1) * self.img.proj(attn[:, :n_img])
        low = low + l_ga.unsqueeze(1) * self.low.proj(attn[:, n_img:])
        # per-modality MLP (dedicated weights)
        img = img + i_gm.unsqueeze(1) * self.img.mlp(_modulate(self.img.norm2(img), i_sm, i_cm))
        low = low + l_gm.unsqueeze(1) * self.low.mlp(_modulate(self.low.norm2(low), l_sm, l_cm))
        return img, low


class JointImageOADiTModel(ModuleAttrMixin):
    def __init__(
        self,
        rgb_shape: Tuple[int, int, int] = (6, 84, 84),
        state_dim: int = 9,
        action_dim: int = 10,
        n_obs_steps: int = 1,
        horizon: int = 8,
        noise_scheduler=None,
        d_model: int = 384,
        num_layers: int = 6,
        num_heads: int = 6,
        time_embed_dim: int = 256,
        unet_channels: Tuple[int, int, int] = (96, 192, 384),
        num_res_blocks: int = 2,
        num_attn_heads: int = 6,
        dropout: float = 0.1,
        num_inference_steps: int | None = None,
        img_loss_weight: float = 1.0,
        state_loss_weight: float = 1.0,
        action_loss_weight: float = 1.0,
    ) -> None:
        super().__init__()
        assert n_obs_steps == 1
        c_img, h_img, w_img = rgb_shape
        assert h_img == 84 and w_img == 84
        self.rgb_shape = (int(c_img), int(h_img), int(w_img))
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.n_obs_steps = int(n_obs_steps)
        self.horizon = int(horizon)
        self.d_model = int(d_model)
        self.img_loss_weight = float(img_loss_weight)
        self.state_loss_weight = float(state_loss_weight)
        self.action_loss_weight = float(action_loss_weight)

        self.noise_scheduler = noise_scheduler
        self.num_inference_steps = (
            int(num_inference_steps)
            if num_inference_steps is not None
            else int(noise_scheduler.config.num_train_timesteps)
        )

        c1, c2, c3 = (int(c) for c in unet_channels)
        nrb = int(num_res_blocks)

        # Shared timestep embedding -> conditioning vector c (B, time_embed_dim) for FiLM
        # (conv ResBlocks) and -> (B, d_model) for the DiT cond.
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_embed_dim),
            nn.Linear(time_embed_dim, 4 * time_embed_dim),
            nn.SiLU(),
            nn.Linear(4 * time_embed_dim, time_embed_dim),
        )
        self.time_to_dmodel = nn.Linear(time_embed_dim, d_model)

        # -------- image stream: conv U-Net (84 -> 42 -> 21 -> 42 -> 84) --------
        # num_res_blocks ResBlocks per resolution level. The first block of a level changes
        # channels; the rest keep them. One skip is saved per level (the level's final
        # output), concatenated at the decoder level start.
        def _level(in_ch: int, out_ch: int) -> nn.ModuleList:
            return nn.ModuleList(
                [
                    ResBlock2d(in_ch if i == 0 else out_ch, out_ch, time_embed_dim, dropout)
                    for i in range(nrb)
                ]
            )

        self.in_conv = nn.Conv2d(self.rgb_shape[0], c1, 3, padding=1)
        self.enc0 = _level(c1, c1)                                       # @84
        self.down1 = Downsample2d(c1, c1)                                # 84->42
        self.enc1 = _level(c1, c2)                                       # @42
        self.down2 = Downsample2d(c2, c2)                                # 42->21
        self.enc2 = _level(c2, c3)                                       # @21
        self.proj_in = nn.Conv2d(c3, d_model, 1)                         # @21 -> d_model
        self.bott_attn = SelfAttention2d(d_model, num_attn_heads)        # @21 self-attn

        # bottleneck spatial size after two downsamples of 84
        self.bott_hw = 84 // 4  # = 21
        self.num_img_tokens = self.bott_hw * self.bott_hw  # 441
        # 2D-factorized learned positional embedding for the image tokens.
        self.img_row_emb = nn.Parameter(torch.zeros(1, self.bott_hw, d_model // 2))
        self.img_col_emb = nn.Parameter(torch.zeros(1, self.bott_hw, d_model // 2))
        nn.init.normal_(self.img_row_emb, std=0.02)
        nn.init.normal_(self.img_col_emb, std=0.02)

        self.proj_out = nn.Conv2d(d_model, c3, 1)                        # d_model -> c3 @21
        self.dec2 = _level(c3, c2)                                       # @21
        self.up2 = Upsample2d(c2, c2)                                    # 21->42
        self.dec1 = _level(c2 + c2, c1)                                  # @42 (+skip enc1)
        self.up1 = Upsample2d(c1, c1)                                    # 42->84
        self.dec0 = _level(c1 + c1, c1)                                  # @84 (+skip enc0)
        self.out_norm_img = nn.GroupNorm(min(8, c1), c1)
        self.out_conv = nn.Conv2d(c1, self.rgb_shape[0], 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

        # -------- action+state stream: token projections --------
        self.state_in = nn.Linear(self.state_dim, d_model)
        self.act_in = nn.Linear(self.action_dim, d_model)
        self.num_lowdim_tokens = self.n_obs_steps + self.horizon  # 1 state + H action
        self.lowdim_pos = nn.Parameter(torch.zeros(1, self.num_lowdim_tokens, d_model))
        nn.init.normal_(self.lowdim_pos, std=0.02)

        # -------- dual-stream MM-DiT bridge (separate per-modality weights, joint attention) --------
        self.blocks = nn.ModuleList(
            [MMDiTBlock(d_model, num_heads, dropout) for _ in range(int(num_layers))]
        )

        # lowdim output heads (zero-init)
        self.state_out = nn.Linear(d_model, self.state_dim)
        self.act_out = nn.Linear(d_model, self.action_dim)
        nn.init.zeros_(self.state_out.weight)
        nn.init.zeros_(self.state_out.bias)
        nn.init.zeros_(self.act_out.weight)
        nn.init.zeros_(self.act_out.bias)

        self.normalizer = LinearNormalizer()

    # ------------------------------------------------------------------ utils
    def set_normalizer(self, normalizer: LinearNormalizer) -> None:
        """Copy the dataset's fitted states/action fields, but OVERRIDE rgb with a fixed
        [0,1]->[-1,1] affine (the dataset's rgb field is identity, wrong for diffusion)."""
        self.normalizer.load_state_dict(normalizer.state_dict())
        one = np.ones((1,), dtype=np.float32)
        self.normalizer["rgb"] = SingleFieldLinearNormalizer.create_manual(
            scale=2.0 * one,
            offset=-1.0 * one,
            input_stats={"min": 0.0 * one, "max": one, "mean": 0.5 * one, "std": 0.5 * one},
        )

    # ------------------------------------------------------------------ core
    def _img_tokens_pos(self) -> torch.Tensor:
        # 2D-factorized pos emb -> (1, HW, d_model)
        h = self.bott_hw
        row = self.img_row_emb.unsqueeze(2).expand(1, h, h, -1)  # (1,H,W,d/2) over rows
        col = self.img_col_emb.unsqueeze(1).expand(1, h, h, -1)  # (1,H,W,d/2) over cols
        pos = torch.cat([row, col], dim=-1)  # (1,H,W,d)
        return pos.reshape(1, h * h, self.d_model)

    def _denoise(
        self,
        rgb_t: torch.Tensor,    # (B, 6, 84, 84)
        state_t: torch.Tensor,  # (B, n_obs, state_dim)
        act_t: torch.Tensor,    # (B, H, action_dim)
        timestep,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b = rgb_t.shape[0]
        device = rgb_t.device
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.long, device=device)
        timestep = timestep.to(device)
        if timestep.ndim == 0:
            timestep = timestep[None]
        timestep = timestep.expand(b)
        temb = self.time_mlp(timestep)          # (B, time_embed_dim) for FiLM
        cond = self.time_to_dmodel(temb)        # (B, d_model) for DiT adaLN

        # --- image encoder ---
        x = self.in_conv(rgb_t)
        for blk in self.enc0:
            x = blk(x, temb)                    # @84, c1
        e0 = x
        x = self.down1(x)
        for blk in self.enc1:
            x = blk(x, temb)                    # @42, c2
        e1 = x
        x = self.down2(x)
        for blk in self.enc2:
            x = blk(x, temb)                    # @21, c3
        x = self.proj_in(x)                     # @21, d_model
        x = self.bott_attn(x)                   # @21 self-attn
        img_tok = x.reshape(b, self.d_model, self.num_img_tokens).transpose(1, 2)
        img_tok = img_tok + self._img_tokens_pos()

        # --- lowdim tokens ---
        state_tok = self.state_in(state_t)      # (B, n_obs, d)
        act_tok = self.act_in(act_t)            # (B, H, d)
        low_tok = torch.cat([state_tok, act_tok], dim=1) + self.lowdim_pos

        # --- dual-stream MM-DiT bridge: separate per-modality weights, joint attention ---
        for block in self.blocks:
            img_tok, low_tok = block(img_tok, low_tok, cond)

        # --- image decoder (skips) ---
        x = img_tok.transpose(1, 2).reshape(b, self.d_model, self.bott_hw, self.bott_hw)
        x = self.proj_out(x)                    # @21, c3
        for blk in self.dec2:
            x = blk(x, temb)                    # @21, c2
        x = self.up2(x)                         # 21->42, c2
        x = torch.cat([x, e1], dim=1)           # +skip enc1
        for blk in self.dec1:
            x = blk(x, temb)                    # @42, c1
        x = self.up1(x)                         # 42->84, c1
        x = torch.cat([x, e0], dim=1)           # +skip enc0
        for blk in self.dec0:
            x = blk(x, temb)                    # @84, c1
        v_img = self.out_conv(F.silu(self.out_norm_img(x)))  # (B, 6, 84, 84)

        # --- lowdim heads ---
        state_x = low_tok[:, : self.n_obs_steps, :]
        act_x = low_tok[:, self.n_obs_steps :, :]
        v_state = self.state_out(state_x)       # (B, n_obs, state_dim)
        v_act = self.act_out(act_x)             # (B, H, action_dim)
        return v_img, v_state, v_act

    def compute_loss(self, batch: Dict) -> torch.Tensor:
        rgb_n = self.normalizer["rgb"].normalize(batch["obs"]["rgb"])      # [0,1]->[-1,1]
        state_n = self.normalizer["states"].normalize(batch["obs"]["states"])
        act_n = self.normalizer["action"].normalize(batch["action"])
        assert self.noise_scheduler.config.prediction_type == "v_prediction", (
            "JointImageOADiTModel assumes v_prediction."
        )

        # n_obs_steps=1: squeeze the singleton obs axis for the conv stream.
        rgb_n2d = rgb_n[:, 0]  # (B, 6, 84, 84)
        b = rgb_n2d.shape[0]
        device = rgb_n2d.device
        timesteps = torch.randint(
            0, int(self.noise_scheduler.config.num_train_timesteps), (b,), device=device
        ).long()

        noise_img = torch.randn_like(rgb_n2d)
        noise_state = torch.randn_like(state_n)
        noise_act = torch.randn_like(act_n)
        rgb_t = self.noise_scheduler.add_noise(rgb_n2d, noise_img, timesteps)
        state_t = self.noise_scheduler.add_noise(state_n, noise_state, timesteps)
        act_t = self.noise_scheduler.add_noise(act_n, noise_act, timesteps)

        v_img, v_state, v_act = self._denoise(rgb_t, state_t, act_t, timesteps)

        # v-prediction targets
        vt_img = self.noise_scheduler.get_velocity(rgb_n2d, noise_img, timesteps)
        vt_state = self.noise_scheduler.get_velocity(state_n, noise_state, timesteps)
        vt_act = self.noise_scheduler.get_velocity(act_n, noise_act, timesteps)

        # per-modality MEAN MSE, weighted sum (each term O(1) regardless of element count)
        loss_img = F.mse_loss(v_img, vt_img)
        loss_state = F.mse_loss(v_state, vt_state)
        loss_act = F.mse_loss(v_act, vt_act)
        # Unweighted per-modality components for separate logging (read by the trainer).
        self._loss_components = {
            "loss_img": loss_img.detach(),
            "loss_state": loss_state.detach(),
            "loss_act": loss_act.detach(),
        }
        return (
            self.img_loss_weight * loss_img
            + self.state_loss_weight * loss_state
            + self.action_loss_weight * loss_act
        )

    def forward(self, batch: Dict) -> torch.Tensor:
        return self.compute_loss(batch)

    # ------------------------------------------------------------------ sampling
    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        device=None,
        generator: torch.Generator | None = None,
        num_inference_steps: int | None = None,
    ) -> Dict:
        """Generate joint (image, state, action). Returns a RAW (dataset-format) dict:
        {"obs": {"rgb": (B, n_obs, 6, 84, 84) in [0,1], "states": (B, n_obs, state_dim) raw},
         "action": (B, horizon, action_dim) raw}."""
        device = device if device is not None else self.device
        steps = int(num_inference_steps) if num_inference_steps is not None else self.num_inference_steps
        c, h, w = self.rgb_shape
        rgb = torch.randn((batch_size, c, h, w), device=device, generator=generator)
        state = torch.randn(
            (batch_size, self.n_obs_steps, self.state_dim), device=device, generator=generator
        )
        act = torch.randn(
            (batch_size, self.horizon, self.action_dim), device=device, generator=generator
        )
        self.noise_scheduler.set_timesteps(steps)
        for t in self.noise_scheduler.timesteps:
            v_img, v_state, v_act = self._denoise(rgb, state, act, t)
            rgb = self.noise_scheduler.step(v_img, t, rgb, generator=generator).prev_sample
            state = self.noise_scheduler.step(v_state, t, state, generator=generator).prev_sample
            act = self.noise_scheduler.step(v_act, t, act, generator=generator).prev_sample

        rgb_raw = self.normalizer["rgb"].unnormalize(rgb).clamp(0.0, 1.0)  # -> [0,1]
        rgb_raw = rgb_raw.unsqueeze(1)  # (B, n_obs=1, 6, 84, 84)
        return {
            "obs": {
                "rgb": rgb_raw,
                "states": self.normalizer["states"].unnormalize(state),
            },
            "action": self.normalizer["action"].unnormalize(act),
        }


def build_joint_image_model_from_cfg(cfg, noise_scheduler) -> JointImageOADiTModel:
    """Construct the generator from a (composed or checkpoint-pickled) joint-generator config."""
    rollout_cfg = cfg.model.rollout
    arch_cfg = cfg.model.arch
    obs_meta = cfg.task.observation.shape_meta.obs
    return JointImageOADiTModel(
        rgb_shape=tuple(int(x) for x in obs_meta.rgb.shape),
        state_dim=int(obs_meta.states.shape[0]),
        action_dim=int(cfg.task.action.model_dim),
        n_obs_steps=int(rollout_cfg.n_obs_steps),
        horizon=int(rollout_cfg.horizon),
        noise_scheduler=noise_scheduler,
        d_model=int(arch_cfg.d_model),
        num_layers=int(arch_cfg.num_layers),
        num_heads=int(arch_cfg.num_heads),
        time_embed_dim=int(arch_cfg.time_embed_dim),
        unet_channels=tuple(int(c) for c in arch_cfg.unet_channels),
        num_res_blocks=int(arch_cfg.num_res_blocks),
        num_attn_heads=int(arch_cfg.num_attn_heads),
        dropout=float(arch_cfg.dropout),
        num_inference_steps=int(cfg.model.scheduler.num_inference_steps),
        img_loss_weight=float(arch_cfg.img_loss_weight),
        state_loss_weight=float(arch_cfg.state_loss_weight),
        action_loss_weight=float(arch_cfg.action_loss_weight),
    )
