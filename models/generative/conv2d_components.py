from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _groups(ch: int, max_groups: int = 8) -> int:
    """GroupNorm groups: the largest divisor of `ch` that is <= max_groups (>=1)."""
    for g in range(min(max_groups, ch), 0, -1):
        if ch % g == 0:
            return g
    return 1


class ResBlock2d(nn.Module):

    def __init__(
        self, in_ch: int, out_ch: int, time_emb_dim: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(_groups(out_ch), out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        # FiLM: time_emb -> (scale, shift) per output channel.
        self.film = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, 2 * out_ch))
        nn.init.zeros_(self.film[1].weight)
        nn.init.zeros_(self.film[1].bias)
        self.skip = (
            nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.film(time_emb).chunk(2, dim=-1)  # (B, out_ch) each
        h = self.norm2(h) * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(h)))
        return h + self.skip(x)


class Downsample2d(nn.Module):
    """Exact stride-2 downsample (HxW -> H/2 x W/2) via a strided 3x3 conv."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample2d(nn.Module):
    """Exact stride-2 upsample (HxW -> 2H x 2W) via a 4x4 transposed conv (k4 s2 p1)."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class SelfAttention2d(nn.Module):
    """Multi-head spatial self-attention over the HxW grid (flatten -> attend -> reshape),
    with a residual connection. Used at the low-resolution bottleneck for global coherence."""

    def __init__(self, ch: int, num_heads: int = 4) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(_groups(ch), ch)
        self.attn = nn.MultiheadAttention(ch, num_heads, batch_first=True)
        self.proj = nn.Conv2d(ch, ch, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        y = self.norm(x).reshape(b, c, h * w).transpose(1, 2)  # (B, HW, C)
        y, _ = self.attn(y, y, y, need_weights=False)
        y = y.transpose(1, 2).reshape(b, c, h, w)
        return x + self.proj(y)
