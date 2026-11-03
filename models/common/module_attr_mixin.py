from __future__ import annotations

import torch
import torch.nn as nn


class ModuleAttrMixin(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._dummy_variable = nn.Parameter(
            torch.zeros((), dtype=torch.float32),
            requires_grad=False,
        )

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype
