from __future__ import annotations

from typing import Dict

import torch

from models.common.module_attr_mixin import ModuleAttrMixin
from utils.common.normalization import LinearNormalizer


class BaseImagePolicy(ModuleAttrMixin):
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.compute_loss(batch)

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError()

    def predict_action(
        self, obs_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError()

    def reset(self) -> None:
        return None

    def set_normalizer(self, normalizer: LinearNormalizer) -> None:
        raise NotImplementedError()
