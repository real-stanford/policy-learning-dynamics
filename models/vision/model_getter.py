from __future__ import annotations

import torch
import torchvision


def get_resnet(name: str, weights=None, **kwargs) -> torch.nn.Module:
    assert hasattr(torchvision.models, name), (
        f"Unsupported torchvision backbone '{name}'."
    )
    builder = getattr(torchvision.models, name)
    model = builder(weights=weights, **kwargs)
    model.fc = torch.nn.Identity()
    return model
