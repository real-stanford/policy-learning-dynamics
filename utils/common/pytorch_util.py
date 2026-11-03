"""Small PyTorch helpers shared by the root training stack."""

from __future__ import annotations

from typing import Callable, Dict

import torch
import torch.nn as nn


def dict_apply(
    value: Dict[str, torch.Tensor], func: Callable[[torch.Tensor], torch.Tensor]
) -> Dict[str, torch.Tensor]:
    result: Dict[str, torch.Tensor] = {}
    for key, item in value.items():
        if isinstance(item, dict):
            result[key] = dict_apply(item, func)
            continue
        result[key] = func(item)
    return result


def replace_submodules(
    root_module: nn.Module,
    predicate: Callable[[nn.Module], bool],
    func: Callable[[nn.Module], nn.Module],
) -> nn.Module:
    if predicate(root_module):
        return func(root_module)

    target_keys = [
        key.split(".")
        for key, module in root_module.named_modules(remove_duplicate=True)
        if predicate(module)
    ]
    for *parent_keys, module_key in target_keys:
        parent_module = root_module
        if parent_keys:
            parent_module = root_module.get_submodule(".".join(parent_keys))
        if isinstance(parent_module, nn.Sequential):
            source_module = parent_module[int(module_key)]
            parent_module[int(module_key)] = func(source_module)
            continue
        source_module = getattr(parent_module, module_key)
        setattr(parent_module, module_key, func(source_module))

    remaining = [
        key
        for key, module in root_module.named_modules(remove_duplicate=True)
        if predicate(module)
    ]
    assert not remaining, f"Failed to replace submodules: {remaining}"
    return root_module
