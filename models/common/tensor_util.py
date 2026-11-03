"""Minimal tensor-shape helpers required by the crop randomizer."""

from __future__ import annotations

from typing import Iterable

import torch


def flatten(x: torch.Tensor, begin_axis: int = 1) -> torch.Tensor:
    shape = list(x.shape)
    if begin_axis < 0:
        begin_axis += len(shape)
    flattened_dim = 1
    for dim in shape[begin_axis:]:
        flattened_dim *= dim
    target_shape = tuple(shape[:begin_axis] + [flattened_dim])
    return x.reshape(target_shape)


def join_dimensions(x: torch.Tensor, begin_axis: int, end_axis: int) -> torch.Tensor:
    shape = list(x.shape)
    if begin_axis < 0:
        begin_axis += len(shape)
    if end_axis < 0:
        end_axis += len(shape)
    assert 0 <= begin_axis <= end_axis < len(shape)
    joined_dim = 1
    for dim in shape[begin_axis : end_axis + 1]:
        joined_dim *= dim
    target_shape = tuple(shape[:begin_axis] + [joined_dim] + shape[end_axis + 1 :])
    return x.reshape(target_shape)


def reshape_dimensions(
    x: torch.Tensor,
    begin_axis: int,
    end_axis: int,
    target_dims: Iterable[int],
) -> torch.Tensor:
    shape = list(x.shape)
    if begin_axis < 0:
        begin_axis += len(shape)
    if end_axis < 0:
        end_axis += len(shape)
    assert 0 <= begin_axis <= end_axis < len(shape)
    target_shape = tuple(shape[:begin_axis] + list(target_dims) + shape[end_axis + 1 :])
    return x.reshape(target_shape)


def unsqueeze_expand_at(x: torch.Tensor, size: int, dim: int) -> torch.Tensor:
    shape = list(x.shape)
    if dim < 0:
        dim += len(shape) + 1
    expanded_shape = shape.copy()
    expanded_shape.insert(dim, size)
    return x.unsqueeze(dim).expand(*expanded_shape)
