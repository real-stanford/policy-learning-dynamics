"""Lightweight normalization utilities for canonical diffusion datasets."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, Iterable

import numpy as np
import torch


def array_to_stats(data) -> Dict[str, np.ndarray]:
    array = np.asarray(data)
    if array.ndim == 0:
        array = array.reshape(1, 1)
    elif array.ndim == 1:
        array = array.reshape(-1, 1)
    else:
        array = array.reshape(-1, array.shape[-1])
    return {
        "min": array.min(axis=0).astype(np.float32),
        "max": array.max(axis=0).astype(np.float32),
        "mean": array.mean(axis=0).astype(np.float32),
        "std": array.std(axis=0).astype(np.float32),
    }


def _as_like(array: np.ndarray, like):
    if torch.is_tensor(like):
        return torch.as_tensor(array, device=like.device, dtype=like.dtype)
    return np.asarray(array, dtype=getattr(like, "dtype", np.float32))


@dataclass
class SingleFieldLinearNormalizer:
    scale: np.ndarray
    offset: np.ndarray
    input_stats: Dict[str, np.ndarray]

    def normalize(self, value):
        scale = _as_like(self.scale, value)
        offset = _as_like(self.offset, value)
        return value * scale + offset

    def unnormalize(self, value):
        scale = _as_like(self.scale, value)
        offset = _as_like(self.offset, value)
        return (value - offset) / scale

    def state_dict(self) -> Dict[str, np.ndarray]:
        return {
            "scale": self.scale.copy(),
            "offset": self.offset.copy(),
            "input_stats": {
                key: value.copy() for key, value in self.input_stats.items()
            },
        }

    def load_state_dict(self, state_dict: Dict[str, object]) -> None:
        self.scale = np.asarray(state_dict["scale"], dtype=np.float32).copy()
        self.offset = np.asarray(state_dict["offset"], dtype=np.float32).copy()
        input_stats = state_dict["input_stats"]
        self.input_stats = {
            key: np.asarray(value, dtype=np.float32).copy()
            for key, value in input_stats.items()
        }

    @classmethod
    def create_fit(
        cls,
        data,
        *,
        output_min: float = -1.0,
        output_max: float = 1.0,
        range_eps: float = 1e-6,
    ) -> "SingleFieldLinearNormalizer":
        stats = array_to_stats(data)
        data_range = stats["max"] - stats["min"]
        safe_range = np.where(data_range < range_eps, 1.0, data_range)
        scale = (output_max - output_min) / safe_range
        offset = output_min - scale * stats["min"]
        midpoint = (output_max + output_min) / 2.0
        offset = np.where(data_range < range_eps, midpoint, offset)
        scale = np.where(data_range < range_eps, 1.0, scale)
        return cls(
            scale=scale.astype(np.float32),
            offset=offset.astype(np.float32),
            input_stats=stats,
        )

    @classmethod
    def create_manual(
        cls,
        scale,
        offset,
        input_stats: Dict[str, np.ndarray],
    ) -> "SingleFieldLinearNormalizer":
        return cls(
            scale=np.asarray(scale, dtype=np.float32),
            offset=np.asarray(offset, dtype=np.float32),
            input_stats={
                key: np.asarray(value, dtype=np.float32)
                for key, value in input_stats.items()
            },
        )


class LinearNormalizer:
    def __init__(self) -> None:
        self._fields: Dict[str, SingleFieldLinearNormalizer] = {}

    def __getitem__(self, key: str) -> SingleFieldLinearNormalizer:
        return self._fields[key]

    def __setitem__(self, key: str, value: SingleFieldLinearNormalizer) -> None:
        self._fields[key] = value

    def keys(self):
        return self._fields.keys()

    def state_dict(self) -> Dict[str, Dict[str, np.ndarray]]:
        return {key: field.state_dict() for key, field in self._fields.items()}

    def load_state_dict(self, state_dict: Dict[str, Dict[str, np.ndarray]]) -> None:
        self._fields = {}
        for key, field_state in state_dict.items():
            field = SingleFieldLinearNormalizer.create_manual(
                scale=np.asarray(field_state["scale"], dtype=np.float32),
                offset=np.asarray(field_state["offset"], dtype=np.float32),
                input_stats={
                    stat_key: np.asarray(stat_value, dtype=np.float32)
                    for stat_key, stat_value in field_state["input_stats"].items()
                },
            )
            self._fields[key] = field

    def copy(self) -> "LinearNormalizer":
        result = LinearNormalizer()
        result.load_state_dict(copy.deepcopy(self.state_dict()))
        return result

    def normalize(self, value):
        if isinstance(value, dict):
            return {
                key: self._fields[key].normalize(item) for key, item in value.items()
            }
        if "_default" not in self._fields:
            raise KeyError(
                "LinearNormalizer has no '_default' field for array normalization."
            )
        return self._fields["_default"].normalize(value)

    def unnormalize(self, value):
        if isinstance(value, dict):
            return {
                key: self._fields[key].unnormalize(item) for key, item in value.items()
            }
        if "_default" not in self._fields:
            raise KeyError(
                "LinearNormalizer has no '_default' field for array normalization."
            )
        return self._fields["_default"].unnormalize(value)


def get_identity_normalizer_from_stat(
    stats: Dict[str, np.ndarray],
) -> SingleFieldLinearNormalizer:
    return SingleFieldLinearNormalizer.create_manual(
        scale=np.ones_like(stats["min"], dtype=np.float32),
        offset=np.zeros_like(stats["min"], dtype=np.float32),
        input_stats=stats,
    )


def get_range_normalizer_from_stat(
    stats: Dict[str, np.ndarray],
) -> SingleFieldLinearNormalizer:
    return SingleFieldLinearNormalizer.create_fit(
        np.stack([stats["min"], stats["max"]], axis=0)
    )


def get_image_identity_normalizer() -> SingleFieldLinearNormalizer:
    stats = {
        "min": np.asarray([0.0], dtype=np.float32),
        "max": np.asarray([1.0], dtype=np.float32),
        "mean": np.asarray([0.5], dtype=np.float32),
        "std": np.asarray([0.5], dtype=np.float32),
    }
    return get_identity_normalizer_from_stat(stats)


def concatenate_normalizers(
    normalizers: Iterable[SingleFieldLinearNormalizer],
) -> SingleFieldLinearNormalizer:
    parts = list(normalizers)
    if not parts:
        raise ValueError("Cannot concatenate an empty normalizer list.")
    return SingleFieldLinearNormalizer.create_manual(
        scale=np.concatenate([part.scale for part in parts], axis=0),
        offset=np.concatenate([part.offset for part in parts], axis=0),
        input_stats={
            key: np.concatenate([part.input_stats[key] for part in parts], axis=0)
            for key in parts[0].input_stats
        },
    )
