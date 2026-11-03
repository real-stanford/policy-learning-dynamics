"""Sequence sampling utilities for canonical zarr datasets."""

from __future__ import annotations

from typing import Optional

import numpy as np


def get_val_mask(n_episodes: int, val_ratio: float, seed: int = 0) -> np.ndarray:
    val_mask = np.zeros(n_episodes, dtype=bool)
    if val_ratio <= 0:
        return val_mask

    n_val = min(max(1, round(n_episodes * val_ratio)), n_episodes - 1)
    rng = np.random.default_rng(seed=seed)
    val_idxs = rng.choice(n_episodes, size=n_val, replace=False)
    val_mask[val_idxs] = True
    return val_mask


def downsample_mask(mask: np.ndarray, max_n: int | None, seed: int = 0) -> np.ndarray:
    if max_n is None or int(mask.sum()) <= max_n:
        return mask
    selected = np.flatnonzero(mask)
    rng = np.random.default_rng(seed=seed)
    keep = rng.choice(selected, size=max_n, replace=False)
    result = np.zeros_like(mask)
    result[keep] = True
    return result


def apply_train_ratio(
    mask: np.ndarray, ratio: float | None, seed: int = 0
) -> np.ndarray:
    if ratio is None:
        return mask
    if ratio <= 0.0 or ratio > 1.0:
        raise ValueError(f"ratio must be in (0, 1], got {ratio}.")
    selected = np.flatnonzero(mask)
    if selected.size == 0:
        raise RuntimeError("Cannot apply train ratio to an empty training mask.")
    if ratio >= 1.0:
        return mask

    keep_count = int(np.floor(selected.size * float(ratio)))
    keep_count = max(1, keep_count)
    rng = np.random.default_rng(seed=seed)
    keep = rng.choice(selected, size=keep_count, replace=False)
    result = np.zeros_like(mask)
    result[keep] = True
    return result


def create_sequence_indices(
    episode_lengths: np.ndarray,
    sequence_length: int,
    *,
    episode_mask: Optional[np.ndarray] = None,
    pad_before: int = 0,
    pad_after: int = 0,
) -> np.ndarray:
    if sequence_length < 1:
        raise ValueError("sequence_length must be positive.")
    if episode_mask is None:
        episode_mask = np.ones(len(episode_lengths), dtype=bool)

    pad_before = min(max(pad_before, 0), sequence_length - 1)
    pad_after = min(max(pad_after, 0), sequence_length - 1)

    indices = []
    for episode_idx, episode_length in enumerate(episode_lengths.tolist()):
        if not episode_mask[episode_idx]:
            continue
        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after
        for idx in range(min_start, max_start + 1):
            buffer_start_idx = max(idx, 0)
            buffer_end_idx = min(idx + sequence_length, episode_length)
            sample_start_idx = buffer_start_idx - idx
            sample_end_idx = sample_start_idx + (buffer_end_idx - buffer_start_idx)
            indices.append(
                [
                    episode_idx,
                    buffer_start_idx,
                    buffer_end_idx,
                    sample_start_idx,
                    sample_end_idx,
                ]
            )
    if not indices:
        return np.zeros((0, 5), dtype=np.int64)
    return np.asarray(indices, dtype=np.int64)


def sample_padded_sequence(
    array,
    *,
    buffer_start_idx: int,
    buffer_end_idx: int,
    sample_start_idx: int,
    sample_end_idx: int,
    sequence_length: int,
) -> np.ndarray:
    sample = np.asarray(array[buffer_start_idx:buffer_end_idx])
    if (
        sample.shape[0] == sequence_length
        and sample_start_idx == 0
        and sample_end_idx == sequence_length
    ):
        return sample

    result = np.zeros((sequence_length,) + sample.shape[1:], dtype=sample.dtype)
    result[sample_start_idx:sample_end_idx] = sample
    if sample_start_idx > 0:
        result[:sample_start_idx] = sample[0]
    if sample_end_idx < sequence_length:
        result[sample_end_idx:] = sample[-1]
    return result
