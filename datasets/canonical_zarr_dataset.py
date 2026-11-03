"""Canonical zarr-backed image dataset for diffusion-policy training.

Every policy, generator and probe in this repo reads training windows through
``CanonicalImageDataset``: one window = ``n_obs_steps`` frames of ``{rgb, states}`` plus an
action chunk of length ``horizon``, enumerated over the train-split episodes with the
padding convention of ``datasets/sampler.py``.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import zarr

from datasets.base_dataset import BaseImageDataset
from datasets.sampler import (
    apply_train_ratio,
    create_sequence_indices,
    downsample_mask,
    get_val_mask,
    sample_padded_sequence,
)
from utils.common.action_pose_repr import (
    pose_repr_from_action_mode,
    validate_action_pose_repr,
)
from utils.common.action_sequence import (
    delta_pose_actions_to_relative,
    relative_pose_actions_to_delta,
)
from utils.common.normalization import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
    array_to_stats,
    concatenate_normalizers,
    get_identity_normalizer_from_stat,
    get_image_identity_normalizer,
    get_range_normalizer_from_stat,
)

logger = logging.getLogger(__name__)


def _read_meta_value(meta_group, key: str):
    value = np.asarray(meta_group[key])
    if value.shape == ():
        return value.item()
    if value.size == 1:
        return value.reshape(-1)[0].item()
    return tuple(value.tolist())


def _load_zarr_into_memory(zarr_path: Path, num_threads: int = 32) -> zarr.Group:
    """Copy every chunk/metadata file of a directory zarr into a ``zarr.MemoryStore``.

    The canonical stores keep one file per chunk (~30k files / 3 GB per task), and
    the training data lives on NFS, where each open/read/close costs 10-90 ms. Read
    lazily per ``__getitem__`` that is ~60 ms/sample versus ~0.5 ms from RAM, so the
    loader -- not the GPU -- bounds the epoch time. Doing the copy ONCE here, before
    the DataLoader forks its (persistent) workers, lets every worker share the bytes
    copy-on-write. The returned group is read-only and exposes the same
    ``root["data"][episode][key]`` interface as the lazy store, so nothing downstream
    changes. Samples are bit-identical either way; only the I/O path differs.
    """
    assert zarr_path.is_dir(), (
        f"in_memory=True needs a directory zarr store, got {zarr_path}; "
        "set dataset.processing.in_memory=false for other store types."
    )
    src = zarr.storage.DirectoryStore(str(zarr_path))
    dst = zarr.storage.MemoryStore()
    t0 = time.perf_counter()
    keys = list(src.keys())
    t_list = time.perf_counter() - t0

    def _copy(key: str) -> int:
        value = src[key]
        dst[key] = value
        return len(value)

    with ThreadPoolExecutor(max_workers=num_threads) as pool:
        n_bytes = sum(pool.map(_copy, keys))
    if os.environ.get("LOCAL_RANK", "0") == "0":
        logger.info(
            "[dataset] loaded %s into memory: %d objects, %.2f GiB in %.1fs "
            "(listing %.1fs, %d threads)",
            zarr_path.name,
            len(keys),
            n_bytes / 2**30,
            time.perf_counter() - t0,
            t_list,
            num_threads,
        )
    return zarr.open_group(store=dst, mode="r")


def _sorted_episode_names(data_group) -> list[str]:
    return sorted(data_group.keys(), key=lambda name: int(name.split("_")[-1]))


def _absolute_rotation6d_normalizer(actions: np.ndarray) -> SingleFieldLinearNormalizer:
    """Range-normalize position and gripper; leave the (unit-norm) rot6d block as is."""
    action_dim = actions.shape[-1]
    if action_dim not in (10, 20):
        return get_range_normalizer_from_stat(array_to_stats(actions))

    parts = []
    num_arms = action_dim // 10
    for arm_idx in range(num_arms):
        offset = arm_idx * 10
        parts.append(
            get_range_normalizer_from_stat(
                array_to_stats(actions[..., offset : offset + 3])
            )
        )
        parts.append(
            get_identity_normalizer_from_stat(
                array_to_stats(actions[..., offset + 3 : offset + 9])
            )
        )
        parts.append(
            get_range_normalizer_from_stat(
                array_to_stats(actions[..., offset + 9 : offset + 10])
            )
        )
    return concatenate_normalizers(parts)


def _build_action_normalizer(
    actions: np.ndarray, action_pose_repr: str, rotation_rep: str
) -> SingleFieldLinearNormalizer:
    resolved_action_pose_repr = validate_action_pose_repr(action_pose_repr)
    if resolved_action_pose_repr == "delta":
        return get_identity_normalizer_from_stat(array_to_stats(actions))
    if (
        resolved_action_pose_repr in {"absolute", "relative"}
        and rotation_rep == "rotation_6d"
    ):
        return _absolute_rotation6d_normalizer(actions)
    if resolved_action_pose_repr in {"absolute", "relative"}:
        return get_range_normalizer_from_stat(array_to_stats(actions))
    raise ValueError(f"Unsupported action_pose_repr '{resolved_action_pose_repr}'.")


def _resolve_action_pose_repr(meta_group) -> str:
    if "action_pose_repr" in meta_group:
        return validate_action_pose_repr(
            str(_read_meta_value(meta_group, "action_pose_repr"))
        )
    if "action_mode" in meta_group:
        return pose_repr_from_action_mode(
            str(_read_meta_value(meta_group, "action_mode"))
        )
    raise KeyError(
        "Missing action pose metadata. Expected key 'meta/action_pose_repr' "
        "(or legacy 'meta/action_mode')."
    )


def _assert_action_pose_repr_compatible(*, requested: str, stored: str) -> None:
    if requested == stored:
        return
    if requested == "relative" and stored == "delta":
        return
    if requested == "delta" and stored == "relative":
        return
    raise AssertionError(
        f"Unsupported action representation conversion: requested '{requested}' "
        f"from stored '{stored}'."
    )


def _convert_action_pose_repr(
    actions: np.ndarray,
    *,
    source_action_pose_repr: str,
    target_action_pose_repr: str,
    rotation_rep: str,
) -> np.ndarray:
    if source_action_pose_repr == target_action_pose_repr:
        return np.asarray(actions, dtype=np.float32)
    if source_action_pose_repr == "delta" and target_action_pose_repr == "relative":
        return delta_pose_actions_to_relative(
            np.asarray(actions, dtype=np.float32),
            rotation_rep=rotation_rep,
        )
    if source_action_pose_repr == "relative" and target_action_pose_repr == "delta":
        return relative_pose_actions_to_delta(
            np.asarray(actions, dtype=np.float32),
            rotation_rep=rotation_rep,
        )
    raise AssertionError(
        f"Unsupported action representation conversion: '{source_action_pose_repr}' -> "
        f"'{target_action_pose_repr}'."
    )


class CanonicalImageDataset(BaseImageDataset):
    """Windows of ``{obs: {rgb, states}, action}`` over the train-split episodes.

    Train/val split: ``get_val_mask(val_ratio, seed)`` marks the validation episodes;
    ``apply_train_ratio`` / ``downsample_mask(max_train_episodes, seed)`` subsample the
    remaining ones. The split is a deterministic function of
    ``(n_episodes, val_ratio, ratio, max_train_episodes, seed)``, so every consumer
    that shares these parameters (policy, generator, probes) sees the identical
    episode list. ``episode_include_path`` replaces the subsampling with an explicit
    JSON list of train episodes (used by the held-out evaluation banks).
    """

    def __init__(
        self,
        zarr_path: str | Path,
        *,
        horizon: int,
        pad_before: int = 0,
        pad_after: int = 0,
        n_obs_steps: int | None = None,
        seed: int = 42,
        val_ratio: float = 0.0,
        ratio: float | None = None,
        max_train_episodes: int | None = None,
        action_pose_repr: str | None = None,
        episode_include_path: str | None = None,
        in_memory: bool = False,
    ) -> None:
        self.zarr_path = Path(zarr_path).expanduser().resolve()
        assert self.zarr_path.exists(), f"{str(self.zarr_path)} does not exist"
        self.in_memory = bool(in_memory)
        if self.in_memory:
            self.root = _load_zarr_into_memory(self.zarr_path)
        else:
            self.root = zarr.open(str(self.zarr_path), mode="r")
        self.data_group = self.root["data"]
        self.meta_group = self.root["meta"]
        self.stored_action_pose_repr = _resolve_action_pose_repr(self.meta_group)
        if action_pose_repr is not None:
            requested_action_pose_repr = validate_action_pose_repr(action_pose_repr)
            _assert_action_pose_repr_compatible(
                requested=requested_action_pose_repr,
                stored=self.stored_action_pose_repr,
            )
            self.action_pose_repr = requested_action_pose_repr
        else:
            self.action_pose_repr = self.stored_action_pose_repr
        self.rotation_rep = str(self._meta("rotation_rep"))
        self.episode_names = _sorted_episode_names(self.data_group)
        self.episode_lengths = np.asarray(
            [
                int(self.data_group[name]["actions"].shape[0])
                for name in self.episode_names
            ],
            dtype=np.int64,
        )
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.n_obs_steps = n_obs_steps
        self.seed = seed
        self.val_ratio = val_ratio
        self.ratio = ratio

        val_mask = get_val_mask(len(self.episode_names), val_ratio=val_ratio, seed=seed)
        self.val_mask = val_mask
        train_mask = apply_train_ratio(~val_mask, ratio=ratio, seed=seed)
        train_mask = downsample_mask(train_mask, max_train_episodes, seed=seed)
        if episode_include_path is not None:
            # EXPLICIT train-episode list: the train mask is exactly the listed episodes;
            # the seeded subsampling knobs must be inert so two selection mechanisms
            # can never silently interact.
            assert max_train_episodes is None, (
                "episode_include_path and max_train_episodes are mutually exclusive."
            )
            assert ratio is None or float(ratio) >= 1.0, (
                "episode_include_path and ratio subsampling are mutually exclusive."
            )
            include_file = Path(episode_include_path).expanduser()
            assert include_file.exists(), f"missing episode list: {include_file}"
            with open(include_file) as f:
                include_idx = json.load(f)
            include_idx = np.asarray(sorted(int(i) for i in include_idx), dtype=np.int64)
            assert include_idx.size > 0
            assert np.unique(include_idx).size == include_idx.size, "duplicate episodes"
            assert include_idx.min() >= 0 and include_idx.max() < len(self.episode_names)
            assert not val_mask[include_idx].any(), (
                "episode_include_path overlaps the val split; set val_ratio=0.0 or "
                "use a disjoint list."
            )
            train_mask = np.zeros_like(train_mask)
            train_mask[include_idx] = True
        self.train_mask = train_mask
        # Log the selected train/val EPISODE indices (rank 0 only) so two runs can be
        # checked for an identical split at a glance (train_sha1) or in full.
        if os.environ.get("LOCAL_RANK", "0") == "0":
            train_idx = np.flatnonzero(train_mask)
            val_idx = np.flatnonzero(val_mask)
            train_sha1 = hashlib.sha1(train_idx.tobytes()).hexdigest()[:12]
            logger.info(
                "[data-split] %s | n_episodes=%d val_ratio=%s max_train_episodes=%s "
                "ratio=%s seed=%s | #train=%d #val=%d train_sha1=%s",
                self.zarr_path.name,
                len(self.episode_names),
                val_ratio,
                max_train_episodes,
                ratio,
                seed,
                train_idx.size,
                val_idx.size,
                train_sha1,
            )
            logger.info("[data-split] train episodes: %s", train_idx.tolist())
            logger.info("[data-split] val   episodes: %s", val_idx.tolist())
        self.indices = create_sequence_indices(
            self.episode_lengths,
            horizon,
            episode_mask=train_mask,
            pad_before=pad_before,
            pad_after=pad_after,
        )

    def get_validation_dataset(self) -> "CanonicalImageDataset":
        val_set = copy.copy(self)
        val_mask = self.val_mask
        val_set.train_mask = val_mask
        val_set.indices = create_sequence_indices(
            self.episode_lengths,
            self.horizon,
            episode_mask=val_mask,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
        )
        return val_set

    def __len__(self) -> int:
        return len(self.indices)

    def _sample_episode_array(
        self, episode_index: int, key: str, sample_index: np.ndarray
    ) -> np.ndarray:
        episode = self.data_group[self.episode_names[episode_index]]
        _, buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = (
            sample_index
        )
        return sample_padded_sequence(
            episode[key],
            buffer_start_idx=int(buffer_start_idx),
            buffer_end_idx=int(buffer_end_idx),
            sample_start_idx=int(sample_start_idx),
            sample_end_idx=int(sample_end_idx),
            sequence_length=self.horizon,
        )

    def _obs_slice(self) -> slice:
        if self.n_obs_steps is None:
            return slice(None)
        return slice(0, self.n_obs_steps)

    def _meta(self, key: str):
        return _read_meta_value(self.meta_group, key)

    def _collect_train_data(self, key: str) -> np.ndarray:
        chunks = []
        for episode_idx, enabled in enumerate(self.train_mask.tolist()):
            if not enabled:
                continue
            episode = self.data_group[self.episode_names[episode_idx]]
            values = np.asarray(episode[key], dtype=np.float32)
            if key == "actions":
                values = _convert_action_pose_repr(
                    values,
                    source_action_pose_repr=self.stored_action_pose_repr,
                    target_action_pose_repr=self.action_pose_repr,
                    rotation_rep=self.rotation_rep,
                )
            chunks.append(values)
        if not chunks:
            raise RuntimeError("Training split is empty.")
        return np.concatenate(chunks, axis=0)

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self._collect_train_data("actions").astype(np.float32))

    def get_normalizer(self) -> LinearNormalizer:
        normalizer = LinearNormalizer()
        states = self._collect_train_data("states").astype(np.float32)
        actions = self._collect_train_data("actions").astype(np.float32)
        normalizer["states"] = get_range_normalizer_from_stat(array_to_stats(states))
        normalizer["action"] = _build_action_normalizer(
            actions,
            action_pose_repr=self.action_pose_repr,
            rotation_rep=self.rotation_rep,
        )
        normalizer["rgb"] = get_image_identity_normalizer()
        return normalizer

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample_index = self.indices[idx]
        episode_index = int(sample_index[0])
        states = self._sample_episode_array(
            episode_index, "states", sample_index
        ).astype(np.float32)
        actions = self._sample_episode_array(
            episode_index, "actions", sample_index
        ).astype(np.float32)
        actions = _convert_action_pose_repr(
            actions,
            source_action_pose_repr=self.stored_action_pose_repr,
            target_action_pose_repr=self.action_pose_repr,
            rotation_rep=self.rotation_rep,
        )
        rgb = (
            self._sample_episode_array(episode_index, "rgb", sample_index).astype(
                np.float32
            )
            / 255.0
        )
        obs_slice = self._obs_slice()
        return {
            "obs": {
                "states": torch.from_numpy(states[obs_slice]),
                "rgb": torch.from_numpy(rgb[obs_slice]),
            },
            "action": torch.from_numpy(actions),
        }
