"""MimicGen HDF5 to canonical zarr conversion."""

from __future__ import annotations

import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
from tqdm.auto import tqdm

from datasets.absolute_action_converter import RobomimicAbsoluteActionConverter
from datasets.action_conversion import convert_action_sequence
from envs.task_registry import get_task_entry
from utils.common.action_pose_repr import (
    is_absolute_action_pose_repr,
    validate_action_pose_repr,
)

_MIMICGEN_WORKER_HANDLE: h5py.File | None = None
_MIMICGEN_WORKER_OUTPUT_PATH: str | None = None
_MIMICGEN_WORKER_MODALITY: str | None = None
_MIMICGEN_WORKER_OBS_KEYS: tuple[str, ...] | None = None
_MIMICGEN_WORKER_RGB_OBS_KEYS: tuple[str, ...] | None = None
_MIMICGEN_WORKER_ACTION_POSE_REPR: str | None = None
_MIMICGEN_WORKER_ROTATION_REP: str | None = None
_MIMICGEN_WORKER_ACTION_CONVERTER: RobomimicAbsoluteActionConverter | None = None
_MIMICGEN_WORKER_TRUNCATE: bool = False


def resolve_mimicgen_jobs(jobs: int | None) -> int:
    if jobs is not None:
        if jobs < 1:
            raise ValueError(f"Expected jobs >= 1, got {jobs}.")
        return jobs
    cpu_count = os.cpu_count()
    assert cpu_count is not None and cpu_count > 0, (
        f"Expected a positive os.cpu_count(), got {cpu_count}."
    )
    return min(8, cpu_count)


def _sorted_demo_keys(handle: h5py.File) -> list[str]:
    return sorted(handle["data"].keys(), key=lambda name: int(name.split("_")[-1]))


def _concat_obs(demo_group: h5py.Group, obs_keys: Iterable[str]) -> np.ndarray:
    chunks = [
        np.asarray(demo_group[f"obs/{key}"], dtype=np.float32) for key in obs_keys
    ]
    return np.concatenate(chunks, axis=-1).astype(np.float32)


def stack_hdf5_rgb_observations(
    demo_group: h5py.Group, rgb_obs_keys: Iterable[str]
) -> np.ndarray:
    rgb_chunks = []
    for key in rgb_obs_keys:
        array = np.asarray(demo_group[f"obs/{key}"])
        if array.ndim != 4:
            raise ValueError(
                f"Expected image observation for key '{key}', got {array.shape}."
            )
        if array.shape[-1] == 3:
            array = np.moveaxis(array, -1, 1)
        rgb_chunks.append(array.astype(np.uint8))
    return np.concatenate(rgb_chunks, axis=1)


def _write_meta_array(group, key: str, value: object) -> None:
    if key in group:
        del group[key]
    if isinstance(value, str):
        group[key] = np.asarray(value, dtype="<U128")
    elif isinstance(value, (list, tuple)) and value and isinstance(value[0], str):
        group[key] = np.asarray(value, dtype="<U128")
    else:
        group[key] = np.asarray(value)


def _optional_model_xml(demo_group: h5py.Group) -> str | bytes | None:
    if "model_file" not in demo_group.attrs:
        return None
    return demo_group.attrs["model_file"]


def _success_array(demo_group: h5py.Group) -> np.ndarray | None:
    if "rewards" not in demo_group:
        return None
    success = float(np.asarray(demo_group["rewards"]).sum() > 0)
    return np.asarray([success], dtype=np.float32)


def _rewards_array(demo_group: h5py.Group) -> np.ndarray | None:
    """Per-timestep rewards (T,) as float32, or None if the demo has no rewards."""
    if "rewards" not in demo_group:
        return None
    return np.asarray(demo_group["rewards"], dtype=np.float32).reshape(-1)


def _truncate_to_single_success(
    states: np.ndarray,
    actions: np.ndarray,
    rgb: np.ndarray | None,
    rewards: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
    """Cut a trajectory to a single terminal success.

    If the demo contains a success (reward > 0), keep timesteps ``[0 : onset+1]``
    where ``onset`` is the first positive-reward index. If it contains no success
    (some MimicGen demos complete the task but never trip the reward checker),
    keep the full trajectory and treat its last recorded state as the success.

    In both cases the reward array is rebuilt as ``0,0,...,0,1`` so the trajectory
    ends with exactly one success, and all per-step arrays (states/actions/rgb/
    rewards) are sliced consistently.
    """
    if rewards is not None and bool((rewards > 0).any()):
        keep = int(np.nonzero(rewards > 0)[0][0]) + 1
    else:
        # No success frame: force the final timestep to be the (single) success.
        keep = int(states.shape[0])
    rewards_out = np.zeros(keep, dtype=np.float32)
    rewards_out[-1] = 1.0
    rgb_out = rgb[:keep] if rgb is not None else None
    return states[:keep], actions[:keep], rgb_out, rewards_out


def _write_episode_group(
    episode,
    *,
    states: np.ndarray,
    actions: np.ndarray,
    rgb: np.ndarray | None,
    episode_success: np.ndarray | None,
    rewards: np.ndarray | None = None,
) -> None:
    episode["states"] = states.astype(np.float32)
    episode["actions"] = actions.astype(np.float32)
    if rgb is not None:
        episode["rgb"] = rgb
    if episode_success is not None:
        episode["episode_success"] = episode_success
    if rewards is not None:
        episode["rewards"] = rewards.astype(np.float32)


def _init_mimicgen_worker(
    input_path: str,
    output_path: str,
    modality: str,
    obs_keys: tuple[str, ...],
    rgb_obs_keys: tuple[str, ...],
    action_pose_repr: str,
    rotation_rep: str,
    truncate: bool,
) -> None:
    global _MIMICGEN_WORKER_HANDLE
    global _MIMICGEN_WORKER_OUTPUT_PATH
    global _MIMICGEN_WORKER_MODALITY
    global _MIMICGEN_WORKER_OBS_KEYS
    global _MIMICGEN_WORKER_RGB_OBS_KEYS
    global _MIMICGEN_WORKER_ACTION_POSE_REPR
    global _MIMICGEN_WORKER_ROTATION_REP
    global _MIMICGEN_WORKER_ACTION_CONVERTER
    global _MIMICGEN_WORKER_TRUNCATE

    _MIMICGEN_WORKER_HANDLE = h5py.File(input_path, "r")
    _MIMICGEN_WORKER_OUTPUT_PATH = output_path
    _MIMICGEN_WORKER_MODALITY = modality
    _MIMICGEN_WORKER_OBS_KEYS = obs_keys
    _MIMICGEN_WORKER_RGB_OBS_KEYS = rgb_obs_keys
    _MIMICGEN_WORKER_ACTION_POSE_REPR = action_pose_repr
    _MIMICGEN_WORKER_ROTATION_REP = rotation_rep
    _MIMICGEN_WORKER_TRUNCATE = truncate
    _MIMICGEN_WORKER_ACTION_CONVERTER = (
        RobomimicAbsoluteActionConverter(input_path)
        if is_absolute_action_pose_repr(action_pose_repr)
        else None
    )


def _convert_mimicgen_episode_worker(job: tuple[int, str]) -> int:
    episode_idx, demo_name = job
    assert _MIMICGEN_WORKER_HANDLE is not None
    assert _MIMICGEN_WORKER_OUTPUT_PATH is not None
    assert _MIMICGEN_WORKER_MODALITY is not None
    assert _MIMICGEN_WORKER_OBS_KEYS is not None
    assert _MIMICGEN_WORKER_RGB_OBS_KEYS is not None
    assert _MIMICGEN_WORKER_ACTION_POSE_REPR is not None
    assert _MIMICGEN_WORKER_ROTATION_REP is not None

    requires_absolute_actions = is_absolute_action_pose_repr(
        _MIMICGEN_WORKER_ACTION_POSE_REPR
    )
    assert (
        _MIMICGEN_WORKER_ACTION_CONVERTER is not None
    ) == requires_absolute_actions, (
        "Action converter state is inconsistent with action_pose_repr."
    )

    demo_group = _MIMICGEN_WORKER_HANDLE[f"data/{demo_name}"]
    model_xml = _optional_model_xml(demo_group)
    if requires_absolute_actions:
        assert model_xml is not None, (
            "MimicGen absolute-action conversion requires per-episode "
            f"'model_file' for demo '{demo_name}'."
        )
    states = _concat_obs(demo_group, _MIMICGEN_WORKER_OBS_KEYS)
    actions = convert_action_sequence(
        np.asarray(demo_group["states"], dtype=np.float32),
        np.asarray(demo_group["actions"], dtype=np.float32),
        action_pose_repr=_MIMICGEN_WORKER_ACTION_POSE_REPR,
        rotation_rep=_MIMICGEN_WORKER_ROTATION_REP,
        absolute_action_converter=_MIMICGEN_WORKER_ACTION_CONVERTER,
        model_xml=model_xml,
    )
    rgb = (
        stack_hdf5_rgb_observations(demo_group, _MIMICGEN_WORKER_RGB_OBS_KEYS)
        if _MIMICGEN_WORKER_MODALITY == "image"
        else None
    )
    rewards = _rewards_array(demo_group)
    if _MIMICGEN_WORKER_TRUNCATE:
        states, actions, rgb, rewards = _truncate_to_single_success(
            states, actions, rgb, rewards
        )
        # Every truncated trajectory ends with exactly one success by construction
        # (including demos that never tripped the reward checker, forced at the end).
        episode_success = np.asarray([1.0], dtype=np.float32)
    else:
        episode_success = _success_array(demo_group)

    import zarr

    root = zarr.open(_MIMICGEN_WORKER_OUTPUT_PATH, mode="a")
    data_group = root["data"]
    episode = data_group.require_group(f"episode_{episode_idx:06d}")
    _write_episode_group(
        episode,
        states=states,
        actions=actions,
        rgb=rgb,
        episode_success=episode_success,
        rewards=rewards,
    )
    return episode_idx


def convert_mimicgen_hdf5_to_zarr(
    input_path: str | Path,
    output_path: str | Path,
    *,
    task_name: str,
    modality: str,
    action_pose_repr: str = "delta",
    rotation_rep: str = "axis_angle",
    max_episodes: int | None = None,
    jobs: int | None = None,
    truncate: bool = False,
) -> Path:
    try:
        import zarr
    except ImportError as exc:
        raise ImportError("zarr is required for canonical dataset conversion.") from exc

    entry = get_task_entry("mimicgen", task_name)
    input_path = Path(input_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    resolved_action_pose_repr = validate_action_pose_repr(action_pose_repr)
    if resolved_action_pose_repr == "relative":
        raise ValueError(
            "action_pose_repr='relative' is sequence-dependent and cannot be stored "
            "as a per-timestep canonical action. Use 'delta' or 'absolute'."
        )
    requires_absolute_actions = is_absolute_action_pose_repr(resolved_action_pose_repr)
    resolved_jobs = resolve_mimicgen_jobs(jobs)

    if modality not in {"lowdim", "image"}:
        raise ValueError(f"Unsupported modality '{modality}'.")

    if output_path.exists():
        shutil.rmtree(output_path)

    root = zarr.open(str(output_path), mode="w")
    data_group = root.require_group("data")
    meta_group = root.require_group("meta")
    absolute_action_converter = None

    try:
        with h5py.File(input_path, "r") as handle:
            demos = _sorted_demo_keys(handle)
            if max_episodes is not None:
                demos = demos[:max_episodes]

            obs_keys = (
                entry.lowdim_obs_keys
                if modality == "lowdim"
                else entry.image_lowdim_obs_keys
            )
            action_dim = (
                entry.absolute_rotation6d_action_dim
                if requires_absolute_actions and rotation_rep == "rotation_6d"
                else entry.delta_action_dim
            )

            if resolved_jobs > 1:
                parallel_jobs = [
                    (episode_idx, demo_name)
                    for episode_idx, demo_name in enumerate(demos)
                ]
                with ProcessPoolExecutor(
                    max_workers=resolved_jobs,
                    initializer=_init_mimicgen_worker,
                    initargs=(
                        str(input_path),
                        str(output_path),
                        modality,
                        tuple(obs_keys),
                        tuple(entry.rgb_obs_keys),
                        resolved_action_pose_repr,
                        rotation_rep,
                        truncate,
                    ),
                ) as executor:
                    for _ in tqdm(
                        executor.map(_convert_mimicgen_episode_worker, parallel_jobs),
                        total=len(parallel_jobs),
                        desc=f"Converting MimicGen {task_name} {modality}",
                        leave=False,
                    ):
                        pass
            else:
                absolute_action_converter = (
                    RobomimicAbsoluteActionConverter(input_path)
                    if requires_absolute_actions
                    else None
                )
                for episode_idx, demo_name in enumerate(
                    tqdm(
                        demos,
                        desc=f"Converting MimicGen {task_name} {modality}",
                        leave=False,
                    )
                ):
                    demo_group = handle[f"data/{demo_name}"]
                    model_xml = _optional_model_xml(demo_group)
                    if requires_absolute_actions:
                        assert model_xml is not None, (
                            "MimicGen absolute-action conversion requires per-episode "
                            f"'model_file' for demo '{demo_name}'."
                        )
                    states = _concat_obs(demo_group, obs_keys)
                    actions = convert_action_sequence(
                        np.asarray(demo_group["states"], dtype=np.float32),
                        np.asarray(demo_group["actions"], dtype=np.float32),
                        action_pose_repr=resolved_action_pose_repr,
                        rotation_rep=rotation_rep,
                        absolute_action_converter=absolute_action_converter,
                        model_xml=model_xml,
                    )
                    rgb = (
                        stack_hdf5_rgb_observations(demo_group, entry.rgb_obs_keys)
                        if modality == "image"
                        else None
                    )
                    rewards = _rewards_array(demo_group)
                    if truncate:
                        states, actions, rgb, rewards = _truncate_to_single_success(
                            states, actions, rgb, rewards
                        )
                        episode_success = np.asarray([1.0], dtype=np.float32)
                    else:
                        episode_success = _success_array(demo_group)
                    episode = data_group.require_group(f"episode_{episode_idx:06d}")
                    _write_episode_group(
                        episode,
                        states=states,
                        actions=actions,
                        rgb=rgb,
                        episode_success=episode_success,
                        rewards=rewards,
                    )

            _write_meta_array(meta_group, "schema_version", "1.0.0")
            _write_meta_array(meta_group, "task_name", entry.task_name)
            _write_meta_array(
                meta_group,
                "observation_type",
                "low_dim" if modality == "lowdim" else "rgb+low_dim",
            )
            _write_meta_array(
                meta_group,
                "action_pose_repr",
                resolved_action_pose_repr,
            )
            _write_meta_array(meta_group, "action_mode", resolved_action_pose_repr)
            _write_meta_array(meta_group, "rotation_rep", rotation_rep)
            _write_meta_array(
                meta_group,
                "obs_dim",
                entry.lowdim_obs_dim
                if modality == "lowdim"
                else entry.image_lowdim_obs_dim,
            )
            _write_meta_array(meta_group, "state_keys", tuple(obs_keys))
            _write_meta_array(meta_group, "action_dim", action_dim)
            _write_meta_array(meta_group, "num_episodes", len(demos))
            _write_meta_array(meta_group, "truncate_single_success", bool(truncate))
            if modality == "image":
                _write_meta_array(meta_group, "camera_names", entry.render_camera_names)
                _write_meta_array(meta_group, "rgb_layout", "camera_concat_chw")
    finally:
        if absolute_action_converter is not None:
            absolute_action_converter.close()

    return output_path
