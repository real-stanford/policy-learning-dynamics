from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from pathlib import Path

import h5py
import numpy as np
import torch
from omegaconf import open_dict

from datasets.canonical_zarr_dataset import _load_zarr_into_memory
from utils.training_utils.factories import build_dataset

from dice_rl.base_policy import FrozenDiffusionBasePolicy
from dice_rl.episodes import EpisodeRecord
from dice_rl.replay_buffer import ChunkReplayBuffer

_SPLIT_LINE = re.compile(r"\[data-split\] train episodes: (\[[^\]]*\])")


def read_train_episode_indices_from_log(run_dir: Path) -> list[int] | None:
    """Parse the pretraining run log for the selected train episode indices.
    Returns None when the log does not exist; raises if it exists but has no
    split line (the check must not silently pass)."""
    log_path = run_dir / "__init__.log"
    if not log_path.exists():
        return None
    text = log_path.read_text(encoding="utf-8", errors="replace")
    match = _SPLIT_LINE.search(text)
    assert match is not None, f"No '[data-split] train episodes' line in {log_path}"
    return [int(v) for v in json.loads(match.group(1))]


def _sorted_demo_keys(handle: h5py.File) -> list[str]:
    return sorted(handle["data"].keys(), key=lambda name: int(name.split("_")[-1]))


NO_SUCCESS_POLICIES = ("force_last", "zero", "drop")


def truncate_to_single_success(
    rewards: np.ndarray, *, no_success_policy: str = "force_last"
) -> tuple[int, np.ndarray] | None:
    """Cut a demo at its FIRST success step (dice-rl process_robomimic_dataset.py
    ``--truncate``: keep ``[0 : first index with cumsum(r) >= 1]`` inclusive, raw
    rewards ``0,...,0,1``). Returns ``(keep_length, rewards_out)``.

    Demos that never trip the reward checker have no dice-rl precedent
    (robomimic PH demos all succeed); ``no_success_policy`` decides:
      force_last -> keep the full demo, last step = the success
                    (datasets/mimicgen_conversion.py::_truncate_to_single_success);
      zero       -> keep the full demo with zero reward (a failed terminal);
      drop       -> exclude the demo (returns None).
    """
    assert no_success_policy in NO_SUCCESS_POLICIES, no_success_policy
    rewards = np.asarray(rewards, dtype=np.float32).reshape(-1)
    if bool((rewards > 0).any()):
        keep = int(np.nonzero(rewards > 0)[0][0]) + 1
        out = np.zeros(keep, dtype=np.float32)
        out[-1] = 1.0
        return keep, out
    keep = int(rewards.shape[0])
    if no_success_policy == "drop":
        return None
    out = np.zeros(keep, dtype=np.float32)
    if no_success_policy == "force_last":
        out[-1] = 1.0
    return keep, out


def expert_cache_key(
    base_policy: FrozenDiffusionBasePolicy,
    train_indices: np.ndarray,
    no_success_policy: str,
    aux_state_zarr: str | None = None,
) -> str:
    fields = {
        "checkpoint": str(base_policy.checkpoint_path),
        "state_key": base_policy.state_key,
        "autocast": base_policy.autocast,
        "no_success_policy": no_success_policy,
        "train_indices": [int(i) for i in train_indices],
        "zarr": str(base_policy.cfg.dataset.source.zarr_path),
        "hdf5": str(base_policy.cfg.task.data.hdf5_path),
    }
    if aux_state_zarr is not None:
        fields["aux_state_zarr"] = str(aux_state_zarr)
    payload = json.dumps(fields, sort_keys=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def _resolve_train_indices(base_policy: FrozenDiffusionBasePolicy) -> tuple[object, np.ndarray]:
    # Force the in-memory zarr loader regardless of the (older) checkpoint cfg. The
    # lazy store reads ~3 small NFS files per episode sequentially just to build the
    # split, then ~30 per episode during encoding; on /store each cold open costs
    # 10 ms on a good minute and >1 s during the server's stall bursts, so init was
    # taking 5-100 min. The in-memory loader fetches the whole store with 32 threads
    # (latency-bound, so ~30x faster) and the split/samples are identical.
    # Only a directory store can be copied; anything else keeps the original lazy path.
    cfg = copy.deepcopy(base_policy.cfg)
    if Path(str(cfg.dataset.source.zarr_path)).expanduser().is_dir():
        with open_dict(cfg):
            cfg.dataset.processing.in_memory = True
    dataset = build_dataset(cfg)
    train_indices = np.flatnonzero(dataset.train_mask).astype(np.int64)
    run_dir = base_policy.checkpoint_path.parent.parent
    logged = read_train_episode_indices_from_log(run_dir)
    if logged is not None:
        assert logged == train_indices.tolist(), (
            "Expert split mismatch: build_dataset selected different train episodes "
            f"than the pretraining run logged in {run_dir / '__init__.log'} "
            f"(#selected={train_indices.size}, #logged={len(logged)})."
        )
    return dataset, train_indices


def build_expert_npz(
    base_policy: FrozenDiffusionBasePolicy,
    *,
    cache_dir: Path,
    encode_batch_size: int,
    no_success_policy: str,
    aux_state_zarr: Path | None = None,
    aux_state_dim: int | None = None,
) -> Path:
    """Encode the checkpoint's train split into a dice-rl-format npz (cached).

    ``aux_state_zarr`` (asymmetric critic): the lowdim zarr of the SAME demos;
    each frame's 'states' vector is appended to the encoded features."""
    assert no_success_policy in NO_SUCCESS_POLICIES, no_success_policy
    cache_dir = Path(cache_dir).expanduser().resolve()
    aux_key = None if aux_state_zarr is None else str(aux_state_zarr)
    # Fast path: the pretraining run log records the exact train split, so the cache
    # key is computable WITHOUT opening the zarr at all. Previously every launch of an
    # already-cached checkpoint still paid the full dataset scan just to learn the key.
    # The slow path below re-derives the split from the dataset and asserts it matches
    # the log, so the key it produces is identical to this one.
    logged = read_train_episode_indices_from_log(base_policy.checkpoint_path.parent.parent)
    if logged is not None:
        key = expert_cache_key(
            base_policy, np.asarray(logged, dtype=np.int64), no_success_policy, aux_state_zarr=aux_key
        )
        cache_path = cache_dir / f"expert_{base_policy.checkpoint_path.stem}_{key}.npz"
        if cache_path.exists():
            print(f"[dice_rl expert] using cached {cache_path} (no dataset scan)", flush=True)
            return cache_path
    dataset, train_indices = _resolve_train_indices(base_policy)
    aux_group = None
    aux_names: list[str] = []
    if aux_state_zarr is not None:
        assert aux_state_dim is not None and int(aux_state_dim) > 0
        # Same reasoning as _resolve_train_indices: pull the (small) lowdim store into
        # RAM in parallel instead of ~3 sequential cold NFS reads per episode.
        aux_path = Path(str(aux_state_zarr)).expanduser()
        if aux_path.is_dir():
            aux_root = _load_zarr_into_memory(aux_path)
        else:
            import zarr as _zarr

            aux_root = _zarr.open(str(aux_state_zarr), mode="r")
        aux_group = aux_root["data"]
        aux_names = sorted(aux_group.keys(), key=lambda name: int(name.split("_")[-1]))
        assert len(aux_names) == len(dataset.episode_names), (
            f"aux state zarr has {len(aux_names)} episodes, image zarr has "
            f"{len(dataset.episode_names)}."
        )
    key = expert_cache_key(base_policy, train_indices, no_success_policy, aux_state_zarr=aux_key)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"expert_{base_policy.checkpoint_path.stem}_{key}.npz"
    if cache_path.exists():
        return cache_path

    assert dataset.stored_action_pose_repr == dataset.action_pose_repr, (
        "dice_rl expects the zarr to store actions in the checkpoint's action "
        f"representation (stored={dataset.stored_action_pose_repr}, "
        f"requested={dataset.action_pose_repr})."
    )
    hdf5_path = Path(str(base_policy.cfg.task.data.hdf5_path))
    assert hdf5_path.exists(), f"Missing MimicGen HDF5 with rewards: {hdf5_path}"
    features_all, actions_all, rewards_all, terminals_all, lengths = [], [], [], [], []
    kept_indices: list[int] = []
    num_no_success = 0
    with h5py.File(str(hdf5_path), "r") as handle:
        demo_keys = _sorted_demo_keys(handle)
        assert len(demo_keys) == len(dataset.episode_names), (
            f"HDF5 has {len(demo_keys)} demos but the zarr has "
            f"{len(dataset.episode_names)} episodes."
        )
        for episode_index in train_indices.tolist():
            episode = dataset.data_group[dataset.episode_names[episode_index]]
            demo = handle["data"][demo_keys[episode_index]]
            states = np.asarray(episode["states"], dtype=np.float32)
            actions = np.asarray(episode["actions"], dtype=np.float32)
            rgb = np.asarray(episode["rgb"])
            hdf5_rewards = np.asarray(demo["rewards"], dtype=np.float32).reshape(-1)
            assert states.shape[0] == actions.shape[0] == hdf5_rewards.shape[0], (
                f"Length mismatch for episode {episode_index}: zarr T={states.shape[0]} "
                f"vs hdf5 T={hdf5_rewards.shape[0]}."
            )
            assert rgb.shape[0] == states.shape[0], (rgb.shape, states.shape)
            assert rgb.dtype == np.uint8, rgb.dtype
            if not bool((hdf5_rewards > 0).any()):
                num_no_success += 1
            truncated = truncate_to_single_success(
                hdf5_rewards, no_success_policy=no_success_policy
            )
            if truncated is None:
                continue
            keep, rewards = truncated
            kept_indices.append(int(episode_index))
            states, actions = states[:keep], actions[:keep]
            rgb = rgb[:keep]
            feats = []
            for start in range(0, keep, encode_batch_size):
                stop = min(start + encode_batch_size, keep)
                feats.append(
                    base_policy.encode_frames(
                        torch.from_numpy(states[start:stop]),
                        torch.from_numpy(rgb[start:stop]),
                    ).cpu()
                )
            encoded = torch.cat(feats, dim=0).numpy().astype(np.float32)
            if aux_group is not None:
                aux_states = np.asarray(
                    aux_group[aux_names[episode_index]]["states"], dtype=np.float32
                )
                assert aux_states.shape == (hdf5_rewards.shape[0], int(aux_state_dim)), (
                    f"aux states {aux_states.shape} vs T={hdf5_rewards.shape[0]}, "
                    f"dim={aux_state_dim} for episode {episode_index}."
                )
                encoded = np.concatenate([encoded, aux_states[:keep]], axis=1)
            features_all.append(encoded)
            actions_all.append(
                base_policy.normalize(torch.from_numpy(actions)).numpy().astype(np.float32)
            )
            rewards_all.append(rewards)
            terminals = np.zeros(keep, dtype=bool)
            terminals[-1] = True
            terminals_all.append(terminals)
            lengths.append(keep)

    print(
        f"[dice_rl expert] {len(kept_indices)}/{train_indices.size} train demos kept "
        f"({num_no_success} without any HDF5 success reward, policy='{no_success_policy}'); "
        f"{int(np.sum(lengths))} transitions after truncation -> {cache_path}",
        flush=True,
    )
    assert kept_indices, "No expert demos left after truncation."
    # Unique temp name: several jobs may build the same cache concurrently; the
    # first rename wins and the others simply reuse the winner's file.
    tmp_path = cache_path.with_name(f"{cache_path.stem}.{os.getpid()}.tmp.npz")
    np.savez(
        tmp_path,
        states=np.concatenate(features_all, axis=0),
        actions=np.concatenate(actions_all, axis=0),
        rewards=np.concatenate(rewards_all, axis=0),
        terminals=np.concatenate(terminals_all, axis=0),
        traj_lengths=np.asarray(lengths, dtype=np.int64),
        episode_indices=np.asarray(kept_indices, dtype=np.int64),
        cache_key=np.asarray(key),
        checkpoint=np.asarray(str(base_policy.checkpoint_path)),
    )
    if cache_path.exists():
        tmp_path.unlink()
    else:
        tmp_path.rename(cache_path)
    return cache_path


def _zero_reward_terminal_record(
    features: np.ndarray, actions: np.ndarray, rewards: np.ndarray
) -> EpisodeRecord:
    """dice-rl semantics for a demo without success (no_success_policy='zero'):
    the last step is done (no bootstrap) and no reward is ever paid. Encoded as a
    truncated episode whose (unused) successor frame repeats the last frame; the
    expert buffer is built with bootstrap_on_truncation=False, so the last
    chunks are terminal with zero return exactly like dice-rl's done=True rows."""
    assert float(rewards.sum()) == 0.0
    obs = np.concatenate([features, features[-1:]], axis=0)
    return EpisodeRecord(
        obs=torch.from_numpy(obs),
        actions=torch.from_numpy(actions),
        rewards=torch.from_numpy(rewards),
        terminated=False,
        truncated=True,
    )


def load_expert_buffer(
    npz_path: Path,
    *,
    base_policy: FrozenDiffusionBasePolicy,
    n_step: int,
    gamma: float,
    device: torch.device | str,
    bootstrap_chunks: int | None = None,
    with_next1: bool = False,
) -> ChunkReplayBuffer:
    data = np.load(str(npz_path), allow_pickle=False)
    features = np.asarray(data["states"], dtype=np.float32)
    actions = np.asarray(data["actions"], dtype=np.float32)
    rewards = np.asarray(data["rewards"], dtype=np.float32)
    terminals = np.asarray(data["terminals"], dtype=bool)
    lengths = np.asarray(data["traj_lengths"], dtype=np.int64)
    assert features.shape[1] == base_policy.feature_dim, (
        f"Expert cache feature dim {features.shape[1]} != base policy {base_policy.feature_dim}; "
        "delete the stale cache."
    )
    assert actions.shape[1] == base_policy.action_dim
    assert int(lengths.sum()) == features.shape[0]
    buffer = ChunkReplayBuffer(
        capacity_frames=expert_buffer_capacity(lengths),
        feature_dim=base_policy.feature_dim,
        action_dim=base_policy.action_dim,
        horizon=base_policy.horizon,
        n_obs_steps=base_policy.n_obs_steps,
        n_step=n_step,
        gamma=gamma,
        bootstrap_on_truncation=False,
        data_source=1,
        device=device,
        bootstrap_chunks=bootstrap_chunks,
        with_next1=with_next1,
    )
    offset = 0
    for length in lengths.tolist():
        sl = slice(offset, offset + length)
        assert bool(terminals[sl][-1]) and not bool(terminals[sl][:-1].any())
        ep_rewards = rewards[sl]
        succeeded = bool(ep_rewards.sum() > 0)
        buffer.add_episode(
            EpisodeRecord(
                obs=torch.from_numpy(features[sl]),
                actions=torch.from_numpy(actions[sl]),
                rewards=torch.from_numpy(ep_rewards),
                terminated=True,
                truncated=False,
            )
            if succeeded
            else _zero_reward_terminal_record(features[sl], actions[sl], ep_rewards)
        )
        offset += length
    assert buffer.num_episodes_live == len(lengths), "expert buffer evicted episodes"
    return buffer


def expert_buffer_capacity(lengths: np.ndarray) -> int:
    return int(lengths.sum()) + int(lengths.shape[0])  # + one spare frame per demo
