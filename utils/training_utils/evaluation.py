"""Closed-loop evaluation of an image policy in parallel MimicGen environments.

Episodes are sharded round-robin over the accelerate ranks; each rank drives
``num_envs_per_rank`` environment subprocesses. Every episode is identified by its
environment seed (``test_start_seed + index``), so the same initial conditions are
replayed for every checkpoint and every pass@k attempt.
"""

from __future__ import annotations

import json
from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from accelerate import Accelerator
from tqdm import tqdm

from envs import ImageEnvSpec, ParallelImageEnv, create_image_env, get_task_entry
from utils.common.action_pose_repr import validate_action_pose_repr, validate_rotation_rep
from utils.common.action_sequence import relative_pose_actions_to_delta
from utils.common.pytorch_util import dict_apply
from utils.common.rotation import rotation_6d_to_rotvec
from utils.training_utils.robomimic_image_wrapper import RobomimicImageWrapper


def _get_distributed_eval_context(
    accelerator: Accelerator | None,
) -> tuple[int, int, bool]:
    if accelerator is None:
        return 0, 1, False
    rank = int(accelerator.process_index)
    world_size = int(accelerator.num_processes)
    assert world_size > 0, f"num_processes must be positive, got {world_size}."
    assert 0 <= rank < world_size, f"Invalid process_index={rank} for world_size={world_size}."
    is_distributed = world_size > 1
    if is_distributed:
        assert dist.is_available(), "torch.distributed is not available."
        assert dist.is_initialized(), "torch.distributed is not initialized."
    return rank, world_size, is_distributed


def _build_distributed_episode_specs(
    *,
    num_episodes: int,
    test_start_seed: int,
    rank: int,
    world_size: int,
) -> list[dict[str, int]]:
    specs = []
    for episode_index in range(num_episodes):
        if (episode_index % world_size) == rank:
            specs.append(
                {
                    "episode_index": int(episode_index),
                    "episode_seed": int(test_start_seed + episode_index),
                }
            )
    return specs


def _gather_distributed_episode_rows(
    local_rows: list[dict], accelerator: Accelerator | None
) -> list[dict]:
    _rank, world_size, is_distributed = _get_distributed_eval_context(accelerator)
    if not is_distributed:
        return local_rows
    gathered_rows: list[list[dict] | None] = [None] * world_size
    dist.all_gather_object(gathered_rows, local_rows)
    flattened_rows = []
    for process_index, process_rows in enumerate(gathered_rows):
        assert process_rows is not None, f"Missing gathered rows for process {process_index}."
        flattened_rows.extend(process_rows)
    return flattened_rows


def _broadcast_eval_summary(summary: dict | None, accelerator: Accelerator | None) -> dict:
    _rank, _world_size, is_distributed = _get_distributed_eval_context(accelerator)
    if not is_distributed:
        assert summary is not None, "Expected a summary in single-process evaluation."
        return summary
    payload = [summary if accelerator.is_main_process else None]
    dist.broadcast_object_list(payload, src=0)
    assert payload[0] is not None, "Expected evaluation summary after broadcast."
    return payload[0]


def _build_image_obs_window(
    obs_history: list[dict[str, np.ndarray]], n_obs_steps: int
) -> dict[str, np.ndarray]:
    if len(obs_history) >= n_obs_steps:
        window = obs_history[-n_obs_steps:]
    else:
        pad = [obs_history[0]] * (n_obs_steps - len(obs_history))
        window = pad + obs_history
    return {
        "states": np.stack(
            [np.asarray(obs["states"], dtype=np.float32) for obs in window], axis=0
        ).astype(np.float32),
        "rgb": np.stack(
            [np.asarray(obs["rgb"], dtype=np.float32) for obs in window], axis=0
        ).astype(np.float32),
    }


def _undo_absolute_action(action: np.ndarray) -> np.ndarray:
    """[pos (3) | rot6d (6) | gripper (1)] -> [pos (3) | rotvec (3) | gripper (1)] per arm."""
    raw_shape = action.shape
    if raw_shape[-1] == 20:
        action = action.reshape(*raw_shape[:-1], 2, 10)

    pos = action[..., :3]
    rot6d = action[..., 3:9]
    gripper = action[..., 9:10]
    rotvec = rotation_6d_to_rotvec(rot6d)
    transformed = np.concatenate((pos, rotvec, gripper), axis=-1)

    if raw_shape[-1] == 20:
        transformed = transformed.reshape(*raw_shape[:-1], 14)
    return transformed.astype(np.float32)


def _prepare_actions_for_env_step(
    action: np.ndarray,
    *,
    action_pose_repr: str,
    rotation_rep: str,
) -> np.ndarray:
    resolved_action_pose_repr = validate_action_pose_repr(action_pose_repr)
    resolved_rotation_rep = validate_rotation_rep(rotation_rep)
    prepared = np.asarray(action, dtype=np.float32)

    if resolved_action_pose_repr == "relative":
        prepared = relative_pose_actions_to_delta(prepared, rotation_rep=resolved_rotation_rep)
        resolved_action_pose_repr = "delta"

    if resolved_rotation_rep == "rotation_6d":
        prepared = _undo_absolute_action(prepared)

    if resolved_action_pose_repr in {"delta", "absolute"}:
        return prepared.astype(np.float32)
    raise ValueError(
        f"Unsupported action_pose_repr '{resolved_action_pose_repr}' after preparation."
    )


@dataclass
class _ActiveImageEpisode:
    spec: dict[str, int]
    obs_history: list[dict[str, np.ndarray]]
    total_reward: float = 0.0
    max_reward: float = field(default_factory=lambda: float("-inf"))
    steps: int = 0


def build_image_env_spec(cfg) -> ImageEnvSpec:
    assert str(cfg.task.benchmark).startswith("mimicgen"), (
        f"Unsupported benchmark '{cfg.task.benchmark}' for evaluation."
    )
    entry = get_task_entry("mimicgen", str(cfg.task.task_name))
    assert entry.image_size is not None, (
        f"Task '{cfg.task.task_name}' is missing image_size metadata."
    )
    return ImageEnvSpec(
        dataset_path=str(cfg.task.evaluation.dataset_path),
        state_obs_keys=tuple(entry.image_lowdim_obs_keys),
        render_camera_names=tuple(entry.render_camera_names),
        render_hw=tuple(entry.image_size),
        action_pose_repr=validate_action_pose_repr(str(cfg.action.action_pose_repr)),
    )


def _episode_row_from_slot(slot: _ActiveImageEpisode) -> dict[str, float | int]:
    return {
        "episode_index": int(slot.spec["episode_index"]),
        "seed": int(slot.spec["episode_seed"]),
        "success": float(slot.max_reward >= 1.0),
        "max_reward": float(slot.max_reward),
        "total_reward": float(slot.total_reward),
        "steps": int(slot.steps),
    }


def _run_single_env_image_episodes(
    cfg,
    policy,
    *,
    env: RobomimicImageWrapper,
    episode_specs: list[dict[str, int]],
    max_steps: int,
    device: torch.device,
    on_episode_complete=None,
) -> list[dict]:
    """Sequential rollouts in one in-process environment (num_envs_per_rank == 1)."""
    n_obs_steps = int(cfg.model.rollout.n_obs_steps)
    n_latency_steps = int(cfg.model.rollout.n_latency_steps)
    action_pose_repr = validate_action_pose_repr(str(cfg.action.action_pose_repr))
    rotation_rep = validate_rotation_rep(str(cfg.action.rotation_rep))
    rows = []
    for spec in episode_specs:
        env.seed(int(spec["episode_seed"]))
        first_obs = env.reset()
        slot = _ActiveImageEpisode(
            spec=spec,
            obs_history=[
                {
                    "states": np.asarray(first_obs["states"], dtype=np.float32),
                    "rgb": np.asarray(first_obs["rgb"], dtype=np.float32),
                }
            ],
        )
        policy.reset()
        done = False
        while not done and slot.steps < max_steps:
            obs_window = _build_image_obs_window(slot.obs_history, n_obs_steps)
            obs_dict = {
                key: torch.from_numpy(value[None]).to(device) for key, value in obs_window.items()
            }
            with torch.no_grad():
                action_dict = policy.predict_action(obs_dict)
            action = dict_apply(action_dict, lambda x: x.detach().cpu().numpy())["action"][0]
            action = _prepare_actions_for_env_step(
                action[n_latency_steps:],
                action_pose_repr=action_pose_repr,
                rotation_rep=rotation_rep,
            )
            for action_step in action:
                next_obs, reward, done, _info = env.step(action_step)
                slot.obs_history.append(
                    {
                        "states": np.asarray(next_obs["states"], dtype=np.float32),
                        "rgb": np.asarray(next_obs["rgb"], dtype=np.float32),
                    }
                )
                slot.total_reward += float(reward)
                slot.max_reward = max(slot.max_reward, float(reward))
                slot.steps += 1
                if done or slot.steps >= max_steps:
                    break
        rows.append(_episode_row_from_slot(slot))
        if on_episode_complete is not None:
            on_episode_complete(slot)
    return rows


def _run_parallel_image_episodes(
    cfg,
    policy,
    *,
    episode_specs: list[dict[str, int]],
    num_envs_per_rank: int,
    max_steps: int,
    device: torch.device,
    on_episode_complete=None,
) -> list[dict]:
    assert num_envs_per_rank > 1, (
        f"Parallel evaluation requires num_envs_per_rank > 1, got {num_envs_per_rank}."
    )
    if not episode_specs:
        return []

    env_spec = build_image_env_spec(cfg)
    n_obs_steps = int(cfg.model.rollout.n_obs_steps)
    n_latency_steps = int(cfg.model.rollout.n_latency_steps)
    action_pose_repr = validate_action_pose_repr(str(cfg.action.action_pose_repr))
    rotation_rep = validate_rotation_rep(str(cfg.action.rotation_rep))
    active_slots: dict[int, _ActiveImageEpisode] = {}
    idle_workers = set(range(min(num_envs_per_rank, len(episode_specs))))
    pending_specs = deque(episode_specs)
    local_episode_rows = []

    def start_episode(env_pool: ParallelImageEnv, worker_index: int) -> None:
        assert pending_specs, "Expected a pending episode when starting a worker."
        spec = pending_specs.popleft()
        first_obs = env_pool.reset(worker_index, int(spec["episode_seed"]))
        active_slots[worker_index] = _ActiveImageEpisode(
            spec=spec,
            obs_history=[
                {
                    "states": np.asarray(first_obs["states"], dtype=np.float32),
                    "rgb": np.asarray(first_obs["rgb"], dtype=np.float32),
                }
            ],
        )

    with ParallelImageEnv(
        spec=env_spec,
        num_envs=min(num_envs_per_rank, len(episode_specs)),
    ) as env_pool:
        while pending_specs and idle_workers:
            start_episode(env_pool, idle_workers.pop())

        while active_slots:
            worker_indices = sorted(active_slots)
            obs_window = [
                _build_image_obs_window(active_slots[worker_index].obs_history, n_obs_steps)
                for worker_index in worker_indices
            ]
            obs_dict = {
                "states": torch.from_numpy(
                    np.stack([window["states"] for window in obs_window], axis=0)
                ).to(device),
                "rgb": torch.from_numpy(
                    np.stack([window["rgb"] for window in obs_window], axis=0)
                ).to(device),
            }
            with torch.no_grad():
                action_dict = policy.predict_action(obs_dict)
            action_batch = dict_apply(action_dict, lambda t: t.detach().cpu().numpy())["action"]
            action_batch = action_batch[:, n_latency_steps:]
            assert action_batch.shape[1] > 0, "Expected at least one action per rollout."
            action_batch = _prepare_actions_for_env_step(
                action_batch,
                action_pose_repr=action_pose_repr,
                rotation_rep=rotation_rep,
            )

            completed_workers = set()
            for action_step_index in range(action_batch.shape[1]):
                step_actions = {}
                for batch_index, worker_index in enumerate(worker_indices):
                    if worker_index in completed_workers:
                        continue
                    step_actions[worker_index] = action_batch[batch_index, action_step_index]
                if not step_actions:
                    break
                step_results = env_pool.step_many(step_actions)
                for worker_index, step_result in step_results.items():
                    slot = active_slots[worker_index]
                    slot.total_reward += float(step_result.reward)
                    slot.max_reward = max(slot.max_reward, float(step_result.reward))
                    slot.obs_history.append(
                        {
                            "states": np.asarray(step_result.observation["states"], dtype=np.float32),
                            "rgb": np.asarray(step_result.observation["rgb"], dtype=np.float32),
                        }
                    )
                    slot.steps += 1
                    if step_result.done or slot.steps >= max_steps:
                        local_episode_rows.append(_episode_row_from_slot(slot))
                        if on_episode_complete is not None:
                            on_episode_complete(slot)
                        completed_workers.add(worker_index)

                for worker_index in completed_workers:
                    if worker_index in active_slots:
                        del active_slots[worker_index]
                        idle_workers.add(worker_index)

            while pending_specs and idle_workers:
                start_episode(env_pool, idle_workers.pop())

    return local_episode_rows


def _summarize_eval(
    *,
    checkpoint_path: str | Path | None,
    policy_state_key: str | None,
    num_episodes: int,
    episode_rows: list[dict],
) -> dict:
    ordered_rows = sorted(episode_rows, key=lambda row: int(row["episode_index"]))
    assert len(ordered_rows) == num_episodes, (
        f"Expected {num_episodes} evaluation rows, got {len(ordered_rows)}."
    )
    successes = np.asarray([row["success"] for row in ordered_rows], dtype=np.float64)
    steps = np.asarray([row["steps"] for row in ordered_rows], dtype=np.float64)
    summary = {
        "num_episodes": int(num_episodes),
        "success_rate": float(np.mean(successes)),
        "avg_steps": float(np.mean(steps)),
        "episodes": ordered_rows,
    }
    if checkpoint_path is not None:
        summary["checkpoint_path"] = str(Path(checkpoint_path).expanduser().resolve())
    if policy_state_key is not None:
        summary["policy_state_key"] = str(policy_state_key)
    return summary


def run_image_policy_eval(
    cfg,
    policy,
    output_dir: str | Path,
    *,
    num_episodes: int,
    test_start_seed: int,
    max_steps: int,
    num_envs_per_rank: int,
    checkpoint_path: str | Path | None = None,
    policy_state_key: str | None = None,
    accelerator: Accelerator | None = None,
) -> dict:
    """Roll out ``policy`` on environment seeds ``test_start_seed .. +num_episodes-1``.

    Writes ``summary.json`` (success_rate, avg_steps, per-episode rows) to ``output_dir``
    on the main process and returns the summary on every rank.
    """
    assert str(cfg.task.evaluation.kind) == "robomimic_image", (
        f"Unsupported image eval kind '{cfg.task.evaluation.kind}'."
    )
    assert num_episodes > 0, f"num_episodes must be positive, got {num_episodes}."
    assert max_steps > 0, f"max_steps must be positive, got {max_steps}."
    assert num_envs_per_rank > 0, f"num_envs_per_rank must be positive, got {num_envs_per_rank}."

    output_path = Path(output_dir).expanduser().resolve()
    device = next(policy.parameters()).device
    policy.eval()
    rank, world_size, _is_distributed = _get_distributed_eval_context(accelerator)
    episode_specs = _build_distributed_episode_specs(
        num_episodes=int(num_episodes),
        test_start_seed=int(test_start_seed),
        rank=rank,
        world_size=world_size,
    )

    progress_context = nullcontext(None)
    if accelerator is None or accelerator.is_main_process:
        progress_desc = "Eval rollouts" if world_size == 1 else f"Eval rollouts rank {rank}"
        progress_context = tqdm(total=len(episode_specs), desc=progress_desc, dynamic_ncols=True)

    with progress_context as episode_progress:
        on_complete = (
            (lambda _slot: episode_progress.update(1)) if episode_progress is not None else None
        )
        if int(num_envs_per_rank) == 1:
            local_episode_rows = []
            if episode_specs:
                env = create_image_env(build_image_env_spec(cfg))
                local_episode_rows = _run_single_env_image_episodes(
                    cfg,
                    policy,
                    env=env,
                    episode_specs=episode_specs,
                    max_steps=int(max_steps),
                    device=device,
                    on_episode_complete=on_complete,
                )
                env.close()
        else:
            local_episode_rows = _run_parallel_image_episodes(
                cfg,
                policy,
                episode_specs=episode_specs,
                num_envs_per_rank=int(num_envs_per_rank),
                max_steps=int(max_steps),
                device=device,
                on_episode_complete=on_complete,
            )

    gathered_episode_rows = _gather_distributed_episode_rows(local_episode_rows, accelerator)
    summary = None
    if accelerator is None or accelerator.is_main_process:
        summary = _summarize_eval(
            checkpoint_path=checkpoint_path,
            policy_state_key=policy_state_key,
            num_episodes=int(num_episodes),
            episode_rows=gathered_episode_rows,
        )
        output_path.mkdir(parents=True, exist_ok=True)
        with open(output_path / "summary.json", "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
    return _broadcast_eval_summary(summary, accelerator)
