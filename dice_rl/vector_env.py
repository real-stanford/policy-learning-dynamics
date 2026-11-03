"""Vectorized RL env over ``dice_rl.env_worker`` processes (spawned via the repo's
``ParallelProcessEnv`` base class) plus the episode bookkeeping that turns raw
frames into ``EpisodeRecord``s in feature space."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import torch

from envs.image_parallel.spec import ImageEnvSpec
from envs.process_vector_env import ParallelProcessEnv
from envs.task_registry import get_task_entry

from dice_rl.actions import EnvActionConverter
from dice_rl.env_worker import RLEnvSpec, rl_env_worker_loop
from dice_rl.episodes import EpisodeRecord


def build_image_env_spec(cfg) -> ImageEnvSpec:
    """Same construction as the repo's evaluation (task registry + checkpoint cfg)."""
    benchmark = str(cfg.task.benchmark)
    assert benchmark.startswith("mimicgen"), f"Unsupported benchmark '{benchmark}'."
    entry = get_task_entry("mimicgen", str(cfg.task.task_name))
    assert entry.image_size is not None
    assert str(cfg.task.evaluation.kind) == "robomimic_image", cfg.task.evaluation.kind
    return ImageEnvSpec(
        dataset_path=str(cfg.task.evaluation.dataset_path),
        state_obs_keys=tuple(entry.image_lowdim_obs_keys),
        render_camera_names=tuple(entry.render_camera_names),
        render_hw=tuple(entry.image_size),
        action_pose_repr=str(cfg.action.action_pose_repr),
    )


@dataclass
class StepChunkResult:
    states: np.ndarray  # (k, S) float32
    rgb: np.ndarray  # (k, C, H, W) uint8
    rewards: np.ndarray  # (k,) float32
    terminated: bool
    truncated: bool
    aux: np.ndarray | None = None  # (k, D_aux) float32 low-dim state (asymmetric critic); None when off


class ParallelRLEnv(ParallelProcessEnv):
    def __init__(self, *, spec: RLEnvSpec, num_envs: int) -> None:
        super().__init__(worker_target=rl_env_worker_loop, spec=spec, num_envs=num_envs)

    def reset(self, worker_index: int, seed: int) -> dict[str, np.ndarray]:
        self._send(worker_index, {"command": "reset", "seed": int(seed)})
        message = self._recv(worker_index)
        obs = {
            "states": np.asarray(message["states"], dtype=np.float32),
            "rgb": np.asarray(message["rgb"], dtype=np.uint8),
        }
        if "aux" in message:
            obs["aux"] = np.asarray(message["aux"], dtype=np.float32)
        return obs

    def step_chunks(self, actions_by_worker: dict[int, np.ndarray]) -> dict[int, StepChunkResult]:
        for worker_index, actions in actions_by_worker.items():
            self._send(
                worker_index,
                {"command": "step_chunk", "actions": np.asarray(actions, dtype=np.float32)},
            )
        results = {}
        for worker_index in actions_by_worker:
            message = self._recv(worker_index)
            results[worker_index] = StepChunkResult(
                states=np.asarray(message["states"], dtype=np.float32),
                rgb=np.asarray(message["rgb"], dtype=np.uint8),
                rewards=np.asarray(message["rewards"], dtype=np.float32),
                terminated=bool(message["terminated"]),
                truncated=bool(message["truncated"]),
                aux=(
                    np.asarray(message["aux"], dtype=np.float32)
                    if "aux" in message
                    else None
                ),
            )
        return results


@dataclass
class _OngoingEpisode:
    seed: int
    features: list[torch.Tensor] = field(default_factory=list)  # per-frame (D,)
    actions: list[torch.Tensor] = field(default_factory=list)  # per-step (A,) normalized
    rewards: list[float] = field(default_factory=list)

    @property
    def num_steps(self) -> int:
        return len(self.actions)


@dataclass
class EpisodeOutcome:
    seed: int
    success: bool
    length: int
    truncated: bool


class RLEnvRunner:
    """Owns ``num_envs`` workers, keeps one ongoing episode per env in feature
    space and hands completed episodes to the caller.

    ``encode`` maps raw frames (states (N,S) float32, rgb (N,C,H,W) uint8) to
    (N, D) features (the frozen base-policy encoder). With ``aux_state_keys``
    (asymmetric critic) the worker also returns the simulator's lowdim state per
    frame, appended to the features.
    """

    def __init__(
        self,
        *,
        cfg,
        num_envs: int,
        max_episode_steps: int,
        success_steps_before_termination: int,
        seed_start: int,
        encode,
        action_converter: EnvActionConverter,
        n_obs_steps: int,
        aux_state_keys: tuple = (),
    ) -> None:
        assert num_envs >= 1
        self.num_envs = int(num_envs)
        self.n_obs_steps = int(n_obs_steps)
        self.encode = encode
        self.action_converter = action_converter
        self.aux_state_keys = tuple(aux_state_keys)
        image_spec = build_image_env_spec(cfg)
        if self.aux_state_keys:
            image_spec = replace(image_spec, aux_obs_keys=self.aux_state_keys)
        self.spec = RLEnvSpec(
            image_spec=image_spec,
            max_episode_steps=int(max_episode_steps),
            success_steps_before_termination=int(success_steps_before_termination),
            aux_state_keys=self.aux_state_keys,
        )
        self.env = ParallelRLEnv(spec=self.spec, num_envs=self.num_envs)
        self._next_seed = int(seed_start)
        self.ongoing: dict[int, _OngoingEpisode] = {}
        self.total_env_steps = 0
        self.completed: list[EpisodeOutcome] = []

    def close(self) -> None:
        self.env.close()

    def _new_seed(self) -> int:
        seed = self._next_seed
        self._next_seed += 1
        return seed

    def reset_all(self) -> None:
        for worker_index in range(self.num_envs):
            self._reset_worker(worker_index)

    def _reset_worker(self, worker_index: int) -> None:
        seed = self._new_seed()
        obs = self.env.reset(worker_index, seed)
        features = self.encode(obs["states"][None], obs["rgb"][None])[0]
        if self.aux_state_keys:
            aux = torch.as_tensor(obs["aux"], dtype=features.dtype, device=features.device)
            features = torch.cat([features, aux], dim=0)
        self.ongoing[worker_index] = _OngoingEpisode(seed=seed, features=[features])

    def current_cond(self) -> torch.Tensor:
        """(num_envs, To * D) conditioning for every env's latest observation
        (history padded by repeating the first frame, as the eval rollout does)."""
        conds = []
        for worker_index in range(self.num_envs):
            frames = self.ongoing[worker_index].features
            window = frames[-self.n_obs_steps :]
            while len(window) < self.n_obs_steps:
                window = [window[0]] + window
            conds.append(torch.stack(window, dim=0).reshape(-1))
        return torch.stack(conds, dim=0)

    def step(
        self, normalized_actions: torch.Tensor, raw_actions: torch.Tensor
    ) -> list[EpisodeRecord]:
        """Execute one chunk in every env.

        normalized_actions: (num_envs, H, A) chunk as stored in the buffer.
        raw_actions:        (num_envs, H, A) unnormalized chunk for the env.
        Returns the episodes that completed during this step.
        """
        assert normalized_actions.shape[0] == self.num_envs
        env_actions = self.action_converter.to_env(raw_actions.detach().cpu().numpy())
        results = self.env.step_chunks(
            {idx: env_actions[idx] for idx in range(self.num_envs)}
        )
        finished: list[EpisodeRecord] = []
        for worker_index in range(self.num_envs):
            result = results[worker_index]
            num_steps = int(result.rewards.shape[0])
            assert 1 <= num_steps <= normalized_actions.shape[1]
            self.total_env_steps += num_steps
            episode = self.ongoing[worker_index]
            features = self.encode(result.states, result.rgb)
            if self.aux_state_keys:
                assert result.aux is not None and result.aux.shape[0] == num_steps
                features = torch.cat(
                    [
                        features,
                        torch.as_tensor(result.aux, dtype=features.dtype, device=features.device),
                    ],
                    dim=1,
                )
            for step_index in range(num_steps):
                episode.features.append(features[step_index])
                episode.actions.append(normalized_actions[worker_index, step_index].detach())
                episode.rewards.append(float(result.rewards[step_index]))
            if result.terminated or result.truncated:
                record = EpisodeRecord(
                    obs=torch.stack(episode.features, dim=0),
                    actions=torch.stack(episode.actions, dim=0),
                    rewards=torch.tensor(episode.rewards, dtype=torch.float32),
                    terminated=bool(result.terminated),
                    truncated=bool(result.truncated),
                )
                finished.append(record)
                self.completed.append(
                    EpisodeOutcome(
                        seed=episode.seed,
                        success=bool(result.terminated),
                        length=episode.num_steps,
                        truncated=bool(result.truncated),
                    )
                )
                self._reset_worker(worker_index)
        return finished

    def pop_completed(self) -> list[EpisodeOutcome]:
        completed = self.completed
        self.completed = []
        return completed
