from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import torch

from dice_rl.episodes import (
    ChunkTable,
    EpisodeRecord,
    assert_single_terminal_reward,
    build_chunk_table,
)


@dataclass
class TransitionBatch:
    cond: torch.Tensor  # (B, To * D) flattened conditioning features
    action: torch.Tensor  # (B, H, A) normalized executed action chunk
    reward: torch.Tensor  # (B, 1) n-step chunk-discounted return
    next_cond: torch.Tensor  # (B, To * D)
    done: torch.Tensor  # (B, 1) float, 1 = no bootstrap
    n_eff: torch.Tensor  # (B, 1) long, bootstrap discount exponent
    mc_return: torch.Tensor  # (B, 1) chunk-discounted Monte Carlo return
    data_source: torch.Tensor  # (B, 1) long, 0 = online, 1 = expert
    # (B, To * D) state one chunk after ``cond`` (critic next-state prediction
    # target); None unless the buffer was built with ``with_next1=True``.
    next1_cond: torch.Tensor | None = None

    def cat(self, other: "TransitionBatch") -> "TransitionBatch":
        fields = {}
        for name in self.__dataclass_fields__:
            a, b = getattr(self, name), getattr(other, name)
            if a is None or b is None:
                assert a is None and b is None, f"{name}: mixing next1-enabled buffers"
                fields[name] = None
            else:
                fields[name] = torch.cat([a, b], dim=0)
        return TransitionBatch(**fields)

    @property
    def batch_size(self) -> int:
        return int(self.cond.shape[0])


@dataclass
class _StoredEpisode:
    frame_start: int
    num_frames: int
    chunk_start: int
    num_chunks: int
    success: bool


def _ranges_overlap(a_start: int, a_len: int, b_start: int, b_len: int) -> bool:
    return a_start < b_start + b_len and b_start < a_start + a_len


class ChunkReplayBuffer:
    def __init__(
        self,
        *,
        capacity_frames: int,
        feature_dim: int,
        action_dim: int,
        horizon: int,
        n_obs_steps: int,
        n_step: int,
        gamma: float,
        bootstrap_on_truncation: bool,
        data_source: int,
        device: torch.device | str,
        bootstrap_chunks: int | None = None,
        obs_dtype: str = "float32",
        with_next1: bool = False,
    ) -> None:
        assert capacity_frames > 0
        assert data_source in (0, 1)
        assert obs_dtype in ("float32", "float16"), obs_dtype
        self.obs_dtype = torch.float32 if obs_dtype == "float32" else torch.float16
        self.bootstrap_chunks = bootstrap_chunks
        # with_next1: also store/gather the 1-chunk-ahead state (critic
        # next-state prediction). Off -> no extra storage, batch.next1_cond=None.
        self.with_next1 = bool(with_next1)
        self.capacity = int(capacity_frames)
        self.feature_dim = int(feature_dim)
        self.action_dim = int(action_dim)
        self.horizon = int(horizon)
        self.n_obs_steps = int(n_obs_steps)
        self.n_step = int(n_step)
        self.gamma = float(gamma)
        self.bootstrap_on_truncation = bool(bootstrap_on_truncation)
        self.data_source = int(data_source)
        self.device = torch.device(device)

        cap = self.capacity
        dev = self.device
        self.obs = torch.zeros((cap, self.feature_dim), dtype=self.obs_dtype, device=dev)
        self.actions = torch.zeros(
            (cap, self.action_dim), dtype=torch.float32, device=dev
        )
        self.rewards = torch.zeros((cap,), dtype=torch.float32, device=dev)
        self.c_state_idx = torch.zeros(
            (cap, self.n_obs_steps), dtype=torch.long, device=dev
        )
        self.c_act_start = torch.zeros((cap,), dtype=torch.long, device=dev)
        self.c_reward = torch.zeros((cap,), dtype=torch.float32, device=dev)
        self.c_done = torch.zeros((cap,), dtype=torch.bool, device=dev)
        self.c_next_idx = torch.zeros(
            (cap, self.n_obs_steps), dtype=torch.long, device=dev
        )
        self.c_n_eff = torch.zeros((cap,), dtype=torch.long, device=dev)
        self.c_mc = torch.zeros((cap,), dtype=torch.float32, device=dev)
        self.c_valid = torch.zeros((cap,), dtype=torch.bool, device=dev)
        self.c_next1_idx = (
            torch.zeros((cap, self.n_obs_steps), dtype=torch.long, device=dev)
            if self.with_next1 else None
        )
        self._valid_rows: torch.Tensor | None = None
        self._chunk_arange = torch.arange(self.horizon, device=dev)

        self.episodes: deque[_StoredEpisode] = deque()
        self.frame_ptr = 0
        self.chunk_ptr = 0
        self.num_chunks_live = 0
        self.num_frames_live = 0
        self.num_episodes_added = 0
        self.num_transitions_added = 0
        self.num_successes_added = 0

    # ------------------------------------------------------------------ insert
    def add_episode(self, episode: EpisodeRecord) -> int:
        """Store one complete episode; returns the number of chunk rows created."""
        assert episode.obs.shape[1] == self.feature_dim, (
            f"feature dim {episode.obs.shape[1]} != buffer feature dim {self.feature_dim}"
        )
        assert episode.actions.shape[1] == self.action_dim
        assert_single_terminal_reward(episode)
        table = build_chunk_table(
            episode,
            horizon=self.horizon,
            n_obs_steps=self.n_obs_steps,
            n_step=self.n_step,
            gamma=self.gamma,
            bootstrap_on_truncation=self.bootstrap_on_truncation,
            bootstrap_chunks=self.bootstrap_chunks,
        )
        num_frames = episode.num_frames
        num_chunks = len(table)
        assert num_frames <= self.capacity and num_chunks <= self.capacity, (
            f"Episode with {num_frames} frames does not fit capacity {self.capacity}."
        )
        self._reserve(num_frames, num_chunks)
        frame_start = self.frame_ptr
        chunk_start = self.chunk_ptr
        self._write_frames(frame_start, episode)
        if num_chunks > 0:
            self._write_chunks(chunk_start, frame_start, table)
        self.episodes.append(
            _StoredEpisode(
                frame_start=frame_start,
                num_frames=num_frames,
                chunk_start=chunk_start,
                num_chunks=num_chunks,
                success=episode.success,
            )
        )
        self.frame_ptr = frame_start + num_frames
        self.chunk_ptr = chunk_start + num_chunks
        self.num_frames_live += num_frames
        self.num_chunks_live += num_chunks
        self.num_episodes_added += 1
        self.num_transitions_added += episode.num_transitions
        self.num_successes_added += int(episode.success)
        self._valid_rows = None
        return num_chunks

    def _reserve(self, num_frames: int, num_chunks: int) -> None:

        if self.frame_ptr + num_frames > self.capacity:
            self.frame_ptr = 0
        if self.chunk_ptr + num_chunks > self.capacity:
            self.chunk_ptr = 0
        survivors: deque[_StoredEpisode] = deque()
        for stored in self.episodes:
            frame_clash = _ranges_overlap(
                self.frame_ptr, num_frames, stored.frame_start, stored.num_frames
            )
            chunk_clash = (
                num_chunks > 0
                and stored.num_chunks > 0
                and _ranges_overlap(
                    self.chunk_ptr, num_chunks, stored.chunk_start, stored.num_chunks
                )
            )
            if frame_clash or chunk_clash:
                self._evict(stored)
            else:
                survivors.append(stored)
        self.episodes = survivors

    def _evict(self, stored: _StoredEpisode) -> None:
        if stored.num_chunks > 0:
            self.c_valid[stored.chunk_start : stored.chunk_start + stored.num_chunks] = (
                False
            )
        self.num_frames_live -= stored.num_frames
        self.num_chunks_live -= stored.num_chunks
        self._valid_rows = None

    def _write_frames(self, frame_start: int, episode: EpisodeRecord) -> None:
        num_frames = episode.num_frames
        num_transitions = episode.num_transitions
        self.obs[frame_start : frame_start + num_frames] = episode.obs.to(
            self.device, dtype=self.obs_dtype
        )
        self.actions[frame_start : frame_start + num_transitions] = episode.actions.to(
            self.device, dtype=torch.float32
        )
        self.rewards[frame_start : frame_start + num_transitions] = episode.rewards.to(
            self.device, dtype=torch.float32
        )

    def _write_chunks(self, chunk_start: int, frame_start: int, table: ChunkTable) -> None:
        rows = slice(chunk_start, chunk_start + len(table))
        dev = self.device
        self.c_state_idx[rows] = torch.from_numpy(table.state_frames + frame_start).to(dev)
        self.c_act_start[rows] = torch.from_numpy(table.action_start + frame_start).to(dev)
        self.c_reward[rows] = torch.from_numpy(table.reward).to(dev)
        self.c_done[rows] = torch.from_numpy(table.done).to(dev)
        self.c_next_idx[rows] = torch.from_numpy(table.next_frames + frame_start).to(dev)
        self.c_n_eff[rows] = torch.from_numpy(table.n_eff).to(dev)
        self.c_mc[rows] = torch.from_numpy(table.mc_return).to(dev)
        self.c_valid[rows] = True
        if self.with_next1:
            assert table.next1_frames is not None
            self.c_next1_idx[rows] = torch.from_numpy(table.next1_frames + frame_start).to(dev)

    # ------------------------------------------------------------------ sample
    def __len__(self) -> int:
        return int(self.num_chunks_live)

    @property
    def num_episodes_live(self) -> int:
        return len(self.episodes)

    def valid_rows(self) -> torch.Tensor:
        if self._valid_rows is None:
            self._valid_rows = torch.nonzero(self.c_valid, as_tuple=False).reshape(-1)
            assert int(self._valid_rows.numel()) == self.num_chunks_live, (
                f"valid-row bookkeeping mismatch: {int(self._valid_rows.numel())} vs "
                f"{self.num_chunks_live}"
            )
        return self._valid_rows

    def sample(self, batch_size: int, generator: torch.Generator | None = None) -> TransitionBatch:
        assert batch_size > 0
        rows_all = self.valid_rows()
        assert rows_all.numel() > 0, "Cannot sample from an empty buffer."
        pick = torch.randint(
            0, int(rows_all.numel()), (batch_size,), device=self.device, generator=generator
        )
        return self.gather(rows_all[pick])

    def gather(self, rows: torch.Tensor) -> TransitionBatch:
        batch_size = int(rows.shape[0])
        cond = self.obs[self.c_state_idx[rows]].reshape(batch_size, -1).to(torch.float32)
        next_cond = self.obs[self.c_next_idx[rows]].reshape(batch_size, -1).to(torch.float32)
        act_idx = self.c_act_start[rows].unsqueeze(1) + self._chunk_arange.unsqueeze(0)
        action = self.actions[act_idx]
        return TransitionBatch(
            cond=cond,
            action=action,
            reward=self.c_reward[rows].unsqueeze(1),
            next_cond=next_cond,
            done=self.c_done[rows].to(torch.float32).unsqueeze(1),
            n_eff=self.c_n_eff[rows].unsqueeze(1),
            mc_return=self.c_mc[rows].unsqueeze(1),
            data_source=torch.full(
                (batch_size, 1), self.data_source, dtype=torch.long, device=self.device
            ),
            next1_cond=(
                self.obs[self.c_next1_idx[rows]].reshape(batch_size, -1).to(torch.float32)
                if self.with_next1 else None
            ),
        )

    def stats(self) -> dict[str, float]:
        return {
            "chunks_live": float(self.num_chunks_live),
            "frames_live": float(self.num_frames_live),
            "episodes_live": float(self.num_episodes_live),
            "episodes_added": float(self.num_episodes_added),
            "transitions_added": float(self.num_transitions_added),
            "successes_added": float(self.num_successes_added),
        }


class MixedSampler:
    """RLPD symmetric sampling: ``ceil/floor(batch * ratio)`` expert rows (like
    dice-rl: ``int(batch * ratio)``) and the rest online. Falls back to online-only
    when the expert share rounds to zero, mirroring dice-rl."""

    def __init__(
        self,
        online: ChunkReplayBuffer,
        expert: ChunkReplayBuffer | None,
        generator: torch.Generator | None = None,
    ) -> None:
        self.online = online
        self.expert = expert
        self.generator = generator

    def sample(self, batch_size: int, expert_ratio: float) -> TransitionBatch:
        assert 0.0 <= expert_ratio <= 1.0, f"expert_ratio out of range: {expert_ratio}"
        if self.expert is None:
            return self.online.sample(batch_size, self.generator)
        expert_bs = int(batch_size * expert_ratio)
        online_bs = batch_size - expert_bs
        if expert_bs == 0:
            return self.online.sample(batch_size, self.generator)
        expert_batch = self.expert.sample(expert_bs, self.generator)
        if online_bs == 0:
            return expert_batch
        return expert_batch.cat(self.online.sample(online_bs, self.generator))


def adaptive_expert_ratio(
    step: int,
    *,
    enabled: bool,
    fixed_ratio: float,
    start: float,
    end: float,
    num_steps: int,
) -> float:
    if not enabled:
        return float(fixed_ratio)
    if step >= num_steps:
        return float(end)
    progress = step / float(num_steps)
    return float(start - (start - end) * progress)
