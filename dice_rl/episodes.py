from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class EpisodeRecord:
    """One complete episode in feature space.

    obs:      (F, D) conditioning features per frame. ``F == T + 1`` (the
              observation after the last transition is available) or ``F == T``
              (allowed only for terminated episodes: expert demos end at the
              success frame and have no successor observation).
    actions:  (T, A) normalized action executed at each frame.
    rewards:  (T,) per-step rewards.
    terminated: the last transition is a true terminal (task success).
    truncated:  the episode was cut by the time limit at its last transition.
    """

    obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    terminated: bool
    truncated: bool

    def __post_init__(self) -> None:
        assert self.obs.ndim == 2, f"obs must be (F, D), got {tuple(self.obs.shape)}"
        assert self.actions.ndim == 2, (
            f"actions must be (T, A), got {tuple(self.actions.shape)}"
        )
        assert self.rewards.ndim == 1, (
            f"rewards must be (T,), got {tuple(self.rewards.shape)}"
        )
        num_transitions = int(self.actions.shape[0])
        num_frames = int(self.obs.shape[0])
        assert self.rewards.shape[0] == num_transitions, (
            f"rewards ({self.rewards.shape[0]}) and actions ({num_transitions}) "
            "length mismatch."
        )
        assert num_transitions >= 1, "Episode must contain at least one transition."
        assert not (self.terminated and self.truncated), (
            "An episode cannot be both terminated and truncated."
        )
        if num_frames == num_transitions:
            assert self.terminated, (
                "Episodes without a successor observation for the last transition "
                "must be terminated (expert demos)."
            )
        else:
            assert num_frames == num_transitions + 1, (
                f"obs has {num_frames} frames but there are {num_transitions} "
                "transitions; expected T or T+1 frames."
            )

    @property
    def num_transitions(self) -> int:
        return int(self.actions.shape[0])

    @property
    def num_frames(self) -> int:
        return int(self.obs.shape[0])

    @property
    def success(self) -> bool:
        return bool(self.terminated)


@dataclass
class ChunkTable:
    """Chunk transitions of one episode. Frame/transition indices are RELATIVE
    to the episode (0 = first frame)."""

    state_frames: np.ndarray  # (N, To) int64 frame indices of the state history
    action_start: np.ndarray  # (N,) int64 first transition index of the chunk
    reward: np.ndarray  # (N,) float32 n-step (chunk-discounted) return
    done: np.ndarray  # (N,) bool no bootstrap
    next_frames: np.ndarray  # (N, To) int64 frame indices of the bootstrap state
    n_eff: np.ndarray  # (N,) int64 number of chunks covered -> gamma ** n_eff
    mc_return: np.ndarray  # (N,) float32 chunk-discounted Monte Carlo return
    # (N, To) int64 frame indices of the state ONE chunk after the start (s + H,
    # clamped to the last frame) -- the critic next-state prediction target.
    # Independent of n_step/bootstrap_chunks; unused unless that knob is on.
    next1_frames: np.ndarray | None = None

    def __len__(self) -> int:
        return int(self.action_start.shape[0])


def _history_frames(frame: int, n_obs_steps: int) -> np.ndarray:
    """Frames ``[frame - To + 1, ..., frame]`` clamped at the episode start (the
    rollout pads the history by repeating the first observation)."""
    frames = np.arange(frame - n_obs_steps + 1, frame + 1, dtype=np.int64)
    return np.maximum(frames, 0)


def build_chunk_table(
    episode: EpisodeRecord,
    *,
    horizon: int,
    n_obs_steps: int,
    n_step: int,
    gamma: float,
    bootstrap_on_truncation: bool,
    bootstrap_chunks: int | None = None,
) -> ChunkTable:
    """Enumerate all chunk transitions of ``episode`` (see module docstring).

    Chunk starts are ``s in [0, T - H]`` (every chunk executes ``H`` real
    actions). Episodes shorter than ``H`` yield an empty table.

    ``bootstrap_chunks`` (default ``n_step``): the bootstrap state is
    ``s + bootstrap_chunks * H`` while reward/done/discount keep the ``n_step``
    window. ``1`` reproduces dice-rl's expert dataset (``sequence.py:374``:
    next state at ``s + H`` with ``gamma ** n_step``), a systematically
    pessimistic target.
    """
    assert horizon >= 1 and n_obs_steps >= 1 and n_step >= 1
    if bootstrap_chunks is None:
        bootstrap_chunks = n_step
    assert 1 <= bootstrap_chunks <= n_step, bootstrap_chunks
    assert 0.0 < gamma <= 1.0, f"gamma must be in (0, 1], got {gamma}."
    num_transitions = episode.num_transitions
    num_frames = episode.num_frames
    rewards = episode.rewards.detach().cpu().numpy().astype(np.float64)
    num_chunks = num_transitions - horizon + 1
    if num_chunks <= 0:
        empty_frames = np.zeros((0, n_obs_steps), dtype=np.int64)
        empty = np.zeros((0,), dtype=np.int64)
        return ChunkTable(
            state_frames=empty_frames,
            action_start=empty,
            reward=np.zeros((0,), dtype=np.float32),
            done=np.zeros((0,), dtype=bool),
            next_frames=empty_frames,
            n_eff=empty,
            mc_return=np.zeros((0,), dtype=np.float32),
            next1_frames=empty_frames,
        )

    episode_end_is_done = bool(episode.terminated) or (
        bool(episode.truncated) and not bootstrap_on_truncation
    )
    window = n_step * horizon
    state_frames = np.zeros((num_chunks, n_obs_steps), dtype=np.int64)
    next_frames = np.zeros((num_chunks, n_obs_steps), dtype=np.int64)
    next1_frames = np.zeros((num_chunks, n_obs_steps), dtype=np.int64)
    action_start = np.arange(num_chunks, dtype=np.int64)
    reward = np.zeros(num_chunks, dtype=np.float64)
    done = np.zeros(num_chunks, dtype=bool)
    n_eff = np.zeros(num_chunks, dtype=np.int64)
    mc_return = np.zeros(num_chunks, dtype=np.float64)

    # Chunk-discounted return to go, computed right-to-left over chunk boundaries
    # relative to each start: mc[s] = sum_{j>=0} gamma^{floor(j/H)} r_{s+j}. There
    # is no recursion over s (the discount grid restarts at every s), so compute
    # it directly with a per-step discount vector.
    step_discount = gamma ** (np.arange(num_transitions, dtype=np.int64) // horizon)
    for start in range(num_chunks):
        state_frames[start] = _history_frames(start, n_obs_steps)
        end = min(start + window, num_transitions)  # exclusive transition index
        segment = rewards[start:end]
        reward[start] = float(np.sum(segment * step_discount[: end - start]))
        n_eff[start] = int(-(-(end - start) // horizon))  # ceil division
        reached_episode_end = end == num_transitions
        done[start] = reached_episode_end and episode_end_is_done
        bootstrap_end = min(start + bootstrap_chunks * horizon, num_transitions)
        next_frame = min(bootstrap_end, num_frames - 1)
        next_frames[start] = _history_frames(next_frame, n_obs_steps)
        next1_frames[start] = _history_frames(min(start + horizon, num_frames - 1), n_obs_steps)
        tail = rewards[start:]
        mc_return[start] = float(np.sum(tail * step_discount[: tail.shape[0]]))

    return ChunkTable(
        state_frames=state_frames,
        action_start=action_start,
        reward=reward.astype(np.float32),
        done=done,
        next_frames=next_frames,
        n_eff=n_eff,
        mc_return=mc_return.astype(np.float32),
        next1_frames=next1_frames,
    )


def assert_single_terminal_reward(episode: EpisodeRecord) -> None:
    """Guard shared by the online and expert paths: a successful episode carries
    exactly one unit reward (at its terminal transition) and nothing else; a
    failed one carries zero reward."""
    rewards = episode.rewards.detach().cpu()
    total = float(rewards.sum())
    if episode.terminated:
        assert abs(total - 1.0) < 1e-6, (
            f"Terminated episode must have cumulative reward 1, got {total}."
        )
        assert float(rewards[-1]) == 1.0, (
            "Terminated episode must place its unit reward on the last transition."
        )
    else:
        assert total == 0.0, f"Non-terminated episode must have zero reward, got {total}."
