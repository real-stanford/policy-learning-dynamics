"""Brute-force reference tests for the chunk transition semantics and the ring
buffer bookkeeping (dice_rl.episodes / dice_rl.replay_buffer)."""

from __future__ import annotations

import math
import unittest

import numpy as np
import torch

from dice_rl.episodes import EpisodeRecord, build_chunk_table
from dice_rl.replay_buffer import (
    ChunkReplayBuffer,
    MixedSampler,
    adaptive_expert_ratio,
)


def _make_episode(
    rng: np.random.Generator,
    *,
    num_transitions: int,
    feature_dim: int,
    action_dim: int,
    terminated: bool,
    truncated: bool,
    with_final_obs: bool,
    tag: float,
) -> EpisodeRecord:
    num_frames = num_transitions + (1 if with_final_obs else 0)
    # Feature column 0 = frame index, column 1 = episode tag: lets the tests
    # verify that gathered states point at the right frame of the right episode.
    obs = rng.standard_normal((num_frames, feature_dim)).astype(np.float32)
    obs[:, 0] = np.arange(num_frames)
    obs[:, 1] = tag
    actions = rng.standard_normal((num_transitions, action_dim)).astype(np.float32)
    actions[:, 0] = np.arange(num_transitions)
    actions[:, 1] = tag
    rewards = np.zeros(num_transitions, dtype=np.float32)
    if terminated:
        rewards[-1] = 1.0
    return EpisodeRecord(
        obs=torch.from_numpy(obs),
        actions=torch.from_numpy(actions),
        rewards=torch.from_numpy(rewards),
        terminated=terminated,
        truncated=truncated,
    )


def _reference_chunks(
    episode: EpisodeRecord,
    *,
    horizon: int,
    n_obs_steps: int,
    n_step: int,
    gamma: float,
    bootstrap_on_truncation: bool,
) -> list[dict]:
    """Independent implementation written as 'n consecutive non-overlapping
    chunks' (the dice-rl description), used to cross-check build_chunk_table."""
    T = episode.num_transitions
    F = episode.num_frames
    r = episode.rewards.numpy().astype(np.float64)
    end_done = episode.terminated or (episode.truncated and not bootstrap_on_truncation)
    out = []
    for s in range(0, T - horizon + 1):
        reward = 0.0
        covered = 0
        ended = False
        for k in range(n_step):
            cs = s + k * horizon
            if cs >= T:
                break
            ce = min(cs + horizon, T)
            reward += (gamma**k) * float(r[cs:ce].sum())
            covered += 1
            if ce == T:
                ended = True
                break
        next_frame = min(s + covered * horizon, F - 1) if not ended else min(T, F - 1)
        if ended:
            next_frame = min(T, F - 1)
        mc = 0.0
        k = 0
        t = s
        while t < T:
            mc += (gamma**k) * float(r[t : min(t + horizon, T)].sum())
            t += horizon
            k += 1
        out.append(
            {
                "state_frames": [max(f, 0) for f in range(s - n_obs_steps + 1, s + 1)],
                "action_start": s,
                "reward": reward,
                "done": ended and end_done,
                "next_frames": [
                    max(f, 0) for f in range(next_frame - n_obs_steps + 1, next_frame + 1)
                ],
                "n_eff": covered,
                "mc_return": mc,
            }
        )
    return out


class TestChunkTable(unittest.TestCase):
    def test_matches_reference_over_random_episodes(self) -> None:
        rng = np.random.default_rng(0)
        cases = 0
        for _ in range(300):
            horizon = int(rng.integers(1, 6))
            n_obs = int(rng.integers(1, 4))
            n_step = int(rng.integers(1, 5))
            gamma = float(rng.choice([1.0, 0.99, 0.9, 0.5]))
            T = int(rng.integers(1, 30))
            kind = rng.integers(0, 4)
            terminated = kind == 0 or kind == 3
            truncated = kind == 1
            with_final = not (kind == 3)  # expert-style demo without final obs
            boot = bool(rng.integers(0, 2))
            ep = _make_episode(
                rng,
                num_transitions=T,
                feature_dim=4,
                action_dim=3,
                terminated=terminated,
                truncated=truncated,
                with_final_obs=with_final,
                tag=1.0,
            )
            table = build_chunk_table(
                ep,
                horizon=horizon,
                n_obs_steps=n_obs,
                n_step=n_step,
                gamma=gamma,
                bootstrap_on_truncation=boot,
            )
            ref = _reference_chunks(
                ep,
                horizon=horizon,
                n_obs_steps=n_obs,
                n_step=n_step,
                gamma=gamma,
                bootstrap_on_truncation=boot,
            )
            self.assertEqual(len(table), len(ref))
            for i, row in enumerate(ref):
                cases += 1
                self.assertEqual(table.state_frames[i].tolist(), row["state_frames"])
                self.assertEqual(int(table.action_start[i]), row["action_start"])
                self.assertTrue(math.isclose(float(table.reward[i]), row["reward"], abs_tol=1e-5))
                self.assertEqual(bool(table.done[i]), row["done"])
                self.assertEqual(table.next_frames[i].tolist(), row["next_frames"])
                self.assertEqual(int(table.n_eff[i]), row["n_eff"])
                self.assertTrue(
                    math.isclose(float(table.mc_return[i]), row["mc_return"], abs_tol=1e-5)
                )
        self.assertGreater(cases, 1000)

    def test_sparse_terminal_reward_semantics(self) -> None:
        # T=10, H=4, n=2, gamma=0.5: success reward at transition 9.
        rng = np.random.default_rng(1)
        ep = _make_episode(
            rng,
            num_transitions=10,
            feature_dim=3,
            action_dim=2,
            terminated=True,
            truncated=False,
            with_final_obs=True,
            tag=0.0,
        )
        table = build_chunk_table(
            ep, horizon=4, n_obs_steps=1, n_step=2, gamma=0.5, bootstrap_on_truncation=False
        )
        # starts 0..6; window covers [s, s+8): reward reaches transition 9 for s>=2
        expected_reward = [0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 1.0]
        # s=6: window [6,14) clipped to [6,10): chunk0 = 6..9 -> reward 1 undiscounted
        expected_done = [False, False, True, True, True, True, True]
        expected_next = [8, 9, 10, 10, 10, 10, 10]
        expected_neff = [2, 2, 2, 2, 2, 2, 1]
        self.assertEqual(table.reward.tolist(), expected_reward)
        self.assertEqual(table.done.tolist(), expected_done)
        self.assertEqual(table.next_frames[:, 0].tolist(), expected_next)
        self.assertEqual(table.n_eff.tolist(), expected_neff)
        # every chunk's MC return is 0.5**(#chunks until the terminal chunk)
        self.assertEqual(table.mc_return.tolist(), [0.25, 0.25, 0.5, 0.5, 0.5, 0.5, 1.0])

    def test_truncation_bootstrap_knob(self) -> None:
        rng = np.random.default_rng(2)
        ep = _make_episode(
            rng,
            num_transitions=6,
            feature_dim=3,
            action_dim=2,
            terminated=False,
            truncated=True,
            with_final_obs=True,
            tag=0.0,
        )
        default = build_chunk_table(
            ep, horizon=2, n_obs_steps=1, n_step=1, gamma=0.9, bootstrap_on_truncation=False
        )
        self.assertEqual(default.done.tolist(), [False, False, False, False, True])
        boot = build_chunk_table(
            ep, horizon=2, n_obs_steps=1, n_step=1, gamma=0.9, bootstrap_on_truncation=True
        )
        self.assertEqual(boot.done.tolist(), [False] * 5)
        self.assertEqual(boot.next_frames[-1, 0], 6)  # bootstraps from the final obs

    def test_rejects_multi_reward_episode(self) -> None:
        rng = np.random.default_rng(3)
        ep = _make_episode(
            rng,
            num_transitions=5,
            feature_dim=3,
            action_dim=2,
            terminated=True,
            truncated=False,
            with_final_obs=True,
            tag=0.0,
        )
        ep.rewards[0] = 1.0
        buffer = ChunkReplayBuffer(
            capacity_frames=100,
            feature_dim=3,
            action_dim=2,
            horizon=2,
            n_obs_steps=1,
            n_step=1,
            gamma=0.99,
            bootstrap_on_truncation=False,
            data_source=0,
            device="cpu",
        )
        with self.assertRaises(AssertionError):
            buffer.add_episode(ep)


class TestRingBuffer(unittest.TestCase):
    def _fill_and_check(self, capacity: int, num_episodes: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        horizon, n_obs, n_step = 3, 2, 2
        buffer = ChunkReplayBuffer(
            capacity_frames=capacity,
            feature_dim=4,
            action_dim=3,
            horizon=horizon,
            n_obs_steps=n_obs,
            n_step=n_step,
            gamma=0.9,
            bootstrap_on_truncation=False,
            data_source=0,
            device="cpu",
        )
        episodes: dict[float, EpisodeRecord] = {}
        for i in range(num_episodes):
            T = int(rng.integers(1, capacity // 2))
            terminated = bool(rng.integers(0, 2))
            ep = _make_episode(
                rng,
                num_transitions=T,
                feature_dim=4,
                action_dim=3,
                terminated=terminated,
                truncated=not terminated,
                with_final_obs=True,
                tag=float(i),
            )
            episodes[float(i)] = ep
            buffer.add_episode(ep)
            self.assertLessEqual(buffer.num_frames_live, capacity)
            self.assertEqual(
                buffer.num_chunks_live,
                sum(e.num_chunks for e in buffer.episodes),
            )
            # live episodes are the most recent ones, in insertion order
            tags = [
                float(buffer.obs[e.frame_start, 1]) for e in buffer.episodes
            ]
            self.assertEqual(tags, sorted(tags))
            self.assertEqual(tags[-1], float(i))
        # Every valid chunk row must reproduce the original episode exactly.
        rows = buffer.valid_rows()
        batch = buffer.gather(rows)
        for b in range(batch.batch_size):
            cond = batch.cond[b].reshape(n_obs, 4)
            tag = float(cond[-1, 1])
            ep = episodes[tag]
            s = int(cond[-1, 0])
            table = build_chunk_table(
                ep,
                horizon=horizon,
                n_obs_steps=n_obs,
                n_step=n_step,
                gamma=0.9,
                bootstrap_on_truncation=False,
            )
            self.assertEqual(cond[:, 0].tolist(), table.state_frames[s].tolist())
            self.assertTrue(torch.equal(cond, ep.obs[table.state_frames[s]]))
            self.assertTrue(
                torch.equal(batch.action[b], ep.actions[s : s + horizon])
            )
            self.assertTrue(
                torch.equal(
                    batch.next_cond[b].reshape(n_obs, 4), ep.obs[table.next_frames[s]]
                )
            )
            self.assertAlmostEqual(float(batch.reward[b]), float(table.reward[s]), places=6)
            self.assertEqual(float(batch.done[b]), float(table.done[s]))
            self.assertEqual(int(batch.n_eff[b]), int(table.n_eff[s]))
            self.assertAlmostEqual(float(batch.mc_return[b]), float(table.mc_return[s]), places=6)
        # No live chunk references a frame outside its own episode's block.
        for e in buffer.episodes:
            if e.num_chunks == 0:
                continue
            idx = buffer.c_state_idx[e.chunk_start : e.chunk_start + e.num_chunks]
            nxt = buffer.c_next_idx[e.chunk_start : e.chunk_start + e.num_chunks]
            for t in (idx, nxt):
                self.assertTrue(bool((t >= e.frame_start).all()))
                self.assertTrue(bool((t < e.frame_start + e.num_frames).all()))

    def test_ring_eviction_keeps_only_consistent_episodes(self) -> None:
        for seed in range(5):
            self._fill_and_check(capacity=64, num_episodes=200, seed=seed)

    def test_sample_and_mixed_sampler(self) -> None:
        rng = np.random.default_rng(7)
        kwargs = dict(
            feature_dim=4,
            action_dim=3,
            horizon=2,
            n_obs_steps=1,
            n_step=1,
            gamma=0.99,
            bootstrap_on_truncation=False,
            device="cpu",
        )
        online = ChunkReplayBuffer(capacity_frames=200, data_source=0, **kwargs)
        expert = ChunkReplayBuffer(capacity_frames=200, data_source=1, **kwargs)
        for i in range(5):
            online.add_episode(
                _make_episode(rng, num_transitions=10, feature_dim=4, action_dim=3,
                              terminated=True, truncated=False, with_final_obs=True, tag=i)
            )
            expert.add_episode(
                _make_episode(rng, num_transitions=8, feature_dim=4, action_dim=3,
                              terminated=True, truncated=False, with_final_obs=False, tag=100 + i)
            )
        sampler = MixedSampler(online, expert, generator=torch.Generator().manual_seed(0))
        batch = sampler.sample(10, expert_ratio=0.3)
        self.assertEqual(batch.batch_size, 10)
        self.assertEqual(int(batch.data_source.sum()), 3)
        self.assertTrue(bool((batch.cond[:3, 1] >= 100).all()))
        self.assertTrue(bool((batch.cond[3:, 1] < 100).all()))
        only_online = sampler.sample(10, expert_ratio=0.05)  # rounds to 0 expert rows
        self.assertEqual(int(only_online.data_source.sum()), 0)
        self.assertEqual(sampler.sample(4, expert_ratio=1.0).data_source.sum().item(), 4)

    def test_adaptive_expert_ratio(self) -> None:
        kw = dict(enabled=True, fixed_ratio=0.3, start=0.5, end=0.1, num_steps=100)
        self.assertAlmostEqual(adaptive_expert_ratio(0, **kw), 0.5)
        self.assertAlmostEqual(adaptive_expert_ratio(50, **kw), 0.3)
        self.assertAlmostEqual(adaptive_expert_ratio(100, **kw), 0.1)
        self.assertAlmostEqual(adaptive_expert_ratio(10_000, **kw), 0.1)
        self.assertAlmostEqual(adaptive_expert_ratio(0, **{**kw, "enabled": False}), 0.3)


if __name__ == "__main__":
    unittest.main()
