"""Parity of dice_rl's transition tables with dice-rl's OWN index code.

The two reference functions below are transcriptions of
/store/real/zhanyis/dice-rl-clean/util/hybrid_replay_buffer.py::_sample_standard
(online n-step over consecutive chunk rows) and
/store/real/zhanyis/dice-rl-clean/agent/dataset/sequence.py::
StitchedSequenceQLearningDataset.__getitem__ (expert n-step), kept as literal
as possible. For sparse terminal rewards (the only reward structure DICE-RL
uses) both must agree with ``build_chunk_table`` on reward, done, the
bootstrap discount exponent and -- except for the known dice-rl expert
``s+H`` bootstrap bug -- the bootstrap state.
"""

from __future__ import annotations

import unittest

import numpy as np
import torch

from dice_rl.episodes import EpisodeRecord, build_chunk_table


def _episode(T: int, terminated: bool, tag: float) -> EpisodeRecord:
    obs = np.zeros((T + 1, 2), dtype=np.float32)
    obs[:, 0] = np.arange(T + 1)
    obs[:, 1] = tag
    actions = np.zeros((T, 1), dtype=np.float32)
    rewards = np.zeros(T, dtype=np.float32)
    if terminated:
        rewards[-1] = 1.0
    return EpisodeRecord(
        obs=torch.from_numpy(obs),
        actions=torch.from_numpy(actions),
        rewards=torch.from_numpy(rewards),
        terminated=terminated,
        truncated=not terminated,
    )


def dice_rl_online_chunks(episodes: list[EpisodeRecord], H: int) -> dict:
    """HybridReplayBuffer._process_complete_episode: one row per chunk start
    s in [0, T-H]; reward = sum r[s:s+H]; done = any(dones[s:s+H]); next = o_{s+H}.
    Rows are stored in insertion order (episode after episode)."""
    rows = {"reward": [], "done": [], "next": [], "ep": [], "s": []}
    for ep_idx, ep in enumerate(episodes):
        T = ep.num_transitions
        r = ep.rewards.numpy()
        dones = np.zeros(T, dtype=bool)
        dones[-1] = True  # terminated or truncated: MultiStep marks done
        for s in range(0, T - H + 1):
            rows["reward"].append(float(r[s : s + H].sum()))
            rows["done"].append(bool(dones[s : s + H].any()))
            rows["next"].append((ep_idx, s + H))
            rows["ep"].append(ep_idx)
            rows["s"].append(s)
    return rows


def dice_rl_online_nstep(rows: dict, idx: int, n_step: int, H: int, gamma: float):
    """HybridReplayBuffer._sample_standard n-step branch for one buffer row."""
    size = len(rows["reward"])
    n_step_reward = 0.0
    n_step_done = False
    max_distance = (n_step - 1) * H + 1
    for k in range(max_distance):
        j = idx + k
        if j >= size or n_step_done:
            break
        discount_power = (k + H - 1) // H
        n_step_reward += (gamma**discount_power) * float(rows["reward"][j])
        n_step_done = n_step_done or bool(rows["done"][j])
    next_row = idx + (n_step - 1) * H
    next_state = rows["next"][next_row] if next_row < size else None
    return n_step_reward, n_step_done, next_state


def dice_rl_expert_item(rewards_flat, dones_flat, start: int, n_step: int, H: int, gamma: float):
    """StitchedSequenceQLearningDataset.__getitem__ (use_n_step branch)."""
    n_step_rewards = 0.0
    n_step_dones = False
    for step in range(n_step):
        chunk_start = start + step * H
        chunk_end = chunk_start + H
        if chunk_end > len(rewards_flat):
            break
        horizon_rewards = rewards_flat[chunk_start:chunk_end]
        horizon_dones = dones_flat[chunk_start:chunk_end]
        chunk_reward = 0.0
        for t in range(H):
            if t < len(horizon_rewards):
                chunk_reward += float(horizon_rewards[t])
                if horizon_dones[t] and t < H - 1:
                    break
        n_step_rewards += (gamma**step) * chunk_reward * (not n_step_dones)
        n_step_dones = n_step_dones or bool(horizon_dones.any())
    return n_step_rewards, n_step_dones


class TestExpertTruncation(unittest.TestCase):
    def test_matches_dice_rl_truncate(self) -> None:
        from dice_rl.expert_data import truncate_to_single_success

        # process_robomimic_dataset.py --truncate: cumsum >= 1 -> keep idx+1 steps
        rewards = np.array([0, 0, 0, 1, 1, 1, 1], dtype=np.float32)
        cumsum = np.cumsum(rewards)
        ref_keep = int(np.where(cumsum >= 1)[0][0]) + 1
        keep, out = truncate_to_single_success(rewards)
        self.assertEqual(keep, ref_keep)
        np.testing.assert_array_equal(out, rewards[:ref_keep])
        # no-success demo policies
        zeros = np.zeros(5, dtype=np.float32)
        self.assertEqual(truncate_to_single_success(zeros, no_success_policy="force_last")[1].tolist(), [0, 0, 0, 0, 1])
        self.assertEqual(truncate_to_single_success(zeros, no_success_policy="zero")[1].tolist(), [0, 0, 0, 0, 0])
        self.assertIsNone(truncate_to_single_success(zeros, no_success_policy="drop"))

    def test_zero_reward_terminal_record_is_terminal_without_bootstrap(self) -> None:
        from dice_rl.expert_data import _zero_reward_terminal_record

        rec = _zero_reward_terminal_record(
            np.zeros((6, 3), np.float32), np.zeros((6, 2), np.float32), np.zeros(6, np.float32)
        )
        table = build_chunk_table(
            rec, horizon=2, n_obs_steps=1, n_step=2, gamma=0.9, bootstrap_on_truncation=False
        )
        self.assertEqual(table.done.tolist(), [False, False, True, True, True])
        self.assertEqual(table.reward.tolist(), [0.0] * 5)


class TestDiceRLParity(unittest.TestCase):
    H = 8
    n_step = 3
    gamma = 0.99

    def test_online_nstep_matches_hybrid_replay_buffer(self) -> None:
        rng = np.random.default_rng(0)
        episodes = [
            _episode(int(rng.integers(8, 60)), terminated=bool(rng.integers(0, 2)), tag=i)
            for i in range(40)
        ]
        rows = dice_rl_online_chunks(episodes, self.H)
        size = len(rows["reward"])
        safe_size = size - (self.n_step - 1) * self.H  # dice-rl never samples the tail
        tables = [
            build_chunk_table(
                ep,
                horizon=self.H,
                n_obs_steps=1,
                n_step=self.n_step,
                gamma=self.gamma,
                bootstrap_on_truncation=False,
            )
            for ep in episodes
        ]
        checked = 0
        for idx in range(safe_size):
            ref_reward, ref_done, ref_next = dice_rl_online_nstep(
                rows, idx, self.n_step, self.H, self.gamma
            )
            ep_idx, s = int(rows["ep"][idx]), int(rows["s"][idx])
            table = tables[ep_idx]
            self.assertAlmostEqual(float(table.reward[s]), ref_reward, places=6)
            self.assertEqual(bool(table.done[s]), ref_done)
            if not ref_done:
                # bootstrap state and exponent only matter when bootstrapping
                self.assertEqual(ref_next, (ep_idx, s + self.n_step * self.H))
                self.assertEqual(int(table.next_frames[s, 0]), s + self.n_step * self.H)
                self.assertEqual(int(table.n_eff[s]), self.n_step)
            checked += 1
        self.assertGreater(checked, 500)

    def test_expert_nstep_matches_stitched_dataset(self) -> None:
        rng = np.random.default_rng(1)
        lengths = [int(rng.integers(8, 80)) for _ in range(30)]
        rewards_flat = np.concatenate(
            [np.concatenate([np.zeros(T - 1), [1.0]]) for T in lengths]
        ).astype(np.float32)
        dones_flat = np.zeros_like(rewards_flat, dtype=bool)
        dones_flat[np.cumsum(lengths) - 1] = True  # dataset marks last step done
        offset = 0
        checked = 0
        for T in lengths:
            ep = EpisodeRecord(
                obs=torch.zeros(T, 2),
                actions=torch.zeros(T, 1),
                rewards=torch.from_numpy(rewards_flat[offset : offset + T]),
                terminated=True,
                truncated=False,
            )
            table = build_chunk_table(
                ep,
                horizon=self.H,
                n_obs_steps=1,
                n_step=self.n_step,
                gamma=self.gamma,
                bootstrap_on_truncation=False,
            )
            for s in range(0, T - self.H + 1):  # make_indices: start in [0, T-H]
                ref_reward, ref_done = dice_rl_expert_item(
                    rewards_flat, dones_flat, offset + s, self.n_step, self.H, self.gamma
                )
                terminal_chunk = (T - 1 - s) // self.H  # window chunk holding r_{T-1}
                terminal_chunk_end = offset + s + (terminal_chunk + 1) * self.H
                if terminal_chunk < self.n_step and terminal_chunk_end > len(rewards_flat):
                    # dice-rl edge bug at the END of the flattened dataset only:
                    # `chunk_end > len(rewards)` breaks before reading the chunk
                    # that holds the terminal reward -> reward 0 / done False for
                    # a window that actually contains the terminal. We do not
                    # reproduce it; the correct values are checked by
                    # test_replay_buffer against the brute-force reference.
                    self.assertEqual((ref_reward, ref_done), (0.0, False))
                    self.assertTrue(bool(table.done[s]))
                    checked += 1
                    continue
                self.assertAlmostEqual(float(table.reward[s]), ref_reward, places=6)
                self.assertEqual(bool(table.done[s]), ref_done)
                if not ref_done:
                    self.assertEqual(int(table.n_eff[s]), self.n_step)
                    # dice-rl bootstraps from s+H here (its known bug); we use s+nH
                    self.assertEqual(int(table.next_frames[s, 0]), s + self.n_step * self.H)
                checked += 1
            offset += T
        self.assertGreater(checked, 500)


if __name__ == "__main__":
    unittest.main()
