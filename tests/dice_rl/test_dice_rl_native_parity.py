"""End-to-end parity against dice-rl's REAL classes (imported from the dice-rl
checkout, not transcribed): ``util.hybrid_replay_buffer.HybridReplayBuffer``
(online n-step) and ``agent.dataset.sequence.StitchedSequenceQLearningDataset``
(expert n-step). Skipped when the checkout is unavailable."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from dice_rl.episodes import EpisodeRecord
from dice_rl.replay_buffer import ChunkReplayBuffer

DICE_RL_ROOT = Path(os.environ.get("DICE_RL_ROOT", "/store/real/zhanyis/dice-rl-clean"))
H, N_STEP, GAMMA = 8, 3, 0.99


def _import_dice_rl():
    if not (DICE_RL_ROOT / "util" / "hybrid_replay_buffer.py").exists():
        return None
    if str(DICE_RL_ROOT) not in sys.path:
        sys.path.insert(0, str(DICE_RL_ROOT))
    from agent.dataset.sequence import StitchedSequenceQLearningDataset  # noqa: E402
    from util.hybrid_replay_buffer import HybridReplayBuffer  # noqa: E402

    return HybridReplayBuffer, StitchedSequenceQLearningDataset


def _episode(rng, T: int, terminated: bool, tag: float, D: int = 3, A: int = 2) -> EpisodeRecord:
    obs = rng.standard_normal((T + 1, D)).astype(np.float32)
    obs[:, 0] = np.arange(T + 1)
    obs[:, 1] = tag
    actions = rng.standard_normal((T, A)).astype(np.float32)
    rewards = np.zeros(T, dtype=np.float32)
    if terminated:
        rewards[-1] = 1.0
    return EpisodeRecord(
        obs=torch.from_numpy(obs), actions=torch.from_numpy(actions),
        rewards=torch.from_numpy(rewards), terminated=terminated, truncated=not terminated,
    )


def _feed_dice_rl_buffer(buffer, episode: EpisodeRecord) -> None:
    """Replay one episode through dice-rl's ``add`` in MultiStepFull chunks of H
    steps (the last chunk may be shorter), with include_initial on the first."""
    T = episode.num_transitions
    obs = episode.obs.numpy()
    act = episode.actions.numpy()
    rew = episode.rewards.numpy()
    dones = np.zeros(T, dtype=bool)
    dones[-1] = True  # MultiStep: terminated or truncated -> done on the last step
    t = 0
    first = True
    while t < T:
        k = min(H, T - t)
        info = {
            "initial_obs": obs[0] if first else None,
            "observations": [obs[t + i + 1] for i in range(k)],
            "actions": [act[t + i] for i in range(k)],
            "rewards": [float(rew[t + i]) for i in range(k)],
            "dones": [bool(dones[t + i]) for i in range(k)],
            "include_initial": first,
        }
        done_flag = torch.tensor([[bool(dones[t + k - 1])]])
        buffer.add(state=None, noise=None, action=None, reward=None, next_state=None,
                   done=done_flag, full_trajectory_info=[info])
        first = False
        t += k


class TestNativeDiceRLParity(unittest.TestCase):
    def setUp(self) -> None:
        mods = _import_dice_rl()
        if mods is None:
            self.skipTest("dice-rl checkout not available")
        self.HybridReplayBuffer, self.StitchedDataset = mods

    def test_online_buffer_rows_match_hybrid_replay_buffer(self) -> None:
        rng = np.random.default_rng(0)
        episodes = [_episode(rng, int(rng.integers(8, 70)), bool(rng.integers(0, 2)), tag=i) for i in range(30)]
        ref = self.HybridReplayBuffer(
            obs_dim=3, action_dim=2, noise_dim=2, max_size=100000, n_envs=1, cond_steps=1,
            horizon_steps=H, device="cpu", gamma=GAMMA, log_q_overestimation=True,
            use_n_step=True, n_step=N_STEP,
        )
        mine = ChunkReplayBuffer(
            capacity_frames=100000, feature_dim=3, action_dim=2, horizon=H, n_obs_steps=1,
            n_step=N_STEP, gamma=GAMMA, bootstrap_on_truncation=False, data_source=0, device="cpu",
        )
        for ep in episodes:
            _feed_dice_rl_buffer(ref, ep)
            mine.add_episode(ep)
        self.assertEqual(ref.size, len(mine))
        # Enumerate every sampleable row of dice-rl's buffer deterministically.
        safe_size = ref.size - (N_STEP - 1) * H
        orig_randint = np.random.randint
        np.random.randint = lambda lo, hi, size=None: np.arange(safe_size)  # type: ignore
        try:
            states, _noise, actions, rewards, next_states, dones, mc, n_steps, _src = ref._sample_standard(safe_size)
        finally:
            np.random.randint = orig_randint
        # Map each dice-rl row to (episode tag, chunk start) via the state's marker dims.
        mine_rows = mine.gather(mine.valid_rows())
        key = lambda cond: (int(round(float(cond[1]))), int(round(float(cond[0]))))  # (tag, s)
        index = {key(mine_rows.cond[i]): i for i in range(mine_rows.batch_size)}
        checked = 0
        for r in range(safe_size):
            i = index[key(states[r, 0])]
            self.assertTrue(torch.equal(actions[r], mine_rows.action[i]))
            self.assertAlmostEqual(float(rewards[r]), float(mine_rows.reward[i]), places=5)
            self.assertEqual(bool(dones[r]), bool(mine_rows.done[i]))
            self.assertAlmostEqual(float(mc[r]), float(mine_rows.mc_return[i]), places=5)
            self.assertEqual(int(n_steps[r]), N_STEP)
            if not bool(dones[r]):
                self.assertEqual(int(mine_rows.n_eff[i]), N_STEP)
                self.assertTrue(torch.equal(next_states[r, 0], mine_rows.next_cond[i]))
            checked += 1
        self.assertGreater(checked, 500)

    def test_expert_rows_match_stitched_dataset(self) -> None:
        rng = np.random.default_rng(1)
        lengths = [int(rng.integers(8, 80)) for _ in range(25)]
        total = sum(lengths)
        states = rng.standard_normal((total, 3)).astype(np.float32)
        states[:, 0] = np.concatenate([np.arange(T) for T in lengths])
        states[:, 1] = np.concatenate([np.full(T, i) for i, T in enumerate(lengths)])
        actions = rng.standard_normal((total, 2)).astype(np.float32)
        rewards = np.concatenate([np.concatenate([np.zeros(T - 1), [1.0]]) for T in lengths]).astype(np.float32)
        terminals = np.zeros(total, dtype=bool)  # dice-rl npz convention; dataset marks last steps
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "train.npz")
            np.savez(path, states=states, actions=actions, rewards=rewards, terminals=terminals,
                     traj_lengths=np.asarray(lengths))
            ref = self.StitchedDataset(
                dataset_path=path, horizon_steps=H, cond_steps=1, img_cond_steps=1, device="cpu",
                discount_factor=GAMMA, gamma=GAMMA, max_n_episodes=len(lengths), use_img=False,
                success_steps_for_termination=1, use_n_step=True, n_step=N_STEP,
            )
            for bootstrap_chunks, expected_next_offset in ((1, H), (None, N_STEP * H)):
                mine = ChunkReplayBuffer(
                    capacity_frames=total + len(lengths), feature_dim=3, action_dim=2, horizon=H,
                    n_obs_steps=1, n_step=N_STEP, gamma=GAMMA, bootstrap_on_truncation=False,
                    data_source=1, device="cpu", bootstrap_chunks=bootstrap_chunks,
                )
                offset = 0
                for T in lengths:
                    sl = slice(offset, offset + T)
                    mine.add_episode(EpisodeRecord(
                        obs=torch.from_numpy(states[sl]), actions=torch.from_numpy(actions[sl]),
                        rewards=torch.from_numpy(rewards[sl]), terminated=True, truncated=False))
                    offset += T
                rows = mine.gather(mine.valid_rows())
                key = lambda cond: (int(round(float(cond[1]))), int(round(float(cond[0]))))
                index = {key(rows.cond[i]): i for i in range(rows.batch_size)}
                self.assertEqual(len(ref), rows.batch_size)
                checked = 0
                for idx in range(len(ref)):
                    tr = ref[idx]
                    i = index[key(tr.conditions["state"][0])]
                    start, _ = ref.indices[idx]
                    s = int(states[start, 0])
                    self.assertTrue(torch.equal(tr.actions, rows.action[i]))
                    # dice-rl edge bug at the flat-dataset end (chunk_end > len -> break)
                    terminal_chunk = (int(lengths[int(states[start, 1])]) - 1 - s) // H
                    end_bug = terminal_chunk < N_STEP and start + (terminal_chunk + 1) * H > total
                    if not end_bug:
                        self.assertAlmostEqual(float(tr.rewards[0]), float(rows.reward[i]), places=5)
                        self.assertEqual(bool(tr.dones[0]), bool(rows.done[i]))
                    self.assertAlmostEqual(float(tr.mc_return[0]), float(rows.mc_return[i]), places=5)
                    if not bool(rows.done[i]):
                        # dice-rl next_state is at s+H; ours at s + bootstrap_chunks*H
                        ref_next = tr.conditions["next_state"][0]
                        self.assertEqual(int(ref_next[0]), s + H)
                        self.assertEqual(int(rows.next_cond[i][0]), s + expected_next_offset)
                        if bootstrap_chunks == 1:
                            self.assertTrue(torch.equal(ref_next, rows.next_cond[i]))
                    checked += 1
                self.assertGreater(checked, 500)


if __name__ == "__main__":
    unittest.main()
