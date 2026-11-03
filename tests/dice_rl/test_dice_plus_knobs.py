"""DICE+ knobs (slides 8/20-9/10/26): critic next-state prediction and the
expert residual imitation loss. Checks: (1) off by default -> no new params,
identical losses; (2) next-state targets live in the critic's OWN cond space
(asymmetric critic: standardized lowdim state only); (3) the aux loss is a
learnable regression; (4) the imitation loss equals ||(base+r)-a_demo||^2 on
expert rows only, vanishes when actions match the demo, and moves the actor;
(5) chunk-table / buffer 1-step next indices are exact."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_model import _FakeBase, _model  # noqa: E402

from dice_rl.episodes import EpisodeRecord, build_chunk_table  # noqa: E402
from dice_rl.model import DistillResidualRLModel  # noqa: E402
from dice_rl.networks import CriticEnsemble, StateCritic  # noqa: E402
from dice_rl.replay_buffer import ChunkReplayBuffer  # noqa: E402


class TestOffByDefault(unittest.TestCase):
    def test_no_new_params_and_identical_losses(self) -> None:
        torch.manual_seed(0)
        a = _model()
        torch.manual_seed(0)
        b = _model(critic_next_state_weight=0.0, expert_imitation_weight=0.0)
        self.assertEqual(sorted(a.state_dict()), sorted(b.state_dict()))
        self.assertNotIn("next_head.weight", " ".join(a.state_dict()))
        cond = torch.randn(5, 6)
        ds = torch.tensor([[0], [1], [1], [0], [0]])
        demo = torch.randn(5, 4, 3)  # drawn BEFORE seeding: must not perturb the RNG stream
        torch.manual_seed(1)
        la = a.actor_loss(cond, data_source=ds, training_step=10, q_overestimation=None)
        torch.manual_seed(1)
        lb = b.actor_loss(cond, data_source=ds, training_step=10, q_overestimation=None,
                          demo_action=demo)  # ignored when off
        self.assertTrue(torch.equal(la["actor_total"], lb["actor_total"]))
        self.assertNotIn("actor_imitation_loss", la)
        act, tgt = torch.randn(5, 4, 3), torch.rand(5, 1)
        ca = a.critic_loss(cond, act, tgt, data_source=ds)
        cb = b.critic_loss(cond, act, tgt, data_source=ds, next1_cond=torch.randn(5, 6))
        self.assertTrue(torch.equal(ca["critic_total"], cb["critic_total"]))
        self.assertNotIn("critic_next_loss", ca)


class TestNextStateTarget(unittest.TestCase):
    def test_plain_critic_target_is_cond_and_head_dim_matches(self) -> None:
        m = _model(critic_next_state_weight=1.0)
        self.assertEqual(m.critic.next_state_dim, 6)
        x = torch.randn(3, 6)
        self.assertTrue(torch.equal(m.critic_next_target(x), x))
        q, nxt = m.critic(x, None, torch.randn(3, 4, 3), return_next=True)
        self.assertEqual(tuple(q.shape), (3, 3, 1))
        self.assertEqual(tuple(nxt.shape), (3, 3, 6))

    def test_q_path_identical_with_return_next(self) -> None:
        torch.manual_seed(0)
        c = CriticEnsemble(cond_dim=6, action_dim=3, horizon=4, hidden_dims=[16, 16],
                           ensemble_size=2, activation_type="GELU", use_layernorm=True,
                           q_depends_on_noise=False, td_loss="mse", next_state_dim=6)
        x, a = torch.randn(5, 6), torch.randn(5, 4, 3)
        q0 = c(x, None, a)
        q1, nxt = c(x, None, a, return_next=True)
        self.assertTrue(torch.equal(q0, q1))
        self.assertEqual(tuple(nxt.shape), (2, 5, 6))

    def test_asymmetric_target_is_standardized_lowdim_state_only(self) -> None:
        """cond = [image feat (F) | state (S)] per frame; the target must be the
        standardized STATE slice only, dim S * n_obs, no image dims."""
        torch.manual_seed(0)
        inner = CriticEnsemble(cond_dim=4 * 2, action_dim=3, horizon=4, hidden_dims=[16],
                               ensemble_size=2, activation_type="GELU", use_layernorm=True,
                               q_depends_on_noise=False, td_loss="mse", next_state_dim=4 * 2)
        sc = StateCritic(critic=inner, n_obs_steps=2, policy_feature_dim=10, state_dim=4)
        mu, sd = torch.randn(4), torch.rand(4) + 0.5
        sc.set_standardization(mu, sd)
        next1 = torch.randn(3, 2 * (10 + 4))
        target = sc.project_cond(next1)
        self.assertEqual(tuple(target.shape), (3, 8))
        frames = next1.reshape(3, 2, 14)
        expect = ((frames[:, :, 10:] - mu) / sd.clamp_min(1e-3)).reshape(3, -1)
        self.assertTrue(torch.allclose(target, expect))
        # image dims must not influence the target
        alt = next1.clone(); alt.reshape(3, 2, 14)[:, :, :10] += 100.0
        self.assertTrue(torch.equal(sc.project_cond(alt), target))
        q, nxt = sc(next1, None, torch.randn(3, 4, 3), return_next=True)
        self.assertEqual(tuple(nxt.shape), (2, 3, 8))

    def test_aux_loss_is_learnable_regression(self) -> None:
        torch.manual_seed(0)
        m = _model(critic_next_state_weight=1.0, disable_td_loss_for_expert_data=False)
        W = torch.randn(6 + 12, 6) * 0.3
        opt = torch.optim.Adam(m.critic.parameters(), lr=3e-3)
        losses = []
        for _ in range(300):
            cond, act = torch.randn(64, 6), torch.randn(64, 4, 3)
            next1 = torch.cat([cond, act.reshape(64, -1)], dim=1) @ W  # known dynamics
            out = m.critic_loss(cond, act, torch.zeros(64, 1),
                                data_source=torch.zeros(64, 1, dtype=torch.long), next1_cond=next1)
            opt.zero_grad(); out["critic_total"].backward(); opt.step()
            losses.append(float(out["critic_next_loss"]))
        self.assertLess(np.mean(losses[-20:]), 0.25 * np.mean(losses[:20]))


class TestExpertImitation(unittest.TestCase):
    def _setup(self):
        torch.manual_seed(0)
        m = _model(expert_imitation_weight=7.0, bc_loss_weight=0.0, num_multi_z_for_actor_loss=3)
        cond = torch.randn(6, 6)
        ds = torch.tensor([[1], [0], [1], [1], [0], [0]])
        return m, cond, ds

    def test_equals_final_action_mse_on_expert_rows(self) -> None:
        m, cond, ds = self._setup()
        demo = torch.randn(6, 4, 3)
        torch.manual_seed(5)
        out = m.actor_loss(cond, data_source=ds, training_step=0, q_overestimation=None, demo_action=demo)
        # recompute by hand with the same noise draw
        torch.manual_seed(5)
        K = 3
        noise = torch.randn(6, K, 4, 3)
        cond_flat = cond.unsqueeze(1).expand(-1, K, -1).reshape(18, -1)
        with torch.no_grad():
            final = m.get_action(cond_flat, noise.reshape(18, 4, 3)).reshape(6, K, 4, 3)
        per = ((final - demo.unsqueeze(1)) ** 2).mean(dim=(-1, -2))  # (6, K)
        expect = per[[0, 2, 3]].sum() / (3 * K)
        self.assertTrue(torch.allclose(out["actor_imitation_loss"], expect, atol=1e-6))
        # contribution to the total = weight * loss (bc weight is 0 here, q term separate)
        torch.manual_seed(5)
        base = _model(expert_imitation_weight=0.0, bc_loss_weight=0.0, num_multi_z_for_actor_loss=3)
        base.load_state_dict(m.state_dict())
        torch.manual_seed(5)
        out0 = base.actor_loss(cond, data_source=ds, training_step=0, q_overestimation=None)
        self.assertTrue(torch.allclose(out["actor_total"] - out0["actor_total"],
                                       7.0 * out["actor_imitation_loss"], atol=1e-5))

    def test_zero_when_actions_match_demo_and_zero_without_expert_rows(self) -> None:
        m, cond, ds = self._setup()
        torch.manual_seed(5)
        K = 3
        noise = torch.randn(6, K, 4, 3)
        with torch.no_grad():
            # demo := the model's own action for the FIRST noise draw; with K=1 loss must be 0
            m1 = _model(expert_imitation_weight=7.0, bc_loss_weight=0.0, num_multi_z_for_actor_loss=1)
            m1.load_state_dict(m.state_dict())
            torch.manual_seed(5)
            n1 = torch.randn(6, 1, 4, 3)
            demo = m1.get_action(cond, n1.reshape(6, 4, 3))
        torch.manual_seed(5)
        out = m1.actor_loss(cond, data_source=ds, training_step=0, q_overestimation=None, demo_action=demo)
        self.assertLess(float(out["actor_imitation_loss"]), 1e-10)
        out2 = m.actor_loss(cond, data_source=torch.zeros(6, 1, dtype=torch.long), training_step=0,
                            q_overestimation=None, demo_action=torch.randn(6, 4, 3))
        self.assertEqual(float(out2["actor_imitation_loss"]), 0.0)

    def test_drives_actions_toward_demo(self) -> None:
        torch.manual_seed(0)
        m = _model(expert_imitation_weight=1.0, bc_loss_weight=0.0, disable_q_loss_for_expert_data=True)
        cond = torch.randn(16, 6)
        ds = torch.ones(16, 1, dtype=torch.long)  # all expert -> q term off
        demo = torch.randn(16, 4, 3)
        opt = torch.optim.Adam(m.actor.parameters(), lr=3e-3)
        first = None
        for step in range(200):
            out = m.actor_loss(cond, data_source=ds, training_step=0, q_overestimation=None, demo_action=demo)
            opt.zero_grad(); out["actor_total"].backward(); opt.step()
            first = float(out["actor_imitation_loss"]) if first is None else first
        self.assertLess(float(out["actor_imitation_loss"]), 0.2 * first)


class TestNext1Index(unittest.TestCase):
    def test_chunk_table_and_buffer_next1(self) -> None:
        T, H, To = 10, 4, 2
        ep = EpisodeRecord(obs=torch.arange(T + 1, dtype=torch.float32).unsqueeze(1).expand(T + 1, 3).clone(),
                           actions=torch.zeros(T, 2), rewards=torch.cat([torch.zeros(T - 1), torch.ones(1)]),
                           terminated=True, truncated=False)
        tab = build_chunk_table(ep, horizon=H, n_obs_steps=To, n_step=3, gamma=0.99,
                                bootstrap_on_truncation=False)
        n = T - H + 1
        for s in range(n):
            f = min(s + H, T)  # next1 frame = s + H clamped to last frame index T
            self.assertEqual(tab.next1_frames[s].tolist(), [max(f - 1, 0), f])
        buf_off = ChunkReplayBuffer(capacity_frames=50, feature_dim=3, action_dim=2, horizon=H,
                                    n_obs_steps=To, n_step=3, gamma=0.99, bootstrap_on_truncation=False,
                                    data_source=0, device="cpu")
        buf_off.add_episode(ep)
        self.assertIsNone(buf_off.c_next1_idx)
        self.assertIsNone(buf_off.gather(torch.arange(n)).next1_cond)
        buf = ChunkReplayBuffer(capacity_frames=50, feature_dim=3, action_dim=2, horizon=H,
                                n_obs_steps=To, n_step=3, gamma=0.99, bootstrap_on_truncation=False,
                                data_source=0, device="cpu", with_next1=True)
        buf.add_episode(ep)
        b = buf.gather(torch.arange(n))
        # obs value == frame index, so next1_cond reveals the gathered frames
        got = b.next1_cond.reshape(n, To, 3)[:, :, 0]
        self.assertEqual(got.long().tolist(), tab.next1_frames.tolist())
        # concatenation keeps / rejects the field consistently
        self.assertEqual(b.cat(b).next1_cond.shape[0], 2 * n)
        with self.assertRaises(AssertionError):
            b.cat(buf_off.gather(torch.arange(n)))


if __name__ == "__main__":
    unittest.main()


class TestFeatureCombos(unittest.TestCase):
    """Both knobs on across the model's other switches: finite losses, both
    metrics present, gradients reach actor / critic (incl. the aux head)."""

    def test_grid(self) -> None:
        import itertools

        grid = itertools.product(
            [False, True],            # condition_residual_on_base_action
            [False, True],            # q_depends_on_noise
            ["mse", "huber", "bce"],  # td_loss
            [1, 4],                   # num_multi_z_for_actor_loss
            [False, True],            # disable_td_loss_for_expert_data
        )
        for cba, qdn, td, K, dtd in grid:
            with self.subTest(cba=cba, qdn=qdn, td=td, K=K, dtd=dtd):
                torch.manual_seed(0)
                m = _model(condition_residual_on_base_action=cba, q_depends_on_noise=qdn,
                           td_loss=td, num_multi_z_for_actor_loss=K,
                           disable_td_loss_for_expert_data=dtd,
                           critic_next_state_weight=0.5, expert_imitation_weight=3.0)
                cond = torch.randn(6, 6)
                ds = torch.tensor([[1], [0], [1], [0], [0], [1]])
                demo = torch.randn(6, 4, 3)
                a = m.actor_loss(cond, data_source=ds, training_step=10,
                                 q_overestimation=None, demo_action=demo)
                self.assertIn("actor_imitation_loss", a)
                self.assertTrue(torch.isfinite(a["actor_total"]))
                a["actor_total"].backward()
                self.assertTrue(all(p.grad is not None for p in m.actor.parameters()))
                noise = m.sample_noise(6)
                tgt = torch.rand(6, 1)
                c = m.critic_loss(cond, torch.randn(6, 4, 3), tgt, data_source=ds,
                                  noise=noise, next1_cond=torch.randn(6, 6))
                self.assertIn("critic_next_loss", c)
                self.assertTrue(torch.isfinite(c["critic_total"]))
                c["critic_total"].backward()
                self.assertTrue(all(p.grad is not None for p in m.critic.next_head.parameters()))
                # target / explore paths untouched by the knobs still run
                m.target_q(cond, tgt, torch.zeros(6, 1), torch.ones(6, 1, dtype=torch.long), 0.99)
                m.explore(cond, num_samples=2, strategy="max_q_min", training_step=100)
