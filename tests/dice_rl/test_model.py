"""Loss/target/exploration semantics of DistillResidualRLModel on a fake,
cheap base policy (no checkpoint needed)."""

from __future__ import annotations

import unittest

import torch

from dice_rl.model import DistillResidualRLModel


class _FakeBase:
    """Deterministic 'base policy': pi_pre(s, z) = tanh(W s)[:, None, :] + 0.1 z."""

    def __init__(self, cond_dim: int, action_dim: int, horizon: int) -> None:
        self.cond_dim = cond_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.device = torch.device("cpu")
        self.n_obs_steps = 1
        self.feature_dim = cond_dim
        g = torch.Generator().manual_seed(0)
        self.w = torch.randn(cond_dim, action_dim, generator=g) / cond_dim**0.5

    def sample_actions(self, cond: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        base = torch.tanh(cond @ self.w).unsqueeze(1).expand(-1, self.horizon, -1)
        return base + 0.1 * noise


def _model(**overrides) -> DistillResidualRLModel:
    kwargs = dict(
        base_policy=_FakeBase(cond_dim=6, action_dim=3, horizon=4),
        actor_hidden_dims=[32, 32],
        critic_hidden_dims=[32, 32],
        activation_type="GELU",
        use_layernorm=True,
        critic_ensemble_size=3,
        q_depends_on_noise=False,
        td_loss="mse",
        bc_loss_weight=50.0,
        use_q_normalization=True,
        num_multi_z_for_actor_loss=4,
        num_next_noise_samples=2,
        condition_residual_on_base_action=False,
        disable_q_loss_for_expert_data=True,
        disable_td_loss_for_expert_data=False,
        always_retain_bc_loss_for_expert_data=True,
        use_soft_q_filtering=False,
        q_filtering_warmup_steps=5,
        q_underestimation_threshold=-0.5,
        replay_flow_warmup_steps=2,
    )
    kwargs.update(overrides)
    return DistillResidualRLModel(**kwargs)


class TestModel(unittest.TestCase):
    def test_actor_loss_updates_actor_only(self) -> None:
        torch.manual_seed(0)
        model = _model()
        cond = torch.randn(5, 6)
        data_source = torch.tensor([[0], [0], [1], [1], [0]])
        out = model.actor_loss(cond, data_source=data_source, training_step=0, q_overestimation=None)
        out["actor_total"].backward()
        self.assertTrue(all(p.grad is not None for p in model.actor.parameters()))
        self.assertTrue(all(p.grad is None for p in model.critic.parameters()))
        # critic params must be trainable again after the frozen context
        self.assertTrue(all(p.requires_grad for p in model.critic.parameters()))
        # bc term is exactly the mean squared residual
        self.assertGreater(float(out["actor_bc_loss"]), 0.0)
        self.assertAlmostEqual(
            float(out["actor_bc_loss"]) ** 0.5, float(out["residual_rms"]), places=5
        )

    def test_bc_only_drives_residual_to_zero(self) -> None:
        torch.manual_seed(0)
        model = _model(bc_loss_weight=1.0)
        # zero-out the Q term by disabling it for all rows (all expert)
        opt = torch.optim.Adam(model.actor.parameters(), lr=3e-3)
        cond = torch.randn(64, 6)
        expert = torch.ones(64, 1, dtype=torch.long)
        first = None
        for _ in range(200):
            out = model.actor_loss(cond, data_source=expert, training_step=0, q_overestimation=None)
            if first is None:
                first = float(out["residual_rms"])
            opt.zero_grad()
            out["actor_total"].backward()
            opt.step()
        self.assertLess(float(out["residual_rms"]), 0.2 * first)

    def test_target_and_critic_loss(self) -> None:
        torch.manual_seed(0)
        model = _model()
        B = 7
        next_cond = torch.randn(B, 6)
        reward = torch.rand(B, 1)
        done = torch.tensor([[1.0], [0.0], [0.0], [1.0], [0.0], [0.0], [0.0]])
        n_eff = torch.tensor([[1], [2], [3], [1], [2], [1], [3]])
        target = model.target_q(next_cond, reward, done, n_eff, gamma=0.9)
        self.assertEqual(tuple(target.shape), (B, 1))
        # done rows equal the reward exactly
        torch.testing.assert_close(target[done.bool()], reward[done.bool()])
        cond = torch.randn(B, 6)
        action = torch.randn(B, 4, 3)
        out = model.critic_loss(cond, action, target, data_source=torch.zeros(B, 1, dtype=torch.long))
        out["critic_total"].backward()
        self.assertTrue(all(p.grad is not None for p in model.critic.parameters()))
        self.assertTrue(all(p.grad is None for p in model.actor.parameters()))
        # loss is a sum of per-member MSEs
        with torch.no_grad():
            q_all = model.critic(cond, None, action)
            ref = ((q_all - target.unsqueeze(0)) ** 2).mean(dim=(1, 2)).sum()
        torch.testing.assert_close(out["critic_total"].detach(), ref)

    def test_explore_shapes_and_warmup(self) -> None:
        torch.manual_seed(0)
        model = _model()
        cond = torch.randn(3, 6)
        for strategy in ("standard", "max_q_min", "max_q_std", "max_q_std_filtered_by_min"):
            for step in (0, 10):
                action, noise = model.explore(cond, num_samples=5, strategy=strategy, training_step=step)
                self.assertEqual(tuple(action.shape), (3, 4, 3))
                self.assertEqual(tuple(noise.shape), (3, 4, 3))
                # the returned noise reproduces the returned action
                torch.testing.assert_close(model.get_action(cond, noise), action)

    def test_soft_update(self) -> None:
        model = _model()
        before = [p.clone() for p in model.target_critic.parameters()]
        with torch.no_grad():
            for p in model.critic.parameters():
                p.add_(1.0)
        model.update_target(tau=0.5)
        for b, t, c in zip(before, model.target_critic.parameters(), model.critic.parameters()):
            torch.testing.assert_close(t, 0.5 * b + 0.5 * c)


if __name__ == "__main__":
    unittest.main()
