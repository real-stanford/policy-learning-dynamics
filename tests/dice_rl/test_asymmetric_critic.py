"""Asymmetric critic (model.asymmetric_critic): the training-time critic reads
the simulator's lowdim state appended to each frame; policy/actor consume only
the pooled slice. Checks: StateCritic slicing + standardization + gradient flow;
model-level dims/losses/explore with an aux-augmented fake base; image-feature
invariance (pooled slice cannot change Q, state slice must); default-off is the
plain CriticEnsemble path."""

from __future__ import annotations

import sys
import unittest

import torch

sys.path.insert(0, "tests/dice_rl")
from test_model import _FakeBase  # noqa: E402

from dice_rl.model import DistillResidualRLModel  # noqa: E402
from dice_rl.networks import CriticEnsemble, StateCritic, soft_update  # noqa: E402


class _FakeStateBase(_FakeBase):
    """_FakeBase whose cond carries [policy features 6 | lowdim state 5] per frame."""

    def __init__(self, policy_feature_dim=6, action_dim=3, horizon=4, state_dim=5):
        super().__init__(cond_dim=policy_feature_dim, action_dim=action_dim, horizon=horizon)
        self.state_features_dim = state_dim
        self.policy_feature_dim = policy_feature_dim
        self.policy_cond_dim = policy_feature_dim
        self.feature_dim = policy_feature_dim + state_dim
        self.cond_dim = self.feature_dim * self.n_obs_steps
        self.aux_features_enabled = True

    def policy_cond(self, cond: torch.Tensor) -> torch.Tensor:
        batch = cond.shape[0]
        frames = cond.reshape(batch, self.n_obs_steps, self.feature_dim)
        return frames[:, :, : self.policy_feature_dim].reshape(batch, -1)

    def sample_actions(self, cond: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        assert cond.shape[1] == self.policy_cond_dim, tuple(cond.shape)  # pooled slice only
        return super().sample_actions(cond, noise)


def _asym_model(**overrides) -> DistillResidualRLModel:
    kwargs = dict(
        base_policy=_FakeStateBase(),
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
        asymmetric_critic={"state_dim": 5},
    )
    kwargs.update(overrides)
    return DistillResidualRLModel(**kwargs)


class TestStateCritic(unittest.TestCase):
    def _wrapped(self, ens=3):
        inner = CriticEnsemble(
            cond_dim=5, action_dim=3, horizon=4, hidden_dims=[32, 32],
            ensemble_size=ens, activation_type="GELU", use_layernorm=True,
            q_depends_on_noise=False, td_loss="mse",
        )
        return StateCritic(critic=inner, n_obs_steps=1, policy_feature_dim=6, state_dim=5)

    def test_slicing_and_grads(self) -> None:
        torch.manual_seed(0)
        wrapped = self._wrapped()
        cond = torch.randn(5, 6 + 5)
        out = wrapped(cond, None, torch.randn(5, 4, 3))
        self.assertEqual(tuple(out.shape), (3, 5, 1))
        out.sum().backward()
        self.assertTrue(all(p.grad is not None for p in wrapped.parameters()))
        # pooled (image) slice must NOT change Q; state slice must
        action = torch.zeros(4, 4, 3)
        c = torch.randn(4, 11)
        with torch.no_grad():
            a = wrapped(c, None, action)
            c2 = c.clone(); c2[:, :6] += 10.0
            b = wrapped(c2, None, action)
            c3 = c.clone(); c3[:, 6:] += 10.0
            d = wrapped(c3, None, action)
        torch.testing.assert_close(a, b)
        self.assertFalse(torch.allclose(a, d))

    def test_standardization(self) -> None:
        torch.manual_seed(0)
        wrapped = self._wrapped()
        mu, sd = torch.randn(5), torch.rand(5) + 0.5
        wrapped.set_standardization(mu, sd)
        c = torch.randn(3, 11)
        with torch.no_grad():
            a = wrapped(c, None, torch.zeros(3, 4, 3))
            # feeding pre-standardized state through an identity-stats twin must agree
            twin = self._wrapped()
            twin.critic.load_state_dict(wrapped.critic.state_dict())
            c_manual = c.clone()
            c_manual[:, 6:] = (c[:, 6:] - mu) / sd.clamp_min(1e-3)
            b = twin(c_manual, None, torch.zeros(3, 4, 3))
        torch.testing.assert_close(a, b)

    def test_soft_update_covers_buffers_and_params(self) -> None:
        import copy
        torch.manual_seed(0)
        wrapped = self._wrapped()
        target = copy.deepcopy(wrapped)
        with torch.no_grad():
            for p in wrapped.parameters():
                p.add_(1.0)
        soft_update(target, wrapped, tau=1.0)
        for p, q in zip(target.parameters(), wrapped.parameters()):
            torch.testing.assert_close(p, q)


class TestModelWithAsymmetricCritic(unittest.TestCase):
    def test_construction_and_dims(self) -> None:
        model = _asym_model()
        self.assertIsInstance(model.critic, StateCritic)
        self.assertIsInstance(model.target_critic, StateCritic)
        self.assertTrue(model.use_asymmetric_critic and model.cond_is_augmented)
        self.assertEqual(model.actor.cond_dim, 6)          # actor: policy features only
        self.assertEqual(model.critic.cond_dim, 6 + 5)     # stored cond: pooled + state
        self.assertEqual(model.critic.critic.cond_dim, 5)  # critic MLP: state only

    def test_losses_explore_target_run_and_separate_grads(self) -> None:
        torch.manual_seed(0)
        model = _asym_model()
        cond = torch.randn(5, 6 + 5)
        data_source = torch.tensor([[0], [0], [1], [1], [0]])
        out = model.actor_loss(cond, data_source=data_source, training_step=10, q_overestimation=None)
        out["actor_total"].backward()
        self.assertTrue(all(p.grad is not None for p in model.actor.parameters()))
        self.assertTrue(all(p.grad is None for p in model.critic.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.critic.parameters()))
        for p in model.actor.parameters():
            p.grad = None

        target = model.target_q(
            torch.randn(5, 11), torch.rand(5, 1),
            torch.tensor([[1.0], [0.0], [0.0], [1.0], [0.0]]),
            torch.tensor([[1], [2], [3], [1], [2]]), gamma=0.9,
        )
        closs = model.critic_loss(
            cond, torch.randn(5, 4, 3), target,
            data_source=torch.zeros(5, 1, dtype=torch.long),
        )
        closs["critic_total"].backward()
        self.assertTrue(all(p.grad is not None for p in model.critic.parameters()))
        self.assertTrue(all(p.grad is None for p in model.actor.parameters()))

        for step in (0, 100):  # warmup and max_q_min exploration paths
            actions, noise = model.explore(cond, num_samples=3, strategy="max_q_min", training_step=step)
            self.assertEqual(tuple(actions.shape), (5, 4, 3))
            self.assertEqual(tuple(noise.shape), (5, 4, 3))

    def test_policy_blind_to_state_and_critic_not(self) -> None:
        """Changing ONLY the aux state block must leave the executed policy
        (base + residual) bit-identical, while the critic must respond."""
        torch.manual_seed(0)
        model = _asym_model()
        cond = torch.randn(6, 6 + 5)
        cond2 = cond.clone()
        cond2[:, 6:] += 5.0  # perturb the state block only
        z = torch.randn(6, 4, 3)
        with torch.no_grad():
            a1 = model.get_action(cond, z)
            a2 = model.get_action(cond2, z)
            q1 = model._q_values(model.critic, cond, None, a1)
            q2 = model._q_values(model.critic, cond2, None, a1)
        self.assertTrue(torch.equal(a1, a2))          # actor + base: state-blind
        self.assertFalse(torch.allclose(q1, q2))      # critic: state-driven
        # exploration executes policy actions: same z-selection universe under
        # both conds during warmup (uniform), bit-identical actions
        torch.manual_seed(3)
        acts1, _ = model.explore(cond, num_samples=2, strategy="max_q_min", training_step=0)
        torch.manual_seed(3)
        acts2, _ = model.explore(cond2, num_samples=2, strategy="max_q_min", training_step=0)
        self.assertTrue(torch.equal(acts1, acts2))

    def test_actor_on_state(self) -> None:
        """r(state, z): actor input is the standardized state slice; the base
        action still comes from the pooled (image) slice; perturbing the pooled
        VISUAL part changes the base action but NOT the residual, and perturbing
        the state changes the residual but NOT the base action."""
        torch.manual_seed(0)
        model = _asym_model(asymmetric_critic={"state_dim": 5, "actor_on_state": True})
        self.assertTrue(model.actor_on_state)
        self.assertEqual(model.actor.cond_dim, 5)  # state only
        cond = torch.randn(4, 6 + 5)
        z = torch.randn(4, 4, 3)
        with torch.no_grad():
            total, base = model.get_action(cond, z, return_base=True)
            res = total - base
            # perturb pooled slice only -> base moves, residual identical
            cond_img = cond.clone(); cond_img[:, :6] += 3.0
            t2, b2 = model.get_action(cond_img, z, return_base=True)
            self.assertFalse(torch.allclose(base, b2))
            torch.testing.assert_close(t2 - b2, res)
            # perturb state slice only -> base identical, residual moves
            cond_st = cond.clone(); cond_st[:, 6:] += 3.0
            t3, b3 = model.get_action(cond_st, z, return_base=True)
            torch.testing.assert_close(b3, base)
            self.assertFalse(torch.allclose(t3 - b3, res))
        # losses still run and gradients stay separated
        out = model.actor_loss(cond, data_source=torch.zeros(4, 1, dtype=torch.long),
                               training_step=10, q_overestimation=None)
        out["actor_total"].backward()
        self.assertTrue(all(p.grad is not None for p in model.actor.parameters()))
        self.assertTrue(all(p.grad is None for p in model.critic.parameters()))

    def test_default_off(self) -> None:
        model = _asym_model(asymmetric_critic=None, base_policy=_FakeBase(cond_dim=6, action_dim=3, horizon=4))
        self.assertIsInstance(model.critic, CriticEnsemble)
        self.assertFalse(model.use_asymmetric_critic or model.cond_is_augmented)


if __name__ == "__main__":
    unittest.main()
