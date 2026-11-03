"""Loss/target parity against dice-rl's REAL ``DistillResidualRLModel``
(model/rl/distill_residual_rl.py from the dice-rl checkout), on identical
weights, identical batches and identical noise draws. The frozen base policy is
stubbed by the same deterministic function on both sides. Compares: TD target
(through the critic loss), actor Q term, actor BC term, exploration selection."""

from __future__ import annotations

import os
import sys
import unittest
from collections import namedtuple
from pathlib import Path

import torch
import torch.nn as nn

from dice_rl.model import DistillResidualRLModel

DICE_RL_ROOT = Path(os.environ.get("DICE_RL_ROOT", "/store/real/zhanyis/dice-rl-clean"))
D, A, H, E = 12, 3, 4, 10  # obs dim, action dim, horizon, ensemble size
HIDDEN = [32, 32, 32]


def _import_ref():
    if not (DICE_RL_ROOT / "model" / "rl" / "distill_residual_rl.py").exists():
        return None
    if str(DICE_RL_ROOT) not in sys.path:
        sys.path.insert(0, str(DICE_RL_ROOT))
    from model.rl.distill_residual_rl import DistillResidualRLModel as Ref  # noqa: E402

    return Ref


_W = torch.randn(D, A, generator=torch.Generator().manual_seed(7)) / D**0.5
Sample = namedtuple("Sample", "trajectories chains")


def _base_fn(state_flat: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    return torch.tanh(state_flat @ _W).unsqueeze(1).expand(-1, H, -1) + 0.1 * noise


class _FakeFlow(nn.Module):
    """dice-rl side: pretrained_flow_policy(cond, deterministic, init_noise)."""

    def forward(self, cond, deterministic=False, init_noise=None):
        return Sample(_base_fn(cond["state"].reshape(cond["state"].shape[0], -1), init_noise), None)


class _FakeBase:
    """dice_rl side: base.sample_actions(cond, noise)."""

    cond_dim, action_dim, horizon, feature_dim, n_obs_steps = D, A, H, D, 1
    device = torch.device("cpu")

    def sample_actions(self, cond, noise):
        return _base_fn(cond, noise)


def _copy_mlp(ref_mlp, mine_seq) -> None:
    ref_lin = [m.linear_1 for m in ref_mlp.moduleList]
    ref_norm = [m.norm_1 for m in ref_mlp.moduleList if hasattr(m, "norm_1")]
    my_lin = [m for m in mine_seq if isinstance(m, nn.Linear)]
    my_norm = [m for m in mine_seq if isinstance(m, nn.LayerNorm)]
    assert len(ref_lin) == len(my_lin) and len(ref_norm) == len(my_norm)
    with torch.no_grad():
        for r, m in zip(ref_lin, my_lin):
            m.weight.copy_(r.weight)
            m.bias.copy_(r.bias)
        for r, m in zip(ref_norm, my_norm):
            m.weight.copy_(r.weight)
            m.bias.copy_(r.bias)


def _copy_critic(ref_critic, mine_critic) -> None:
    ens_lin = [m for m in mine_critic.net if hasattr(m, "load_member_from_linear")]
    ens_norm = [m for m in mine_critic.net if m.__class__.__name__ == "EnsembleLayerNorm"]
    for e, member in enumerate(ref_critic.Q_ensemble):
        lin = [m.linear_1 for m in member.moduleList]
        norms = [m.norm_1 for m in member.moduleList if hasattr(m, "norm_1")]
        assert len(lin) == len(ens_lin) and len(norms) == len(ens_norm)
        with torch.no_grad():
            for r, ens in zip(lin, ens_lin):
                ens.load_member_from_linear(e, r)
            for r, ens in zip(norms, ens_norm):
                ens.weight[e, 0].copy_(r.weight)
                ens.bias[e, 0].copy_(r.bias)


def _build(Ref, K: int, K_next: int, training_step_warmup: int):
    Ref._load_pretrained_policy = lambda self, path, device: _FakeFlow()  # stub the checkpoint load
    ref = Ref(
        obs_dim=D, action_dim=A, pretrained_flow_policy_path="stub",
        actor_hidden_dims=HIDDEN, critic_hidden_dims=HIDDEN, activation_type="GELU",
        bc_loss_weight=50.0, critic_weight=1.0, use_soft_q_filtering=False,
        q_filtering_warmup_steps=training_step_warmup, q_underestimation_threshold=-0.5,
        replay_flow_warmup_steps=3000, use_q_normalization=True, q_depends_on_noise=False,
        multi_sample_next_noise=True, num_next_noise_samples=K_next, critic_ensemble_size=E,
        cond_steps=1, horizon_steps=H, use_n_step=True, n_step=3,
        disable_q_loss_for_expert_data=True, disable_td_loss_for_expert_data=False,
        always_retain_bc_loss_for_expert_data=True, td_loss="mse",
        sample_multi_z_for_actor_loss=True, num_multi_z_for_actor_loss=K,
        condition_residual_on_base_action=False, device="cpu", use_layernorm=True,
    )
    mine = DistillResidualRLModel(
        base_policy=_FakeBase(), actor_hidden_dims=HIDDEN, critic_hidden_dims=HIDDEN,
        activation_type="GELU", use_layernorm=True, critic_ensemble_size=E, q_depends_on_noise=False,
        td_loss="mse", bc_loss_weight=50.0, use_q_normalization=True, num_multi_z_for_actor_loss=K,
        num_next_noise_samples=K_next, condition_residual_on_base_action=False,
        disable_q_loss_for_expert_data=True, disable_td_loss_for_expert_data=False,
        always_retain_bc_loss_for_expert_data=True, use_soft_q_filtering=False,
        q_filtering_warmup_steps=training_step_warmup, q_underestimation_threshold=-0.5,
        replay_flow_warmup_steps=3000,
    )
    _copy_mlp(ref.actor.mlp, mine.actor.mlp.net)
    _copy_critic(ref.critic, mine.critic)
    _copy_critic(ref.target_critic, mine.target_critic)
    return ref, mine


def _batch(B: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    state = torch.randn(B, D, generator=g)
    action = torch.randn(B, H, A, generator=g)
    next_state = torch.randn(B, D, generator=g)
    reward = (torch.rand(B, 1, generator=g) < 0.2).float() * 0.97
    done = (torch.rand(B, 1, generator=g) < 0.1).float()
    data_source = (torch.rand(B, 1, generator=g) < 0.4).long()
    return state, action, next_state, reward, done, data_source


class TestNativeLossParity(unittest.TestCase):
    def setUp(self) -> None:
        self.Ref = _import_ref()
        if self.Ref is None:
            self.skipTest("dice-rl checkout not available")

    def _check(self, training_step: int, warmup: int) -> None:
        B, K, K_next = 16, 1, 1
        ref, mine = _build(self.Ref, K, K_next, warmup)
        state, action, next_state, reward, done, data_source = _batch(B, 3)
        n_steps = torch.full((B, 1), 3, dtype=torch.long)
        gamma = 0.99

        # --- dice-rl: one call draws next_noise (B,H,A) [unused], next_noise_samples
        # (K',B,H,A), then the actor's noise_samples (B,K,H,A).
        torch.manual_seed(11)
        out_ref = ref.loss(
            state=state.unsqueeze(1), noise=torch.zeros(B, H, A), action=action,
            next_state=next_state.unsqueeze(1), reward=reward, done=done, gamma=gamma,
            training_step=training_step, q_overestimation=None, n_steps=n_steps,
            data_source=data_source,
        )
        # --- ours: replay the identical draw sequence.
        torch.manual_seed(11)
        _ = torch.randn(B, H, A)  # dice-rl's unused single next_noise draw
        target = mine.target_q(next_state, reward, done, n_steps, gamma)
        out_actor = mine.actor_loss(state, data_source=data_source, training_step=training_step, q_overestimation=None)
        out_critic = mine.critic_loss(state, action, target, data_source=data_source)

        torch.testing.assert_close(out_critic["critic_total"], out_ref["critic_loss"], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(out_actor["actor_q_loss"], out_ref["actor_q_loss"], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(out_actor["actor_bc_loss"], out_ref["actor_bc_loss"], atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(out_actor["actor_total"], out_ref["actor_total"], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(out_actor["current_q_mean"], out_ref["current_q_mean"], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(out_actor["base_q_mean"], out_ref["pretrained_q_mean"], atol=1e-5, rtol=1e-5)

        # Gradients of the actor loss w.r.t. the actor parameters must match too.
        ref.actor.zero_grad(); mine.actor.zero_grad()
        out_ref["actor_total"].backward(); out_actor["actor_total"].backward()
        for r, m in zip([p for p in ref.actor.parameters()], [p for p in mine.actor.parameters()]):
            torch.testing.assert_close(m.grad, r.grad, atol=1e-5, rtol=1e-4)

    def test_warmup_phase(self) -> None:
        self._check(training_step=10, warmup=2000)

    def test_post_warmup_phase(self) -> None:
        self._check(training_step=5000, warmup=2000)

    def test_exploration_selection(self) -> None:
        ref, mine = _build(self.Ref, 1, 1, 2000)
        state = torch.randn(5, D, generator=torch.Generator().manual_seed(4))
        torch.manual_seed(21)
        a_ref, z_ref = ref.get_exploration_action(state.unsqueeze(1), num_samples=6, exploration_strategy="max_q_min", training_step=10**6)
        torch.manual_seed(21)
        a_mine, z_mine = mine.explore(state, num_samples=6, strategy="max_q_min", training_step=10**6)
        torch.testing.assert_close(a_mine, a_ref, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(z_mine, z_ref, atol=1e-6, rtol=1e-6)

    def test_target_update(self) -> None:
        ref, mine = _build(self.Ref, 1, 1, 2000)
        with torch.no_grad():
            for p in ref.critic.parameters():
                p.add_(1.0)
        _copy_critic(ref.critic, mine.critic)
        ref.update_target_networks(tau=0.01); mine.update_target(0.01)
        b = _batch(8, 5)
        q_ref = torch.stack(ref.target_critic(b[0].unsqueeze(1), None, b[1], return_all=True), 0)
        q_mine = mine.target_critic(b[0], None, b[1])
        torch.testing.assert_close(q_mine, q_ref, atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
