from __future__ import annotations

import unittest

import torch

from dice_rl.networks import MLP, CriticEnsemble, ResidualActor, soft_update


class TestNetworks(unittest.TestCase):
    def test_ensemble_matches_independent_mlps(self) -> None:
        torch.manual_seed(0)
        cond_dim, action_dim, horizon, hidden, E = 7, 3, 4, [16, 8], 5
        critic = CriticEnsemble(
            cond_dim=cond_dim,
            action_dim=action_dim,
            horizon=horizon,
            hidden_dims=hidden,
            ensemble_size=E,
            activation_type="GELU",
            use_layernorm=True,
            q_depends_on_noise=False,
            td_loss="mse",
        )
        in_dim = cond_dim + horizon * action_dim
        members = [
            MLP([in_dim, *hidden, 1], activation_type="GELU", use_layernorm=True)
            for _ in range(E)
        ]
        # copy the ensemble parameters into the reference per-member MLPs
        linear_layers = [m for m in critic.net if hasattr(m, "load_member_from_linear")]
        norm_layers = [m for m in critic.net if m.__class__.__name__ == "EnsembleLayerNorm"]
        for e, member in enumerate(members):
            lin = [m for m in member.net if isinstance(m, torch.nn.Linear)]
            norms = [m for m in member.net if isinstance(m, torch.nn.LayerNorm)]
            self.assertEqual(len(lin), len(linear_layers))
            self.assertEqual(len(norms), len(norm_layers))
            for ens, ref in zip(linear_layers, lin):
                with torch.no_grad():
                    ref.weight.copy_(ens.weight[e].T)
                    ref.bias.copy_(ens.bias[e, 0])
            for ens, ref in zip(norm_layers, norms):
                with torch.no_grad():
                    ens.weight[e, 0].copy_(torch.randn_like(ref.weight))
                    ens.bias[e, 0].copy_(torch.randn_like(ref.bias))
                    ref.weight.copy_(ens.weight[e, 0])
                    ref.bias.copy_(ens.bias[e, 0])
        cond = torch.randn(11, cond_dim)
        action = torch.randn(11, horizon, action_dim)
        out = critic(cond, None, action)
        self.assertEqual(tuple(out.shape), (E, 11, 1))
        for e, member in enumerate(members):
            ref = member(torch.cat([cond, action.reshape(11, -1)], dim=-1))
            torch.testing.assert_close(out[e], ref, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(critic.conservative(cond, None, action), out.min(0)[0])

    def test_members_are_independent(self) -> None:
        torch.manual_seed(1)
        critic = CriticEnsemble(
            cond_dim=4, action_dim=2, horizon=3, hidden_dims=[8], ensemble_size=3,
            activation_type="GELU", use_layernorm=True, q_depends_on_noise=False, td_loss="mse",
        )
        out = critic(torch.randn(5, 4), None, torch.randn(5, 3, 2))
        self.assertFalse(torch.allclose(out[0], out[1]))
        # gradient of member 0's loss must not touch member 1's weights
        out[0].sum().backward()
        first = [m for m in critic.net if hasattr(m, "load_member_from_linear")][0]
        self.assertTrue(bool((first.weight.grad[1] == 0).all()))
        self.assertTrue(bool((first.weight.grad[0] != 0).any()))

    def test_actor_shapes_and_soft_update(self) -> None:
        actor = ResidualActor(
            cond_dim=6, action_dim=2, horizon=4, hidden_dims=[8, 8],
            activation_type="GELU", use_layernorm=True,
        )
        out = actor(torch.randn(3, 6), torch.randn(3, 4, 2))
        self.assertEqual(tuple(out.shape), (3, 4, 2))
        a = ResidualActor(cond_dim=6, action_dim=2, horizon=4, hidden_dims=[8],
                          activation_type="GELU", use_layernorm=False)
        b = ResidualActor(cond_dim=6, action_dim=2, horizon=4, hidden_dims=[8],
                          activation_type="GELU", use_layernorm=False)
        pa = [p.clone() for p in a.parameters()]
        pb = [p.clone() for p in b.parameters()]
        soft_update(a, b, tau=0.25)
        for p, qa, qb in zip(a.parameters(), pa, pb):
            torch.testing.assert_close(p, 0.75 * qa + 0.25 * qb)


if __name__ == "__main__":
    unittest.main()
