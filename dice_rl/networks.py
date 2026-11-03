from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

_ACTIVATIONS = {
    "ReLU": nn.ReLU,
    "ELU": nn.ELU,
    "GELU": nn.GELU,
    "Tanh": nn.Tanh,
    "Mish": nn.Mish,
    "Identity": nn.Identity,
    "Softplus": nn.Softplus,
    "Sigmoid": nn.Sigmoid,
}


def make_activation(name: str) -> nn.Module:
    assert name in _ACTIVATIONS, f"Unknown activation '{name}'; choose from {list(_ACTIVATIONS)}"
    return _ACTIVATIONS[name]()


class MLP(nn.Module):
    """dice-rl ``model/common/mlp.py::MLP`` (no append layers / dropout)."""

    def __init__(
        self,
        dim_list: Sequence[int],
        *,
        activation_type: str = "Mish",
        out_activation_type: str = "Identity",
        use_layernorm: bool = False,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        num_layer = len(dim_list) - 1
        for idx in range(num_layer):
            layers.append(nn.Linear(int(dim_list[idx]), int(dim_list[idx + 1])))
            is_last = idx == num_layer - 1
            if use_layernorm and not is_last:
                layers.append(nn.LayerNorm(int(dim_list[idx + 1])))
            layers.append(
                make_activation(out_activation_type if is_last else activation_type)
            )
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StateCritic(nn.Module):

    def __init__(
        self,
        *,
        critic: "CriticEnsemble",
        n_obs_steps: int,
        policy_feature_dim: int,
        state_dim: int,
    ) -> None:
        super().__init__()
        self.critic = critic
        self.n_obs_steps = int(n_obs_steps)
        self.policy_feature_dim = int(policy_feature_dim)
        self.state_dim = int(state_dim)
        assert self.state_dim > 0
        self.frame_dim = self.policy_feature_dim + self.state_dim
        self.cond_dim = self.frame_dim * self.n_obs_steps
        self.register_buffer("state_mu", torch.zeros(self.state_dim))
        self.register_buffer("state_sd", torch.ones(self.state_dim))
        # mirrored for call sites that inspect the critic
        self.td_loss = critic.td_loss
        self.ensemble_size = critic.ensemble_size
        self.action_dim = critic.action_dim
        self.horizon = critic.horizon

    @torch.no_grad()
    def set_standardization(self, mu: torch.Tensor, sd: torch.Tensor) -> None:
        assert mu.shape == (self.state_dim,) and sd.shape == (self.state_dim,)
        self.state_mu.copy_(mu.to(self.state_mu))
        self.state_sd.copy_(sd.clamp_min(1e-3).to(self.state_sd))

    def project_cond(self, cond: torch.Tensor) -> torch.Tensor:
        batch = cond.shape[0]
        assert cond.shape == (batch, self.cond_dim), tuple(cond.shape)
        frames = cond.reshape(batch, self.n_obs_steps, self.frame_dim)
        state = frames[:, :, self.policy_feature_dim :]
        state = (state - self.state_mu) / self.state_sd
        return state.reshape(batch, -1)

    def forward(
        self, cond: torch.Tensor, noise: torch.Tensor | None, action: torch.Tensor,
        return_next: bool = False,
    ):
        return self.critic(self.project_cond(cond), noise, action, return_next=return_next)


class ResidualActor(nn.Module):
    """r_theta(s, z): flattened conditioning features + flattened noise chunk ->
    residual action chunk. ``condition_on_base_action`` swaps z for the base
    policy's action chunk (dice-rl ``condition_residual_on_base_action``)."""

    def __init__(
        self,
        *,
        cond_dim: int,
        action_dim: int,
        horizon: int,
        hidden_dims: Sequence[int],
        activation_type: str,
        use_layernorm: bool,
        zero_init_last_layer: bool = False,
    ) -> None:
        super().__init__()
        self.cond_dim = int(cond_dim)
        self.action_dim = int(action_dim)
        self.horizon = int(horizon)
        chunk_dim = self.horizon * self.action_dim
        self.mlp = MLP(
            [self.cond_dim + chunk_dim, *hidden_dims, chunk_dim],
            activation_type=activation_type,
            out_activation_type="Identity",
            use_layernorm=use_layernorm,
        )
        if zero_init_last_layer:
            last = [m for m in self.mlp.net if isinstance(m, nn.Linear)][-1]
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def forward(self, cond: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        batch = noise.shape[0]
        assert cond.shape == (batch, self.cond_dim), (
            f"cond must be ({batch}, {self.cond_dim}), got {tuple(cond.shape)}"
        )
        assert noise.shape == (batch, self.horizon, self.action_dim), (
            f"noise must be ({batch}, {self.horizon}, {self.action_dim}), got "
            f"{tuple(noise.shape)}"
        )
        x = torch.cat([cond, noise.reshape(batch, -1)], dim=-1)
        return self.mlp(x).reshape(batch, self.horizon, self.action_dim)


class EnsembleLinear(nn.Module):
    """E independent Linear layers applied to an (E, B, in) input."""

    def __init__(self, ensemble_size: int, in_features: int, out_features: int) -> None:
        super().__init__()
        self.ensemble_size = int(ensemble_size)
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.weight = nn.Parameter(torch.empty(ensemble_size, in_features, out_features))
        self.bias = nn.Parameter(torch.empty(ensemble_size, 1, out_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # nn.Linear default init, applied per member.
        bound = 1.0 / math.sqrt(self.in_features)
        for member in range(self.ensemble_size):
            nn.init.kaiming_uniform_(self.weight[member].T, a=math.sqrt(5))
            nn.init.uniform_(self.bias[member], -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[0] == self.ensemble_size and x.shape[-1] == self.in_features
        return torch.baddbmm(self.bias, x, self.weight)

    def load_member_from_linear(self, member: int, linear: nn.Linear) -> None:
        with torch.no_grad():
            self.weight[member].copy_(linear.weight.T)
            self.bias[member, 0].copy_(linear.bias)


class EnsembleLayerNorm(nn.Module):
    """Per-member LayerNorm over the last dim of an (E, B, C) tensor."""

    def __init__(self, ensemble_size: int, num_features: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(ensemble_size, 1, num_features))
        self.bias = nn.Parameter(torch.zeros(ensemble_size, 1, num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = F.layer_norm(x, (x.shape[-1],), eps=self.eps)
        return normalized * self.weight + self.bias


class CriticEnsemble(nn.Module):

    def __init__(
        self,
        *,
        cond_dim: int,
        action_dim: int,
        horizon: int,
        hidden_dims: Sequence[int],
        ensemble_size: int,
        activation_type: str,
        use_layernorm: bool,
        q_depends_on_noise: bool,
        td_loss: str,
        next_state_dim: int | None = None,
    ) -> None:
        super().__init__()
        assert ensemble_size >= 1
        assert td_loss in {"mse", "huber", "bce"}, f"Unknown td_loss '{td_loss}'"
        self.cond_dim = int(cond_dim)
        self.action_dim = int(action_dim)
        self.horizon = int(horizon)
        self.ensemble_size = int(ensemble_size)
        self.q_depends_on_noise = bool(q_depends_on_noise)
        self.td_loss = str(td_loss)
        chunk_dim = self.horizon * self.action_dim
        input_dim = self.cond_dim + chunk_dim * (2 if self.q_depends_on_noise else 1)
        dims = [input_dim, *[int(h) for h in hidden_dims], 1]
        layers: list[nn.Module] = []
        for idx in range(len(dims) - 1):
            layers.append(EnsembleLinear(self.ensemble_size, dims[idx], dims[idx + 1]))
            is_last = idx == len(dims) - 2
            if use_layernorm and not is_last:
                layers.append(EnsembleLayerNorm(self.ensemble_size, dims[idx + 1]))
            if not is_last:
                layers.append(make_activation(activation_type))
        self.net = nn.Sequential(*layers)

        self.next_state_dim = None if next_state_dim is None else int(next_state_dim)
        self.next_head = (
            EnsembleLinear(self.ensemble_size, dims[-2], self.next_state_dim)
            if self.next_state_dim is not None else None
        )

    def _input(self, cond: torch.Tensor, noise: torch.Tensor | None, action: torch.Tensor) -> torch.Tensor:
        batch = cond.shape[0]
        assert cond.shape == (batch, self.cond_dim), tuple(cond.shape)
        assert action.shape == (batch, self.horizon, self.action_dim), tuple(action.shape)
        parts = [cond, action.reshape(batch, -1)]
        if self.q_depends_on_noise:
            assert noise is not None, "q_depends_on_noise requires the noise input."
            parts.insert(1, noise.reshape(batch, -1))
        x = torch.cat(parts, dim=-1)
        return x.unsqueeze(0).expand(self.ensemble_size, -1, -1)

    def forward(
        self, cond: torch.Tensor, noise: torch.Tensor | None, action: torch.Tensor,
        return_next: bool = False,
    ):
        """All member outputs, raw (logits when td_loss == 'bce'): (E, B, 1).
        return_next=True (requires next_state_dim) -> (q (E, B, 1), next-state
        prediction (E, B, next_state_dim)); the q path is the same computation."""
        if not return_next:
            return self.net(self._input(cond, noise, action))
        assert self.next_head is not None, "return_next requires next_state_dim."
        layers = list(self.net)
        h = self._input(cond, noise, action)
        for layer in layers[:-1]:  # identical op sequence to self.net up to the head
            h = layer(h)
        return layers[-1](h), self.next_head(h)

    def values(self, cond: torch.Tensor, noise: torch.Tensor | None, action: torch.Tensor) -> torch.Tensor:
        """Member Q-values in value space (sigmoid applied for bce): (E, B, 1)."""
        out = self.forward(cond, noise, action)
        if self.td_loss == "bce":
            out = torch.sigmoid(out)
        return out

    def conservative(self, cond: torch.Tensor, noise: torch.Tensor | None, action: torch.Tensor) -> torch.Tensor:
        """min over the ensemble: (B, 1)."""
        return self.values(cond, noise, action).min(dim=0)[0]


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    """target <- tau * source + (1 - tau) * target (dice-rl ``update_target_networks``)."""
    with torch.no_grad():
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.lerp_(param.data, tau)
