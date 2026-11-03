from __future__ import annotations

import contextlib
import copy
from typing import Callable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from dice_rl.base_policy import FrozenDiffusionBasePolicy
from dice_rl.networks import CriticEnsemble, ResidualActor, StateCritic, soft_update

EXPLORATION_STRATEGIES = (
    "standard",
    "max_q_min",
    "max_q_std",
    "max_q_std_filtered_by_min",
)


@contextlib.contextmanager
def _frozen(module: nn.Module):
    flags = [p.requires_grad for p in module.parameters()]
    for p in module.parameters():
        p.requires_grad_(False)
    try:
        yield
    finally:
        for p, flag in zip(module.parameters(), flags):
            p.requires_grad_(flag)


class DistillResidualRLModel(nn.Module):
    def __init__(
        self,
        *,
        base_policy: FrozenDiffusionBasePolicy,
        actor_hidden_dims: Sequence[int],
        critic_hidden_dims: Sequence[int],
        activation_type: str,
        use_layernorm: bool,
        critic_ensemble_size: int,
        q_depends_on_noise: bool,
        td_loss: str,
        bc_loss_weight: float,
        use_q_normalization: bool,
        num_multi_z_for_actor_loss: int,
        num_next_noise_samples: int,
        condition_residual_on_base_action: bool,
        disable_q_loss_for_expert_data: bool,
        disable_td_loss_for_expert_data: bool,
        always_retain_bc_loss_for_expert_data: bool,
        use_soft_q_filtering: bool,
        q_filtering_warmup_steps: int,
        q_underestimation_threshold: float,
        replay_flow_warmup_steps: int,
        actor_zero_init_last_layer: bool = False,
        clip_q: tuple[float | None, float | None] | None = None,
        asymmetric_critic: dict | None = None,
        critic_next_state_weight: float = 0.0,
        expert_imitation_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.critic_next_state_weight = float(critic_next_state_weight)
        self.expert_imitation_weight = float(expert_imitation_weight)
        assert self.critic_next_state_weight >= 0.0 and self.expert_imitation_weight >= 0.0
        # next-state head dim = the inner ensemble's cond_dim (set per branch below)
        _nsd = lambda inner_cond_dim: (
            int(inner_cond_dim) if self.critic_next_state_weight > 0 else None
        )
        # Not a registered submodule: keeps the frozen 260M-param policy out of
        # state_dict / DDP / optimizers.
        object.__setattr__(self, "base", base_policy)
        self.cond_dim = int(base_policy.cond_dim)
        self.action_dim = int(base_policy.action_dim)
        self.horizon = int(base_policy.horizon)
        self.device = base_policy.device

        self.use_asymmetric_critic = asymmetric_critic is not None
        self.cond_is_augmented = self.use_asymmetric_critic
        policy_cond_dim = self.cond_dim
        self.actor_on_state = False
        if self.use_asymmetric_critic:
            assert getattr(base_policy, "aux_features_enabled", False), (
                "asymmetric_critic requires base_policy.enable_state_features() first."
            )
            assert int(base_policy.state_features_dim) == int(asymmetric_critic["state_dim"])
            policy_cond_dim = int(base_policy.policy_cond_dim)
            self.actor_on_state = bool(asymmetric_critic.get("actor_on_state", False))

        actor_cond_dim = policy_cond_dim
        if self.actor_on_state:
            actor_cond_dim = int(asymmetric_critic["state_dim"]) * int(base_policy.n_obs_steps)
        self.actor = ResidualActor(
            cond_dim=actor_cond_dim,
            action_dim=self.action_dim,
            horizon=self.horizon,
            hidden_dims=actor_hidden_dims,
            activation_type=activation_type,
            use_layernorm=use_layernorm,
            zero_init_last_layer=actor_zero_init_last_layer,
        )
        if self.use_asymmetric_critic:
            state_dim = int(asymmetric_critic["state_dim"])
            inner = CriticEnsemble(
                cond_dim=state_dim * int(base_policy.n_obs_steps),
                action_dim=self.action_dim,
                horizon=self.horizon,
                hidden_dims=critic_hidden_dims,
                ensemble_size=critic_ensemble_size,
                activation_type=activation_type,
                use_layernorm=use_layernorm,
                q_depends_on_noise=q_depends_on_noise,
                td_loss=td_loss,
                # next-state target = standardized lowdim state only (project_cond)
                next_state_dim=_nsd(state_dim * int(base_policy.n_obs_steps)),
            )
            self.critic = StateCritic(
                critic=inner,
                n_obs_steps=int(base_policy.n_obs_steps),
                policy_feature_dim=int(base_policy.policy_feature_dim),
                state_dim=state_dim,
            )
        else:
            self.critic = CriticEnsemble(
                cond_dim=self.cond_dim,
                action_dim=self.action_dim,
                horizon=self.horizon,
                hidden_dims=critic_hidden_dims,
                ensemble_size=critic_ensemble_size,
                activation_type=activation_type,
                use_layernorm=use_layernorm,
                q_depends_on_noise=q_depends_on_noise,
                td_loss=td_loss,
                next_state_dim=_nsd(self.cond_dim),
            )
        self.target_critic = copy.deepcopy(self.critic)
        for p in self.target_critic.parameters():
            p.requires_grad_(False)
        self.actor_fwd: Callable = self.actor
        self.critic_fwd: Callable = self.critic

        self.td_loss = str(td_loss)
        self.bc_loss_weight = float(bc_loss_weight)
        self.use_q_normalization = bool(use_q_normalization)
        self.num_multi_z = int(num_multi_z_for_actor_loss)
        self.num_next_noise_samples = int(num_next_noise_samples)
        self.condition_on_base_action = bool(condition_residual_on_base_action)
        self.disable_q_loss_for_expert_data = bool(disable_q_loss_for_expert_data)
        self.disable_td_loss_for_expert_data = bool(disable_td_loss_for_expert_data)
        self.always_retain_bc_loss_for_expert_data = bool(always_retain_bc_loss_for_expert_data)
        self.use_soft_q_filtering = bool(use_soft_q_filtering)
        self.q_filtering_warmup_steps = int(q_filtering_warmup_steps)
        self.q_underestimation_threshold = float(q_underestimation_threshold)
        self.replay_flow_warmup_steps = int(replay_flow_warmup_steps)
        assert self.num_multi_z >= 1 and self.num_next_noise_samples >= 1
        self.clip_q = None
        if clip_q is not None:
            lo, hi = clip_q
            assert lo is None or hi is None or lo <= hi, clip_q
            self.clip_q = (None if lo is None else float(lo), None if hi is None else float(hi))

    # ------------------------------------------------------------ primitives
    def sample_noise(self, batch: int) -> torch.Tensor:
        return torch.randn(batch, self.horizon, self.action_dim, device=self.device)

    def _q_values(self, critic: nn.Module, cond, noise, action) -> torch.Tensor:
        """(E, B, 1) member values in value space via the given (possibly
        wrapped) critic forward."""
        out = critic(cond, noise, action)
        if self.td_loss == "bce":
            out = torch.sigmoid(out)
        return out

    def _q_min(self, critic: nn.Module, cond, noise, action) -> torch.Tensor:
        return self._q_values(critic, cond, noise, action).min(dim=0)[0]

    def get_action(
        self,
        cond: torch.Tensor,
        noise: torch.Tensor,
        *,
        actor: nn.Module | None = None,
        return_base: bool = False,
    ):
        """a = pi_pre(s, z) + r_theta(s, z). Base actions carry no gradient."""
        actor_cond = cond
        if self.cond_is_augmented:
            if self.actor_on_state:
                # standardized state slice (same transform as the critic's input);
                # buffers only, so no gradient path into the critic.
                actor_cond = self.critic.project_cond(cond)
            cond = self.base.policy_cond(cond)
            if not self.actor_on_state:
                actor_cond = cond
        base = self.base.sample_actions(cond, noise).detach()
        actor = self.actor if actor is None else actor
        residual = actor(actor_cond, base if self.condition_on_base_action else noise)
        total = base + residual
        if return_base:
            return total, base
        return total

    def update_target(self, tau: float) -> None:
        soft_update(self.target_critic, self.critic, tau)

    # ------------------------------------------------------------ exploration
    @torch.no_grad()
    def explore(
        self,
        cond: torch.Tensor,
        *,
        num_samples: int,
        strategy: str,
        training_step: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert strategy in EXPLORATION_STRATEGIES, strategy
        batch = cond.shape[0]
        if strategy == "standard" or training_step <= self.replay_flow_warmup_steps:
            noise = self.sample_noise(batch)
            return self.get_action(cond, noise), noise
        assert num_samples >= 1
        noise = torch.randn(
            num_samples, batch, self.horizon, self.action_dim, device=self.device
        )
        cond_flat = cond.unsqueeze(0).expand(num_samples, -1, -1).reshape(num_samples * batch, -1)
        noise_flat = noise.reshape(num_samples * batch, self.horizon, self.action_dim)
        actions_flat = self.get_action(cond_flat, noise_flat)
        q_all = self._q_values(self.critic, cond_flat, noise_flat, actions_flat)  # (E, N*B, 1)
        q = q_all.reshape(q_all.shape[0], num_samples, batch)
        if strategy == "max_q_min":
            selection = q.min(dim=0)[0].argmax(dim=0)  # (B,)
        elif strategy == "max_q_std":
            selection = q.std(dim=0).argmax(dim=0)
        elif strategy == "max_q_std_filtered_by_min":
            top_k = min(3, num_samples)
            q_min = q.min(dim=0)[0]  # (N, B)
            _, top_idx = q_min.topk(top_k, dim=0)  # (k, B)
            q_top = torch.gather(q, 1, top_idx.unsqueeze(0).expand(q.shape[0], -1, -1))
            best_in_top = q_top.std(dim=0).argmax(dim=0)  # (B,)
            selection = torch.gather(top_idx, 0, best_in_top.unsqueeze(0)).squeeze(0)
        else:
            raise ValueError(strategy)
        actions = actions_flat.reshape(num_samples, batch, self.horizon, self.action_dim)
        rows = torch.arange(batch, device=self.device)
        return actions[selection, rows], noise[selection, rows]

    # ---------------------------------------------------------------- losses
    def actor_loss(
        self,
        cond: torch.Tensor,
        *,
        data_source: torch.Tensor,
        training_step: int,
        q_overestimation: torch.Tensor | None,
        demo_action: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch = cond.shape[0]
        K = self.num_multi_z
        noise = torch.randn(batch, K, self.horizon, self.action_dim, device=self.device)
        cond_flat = cond.unsqueeze(1).expand(-1, K, -1).reshape(batch * K, -1)
        noise_flat = noise.reshape(batch * K, self.horizon, self.action_dim)
        actions_flat, base_flat = self.get_action(
            cond_flat, noise_flat, actor=self.actor_fwd, return_base=True
        )
        with _frozen(self.critic):
            q_current = self._q_min(self.critic, cond_flat, noise_flat, actions_flat).reshape(batch, K)
        with torch.no_grad():
            q_base = self._q_min(self.critic, cond_flat, noise_flat, base_flat).reshape(batch, K)

        online_mask = None
        q_per_row = q_current.mean(dim=1)  # (B,)
        if self.disable_q_loss_for_expert_data:
            online_mask = (data_source == 0).to(q_per_row.dtype).reshape(-1)  # (B,)
            q_per_row = q_per_row * online_mask
        q_loss = -q_per_row.mean()
        if self.use_q_normalization:
            if online_mask is not None:
                online_count = online_mask.sum() * K
                if float(online_count) > 0:
                    q_abs_mean = (
                        (q_current.abs() * online_mask.unsqueeze(1)).sum() / online_count
                    ).detach()
                else:
                    q_abs_mean = q_current.abs().mean().detach()
            else:
                q_abs_mean = q_current.abs().mean().detach()
            if float(q_abs_mean) > 1e-8:
                q_loss = q_loss / q_abs_mean

        residual = actions_flat.reshape(batch, K, self.horizon, self.action_dim) - base_flat.reshape(
            batch, K, self.horizon, self.action_dim
        )
        mse_per_sample = (residual**2).mean(dim=(-1, -2))  # (B, K)

        in_warmup = training_step <= self.q_filtering_warmup_steps
        if in_warmup:
            bc_filter = torch.ones(batch, K, device=self.device)
            q_advantage = torch.zeros(batch, device=self.device)
            better_frac = torch.tensor(0.0, device=self.device)
        else:
            with torch.no_grad():
                adv = q_current.detach() - q_base  # (B, K)
                better = (adv > 0).to(torch.float32)
                q_advantage = adv.mean(dim=1)
                better_frac = better.mean()
                if self.use_soft_q_filtering:
                    if q_overestimation is not None:
                        under = (q_overestimation.reshape(-1, 1) < self.q_underestimation_threshold).to(
                            torch.float32
                        )
                        bc_filter = 1.0 - better * under
                    else:
                        bc_filter = 1.0 - better
                else:
                    bc_filter = torch.ones_like(better)
                if self.always_retain_bc_loss_for_expert_data:
                    expert = (data_source == 1).to(torch.float32).reshape(-1, 1)
                    bc_filter = expert + (1.0 - expert) * bc_filter
        bc_loss = (bc_filter * mse_per_sample).sum(dim=1).div(K).mean()
        total = q_loss + self.bc_loss_weight * bc_loss

        imitation_loss = None
        if self.expert_imitation_weight > 0:
            assert demo_action is not None, "expert_imitation_weight > 0 needs demo_action."
            assert demo_action.shape == (batch, self.horizon, self.action_dim), demo_action.shape
            expert_rows = (data_source == 1).to(actions_flat.dtype).reshape(batch)
            diff = actions_flat.reshape(batch, K, self.horizon, self.action_dim) - demo_action.unsqueeze(1)
            imit_per_sample = (diff**2).mean(dim=(-1, -2))  # (B, K)
            n_terms = expert_rows.sum() * K
            imitation_loss = (imit_per_sample * expert_rows.unsqueeze(1)).sum() / n_terms.clamp_min(1.0)
            total = total + self.expert_imitation_weight * imitation_loss

        with torch.no_grad():
            metrics = {
                "actor_q_loss": q_loss.detach(),
                "actor_bc_loss": bc_loss.detach(),
                "residual_rms": residual.pow(2).mean().sqrt(),
                "q_advantage_mean": q_advantage.mean(),
                "better_than_base_frac": better_frac,
                "base_q_mean": q_base.mean(),
                "current_q_mean": q_current.detach().mean(),
                "bc_filter_mean": bc_filter.mean(),
            }
            if imitation_loss is not None:
                metrics["actor_imitation_loss"] = imitation_loss.detach()
        return {"actor_total": total, **metrics}

    @torch.no_grad()
    def target_q(
        self,
        next_cond: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        n_eff: torch.Tensor,
        gamma: float,
    ) -> torch.Tensor:
        """r + gamma^{n_eff} (1 - done) mean_k min_E Q_target(s', pi(s', z_k))."""
        batch = next_cond.shape[0]
        K = self.num_next_noise_samples
        noise = torch.randn(K, batch, self.horizon, self.action_dim, device=self.device)
        cond_flat = next_cond.unsqueeze(0).expand(K, -1, -1).reshape(K * batch, -1)
        noise_flat = noise.reshape(K * batch, self.horizon, self.action_dim)
        next_actions = self.get_action(cond_flat, noise_flat)
        q_next = self._q_min(self.target_critic, cond_flat, noise_flat, next_actions)
        q_next = q_next.reshape(K, batch, 1).mean(dim=0)
        if self.clip_q is not None:
            q_next = q_next.clamp(min=self.clip_q[0], max=self.clip_q[1])
        discount = gamma ** n_eff.to(torch.float32)
        return reward + discount * (1.0 - done) * q_next

    def critic_loss(
        self,
        cond: torch.Tensor,
        action: torch.Tensor,
        target: torch.Tensor,
        *,
        data_source: torch.Tensor,
        noise: torch.Tensor | None = None,
        next1_cond: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Sum over ensemble members of the per-member TD loss (+ the optional
        next-state prediction auxiliary, see __init__)."""
        next_pred = None
        if self.critic_next_state_weight > 0:
            assert next1_cond is not None, "critic_next_state_weight > 0 needs next1_cond."
            # same forward op sequence as below, plus the aux head (one DDP call)
            q_all, next_pred = self.critic_fwd(cond, noise, action, return_next=True)
        else:
            q_all = self.critic_fwd(cond, noise, action)  # (E, B, 1) raw
        target_all = target.unsqueeze(0).expand_as(q_all)
        if self.td_loss == "mse":
            per_sample = F.mse_loss(q_all, target_all, reduction="none")
        elif self.td_loss == "huber":
            per_sample = F.smooth_l1_loss(q_all, target_all, reduction="none", beta=1.0)
        elif self.td_loss == "bce":
            assert bool((target >= 0).all()) and bool((target <= 1).all()), (
                "BCE TD loss requires targets in [0, 1]."
            )
            per_sample = F.binary_cross_entropy_with_logits(q_all, target_all, reduction="none")
        else:
            raise ValueError(self.td_loss)
        if self.disable_td_loss_for_expert_data:
            online = (data_source == 0).to(per_sample.dtype).reshape(1, -1, 1)
            per_member = (per_sample * online).mean(dim=(1, 2))
        else:
            per_member = per_sample.mean(dim=(1, 2))
        loss = per_member.sum()
        next_loss = None
        if next_pred is not None:
            next_target = self.critic_next_target(next1_cond)
            next_loss = (
                F.mse_loss(next_pred, next_target.unsqueeze(0).expand_as(next_pred), reduction="none")
                .mean(dim=(1, 2))
                .sum()
            )
            loss = loss + self.critic_next_state_weight * next_loss
        with torch.no_grad():
            q_values = torch.sigmoid(q_all) if self.td_loss == "bce" else q_all
            metrics = {
                "critic_loss": loss.detach(),
                "critic_q_mean": q_values.mean(),
                "critic_q_std_across_ensemble": q_values.std(dim=0).mean(),
                "target_q_mean": target.mean(),
                "target_q_max": target.max(),
            }
            if next_loss is not None:
                metrics["critic_next_loss"] = next_loss.detach()
        return {"critic_total": loss, **metrics}

    @torch.no_grad()
    def critic_next_target(self, next1_cond: torch.Tensor) -> torch.Tensor:
        """Next-state prediction target in the critic's own cond space."""
        project = getattr(self.critic, "project_cond", None)
        return project(next1_cond) if project is not None else next1_cond
