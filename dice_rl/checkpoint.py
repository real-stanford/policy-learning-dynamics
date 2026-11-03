from __future__ import annotations

import copy
from pathlib import Path

import torch
from omegaconf import OmegaConf


def save_rl_checkpoint(
    path: Path,
    *,
    cfg,
    model,
    actor_optimizer,
    critic_optimizer,
    actor_scheduler,
    critic_scheduler,
    iteration: int,
    env_steps: int,
    base_policy_summary: dict,
) -> None:
    payload = {
        "cfg": OmegaConf.to_container(cfg, resolve=True),
        "actor_state": copy.deepcopy(model.actor.state_dict()),
        "critic_state": copy.deepcopy(model.critic.state_dict()),
        "target_critic_state": copy.deepcopy(model.target_critic.state_dict()),
        "actor_optimizer_state": actor_optimizer.state_dict(),
        "critic_optimizer_state": critic_optimizer.state_dict(),
        "actor_scheduler_state": None if actor_scheduler is None else actor_scheduler.state_dict(),
        "critic_scheduler_state": None if critic_scheduler is None else critic_scheduler.state_dict(),
        "iteration": int(iteration),
        "env_steps": int(env_steps),
        "base_policy": base_policy_summary,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.rename(path)


def load_rl_checkpoint(
    path: Path,
    *,
    model,
    actor_optimizer,
    critic_optimizer,
    actor_scheduler,
    critic_scheduler,
    map_location,
) -> dict:
    payload = torch.load(str(path), map_location=map_location, weights_only=False)
    model.actor.load_state_dict(payload["actor_state"])
    model.critic.load_state_dict(payload["critic_state"])
    model.target_critic.load_state_dict(payload["target_critic_state"])
    actor_optimizer.load_state_dict(payload["actor_optimizer_state"])
    critic_optimizer.load_state_dict(payload["critic_optimizer_state"])
    if actor_scheduler is not None:
        assert payload["actor_scheduler_state"] is not None
        actor_scheduler.load_state_dict(payload["actor_scheduler_state"])
    if critic_scheduler is not None:
        assert payload["critic_scheduler_state"] is not None
        critic_scheduler.load_state_dict(payload["critic_scheduler_state"])
    return payload
