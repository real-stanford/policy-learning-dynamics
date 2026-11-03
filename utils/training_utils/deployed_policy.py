"""Load a checkpoint exactly as deployed and expose its sampler for offline probes.

Used by the held-out error cache (coverage-reliability checkpoint selection) and the
generator nearest-neighbour probe. ``ddim_record`` is the policy's own
``conditional_sample`` loop with the initial noise injected, so a fixed noise bank
gives paired samples across checkpoints.
"""

from __future__ import annotations

import torch
from omegaconf import open_dict

from utils.common.pytorch_util import dict_apply
from utils.training_utils.checkpointing import load_checkpoint_payload
from utils.training_utils.factories import build_policy


def configure_determinism() -> None:
    """Diagnostics run without TF32: with it, batched and per-sample results differ."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_deployed_policy(ckpt_path: str, num_inference_steps: int, eta: float, device):
    """EMA weights, eval mode, DDIM with the given step count / eta (the deployed sampler)."""
    payload = load_checkpoint_payload(ckpt_path, map_location="cpu")
    cfg = payload["cfg"]
    with open_dict(cfg):
        # Checkpoints written by the original research repo carry loss-shaping blocks
        # (all disabled); the only one this repo still knows is the generator mixer,
        # which must not be loaded for a probe.
        if "jointaug_w_oa_joint_model" in cfg.training:
            cfg.training.jointaug_w_oa_joint_model.enabled = False
    policy = build_policy(cfg)
    assert payload.get("ema_state") is not None, f"{ckpt_path}: no ema_state"
    policy.load_state_dict(payload["ema_state"])
    policy.normalizer.load_state_dict(payload["normalizer_state"])
    policy.to(device).eval()
    for p in policy.parameters():
        p.requires_grad_(False)
    assert policy.obs_as_global_cond, "probes assume obs_conditioning=global."
    policy.num_inference_steps = int(num_inference_steps)
    policy.scheduler_step_kwargs = dict(policy.scheduler_step_kwargs)
    policy.scheduler_step_kwargs["eta"] = float(eta)
    return policy


@torch.no_grad()
def encode_obs(policy, rgb_u8: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
    """RAW obs -> global-cond features, exactly the deployed predict_action path.
    rgb_u8 (B, n_obs, 6, 84, 84) uint8 on device; states (B, n_obs, sd) fp32."""
    obs = {"rgb": rgb_u8.float().div_(255.0), "states": states}
    nobs = policy.normalizer.normalize(obs)
    enc_in = dict_apply(
        nobs, lambda x: x[:, : policy.n_obs_steps, ...].reshape(-1, *x.shape[2:])
    )
    return policy.obs_encoder(enc_in).reshape(rgb_u8.shape[0], -1)


@torch.no_grad()
def ddim_record(policy, global_cond: torch.Tensor, z: torch.Tensor) -> list[torch.Tensor]:
    """Deployed DDIM chain from injected init noise ``z`` (normalized action space).
    Returns the S+1 carried sampler states, states[0] == z; states[-1] is the sample.
    Identical to ``policy.conditional_sample`` driven by a generator that draws ``z``."""
    traj = z.clone()
    states = [traj.clone()]
    policy.noise_scheduler.set_timesteps(policy.num_inference_steps)
    for t in policy.noise_scheduler.timesteps:
        out = policy.model(traj, t, local_cond=None, global_cond=global_cond)
        traj = policy.noise_scheduler.step(out, t, traj, **policy.scheduler_step_kwargs).prev_sample
        states.append(traj.clone())
    return states
