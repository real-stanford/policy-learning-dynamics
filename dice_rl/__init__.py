"""DICE-RL residual finetuning of frozen image diffusion-policy checkpoints.

Self-contained port (read-only use of the training stack: ``build_policy``,
``build_dataset``, checkpoint loading, the parallel image env) of dice-rl
(model/rl/distill_residual_rl*.py, agent/finetune/train_distill_residual_flow*_agent.py,
util/hybrid_replay_buffer.py) adapted to this repo's ``DiffusionUnetImagePolicy``:

* base policy  pi_pre(s, z): frozen pretrained diffusion policy, deterministic
  DDIM (eta=0) denoising from an explicit initial noise z, conditioned on the
  frozen encoder features s (global_cond);
* residual actor r_theta(s, z) -> a = pi_pre(s, z) + r_theta(s, z);
* critic ensemble Q_phi(s, a) (10 members by default) with n-step TD targets;
  optionally an ASYMMETRIC critic that reads the simulator's lowdim state;
* RLPD-style mixed sampling between the online chunk buffer and the expert
  (pretraining) demonstrations truncated to a single terminal success.
"""
