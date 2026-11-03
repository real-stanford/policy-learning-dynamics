"""pass@k closed-loop evaluation of saved checkpoints (multi-GPU, parallel envs).

For every checkpoint, ``--pass_at_k`` independent rollouts are run on each of the
``--num_episodes`` held-out environment seeds (``task.evaluation.test_start_seed`` + i)
with the chosen sampler (default DDIM-10, eta 0 = the deployed sampler). The per-attempt
per-episode successes are stored, so pass@k for any k <= pass_at_k can be computed
afterwards with the unbiased estimator 1 - C(K - c, k) / C(K, k) (scripts/crb/analyze.py).

    CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --multi_gpu --num_processes 4 \
        --num_machines 1 --mixed_precision no --dynamo_backend no \
        -m scripts.eval_checkpoints \
        --run_dir outputs/training/threading_d2_image/<run> \
        --pass_at_k 10 --num_episodes 500 --num_envs_per_rank 24 \
        --scheduler ddim --num_inference_steps 10 --eta 0.0

Outputs: <run_dir>/eval_<scheduler>_num_steps_<S>_eta_<eta>_pass_<K>/epoch_XXXX/
  summary.json           pass@K (OR over attempts), per_pass_success_rates, per-episode rows
  pass_<a>/summary.json  attempt a >= 1 (attempt 0 is the directory itself)
plus all_results.json and eval_across_epochs.png at the top level.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler, DDPMSchedulerOutput
from omegaconf import open_dict

from utils.training_utils.checkpointing import load_checkpoint_payload
from utils.training_utils.compile import load_compiled_state_dict
from utils.training_utils.evaluation import run_image_policy_eval
from utils.training_utils.factories import build_policy


class StridedDDPMScheduler(DDPMScheduler):
    """diffusers 0.11.1's DDPMScheduler.step always steps t -> t-1 regardless of
    set_timesteps(), so with fewer inference steps than training steps every update uses
    a one-step posterior across a strided jump. This subclass steps to the previous
    timestep of the INFERENCE grid (as diffusers >= 0.15 does); with stride 1 it reduces
    to the original formulas."""

    def _stride(self) -> int:
        n = self.num_inference_steps or self.config.num_train_timesteps
        return self.config.num_train_timesteps // n

    def _transition(self, t):
        stride = self._stride()
        prev_t = t - stride
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        if stride == 1:
            return alpha_prod_t, alpha_prod_t_prev, self.alphas[t], self.betas[t]
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        return alpha_prod_t, alpha_prod_t_prev, current_alpha_t, 1 - current_alpha_t

    def _get_variance(self, t, predicted_variance=None, variance_type=None):
        alpha_prod_t, alpha_prod_t_prev, _, current_beta_t = self._transition(t)
        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t
        if variance_type is None:
            variance_type = self.config.variance_type
        assert variance_type == "fixed_small", variance_type
        return torch.clamp(variance, min=1e-20)

    def step(self, model_output, timestep, sample, generator=None, return_dict=True, **kwargs):
        t = int(timestep)
        alpha_prod_t, alpha_prod_t_prev, current_alpha_t, current_beta_t = self._transition(t)
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        assert self.config.prediction_type == "epsilon", self.config.prediction_type
        pred_original_sample = (sample - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5
        if self.config.clip_sample:
            pred_original_sample = torch.clamp(pred_original_sample, -1, 1)
        pred_original_sample_coeff = (alpha_prod_t_prev**0.5 * current_beta_t) / beta_prod_t
        current_sample_coeff = current_alpha_t**0.5 * beta_prod_t_prev / beta_prod_t
        pred_prev_sample = (
            pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample
        )
        if t > 0:
            noise = torch.randn(
                model_output.shape,
                generator=generator,
                device=model_output.device,
                dtype=model_output.dtype,
            )
            pred_prev_sample = pred_prev_sample + (self._get_variance(t) ** 0.5) * noise
        if not return_dict:
            return (pred_prev_sample,)
        return DDPMSchedulerOutput(
            prev_sample=pred_prev_sample, pred_original_sample=pred_original_sample
        )


def make_scheduler(scheduler_type: str):
    """Inference scheduler with the training forward process (100 steps, squaredcos_cap_v2)."""
    common = dict(
        num_train_timesteps=100,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )
    if scheduler_type == "ddim":
        return DDIMScheduler(**common, set_alpha_to_one=True, steps_offset=0)
    if scheduler_type == "ddpm":
        return StridedDDPMScheduler(**common, variance_type="fixed_small")
    raise ValueError(f"Unknown scheduler: {scheduler_type}")


def load_policy_for_eval(checkpoint_path: Path, device, args):
    payload = load_checkpoint_payload(str(checkpoint_path), map_location=device)
    cfg = payload["cfg"]
    with open_dict(cfg):
        # The generated-sample mixer is a training-only hook (and its generator checkpoints
        # need not exist at eval time): never attach it when evaluating a checkpoint.
        if "jointaug_w_oa_joint_model" in cfg.training:
            cfg.training.jointaug_w_oa_joint_model.enabled = False
    policy = build_policy(cfg)
    state_key = "ema_state" if "ema_state" in payload else "model_state"
    load_compiled_state_dict(policy, payload[state_key])
    policy.normalizer.load_state_dict(payload["normalizer_state"])
    policy.to(device)
    policy.eval()
    policy.noise_scheduler = make_scheduler(args.scheduler)
    policy.num_inference_steps = int(args.num_inference_steps)
    if args.scheduler == "ddim":
        policy.scheduler_step_kwargs["eta"] = float(args.eta)
    return policy, cfg, state_key


def save_pass_at_k_plot(all_pass_episode_successes, output_path, test_start_seed) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    k = len(all_pass_episode_successes)
    n_eps = len(all_pass_episode_successes[0])
    matrix = np.array(all_pass_episode_successes)
    means = matrix.mean(axis=0)
    pass_at_k_rate = matrix.max(axis=0).mean()
    fig, axes = plt.subplots(
        2, 1, figsize=(max(n_eps * 0.15, 8), 4), gridspec_kw={"height_ratios": [k, 1]}, sharex=True
    )
    axes[0].imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1, interpolation="nearest")
    axes[0].set_ylabel("Trial")
    axes[0].set_yticks(range(k))
    axes[0].set_yticklabels([f"{i + 1}" for i in range(k)])
    axes[0].set_title(f"pass@{k}={pass_at_k_rate:.2f}, avg={means.mean():.2f} ({n_eps} configs)")
    axes[1].bar(range(n_eps), means, color=["green" if m > 0 else "red" for m in means], width=1.0, edgecolor="none")
    axes[1].set_ylabel("Mean")
    axes[1].set_ylim(0, 1)
    axes[1].set_xlabel(f"Config index (seed {test_start_seed}+)")
    axes[1].set_xlim(-0.5, n_eps - 0.5)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close()


def eval_single_checkpoint(checkpoint_path: Path, accelerator: Accelerator, args, output_dir: Path) -> dict:
    """pass@k of one checkpoint: K independent attempts over the same episode seeds."""
    device = accelerator.device
    policy, cfg, state_key = load_policy_for_eval(checkpoint_path, device, args)
    eval_cfg = cfg.task.evaluation
    test_start_seed = int(eval_cfg.test_start_seed)
    max_steps = int(eval_cfg.max_steps)

    all_pass_episode_successes = []
    first_summary = None
    for attempt in range(int(args.pass_at_k)):
        summary = run_image_policy_eval(
            cfg=cfg,
            policy=policy,
            output_dir=output_dir if attempt == 0 else output_dir / f"pass_{attempt}",
            num_episodes=int(args.num_episodes),
            test_start_seed=test_start_seed,
            max_steps=max_steps,
            num_envs_per_rank=int(args.num_envs_per_rank),
            checkpoint_path=str(checkpoint_path),
            policy_state_key=state_key,
            accelerator=accelerator,
        )
        if first_summary is None:
            first_summary = summary
        all_pass_episode_successes.append([ep["success"] for ep in summary["episodes"]])

    matrix = np.asarray(all_pass_episode_successes, dtype=np.float64)   # (K, n_eps)
    first_summary["pass_at_k"] = int(args.pass_at_k)
    first_summary["success_rate_mean"] = float(matrix.max(axis=0).mean())   # pass@K
    first_summary["per_pass_success_rates"] = [float(r) for r in matrix.mean(axis=1)]
    first_summary["scheduler"] = args.scheduler
    first_summary["num_inference_steps"] = int(args.num_inference_steps)
    first_summary["eta"] = float(args.eta) if args.scheduler == "ddim" else None
    if accelerator.is_main_process:
        save_pass_at_k_plot(all_pass_episode_successes, output_dir / "pass_at_k_stats.png", test_start_seed)
    return first_summary


def plot_across_epochs(all_results: dict, output_dir: Path, args) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epoch_data = sorted(
        (int(name.split("_")[1]), data) for name, data in all_results.items() if name.startswith("epoch_")
    )
    if not epoch_data:
        return
    epochs = [e for e, _ in epoch_data]
    pass_at_k_rates = [d["success_rate_mean"] for _, d in epoch_data]
    avg_pass1_rates = [float(np.mean(d["per_pass_success_rates"])) for _, d in epoch_data]
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(epochs, pass_at_k_rates, "o-", color="tab:blue", label=f"pass@{args.pass_at_k}", linewidth=2, markersize=5)
    ax.plot(epochs, avg_pass1_rates, "s-", color="tab:orange", label="avg pass@1", linewidth=2, markersize=5)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Success Rate", fontsize=12)
    ax.set_title(f"Eval ({args.scheduler}, {args.num_inference_steps} steps)", fontsize=14)
    ax.legend(fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(output_dir / "eval_across_epochs.png"), dpi=150, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="pass@k evaluation of checkpoints")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run_dir", type=str, help="Run directory (evaluates every checkpoints/epoch_*.ckpt)")
    group.add_argument("--checkpoints", type=str, nargs="+", help="Specific checkpoint paths")
    parser.add_argument("--pass_at_k", type=int, default=10, help="independent attempts per episode seed")
    parser.add_argument("--num_episodes", type=int, default=500, help="held-out environment seeds")
    parser.add_argument("--num_envs_per_rank", type=int, default=24)
    parser.add_argument("--scheduler", type=str, default="ddim", choices=["ddpm", "ddim"])
    parser.add_argument("--num_inference_steps", type=int, default=10)
    parser.add_argument("--eta", type=float, default=0.0, help="DDIM eta: 0 = deterministic, 1 = stochastic")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--min_epoch", type=int, default=0, help="skip epoch_XXXX checkpoints below this epoch")
    parser.add_argument("--max_epoch", type=int, default=None, help="skip epoch_XXXX checkpoints above this epoch")
    args = parser.parse_args()

    accelerator = Accelerator()

    if args.run_dir:
        run_dir = Path(args.run_dir)
        ckpt_paths = sorted((run_dir / "checkpoints").glob("epoch_*.ckpt"))
        assert ckpt_paths, f"No checkpoints found in {run_dir / 'checkpoints'}"
        eta_str = f"{args.eta}" if args.scheduler == "ddim" else "na"
        base_output_dir = Path(
            args.output_dir
            or run_dir / f"eval_{args.scheduler}_num_steps_{args.num_inference_steps}_eta_{eta_str}_pass_{args.pass_at_k}"
        )
    else:
        ckpt_paths = [Path(p) for p in args.checkpoints]
        assert args.output_dir is not None, "--checkpoints requires --output_dir"
        base_output_dir = Path(args.output_dir)

    def _epoch_num(p: Path):
        m = re.match(r"epoch_(\d+)$", p.stem)
        return int(m.group(1)) if m else None

    max_ep = float("inf") if args.max_epoch is None else args.max_epoch
    ckpt_paths = [
        p for p in ckpt_paths if _epoch_num(p) is None or args.min_epoch <= _epoch_num(p) <= max_ep
    ]

    if accelerator.is_main_process:
        print(f"Scheduler: {args.scheduler} ({args.num_inference_steps} steps, eta {args.eta})")
        print(
            f"pass@{args.pass_at_k}, {args.num_episodes} episodes, "
            f"{args.num_envs_per_rank} envs/rank, {accelerator.num_processes} GPUs"
        )
        print(f"Checkpoints: {len(ckpt_paths)}")
        print(f"Output: {base_output_dir}")

    all_results = {}
    for ckpt_path in ckpt_paths:
        ckpt_name = ckpt_path.stem
        output_dir = base_output_dir / ckpt_name
        if accelerator.is_main_process:
            print(f"\n{'=' * 60}\nEvaluating: {ckpt_path.name}", flush=True)
        summary = eval_single_checkpoint(ckpt_path, accelerator, args, output_dir)
        if accelerator.is_main_process:
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(output_dir / "summary.json", "w") as f:
                json.dump(summary, f, indent=2)
            rate = summary["success_rate_mean"]
            per_pass = summary["per_pass_success_rates"]
            print(f"  pass@{args.pass_at_k}: {rate:.3f}, per-pass: {[f'{r:.3f}' for r in per_pass]}", flush=True)
            all_results[ckpt_name] = {
                "success_rate_mean": rate,
                "per_pass_success_rates": per_pass,
            }

    if accelerator.is_main_process and all_results:
        with open(base_output_dir / "all_results.json", "w") as f:
            json.dump(all_results, f, indent=2)
        plot_across_epochs(all_results, base_output_dir, args)
        print(f"\nAll results saved to {base_output_dir}")


if __name__ == "__main__":
    main()
