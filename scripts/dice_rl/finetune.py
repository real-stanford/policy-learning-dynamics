"""DICE-RL residual finetuning of a pretrained diffusion-policy checkpoint.

    source set_env.sh
    CUDA_VISIBLE_DEVICES=0,1 accelerate launch --multi_gpu --num_processes 2 \
        --num_machines 1 --mixed_precision no --dynamo_backend no \
        -m scripts.dice_rl.finetune \
        base_policy.checkpoint=outputs/training/threading_d2_image/<run>/checkpoints/epoch_0300.ckpt \
        --exp_suffix ep300

``--smoke`` runs a tiny end-to-end loop (see ``smoke:`` in the config).
"""

from __future__ import annotations

import os
import pathlib
import sys

import hydra
from omegaconf import OmegaConf

from utils.training_utils.runtime import (
    configure_torch_runtime,
    register_training_resolvers,
    sanitize_exp_suffix,
)


def _checkpoint_task_dirname(checkpoint: str) -> str:
    """outputs/training/<task>_<modality>/<run>/checkpoints/<ckpt> -> <task>_<modality>."""
    path = pathlib.Path(str(checkpoint)).expanduser()
    assert path.parent.name == "checkpoints", (
        f"Expected <run_dir>/checkpoints/<ckpt>, got {path}"
    )
    return path.parent.parent.parent.name


register_training_resolvers()
OmegaConf.register_new_resolver("ckpt_task", _checkpoint_task_dirname, replace=True)


def _consume_cli_args() -> None:
    argv = list(sys.argv)
    passthrough = [argv[0]]
    index = 1
    while index < len(argv):
        arg = argv[index]
        if arg == "--exp_suffix":
            assert index + 1 < len(argv), "--exp_suffix requires a value."
            passthrough.append(f"runtime.exp_suffix={sanitize_exp_suffix(argv[index + 1])}")
            index += 2
            continue
        if arg in {"--smoke", "--smoke-test"}:
            passthrough.append("smoke.enabled=true")
            index += 1
            continue
        passthrough.append(arg)
        index += 1
    sys.argv[:] = passthrough


def apply_smoke_overrides(cfg) -> None:
    if not bool(cfg.smoke.enabled):
        return
    smoke = cfg.smoke
    cfg.train.num_train_iters = int(smoke.num_train_iters)
    cfg.env.n_envs = int(smoke.n_envs)
    cfg.env.max_episode_steps = int(smoke.max_episode_steps)
    cfg.train.gradient_steps = int(smoke.gradient_steps)
    cfg.train.batch_size = int(smoke.batch_size)
    cfg.train.log_freq = int(smoke.log_freq)
    cfg.train.save_freq = int(smoke.save_freq)
    cfg.train.num_exploration_samples = int(smoke.num_exploration_samples)
    cfg.model.replay_flow_warmup_steps = int(smoke.replay_flow_warmup_steps)
    cfg.model.q_filtering_warmup_steps = int(smoke.q_filtering_warmup_steps)
    cfg.expert.adaptive_steps = int(smoke.adaptive_steps)
    cfg.logging.wandb.enabled = False


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).resolve().parents[2] / "configs" / "dice_rl"),
    config_name="finetune",
)
def main(cfg) -> None:
    OmegaConf.resolve(cfg)
    configure_torch_runtime()
    apply_smoke_overrides(cfg)
    output_dir = pathlib.Path(str(cfg.runtime.output_dir)).expanduser().resolve()
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        print(f"Run directory: {output_dir}", flush=True)

    from dice_rl.trainer import DiceRLTrainer

    trainer = DiceRLTrainer(cfg=cfg, output_dir=output_dir)
    summary = trainer.run()
    if trainer.is_main:
        print(f"Finished: {summary}", flush=True)


if __name__ == "__main__":
    _consume_cli_args()
    main()
