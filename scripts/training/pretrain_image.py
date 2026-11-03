"""Behaviour cloning of an image diffusion policy.

    source set_env.sh
    CUDA_VISIBLE_DEVICES=0,1 accelerate launch --multi_gpu --num_processes 2 \
        --num_machines 1 --mixed_precision no --dynamo_backend no \
        -m scripts.training.pretrain_image task=mimicgen/threading_d2_image \
        runtime.exp_suffix=bc

Training only writes checkpoints (every ``save_every`` epochs). Evaluate them
afterwards with ``scripts/eval_checkpoints.py``.
"""

from __future__ import annotations

import os
import pathlib

import hydra
from omegaconf import OmegaConf
from torch.distributed.elastic.multiprocessing.errors import record

from utils.training_utils.factories import build_dataset, build_output_dir, build_policy
from utils.training_utils.runtime import (
    apply_smoke_overrides,
    configure_torch_runtime,
    consume_training_cli_args,
    register_training_resolvers,
)
from utils.training_utils.supervised_trainer import SupervisedTrainer

register_training_resolvers()


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).resolve().parents[2] / "configs"),
    config_name="diffusion_image",
)
@record
def main(cfg) -> None:
    OmegaConf.resolve(cfg)
    configure_torch_runtime()
    apply_smoke_overrides(cfg)
    output_dir = build_output_dir(cfg)
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        print(f"Run directory: {output_dir}", flush=True)
        print(f"Checkpoint directory: {output_dir / 'checkpoints'}", flush=True)
        print(f"Log file: {output_dir / 'logs.json.txt'}", flush=True)
    dataset = build_dataset(cfg)
    policy = build_policy(cfg)
    trainer = SupervisedTrainer(cfg=cfg, policy=policy, dataset=dataset, output_dir=output_dir)
    trainer.run()


if __name__ == "__main__":
    consume_training_cli_args()
    main()
