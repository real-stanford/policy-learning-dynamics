"""Train the joint (image-obs, state, action) generative model on the policy's windows.

    source set_env.sh
    CUDA_VISIBLE_DEVICES=0,1 accelerate launch --multi_gpu --num_processes 2 \
        --mixed_precision bf16 --dynamo_backend no \
        -m scripts.training.joint_generator_image \
        task=mimicgen/mug_cleanup_d1_image action=absolute_rotation6d \
        dataset.processing.max_train_episodes=200 \
        training.dataloader.batch_size=128 training.training.gradient_accumulate_every=2 \
        --exp_suffix joint_gen_img

The dataset split parameters (val_ratio, max_train_episodes, seed) are shared with the
policy config, so the generator trains on exactly the policy's 200 demos. Checkpoints are
written every ``save_every`` epochs; pick the mixture with scripts/eval_joint_gen_nn_ratio.py.
"""

from __future__ import annotations

import os
import pathlib

import hydra
from omegaconf import OmegaConf

from datasets.canonical_zarr_dataset import CanonicalImageDataset
from models.generative.joint_image_oa_dit import (
    JointImageOADiTModel,
    build_joint_image_model_from_cfg,
)
from utils.training_utils.factories import build_noise_scheduler
from utils.training_utils.generator_trainer import GeneratorTrainer
from utils.training_utils.runtime import (
    configure_torch_runtime,
    consume_training_cli_args,
    register_training_resolvers,
)

register_training_resolvers()


def build_joint_image_dataset(cfg) -> CanonicalImageDataset:
    sampling_cfg = cfg.dataset.sampling
    processing_cfg = cfg.dataset.processing
    return CanonicalImageDataset(
        zarr_path=str(cfg.dataset.source.zarr_path),
        horizon=int(sampling_cfg.horizon),
        pad_before=int(sampling_cfg.pad_before),
        pad_after=int(sampling_cfg.pad_after),
        n_obs_steps=int(sampling_cfg.n_obs_steps),
        seed=int(processing_cfg.seed),
        val_ratio=float(processing_cfg.val_ratio),
        ratio=float(processing_cfg.ratio),
        max_train_episodes=processing_cfg.max_train_episodes,
        action_pose_repr=str(processing_cfg.action_pose_repr),
        # .get -> configs pickled into older generator checkpoints may lack these keys.
        episode_include_path=processing_cfg.get("episode_include_path", None),
        in_memory=bool(processing_cfg.get("in_memory", False)),
    )


def build_joint_image_model(cfg) -> JointImageOADiTModel:
    return build_joint_image_model_from_cfg(cfg, build_noise_scheduler(cfg))


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).resolve().parents[2] / "configs"),
    config_name="joint_generator_image",
)
def main(cfg) -> None:
    OmegaConf.resolve(cfg)
    configure_torch_runtime()

    output_dir = pathlib.Path(str(cfg.runtime.output_dir)).expanduser().resolve()
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        print(f"Run directory: {output_dir}", flush=True)
        print(f"Checkpoint directory: {output_dir / 'checkpoints'}", flush=True)
        print(f"Log file: {output_dir / 'logs.json.txt'}", flush=True)

    dataset = build_joint_image_dataset(cfg)
    model = build_joint_image_model(cfg)
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        n_params = sum(p.numel() for p in model.parameters())
        print(
            f"Joint image generator: rgb={model.rgb_shape} state_dim={model.state_dim} "
            f"action_dim={model.action_dim} horizon={model.horizon} "
            f"img_tokens={model.num_img_tokens} params={n_params / 1e6:.2f}M",
            flush=True,
        )
    trainer = GeneratorTrainer(cfg=cfg, model=model, dataset=dataset, output_dir=output_dir)
    trainer.run()


if __name__ == "__main__":
    consume_training_cli_args()
    main()
