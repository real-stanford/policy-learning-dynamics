"""Build the dataset, policy and optimizer from a composed Hydra config."""

from __future__ import annotations

from pathlib import Path

import torch
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from datasets.canonical_zarr_dataset import CanonicalImageDataset
from envs.task_registry import get_task_entry
from models.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
from models.vision.model_getter import get_resnet
from models.vision.multi_image_obs_encoder import MultiImageObsEncoder
from utils.common.action_pose_repr import validate_action_pose_repr


def _validate_task_action_contract(cfg) -> None:
    action_pose_repr = validate_action_pose_repr(str(cfg.action.action_pose_repr))
    dataset_action_pose_repr = validate_action_pose_repr(
        str(cfg.dataset.processing.action_pose_repr)
    )
    assert action_pose_repr == dataset_action_pose_repr, (
        "Config mismatch: action.action_pose_repr does not match "
        "dataset.processing.action_pose_repr."
    )


def _resolve_task_family_from_benchmark(benchmark: str) -> str:
    if benchmark.startswith("mimicgen"):
        return "mimicgen"
    raise ValueError(
        f"Unsupported task benchmark '{benchmark}'. Cannot resolve task family."
    )


def _validate_image_task_registry_contract(cfg) -> None:
    benchmark = str(cfg.task.benchmark)
    family = _resolve_task_family_from_benchmark(benchmark)
    task_name = str(cfg.task.task_name)
    entry = get_task_entry(family, task_name)
    shape_meta = cfg.task.observation.shape_meta
    configured_state_dim = int(shape_meta.obs.states.shape[0])
    expected_state_dim = int(entry.image_lowdim_obs_dim)
    assert configured_state_dim == expected_state_dim, (
        "Task image config mismatch with registry for state dim. "
        f"Expected {expected_state_dim}, got {configured_state_dim} for "
        f"{family}/{task_name}."
    )
    configured_rgb_channels = int(shape_meta.obs.rgb.shape[0])
    expected_rgb_channels = int(3 * len(entry.rgb_obs_keys))
    assert configured_rgb_channels == expected_rgb_channels, (
        "Task image config mismatch with registry for rgb channels. "
        f"Expected {expected_rgb_channels}, got {configured_rgb_channels} for "
        f"{family}/{task_name}."
    )
    configured_rgb_height = int(shape_meta.obs.rgb.shape[1])
    configured_rgb_width = int(shape_meta.obs.rgb.shape[2])
    expected_rgb_height, expected_rgb_width = entry.image_size
    assert configured_rgb_height == int(expected_rgb_height), (
        "Task image config mismatch with registry for rgb height. "
        f"Expected {expected_rgb_height}, got {configured_rgb_height} for "
        f"{family}/{task_name}."
    )
    assert configured_rgb_width == int(expected_rgb_width), (
        "Task image config mismatch with registry for rgb width. "
        f"Expected {expected_rgb_width}, got {configured_rgb_width} for "
        f"{family}/{task_name}."
    )
    expected_axis_angle_dim = int(cfg.task.action.axis_angle_dim)
    expected_rotation6d_dim = int(cfg.task.action.rotation6d_dim)
    assert expected_axis_angle_dim == int(entry.delta_action_dim), (
        "Task config mismatch with registry for axis-angle action dim. "
        f"Expected registry delta_action_dim={entry.delta_action_dim}, got "
        f"cfg.task.action.axis_angle_dim={expected_axis_angle_dim} for "
        f"{family}/{task_name}."
    )
    assert expected_rotation6d_dim == int(entry.absolute_rotation6d_action_dim), (
        "Task config mismatch with registry for rotation-6d action dim. "
        f"Expected registry absolute_rotation6d_action_dim="
        f"{entry.absolute_rotation6d_action_dim}, got "
        f"cfg.task.action.rotation6d_dim={expected_rotation6d_dim} for "
        f"{family}/{task_name}."
    )


def build_dataset(cfg) -> CanonicalImageDataset:
    _validate_task_action_contract(cfg)
    assert str(cfg.task.modality) == "image", (
        f"Only image tasks are supported, got modality '{cfg.task.modality}'."
    )
    _validate_image_task_registry_contract(cfg)
    dataset_cfg = cfg.dataset
    sampling_cfg = dataset_cfg.sampling
    processing_cfg = dataset_cfg.processing
    return CanonicalImageDataset(
        zarr_path=dataset_cfg.source.zarr_path,
        horizon=sampling_cfg.horizon,
        pad_before=sampling_cfg.pad_before,
        pad_after=sampling_cfg.pad_after,
        n_obs_steps=sampling_cfg.n_obs_steps,
        seed=processing_cfg.seed,
        val_ratio=processing_cfg.val_ratio,
        ratio=processing_cfg.ratio,
        max_train_episodes=processing_cfg.max_train_episodes,
        action_pose_repr=processing_cfg.action_pose_repr,
        # .get -> configs pickled into older checkpoints may lack these keys.
        episode_include_path=processing_cfg.get("episode_include_path", None),
        in_memory=bool(processing_cfg.get("in_memory", False)),
    )


def build_noise_scheduler(cfg):
    scheduler_cfg = cfg.model.scheduler
    scheduler_type = getattr(scheduler_cfg, "type", "ddpm")
    if scheduler_type == "ddim":
        return DDIMScheduler(
            num_train_timesteps=scheduler_cfg.num_train_timesteps,
            beta_start=scheduler_cfg.beta_start,
            beta_end=scheduler_cfg.beta_end,
            beta_schedule=scheduler_cfg.beta_schedule,
            clip_sample=scheduler_cfg.clip_sample,
            prediction_type=scheduler_cfg.prediction_type,
            set_alpha_to_one=getattr(scheduler_cfg, "set_alpha_to_one", True),
            steps_offset=getattr(scheduler_cfg, "steps_offset", 0),
        )
    return DDPMScheduler(
        num_train_timesteps=scheduler_cfg.num_train_timesteps,
        beta_start=scheduler_cfg.beta_start,
        beta_end=scheduler_cfg.beta_end,
        beta_schedule=scheduler_cfg.beta_schedule,
        variance_type=scheduler_cfg.variance_type,
        clip_sample=scheduler_cfg.clip_sample,
        prediction_type=scheduler_cfg.prediction_type,
    )


def _resolve_torchvision_weights(weights_name: str | None):
    if weights_name in {None, "", "none", "None"}:
        return None
    from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights

    return {
        "resnet18": ResNet18_Weights,
        "resnet34": ResNet34_Weights,
        "resnet50": ResNet50_Weights,
    }


def build_obs_encoder(cfg) -> MultiImageObsEncoder:
    encoder_cfg = cfg.model.obs_encoder
    backbone_name = str(encoder_cfg.backbone)
    weights_lookup = _resolve_torchvision_weights(encoder_cfg.get("weights", None))
    weights = None
    if weights_lookup is not None:
        assert backbone_name in weights_lookup, (
            f"Unsupported backbone '{backbone_name}'."
        )
        weights = getattr(weights_lookup[backbone_name], str(encoder_cfg.weights))
    rgb_model = get_resnet(backbone_name, weights=weights)
    return MultiImageObsEncoder(
        shape_meta=cfg.task.observation.shape_meta,
        rgb_model=rgb_model,
        resize_shape=encoder_cfg.resize_shape,
        crop_shape=encoder_cfg.crop_shape,
        random_crop=encoder_cfg.random_crop,
        use_group_norm=encoder_cfg.group_norm,
        share_rgb_model=encoder_cfg.share_backbone,
        imagenet_norm=encoder_cfg.imagenet_norm,
    )


def build_image_policy(cfg) -> DiffusionUnetImagePolicy:
    rollout_cfg = cfg.model.rollout
    unet_cfg = cfg.model.unet
    policy = DiffusionUnetImagePolicy(
        shape_meta=cfg.task.observation.shape_meta,
        noise_scheduler=build_noise_scheduler(cfg),
        obs_encoder=build_obs_encoder(cfg),
        horizon=rollout_cfg.horizon,
        n_action_steps=rollout_cfg.n_action_steps,
        n_obs_steps=rollout_cfg.n_obs_steps,
        num_inference_steps=cfg.model.scheduler.num_inference_steps,
        obs_as_global_cond=rollout_cfg.obs_conditioning == "global",
        diffusion_step_embed_dim=unet_cfg.diffusion_step_embed_dim,
        down_dims=tuple(unet_cfg.down_dims),
        kernel_size=unet_cfg.kernel_size,
        n_groups=unet_cfg.n_groups,
        cond_predict_scale=unet_cfg.cond_predict_scale,
    )

    # Optional joint (image, state, action) generator mixing. Loads a frozen mixture of
    # JointImageOADiTModel checkpoints as a plain (non-Module) mixer, so they never enter
    # the policy state_dict / EMA / compile graph. Disabled (or absent, in configs pickled
    # into older checkpoints) -> standard training.
    oa_cfg = cfg.training.get("jointaug_w_oa_joint_model", None)
    if oa_cfg is not None and bool(oa_cfg.enabled):
        from utils.oa_joint_image_mixer import load_oa_joint_image_mixer

        obs_meta = cfg.task.observation.shape_meta.obs
        policy.oa_joint_mixer = load_oa_joint_image_mixer(
            oa_cfg,
            expected_rgb_shape=tuple(int(x) for x in obs_meta.rgb.shape),
            expected_state_dim=int(obs_meta.states.shape[0]),
            expected_action_dim=int(cfg.task.action.model_dim),
            expected_n_obs_steps=int(cfg.model.rollout.n_obs_steps),
            expected_horizon=int(cfg.model.rollout.horizon),
            map_location="cpu",
        )
    return policy


def build_policy(cfg) -> DiffusionUnetImagePolicy:
    _validate_task_action_contract(cfg)
    assert str(cfg.task.modality) == "image", (
        f"Only image policies are supported, got modality '{cfg.task.modality}'."
    )
    return build_image_policy(cfg)


def build_optimizer(cfg, policy) -> torch.optim.Optimizer:
    optimizer_cfg = cfg.training.optimizer
    assert optimizer_cfg.type == "adamw", (
        f"Unsupported optimizer type '{optimizer_cfg.type}'."
    )
    return torch.optim.AdamW(
        params=policy.parameters(),
        lr=optimizer_cfg.lr,
        betas=tuple(optimizer_cfg.betas),
        eps=optimizer_cfg.eps,
        weight_decay=optimizer_cfg.weight_decay,
    )


def build_output_dir(cfg) -> Path:
    return Path(str(cfg.runtime.output_dir)).expanduser().resolve()
