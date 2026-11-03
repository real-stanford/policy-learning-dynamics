from __future__ import annotations

import os

import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
import robosuite
import torch

from envs.image_parallel.spec import ImageEnvSpec
from utils.common.action_pose_repr import is_absolute_action_pose_repr
from utils.common.egl_device import cuda_to_egl_device_index
from utils.training_utils.robomimic_image_wrapper import RobomimicImageWrapper


def ensure_robosuite_env_registered(env_name: str) -> None:
    """MimicGen registers its environments with robosuite on import."""
    assert env_name, "Environment name must be non-empty."
    if env_name in robosuite.ALL_ENVIRONMENTS:
        return
    import mimicgen  # noqa: F401

    assert env_name in robosuite.ALL_ENVIRONMENTS, (
        f"Environment '{env_name}' is not registered in robosuite even after "
        "importing mimicgen."
    )


def _configure_mujoco_egl_device_id() -> None:
    if os.environ.get("MUJOCO_EGL_DEVICE_ID") is not None:
        return

    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices:
        visible_device_ids = [int(value) for value in visible_devices.split(",")]
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        assert 0 <= local_rank < len(visible_device_ids), (
            f"LOCAL_RANK={local_rank} is out of range for CUDA_VISIBLE_DEVICES="
            f"{visible_devices}."
        )
        # MUJOCO_EGL_DEVICE_ID indexes EGL's device list, whose order need not
        # match CUDA's; translate so rendering lands on this rank's own GPU.
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(
            cuda_to_egl_device_index(visible_device_ids[local_rank])
        )
        return

    if torch.cuda.is_available():
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(
            cuda_to_egl_device_index(torch.cuda.current_device())
        )


def create_image_env(spec: ImageEnvSpec) -> RobomimicImageWrapper:
    _configure_mujoco_egl_device_id()
    aux_obs_keys = list(spec.aux_obs_keys)
    # order-preserving union; with no aux keys this is exactly the state key list
    low_dim_keys = list(dict.fromkeys(list(spec.state_obs_keys) + aux_obs_keys))
    ObsUtils.initialize_obs_modality_mapping_from_dict({"low_dim": low_dim_keys})
    env_meta = FileUtils.get_env_metadata_from_dataset(spec.dataset_path)
    ensure_robosuite_env_registered(str(env_meta["env_name"]))
    if is_absolute_action_pose_repr(spec.action_pose_repr):
        env_meta["env_kwargs"]["controller_configs"]["control_delta"] = False
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=True,
        use_image_obs=True,
    )
    return RobomimicImageWrapper(
        env=env,
        state_obs_keys=list(spec.state_obs_keys),
        render_camera_names=list(spec.render_camera_names),
        init_state=None,
        render_hw=tuple(spec.render_hw),
        aux_obs_keys=aux_obs_keys or None,
    )
