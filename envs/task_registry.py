"""Local task registry for the MimicGen datasets used in this repo.

This file is the single source of truth for supported task names, download URLs,
horizons, observation / camera conventions and action-convention metadata.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Tuple

MIMICGEN_HF_REPO = "amandlek/mimicgen_datasets"


def _mimicgen_hf_url(dataset_type: str, filename: str) -> str:
    return (
        "https://huggingface.co/datasets/"
        f"{MIMICGEN_HF_REPO}/resolve/main/{dataset_type}/{filename}"
    )


@dataclass(frozen=True)
class TaskRegistryEntry:
    """Shared task metadata used by dataset scripts and configs."""

    family: str
    task_name: str
    dataset_type: str
    relative_dir: Tuple[str, ...]
    artifact_filenames: Mapping[str, str]
    download_urls: Mapping[str, str]
    horizon: int
    upstream_action_mode: str
    upstream_rotation_rep: str
    supported_action_schemas: Tuple[str, ...]
    lowdim_obs_keys: Tuple[str, ...]
    image_lowdim_obs_keys: Tuple[str, ...]
    rgb_obs_keys: Tuple[str, ...]
    render_camera_names: Tuple[str, ...]
    image_size: Tuple[int, int] | None
    lowdim_obs_dim: int | None
    image_lowdim_obs_dim: int | None
    delta_action_dim: int
    absolute_rotation6d_action_dim: int
    notes: Tuple[str, ...] = ()

    def artifact_path(self, data_root: str | os.PathLike[str], artifact: str) -> Path:
        if artifact not in self.artifact_filenames:
            raise KeyError(
                f"Artifact '{artifact}' is not defined for {self.family}:{self.task_name}."
            )
        return (
            Path(data_root)
            .expanduser()
            .resolve()
            .joinpath(*self.relative_dir)
            .joinpath(self.artifact_filenames[artifact])
        )

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _mimicgen_core_image_task(
    task_name: str,
    horizon: int,
    *,
    lowdim_obs_dim: int,
) -> TaskRegistryEntry:
    filename = f"{task_name}.hdf5"
    return TaskRegistryEntry(
        family="mimicgen",
        task_name=task_name,
        dataset_type="core",
        relative_dir=("core",),
        artifact_filenames={
            "hdf5": filename,
            "lowdim_zarr": f"{task_name}_lowdim.zarr",
            "image_zarr": f"{task_name}_image.zarr",
        },
        download_urls={"hdf5": _mimicgen_hf_url("core", filename)},
        horizon=horizon,
        upstream_action_mode="delta",
        upstream_rotation_rep="axis_angle",
        supported_action_schemas=("delta_axis_angle", "absolute_rotation6d"),
        lowdim_obs_keys=(
            "object",
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
        ),
        image_lowdim_obs_keys=(
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
        ),
        rgb_obs_keys=(
            "agentview_image",
            "robot0_eye_in_hand_image",
        ),
        render_camera_names=(
            "agentview",
            "robot0_eye_in_hand",
        ),
        image_size=(84, 84),
        lowdim_obs_dim=lowdim_obs_dim,
        image_lowdim_obs_dim=9,
        delta_action_dim=7,
        absolute_rotation6d_action_dim=10,
        notes=(
            "Public MimicGen releases are downloaded as single HDF5 files.",
            "Absolute rotation_6d variants are derived from the upstream delta-action HDF5s.",
        ),
    )


MIMICGEN_TASK_REGISTRY: Dict[str, TaskRegistryEntry] = {
    "square_d2": _mimicgen_core_image_task("square_d2", 400, lowdim_obs_dim=26),
    "threading_d2": _mimicgen_core_image_task("threading_d2", 400, lowdim_obs_dim=37),
    "three_piece_assembly_d1": _mimicgen_core_image_task(
        "three_piece_assembly_d1", 500, lowdim_obs_dim=51
    ),
    "hammer_cleanup_d1": _mimicgen_core_image_task(
        "hammer_cleanup_d1", 500, lowdim_obs_dim=37
    ),
    "mug_cleanup_d1": _mimicgen_core_image_task("mug_cleanup_d1", 500, lowdim_obs_dim=38),
}


TASK_REGISTRIES: Dict[str, Dict[str, TaskRegistryEntry]] = {
    "mimicgen": MIMICGEN_TASK_REGISTRY,
}


def get_family_data_root(family: str, explicit_root: str | None = None) -> Path:
    if explicit_root:
        return Path(explicit_root).expanduser().resolve()
    env_names = {"mimicgen": "MIMICGEN_DATA_ROOT"}
    assert family in env_names, f"Unsupported family '{family}'."
    env_name = env_names[family]
    assert env_name in os.environ, (
        f"Expected environment variable '{env_name}' to be set for family '{family}'."
    )
    return Path(os.environ[env_name]).expanduser().resolve()


def get_task_entry(family: str, task_name: str) -> TaskRegistryEntry:
    assert family in TASK_REGISTRIES, f"Unsupported family '{family}'."
    family_registry = TASK_REGISTRIES[family]
    if task_name not in family_registry:
        supported = ", ".join(sorted(family_registry))
        raise KeyError(
            f"Unsupported task '{task_name}' for family '{family}'. Supported: {supported}"
        )
    return family_registry[task_name]


def list_task_names(family: str) -> Tuple[str, ...]:
    return tuple(sorted(TASK_REGISTRIES[family]))


def infer_task_name_from_path(family: str, path: str | os.PathLike[str]) -> str | None:
    assert family == "mimicgen", f"Unsupported family '{family}'."
    candidate = Path(path).expanduser().resolve().stem
    if candidate in MIMICGEN_TASK_REGISTRY:
        return candidate
    return None


def task_entry_to_dict(entry: TaskRegistryEntry) -> Dict[str, object]:
    return entry.to_dict()


def iter_registry_entries(family: str) -> Iterable[TaskRegistryEntry]:
    return TASK_REGISTRIES[family].values()
