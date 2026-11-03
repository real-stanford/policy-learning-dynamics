"""Canonical storage schema definitions for supervised diffusion datasets.

This module defines the minimal zarr contract for the root repository.

Design goals:

1. support supervised imitation only,
2. work for both robomimic and MimicGen,
3. stay minimal for low-dimensional training,
4. keep one optional image key for future image baselines.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

from utils.common.action_pose_repr import pose_repr_from_action_mode


@dataclass(frozen=True)
class ArraySpec:
    """Description of one stored array."""

    key: str
    dtype: str
    axes: Tuple[str, ...]
    description: str


# ---------------------------------------------------------------------------
# Canonical supervised zarr schema
# ---------------------------------------------------------------------------

CANONICAL_STEP_ARRAYS: Tuple[ArraySpec, ...] = (
    ArraySpec(
        key="states",
        dtype="float32",
        axes=("T", "obs_dim"),
        description="Low-dimensional state vector per timestep.",
    ),
    ArraySpec(
        key="actions",
        dtype="float32",
        axes=("T", "action_dim"),
        description="Action vector per timestep.",
    ),
    ArraySpec(
        key="rgb",
        dtype="uint8",
        axes=("T", "channels", "height", "width"),
        description=(
            "Optional image tensor with cameras concatenated along channels in a "
            "fixed order recorded in metadata."
        ),
    ),
    ArraySpec(
        key="rewards",
        dtype="float32",
        axes=("T",),
        description=(
            "Optional per-timestep scalar reward. For MimicGen this is the sparse "
            "binary task-completion signal (0 until solved, then 1). When the "
            "dataset is built with truncate=True each trajectory is cut to a single "
            "terminal success, so rewards are 0,...,0,1."
        ),
    ),
)

CANONICAL_EPISODE_METADATA_ARRAYS: Tuple[ArraySpec, ...] = (
    ArraySpec(
        key="episode_success",
        dtype="float32",
        axes=("1",),
        description="Episode success flag.",
    ),
    ArraySpec(
        key="episode_max_coverage",
        dtype="float32",
        axes=("1",),
        description="Episode-level task progress metric when available.",
    ),
)

CANONICAL_META_ARRAYS: Tuple[ArraySpec, ...] = (
    ArraySpec(
        key="schema_version",
        dtype="str",
        axes=(),
        description="Canonical schema version string.",
    ),
    ArraySpec(
        key="task_name",
        dtype="str",
        axes=(),
        description="Task identifier, for example transport_mh.",
    ),
    ArraySpec(
        key="observation_type",
        dtype="str",
        axes=(),
        description='Observation type string such as "low_dim" or "rgb+low_dim".',
    ),
    ArraySpec(
        key="action_pose_repr",
        dtype="str",
        axes=(),
        description='Action pose parameterization such as "delta", "relative", or "absolute".',
    ),
    ArraySpec(
        key="action_mode",
        dtype="str",
        axes=(),
        description='Legacy alias for action_pose_repr, such as "delta" or "absolute".',
    ),
    ArraySpec(
        key="rotation_rep",
        dtype="str",
        axes=(),
        description='Rotation representation such as "axis_angle" or "rotation_6d".',
    ),
    ArraySpec(
        key="obs_dim",
        dtype="int64",
        axes=(),
        description="Flattened low-dimensional observation dimension.",
    ),
    ArraySpec(
        key="action_dim",
        dtype="int64",
        axes=(),
        description="Action dimension.",
    ),
    ArraySpec(
        key="num_episodes",
        dtype="int64",
        axes=(),
        description="Number of episodes stored in the dataset.",
    ),
    ArraySpec(
        key="camera_names",
        dtype="str",
        axes=("num_cameras",),
        description="Optional ordered list of camera names used to build rgb.",
    ),
    ArraySpec(
        key="rgb_layout",
        dtype="str",
        axes=(),
        description='Optional rgb storage layout descriptor, for example "camera_concat_chw".',
    ),
)


# ---------------------------------------------------------------------------
# Legacy / reference conventions that informed the migration
# ---------------------------------------------------------------------------

LEGACY_LEARNING_WITH_PLANNER_STEP_ARRAYS: Tuple[ArraySpec, ...] = (
    ArraySpec(
        key="js_fb_0",
        dtype="float32",
        axes=("T", "joint_dim"),
        description="Robot joint or pusher feedback used as sparse observation input.",
    ),
    ArraySpec(
        key="js_command_0",
        dtype="float32",
        axes=("T", "action_dim"),
        description="Executed action sequence used as the diffusion training target.",
    ),
    ArraySpec(
        key="item_poses_0",
        dtype="float32",
        axes=("T", "pose_dim"),
        description="Raw object pose array; current PushT mapping splits this into position and yaw.",
    ),
    ArraySpec(
        key="goal_pose2d",
        dtype="float32",
        axes=("goal_dim",),
        description="Optional task-level PushT goal pose when present.",
    ),
)

LEGACY_LEARNING_WITH_PLANNER_STEP_METADATA_ARRAYS: Tuple[ArraySpec, ...] = (
    ArraySpec(
        key="action_source_t",
        dtype="int32",
        axes=("T",),
        description="Per-timestep action source identifier, used for planner-only filtering and diagnostics.",
    ),
    ArraySpec(
        key="is_correction_t",
        dtype="float32",
        axes=("T",),
        description="Per-timestep correction flag.",
    ),
    ArraySpec(
        key="value_estimate_t",
        dtype="float32",
        axes=("T",),
        description="Per-timestep scalar value estimate.",
    ),
    ArraySpec(
        key="indicator_t",
        dtype="float32",
        axes=("T",),
        description="Per-timestep indicator value used by some filtering or weighting paths.",
    ),
    ArraySpec(
        key="state_based_distance_reward",
        dtype="float32",
        axes=("T",),
        description="Per-timestep reward used for Monte Carlo advantage computation.",
    ),
    ArraySpec(
        key="q_raw_t",
        dtype="float32",
        axes=("T",),
        description="Unnormalized Q estimate.",
    ),
    ArraySpec(
        key="v_raw_t",
        dtype="float32",
        axes=("T",),
        description="Unnormalized V estimate.",
    ),
    ArraySpec(
        key="adv_raw_t",
        dtype="float32",
        axes=("T",),
        description="Unnormalized advantage estimate.",
    ),
    ArraySpec(
        key="q_normalized_t",
        dtype="float32",
        axes=("T",),
        description="Normalized Q estimate used by weighting-aware training paths.",
    ),
    ArraySpec(
        key="v_normalized_t",
        dtype="float32",
        axes=("T",),
        description="Normalized V estimate used by weighting-aware training paths.",
    ),
    ArraySpec(
        key="adv_normalized_t",
        dtype="float32",
        axes=("T",),
        description="Normalized advantage estimate used by weighting-aware training paths.",
    ),
)

LEGACY_LEARNING_WITH_PLANNER_EPISODE_METADATA_ARRAYS: Tuple[ArraySpec, ...] = (
    ArraySpec(
        key="episode_success",
        dtype="float32",
        axes=("1",),
        description="Episode success flag.",
    ),
    ArraySpec(
        key="episode_max_coverage",
        dtype="float32",
        axes=("1",),
        description="Episode-level coverage or progress metric.",
    ),
    ArraySpec(
        key="collection_mode",
        dtype="str",
        axes=(),
        description="Collection mode label for the episode.",
    ),
    ArraySpec(
        key="episode_planning_time",
        dtype="float32",
        axes=("1",),
        description="Optional total planning time for the episode.",
    ),
    ArraySpec(
        key="planner_invocations",
        dtype="int32",
        axes=("1",),
        description="Optional number of planner invocations in the episode.",
    ),
)


# ---------------------------------------------------------------------------
# Representative `dice-rl` processed dataset conventions
# ---------------------------------------------------------------------------

# These are included because the migration path needs to convert them into the
# canonical zarr layout, even though they are not currently zarr-backed.
DICE_RL_NPZ_ARRAYS: Tuple[ArraySpec, ...] = (
    ArraySpec(
        key="states",
        dtype="float32",
        axes=("sum_T", "obs_dim"),
        description="Flattened low-dimensional state array over all trajectories.",
    ),
    ArraySpec(
        key="actions",
        dtype="float32",
        axes=("sum_T", "action_dim"),
        description="Flattened action array over all trajectories.",
    ),
    ArraySpec(
        key="rewards",
        dtype="float32",
        axes=("sum_T",),
        description="Flattened reward array over all trajectories.",
    ),
    ArraySpec(
        key="terminals",
        dtype="bool",
        axes=("sum_T",),
        description="Flattened terminal flag array over all trajectories.",
    ),
    ArraySpec(
        key="traj_lengths",
        dtype="int64",
        axes=("num_trajectories",),
        description="Trajectory boundary index needed to reconstruct episodes.",
    ),
    ArraySpec(
        key="images",
        dtype="uint8",
        axes=("sum_T", "channels", "height", "width"),
        description="Optional concatenated multi-camera image array.",
    ),
)


def _spec_dict(specs: Iterable[ArraySpec]) -> Dict[str, ArraySpec]:
    return {spec.key: spec for spec in specs}


CANONICAL_STEP_ARRAYS_BY_KEY = _spec_dict(CANONICAL_STEP_ARRAYS)
CANONICAL_EPISODE_METADATA_ARRAYS_BY_KEY = _spec_dict(CANONICAL_EPISODE_METADATA_ARRAYS)
CANONICAL_META_ARRAYS_BY_KEY = _spec_dict(CANONICAL_META_ARRAYS)
DICE_RL_NPZ_ARRAYS_BY_KEY = _spec_dict(DICE_RL_NPZ_ARRAYS)


def _dtype_matches(actual_dtype, expected_dtype: str) -> bool:
    actual = str(actual_dtype)
    if expected_dtype == "str":
        return actual_dtype.kind in {"U", "S", "O"}
    if expected_dtype == "int64":
        return actual == "int64"
    if expected_dtype == "float32":
        return actual == "float32"
    if expected_dtype == "uint8":
        return actual == "uint8"
    if expected_dtype == "bool":
        return actual == "bool"
    return actual == expected_dtype


def validate_canonical_zarr(zarr_path: str | Path) -> List[str]:
    """Validate a canonical zarr store and return a list of schema errors."""

    try:
        import zarr
    except ImportError as exc:
        raise ImportError("zarr is required to validate canonical datasets.") from exc

    root = zarr.open(str(Path(zarr_path).expanduser().resolve()), mode="r")
    errors: List[str] = []

    if "data" not in root:
        errors.append("Missing top-level 'data' group.")
    if "meta" not in root:
        errors.append("Missing top-level 'meta' group.")
    if errors:
        return errors

    data_group = root["data"]
    meta_group = root["meta"]
    required_meta_keys = (
        "schema_version",
        "task_name",
        "observation_type",
        "rotation_rep",
        "obs_dim",
        "action_dim",
        "num_episodes",
    )
    for key in required_meta_keys:
        if key not in meta_group:
            errors.append(f"Missing meta key '{key}'.")
            continue
        spec = CANONICAL_META_ARRAYS_BY_KEY[key]
        if not _dtype_matches(meta_group[key].dtype, spec.dtype):
            errors.append(
                f"Meta key '{key}' has dtype {meta_group[key].dtype}, expected {spec.dtype}."
            )
    if "action_pose_repr" in meta_group:
        spec = CANONICAL_META_ARRAYS_BY_KEY["action_pose_repr"]
        if not _dtype_matches(meta_group["action_pose_repr"].dtype, spec.dtype):
            errors.append(
                "Meta key 'action_pose_repr' has dtype "
                f"{meta_group['action_pose_repr'].dtype}, expected {spec.dtype}."
            )
    elif "action_mode" in meta_group:
        spec = CANONICAL_META_ARRAYS_BY_KEY["action_mode"]
        if not _dtype_matches(meta_group["action_mode"].dtype, spec.dtype):
            errors.append(
                "Meta key 'action_mode' has dtype "
                f"{meta_group['action_mode'].dtype}, expected {spec.dtype}."
            )
        else:
            try:
                pose_repr_from_action_mode(
                    str(np.asarray(meta_group["action_mode"])[()])
                )
            except ValueError as exc:
                errors.append(str(exc))
    else:
        errors.append(
            "Missing action pose metadata. Expected 'meta/action_pose_repr' "
            "(or legacy 'meta/action_mode')."
        )

    observation_type = None
    if "observation_type" in meta_group:
        observation_type = str(np.asarray(meta_group["observation_type"])[()])
        if observation_type == "rgb+low_dim":
            for key in ("camera_names", "rgb_layout"):
                if key not in meta_group:
                    errors.append(
                        f"Image dataset is missing required meta key '{key}'."
                    )

    episode_names = sorted(
        [name for name in data_group.group_keys() if str(name).startswith("episode_")]
    )
    if "num_episodes" in meta_group:
        expected = int(np.asarray(meta_group["num_episodes"])[()])
        if expected != len(episode_names):
            errors.append(
                f"Meta num_episodes={expected} does not match stored episodes={len(episode_names)}."
            )

    for episode_name in episode_names:
        episode_group = data_group[episode_name]
        for key in ("states", "actions"):
            if key not in episode_group:
                errors.append(f"{episode_name} is missing required array '{key}'.")
        if "states" not in episode_group or "actions" not in episode_group:
            continue

        states = episode_group["states"]
        actions = episode_group["actions"]
        if states.shape[0] != actions.shape[0]:
            errors.append(
                f"{episode_name} has mismatched time dimension: states={states.shape[0]}, actions={actions.shape[0]}."
            )
        if states.ndim != 2:
            errors.append(
                f"{episode_name}/states must be rank-2, got shape {states.shape}."
            )
        if actions.ndim != 2:
            errors.append(
                f"{episode_name}/actions must be rank-2, got shape {actions.shape}."
            )
        if not _dtype_matches(states.dtype, "float32"):
            errors.append(f"{episode_name}/states must be float32, got {states.dtype}.")
        if not _dtype_matches(actions.dtype, "float32"):
            errors.append(
                f"{episode_name}/actions must be float32, got {actions.dtype}."
            )

        if observation_type == "rgb+low_dim":
            if "rgb" not in episode_group:
                errors.append(f"{episode_name} is missing required array 'rgb'.")
            else:
                rgb = episode_group["rgb"]
                if rgb.shape[0] != states.shape[0]:
                    errors.append(
                        f"{episode_name}/rgb has mismatched time dimension {rgb.shape[0]}."
                    )
                if rgb.ndim != 4:
                    errors.append(
                        f"{episode_name}/rgb must be rank-4, got shape {rgb.shape}."
                    )
                if not _dtype_matches(rgb.dtype, "uint8"):
                    errors.append(f"{episode_name}/rgb must be uint8, got {rgb.dtype}.")

    return errors


LEGACY_LEARNING_WITH_PLANNER_STEP_ARRAYS_BY_KEY = _spec_dict(
    LEGACY_LEARNING_WITH_PLANNER_STEP_ARRAYS
)
LEGACY_LEARNING_WITH_PLANNER_STEP_METADATA_ARRAYS_BY_KEY = _spec_dict(
    LEGACY_LEARNING_WITH_PLANNER_STEP_METADATA_ARRAYS
)
LEGACY_LEARNING_WITH_PLANNER_EPISODE_METADATA_ARRAYS_BY_KEY = _spec_dict(
    LEGACY_LEARNING_WITH_PLANNER_EPISODE_METADATA_ARRAYS
)


def list_canonical_zarr_keys() -> Tuple[str, ...]:
    """Return all keys in the canonical supervised zarr contract."""

    return tuple(
        spec.key
        for spec in (
            *CANONICAL_STEP_ARRAYS,
            *CANONICAL_EPISODE_METADATA_ARRAYS,
            *CANONICAL_META_ARRAYS,
        )
    )


def describe_canonical_zarr() -> Dict[str, Dict[str, ArraySpec]]:
    """Return the grouped canonical supervised zarr schema."""

    return {
        "step_arrays": CANONICAL_STEP_ARRAYS_BY_KEY,
        "episode_metadata_arrays": CANONICAL_EPISODE_METADATA_ARRAYS_BY_KEY,
        "meta_arrays": CANONICAL_META_ARRAYS_BY_KEY,
    }


def describe_legacy_learning_with_planner_zarr() -> Dict[str, Dict[str, ArraySpec]]:
    """Return the legacy zarr conventions from `learning_with_planner`."""

    return {
        "raw_step_arrays": LEGACY_LEARNING_WITH_PLANNER_STEP_ARRAYS_BY_KEY,
        "step_metadata_arrays": LEGACY_LEARNING_WITH_PLANNER_STEP_METADATA_ARRAYS_BY_KEY,
        "episode_metadata_arrays": LEGACY_LEARNING_WITH_PLANNER_EPISODE_METADATA_ARRAYS_BY_KEY,
    }


def describe_dice_rl_processed_dataset() -> Dict[str, ArraySpec]:
    """Return the representative `dice-rl` processed dataset fields."""

    return DICE_RL_NPZ_ARRAYS_BY_KEY
