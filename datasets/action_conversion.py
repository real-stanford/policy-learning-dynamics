"""Action representation conversion used by the MimicGen HDF5 -> zarr converter.

Source datasets store delta axis-angle actions. The canonical zarrs used by every
experiment in this repo store ABSOLUTE end-effector poses with a rotation-6d block:
``[pos (3) | rot6d (6) | gripper (1)]`` per arm. The absolute pose is recovered by
replaying the controller (``RobomimicAbsoluteActionConverter``); the rotation block is
then re-expressed as the first two columns of the rotation matrix.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from datasets.absolute_action_converter import RobomimicAbsoluteActionConverter
from utils.common.action_pose_repr import validate_action_pose_repr
from utils.common.rotation import matrix_to_rotation_6d


def _axis_angle_to_rotation6d(rotvec: np.ndarray) -> np.ndarray:
    matrices = Rotation.from_rotvec(rotvec.reshape(-1, 3)).as_matrix()
    rot6d = matrix_to_rotation_6d(matrices).reshape(rotvec.shape[:-1] + (6,))
    return rot6d.astype(np.float32)


def convert_actions(
    raw_actions: np.ndarray,
    *,
    action_pose_repr: str,
    rotation_rep: str,
    absolute_actions: np.ndarray | None = None,
) -> np.ndarray:
    resolved_action_pose_repr = validate_action_pose_repr(action_pose_repr)
    raw_actions = np.asarray(raw_actions, dtype=np.float32)
    if resolved_action_pose_repr == "delta":
        assert rotation_rep == "axis_angle", (
            f"Delta actions must use axis_angle, got '{rotation_rep}'."
        )
        return raw_actions
    if resolved_action_pose_repr == "relative":
        raise ValueError(
            "action_pose_repr='relative' is sequence-dependent and cannot be stored "
            "as a per-timestep canonical dataset action in this pipeline. "
            "Use action_pose_repr='delta' or 'absolute' during conversion."
        )

    assert absolute_actions is not None, (
        "action_pose_repr='absolute' requires true absolute controller actions. "
        "Pass absolute_actions instead of reinterpreting delta actions."
    )
    absolute_actions = np.asarray(absolute_actions, dtype=np.float32)
    assert absolute_actions.shape == raw_actions.shape, (
        f"Expected absolute actions to match raw shape {raw_actions.shape}, got "
        f"{absolute_actions.shape}."
    )
    if rotation_rep == "axis_angle":
        return absolute_actions
    if rotation_rep != "rotation_6d":
        raise ValueError(
            f"Unsupported absolute-action rotation representation '{rotation_rep}'."
        )
    if absolute_actions.shape[-1] not in (7, 14):
        raise ValueError(
            f"Expected 7D or 14D absolute axis-angle actions, got shape "
            f"{absolute_actions.shape}."
        )

    dual_arm = absolute_actions.shape[-1] == 14
    actions = (
        absolute_actions.reshape(-1, 2, 7)
        if dual_arm
        else absolute_actions.reshape(-1, 1, 7)
    )
    pos = actions[..., :3]
    rot = actions[..., 3:6]
    grip = actions[..., 6:7]
    rot6d = _axis_angle_to_rotation6d(rot)
    converted = np.concatenate([pos, rot6d, grip], axis=-1).astype(np.float32)
    if dual_arm:
        return converted.reshape(absolute_actions.shape[0], 20)
    return converted.reshape(absolute_actions.shape[0], 10)


def convert_action_sequence(
    states: np.ndarray,
    raw_actions: np.ndarray,
    *,
    action_pose_repr: str,
    rotation_rep: str,
    absolute_action_converter: RobomimicAbsoluteActionConverter | None = None,
    model_xml: str | bytes | None = None,
) -> np.ndarray:
    resolved_action_pose_repr = validate_action_pose_repr(action_pose_repr)
    if resolved_action_pose_repr != "absolute":
        return convert_actions(
            raw_actions,
            action_pose_repr=resolved_action_pose_repr,
            rotation_rep=rotation_rep,
        )
    assert absolute_action_converter is not None, (
        "action_pose_repr='absolute' requires an initialized "
        "RobomimicAbsoluteActionConverter."
    )
    absolute_actions = absolute_action_converter.convert_actions(
        states,
        raw_actions,
        model_xml=model_xml,
    )
    return convert_actions(
        raw_actions,
        action_pose_repr="absolute",
        rotation_rep=rotation_rep,
        absolute_actions=absolute_actions,
    )
