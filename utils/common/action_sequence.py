from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from utils.common.action_pose_repr import validate_rotation_rep
from utils.common.rotation import matrix_to_rotation_6d, rotation_6d_to_matrix


def _pose_block_dim(rotation_rep: str) -> int:
    resolved_rotation_rep = validate_rotation_rep(rotation_rep)
    if resolved_rotation_rep == "axis_angle":
        return 7
    if resolved_rotation_rep == "rotation_6d":
        return 10
    raise ValueError(f"Unsupported rotation representation '{resolved_rotation_rep}'.")


def _reshape_pose_actions(actions: np.ndarray, rotation_rep: str) -> np.ndarray:
    block_dim = _pose_block_dim(rotation_rep)
    if actions.shape[-1] % block_dim != 0:
        raise ValueError(
            f"Action dimension {actions.shape[-1]} is not divisible by "
            f"pose block dim {block_dim} for rotation_rep='{rotation_rep}'."
        )
    num_arms = actions.shape[-1] // block_dim
    return actions.reshape(actions.shape[:-1] + (num_arms, block_dim))


def _pose_blocks_to_matrices(
    action_blocks: np.ndarray, rotation_rep: str
) -> tuple[np.ndarray, np.ndarray]:
    pos = action_blocks[..., :3]
    if rotation_rep == "axis_angle":
        rotvec = action_blocks[..., 3:6]
        matrices = Rotation.from_rotvec(rotvec.reshape(-1, 3)).as_matrix()
        rot_mat = matrices.reshape(rotvec.shape[:-1] + (3, 3))
    elif rotation_rep == "rotation_6d":
        rot_mat = rotation_6d_to_matrix(action_blocks[..., 3:9])
    else:
        raise ValueError(f"Unsupported rotation representation '{rotation_rep}'.")
    gripper = action_blocks[..., -1:]

    pose_mat = np.zeros(action_blocks.shape[:-1] + (4, 4), dtype=np.float64)
    pose_mat[..., :3, :3] = rot_mat
    pose_mat[..., :3, 3] = pos
    pose_mat[..., 3, 3] = 1.0
    return pose_mat, gripper.astype(np.float32)


def _pose_matrices_to_blocks(
    pose_mats: np.ndarray, gripper: np.ndarray, rotation_rep: str
) -> np.ndarray:
    pos = pose_mats[..., :3, 3]
    if rotation_rep == "axis_angle":
        rotvec = Rotation.from_matrix(
            pose_mats[..., :3, :3].reshape(-1, 3, 3)
        ).as_rotvec()
        rot = rotvec.reshape(pos.shape[:-1] + (3,))
    elif rotation_rep == "rotation_6d":
        rot = matrix_to_rotation_6d(pose_mats[..., :3, :3]).astype(np.float32)
    else:
        raise ValueError(f"Unsupported rotation representation '{rotation_rep}'.")
    return np.concatenate([pos, rot, gripper], axis=-1).astype(np.float32)


def relative_pose_actions_to_delta(
    relative_actions: np.ndarray,
    *,
    rotation_rep: str,
) -> np.ndarray:
    resolved_rotation_rep = validate_rotation_rep(rotation_rep)
    relative_actions = np.asarray(relative_actions, dtype=np.float32)
    if relative_actions.ndim < 2:
        raise ValueError(
            f"Expected relative actions with shape (..., T, action_dim), got "
            f"{relative_actions.shape}."
        )
    action_blocks = _reshape_pose_actions(relative_actions, resolved_rotation_rep)
    if action_blocks.shape[-3] < 1:
        raise ValueError("Action sequence must have at least one timestep.")

    pose_mats, gripper = _pose_blocks_to_matrices(action_blocks, resolved_rotation_rep)
    time_axis = pose_mats.ndim - 4
    identity_shape = (
        pose_mats.shape[:time_axis] + (1,) + pose_mats.shape[time_axis + 1 :]
    )
    identity = np.zeros(identity_shape, dtype=np.float64)
    identity[..., :3, :3] = np.eye(3, dtype=np.float64)
    identity[..., 3, 3] = 1.0
    prev_pose = np.concatenate(
        [
            identity,
            np.take(
                pose_mats, indices=range(pose_mats.shape[time_axis] - 1), axis=time_axis
            ),
        ],
        axis=time_axis,
    )
    delta_pose = np.matmul(pose_mats, np.linalg.inv(prev_pose))
    delta_blocks = _pose_matrices_to_blocks(
        delta_pose,
        gripper=gripper,
        rotation_rep=resolved_rotation_rep,
    )
    return delta_blocks.reshape(relative_actions.shape).astype(np.float32)


def delta_pose_actions_to_relative(
    delta_actions: np.ndarray,
    *,
    rotation_rep: str,
) -> np.ndarray:
    resolved_rotation_rep = validate_rotation_rep(rotation_rep)
    delta_actions = np.asarray(delta_actions, dtype=np.float32)
    if delta_actions.ndim < 2:
        raise ValueError(
            f"Expected delta actions with shape (..., T, action_dim), got "
            f"{delta_actions.shape}."
        )
    action_blocks = _reshape_pose_actions(delta_actions, resolved_rotation_rep)
    if action_blocks.shape[-3] < 1:
        raise ValueError("Action sequence must have at least one timestep.")

    delta_pose, gripper = _pose_blocks_to_matrices(action_blocks, resolved_rotation_rep)
    time_axis = delta_pose.ndim - 4
    delta_pose_time_major = np.moveaxis(delta_pose, time_axis, 0)
    relative_pose_time_major = np.zeros_like(delta_pose_time_major)

    curr_pose = np.zeros(delta_pose_time_major.shape[1:], dtype=np.float64)
    curr_pose[..., :3, :3] = np.eye(3, dtype=np.float64)
    curr_pose[..., 3, 3] = 1.0
    for step in range(delta_pose_time_major.shape[0]):
        curr_pose = np.matmul(delta_pose_time_major[step], curr_pose)
        relative_pose_time_major[step] = curr_pose

    relative_pose = np.moveaxis(relative_pose_time_major, 0, time_axis)
    relative_blocks = _pose_matrices_to_blocks(
        relative_pose,
        gripper=gripper,
        rotation_rep=resolved_rotation_rep,
    )
    return relative_blocks.reshape(delta_actions.shape).astype(np.float32)
