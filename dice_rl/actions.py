from __future__ import annotations

import numpy as np

from utils.common.action_pose_repr import (
    validate_action_pose_repr,
    validate_rotation_rep,
)
from utils.common.rotation import rotation_6d_to_rotvec


def rotation6d_actions_to_axis_angle(action: np.ndarray) -> np.ndarray:
    """(..., 10) [pos3, rot6d, gripper1] -> (..., 7) [pos3, rotvec3, gripper1]."""
    assert action.shape[-1] == 10, (
        f"Expected 10-dim single-arm rotation-6d actions, got {action.shape}."
    )
    pos = action[..., :3]
    rot6d = action[..., 3:9]
    gripper = action[..., 9:10]
    rotvec = rotation_6d_to_rotvec(rot6d)
    return np.concatenate((pos, rotvec, gripper), axis=-1).astype(np.float32)


class EnvActionConverter:
    """Raw (unnormalized) policy actions -> robosuite controller actions."""

    def __init__(self, *, action_pose_repr: str, rotation_rep: str) -> None:
        self.action_pose_repr = validate_action_pose_repr(action_pose_repr)
        self.rotation_rep = validate_rotation_rep(rotation_rep)
        assert self.action_pose_repr in {"absolute", "delta"}, (
            "dice_rl supports absolute or delta pose actions only, got "
            f"'{self.action_pose_repr}'."
        )

    @property
    def env_action_dim(self) -> int:
        return 7

    def to_env(self, action: np.ndarray) -> np.ndarray:
        prepared = np.asarray(action, dtype=np.float32)
        if self.rotation_rep == "rotation_6d":
            prepared = rotation6d_actions_to_axis_angle(prepared)
        assert prepared.shape[-1] == self.env_action_dim, (
            f"Expected {self.env_action_dim}-dim env actions, got {prepared.shape}."
        )
        return prepared.astype(np.float32)
