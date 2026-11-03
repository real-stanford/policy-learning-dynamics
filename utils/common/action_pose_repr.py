from __future__ import annotations

from typing import Final

SUPPORTED_ACTION_POSE_REPRS: Final[tuple[str, ...]] = (
    "delta",
    "relative",
    "absolute",
)
SUPPORTED_ROTATION_REPS: Final[tuple[str, ...]] = (
    "axis_angle",
    "rotation_6d",
    # "none": the action carries no rotation block (planar Push-T: 2-d target position).
    # Additive; every existing code path keyed on "axis_angle" / "rotation_6d" is untouched,
    # and an absolute action with rotation_rep "none" takes the plain range-normalizer branch.
    "none",
)


def validate_action_pose_repr(action_pose_repr: str) -> str:
    value = str(action_pose_repr)
    if value not in SUPPORTED_ACTION_POSE_REPRS:
        raise ValueError(
            f"Unsupported action_pose_repr '{value}'. Expected one of "
            f"{SUPPORTED_ACTION_POSE_REPRS}."
        )
    return value


def validate_rotation_rep(rotation_rep: str) -> str:
    value = str(rotation_rep)
    if value not in SUPPORTED_ROTATION_REPS:
        raise ValueError(
            f"Unsupported rotation_rep '{value}'. Expected one of "
            f"{SUPPORTED_ROTATION_REPS}."
        )
    return value


def is_absolute_action_pose_repr(action_pose_repr: str) -> bool:
    return validate_action_pose_repr(action_pose_repr) == "absolute"


def pose_repr_from_action_mode(action_mode: str) -> str:
    value = str(action_mode)
    if value == "delta":
        return "delta"
    if value == "relative":
        return "relative"
    if value == "absolute":
        return "absolute"
    raise ValueError(
        f"Unsupported legacy action_mode '{value}'. Expected one of "
        "('delta', 'relative', 'absolute')."
    )
