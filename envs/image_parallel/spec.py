from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ImageEnvSpec:
    dataset_path: str
    state_obs_keys: tuple[str, ...]
    render_camera_names: tuple[str, ...]
    render_hw: tuple[int, int]
    action_pose_repr: str
    # Extra low-dim observation keys returned per frame as "aux_states", built
    # from the SAME raw observation dict as "states" (dice_rl asymmetric critic).
    # Empty (default) -> wrapper outputs are byte-identical to before.
    aux_obs_keys: tuple[str, ...] = ()


@dataclass(frozen=True)
class ImageStepResult:
    observation: object
    reward: float
    done: bool
