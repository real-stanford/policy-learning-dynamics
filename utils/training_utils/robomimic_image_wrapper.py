from __future__ import annotations

from typing import Optional

import numpy as np
from robomimic.envs.env_robosuite import EnvRobosuite


class RobomimicImageWrapper:
    def __init__(
        self,
        env: EnvRobosuite,
        state_obs_keys: list[str],
        render_camera_names: list[str],
        init_state: Optional[np.ndarray | dict[str, object]] = None,
        render_hw: tuple[int, int] = (84, 84),
        aux_obs_keys: Optional[list[str]] = None,
    ) -> None:
        self.env = env
        self.state_obs_keys = list(state_obs_keys)
        self.render_camera_names = list(render_camera_names)
        self.init_state = init_state
        self.render_hw = tuple(render_hw)
        # Optional extra low-dim keys returned as "aux_states", built from the
        # SAME raw_obs dict as "states" (same step-sampled convention as the
        # recorded demos). None (default) -> outputs unchanged.
        self.aux_obs_keys = None if not aux_obs_keys else list(aux_obs_keys)
        self.seed_state_map: dict[int, dict[str, object]] = {}
        self._seed: int | None = None

    @staticmethod
    def _copy_reset_state(reset_state: dict[str, object]) -> dict[str, object]:
        copied = {}
        for key, value in reset_state.items():
            if isinstance(value, np.ndarray):
                copied[key] = np.array(value, copy=True)
            else:
                copied[key] = value
        return copied

    def _states_from_raw_obs(self, raw_obs: dict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate(
            [np.asarray(raw_obs[key], dtype=np.float32) for key in self.state_obs_keys],
            axis=0,
        )

    def _render_rgb(self) -> np.ndarray:
        height, width = self.render_hw
        rgb_chunks = []
        for camera_name in self.render_camera_names:
            image = self.env.render(
                mode="rgb_array",
                height=height,
                width=width,
                camera_name=camera_name,
            )
            assert image.ndim == 3, (
                f"Expected rank-3 image for camera '{camera_name}', got {image.shape}."
            )
            assert image.shape[-1] == 3, (
                f"Expected HWC image with 3 channels for camera '{camera_name}', "
                f"got {image.shape}."
            )
            rgb_chunks.append(np.moveaxis(image, -1, 0))
        return np.concatenate(rgb_chunks, axis=0).astype(np.float32) / 255.0

    def _aux_from_raw_obs(self, raw_obs: dict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate(
            [
                np.asarray(raw_obs[key], dtype=np.float32).reshape(-1)
                for key in self.aux_obs_keys
            ]
        )

    def get_observation(self) -> dict[str, np.ndarray]:
        raw_obs = self.env.get_observation()
        obs = {
            "states": self._states_from_raw_obs(raw_obs),
            "rgb": self._render_rgb(),
        }
        if self.aux_obs_keys is not None:
            obs["aux_states"] = self._aux_from_raw_obs(raw_obs)
        return obs

    def seed(self, seed: int | None = None) -> None:
        if seed is not None:
            np.random.seed(seed=seed)
        self._seed = seed

    def reset(self) -> dict[str, np.ndarray]:
        if self.init_state is not None:
            if isinstance(self.init_state, dict):
                self.env.reset_to(self._copy_reset_state(self.init_state))
            else:
                self.env.reset_to({"states": self.init_state})
        elif self._seed is not None:
            seed = int(self._seed)
            if seed in self.seed_state_map:
                self.env.reset_to(self._copy_reset_state(self.seed_state_map[seed]))
            else:
                np.random.seed(seed=seed)
                self.env.reset()
                self.seed_state_map[seed] = self._copy_reset_state(self.env.get_state())
            self._seed = None
        else:
            self.env.reset()
        return self.get_observation()

    def step(self, action: np.ndarray):
        raw_obs, reward, done, info = self.env.step(action)
        obs = {
            "states": self._states_from_raw_obs(raw_obs),
            "rgb": self._render_rgb(),
        }
        if self.aux_obs_keys is not None:
            obs["aux_states"] = self._aux_from_raw_obs(raw_obs)
        return obs, reward, done, info

    def close(self) -> None:
        close_fn = getattr(self.env, "close", None)
        if close_fn is not None:
            close_fn()
