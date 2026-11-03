"""Absolute-action conversion helpers for robomimic-style datasets."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.config import config_factory
from scipy.spatial.transform import Rotation

from envs.image_parallel.factory import ensure_robosuite_env_registered


class RobomimicAbsoluteActionConverter:
    """Convert delta controller actions into absolute controller targets."""

    def __init__(self, dataset_path: str | Path, algo_name: str = "bc") -> None:
        self.dataset_path = Path(dataset_path).expanduser().resolve()

        config = config_factory(algo_name=algo_name)
        ObsUtils.initialize_obs_utils_with_config(config)

        env_meta = FileUtils.get_env_metadata_from_dataset(str(self.dataset_path))
        ensure_robosuite_env_registered(str(env_meta["env_name"]))
        abs_env_meta = copy.deepcopy(env_meta)
        abs_env_meta["env_kwargs"]["controller_configs"]["control_delta"] = False

        self.env = EnvUtils.create_env_from_metadata(
            env_meta=env_meta,
            render=False,
            render_offscreen=False,
            use_image_obs=False,
        )
        self.abs_env = EnvUtils.create_env_from_metadata(
            env_meta=abs_env_meta,
            render=False,
            render_offscreen=False,
            use_image_obs=False,
        )
        assert len(self.env.env.robots) in (1, 2), (
            f"Expected one or two robots, got {len(self.env.env.robots)}."
        )
        assert not self.abs_env.env.robots[0].controller.use_delta, (
            "Absolute-action converter expected a non-delta controller."
        )

    def close(self) -> None:
        env_close = getattr(self.env, "close", None)
        if callable(env_close):
            env_close()
        abs_env_close = getattr(self.abs_env, "close", None)
        if callable(abs_env_close):
            abs_env_close()

    def _build_reset_payload(
        self,
        *,
        state: np.ndarray,
        model_xml: str | bytes | None,
    ) -> dict[str, object]:
        payload: dict[str, object] = {"states": state}
        if model_xml is None:
            return payload
        if isinstance(model_xml, bytes):
            payload["model"] = model_xml.decode("utf-8")
            return payload
        payload["model"] = str(model_xml)
        return payload

    def convert_actions(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        *,
        model_xml: str | bytes | None = None,
    ) -> np.ndarray:
        states_array = np.asarray(states, dtype=np.float32)
        actions_array = np.asarray(actions, dtype=np.float32)
        assert states_array.ndim == 2, (
            f"Expected states with rank 2, got {states_array.shape}."
        )
        assert actions_array.ndim == 2, (
            f"Expected actions with rank 2, got {actions_array.shape}."
        )
        assert states_array.shape[0] == actions_array.shape[0], (
            f"Expected matching timesteps, got states={states_array.shape[0]} and "
            f"actions={actions_array.shape[0]}."
        )
        assert actions_array.shape[-1] in (7, 14), (
            f"Expected 7D or 14D delta actions, got {actions_array.shape}."
        )
        assert states_array.shape[0] > 0, "Expected at least one state frame."

        stacked_actions = actions_array.reshape(actions_array.shape[0], -1, 7)
        action_goal_pos = np.zeros(stacked_actions.shape[:-1] + (3,), dtype=np.float32)
        action_goal_ori = np.zeros(stacked_actions.shape[:-1] + (3,), dtype=np.float32)
        action_gripper = stacked_actions[..., -1:].astype(np.float32)

        def _record_goal_for_step(
            step_idx: int, reset_payload: dict[str, object]
        ) -> None:
            self.env.reset_to(reset_payload)
            for robot_idx, robot in enumerate(self.env.env.robots):
                robot.control(stacked_actions[step_idx, robot_idx], policy_step=True)
                controller = robot.controller
                action_goal_pos[step_idx, robot_idx] = controller.goal_pos.astype(
                    np.float32
                )
                action_goal_ori[step_idx, robot_idx] = (
                    Rotation.from_matrix(controller.goal_ori)
                    .as_rotvec()
                    .astype(np.float32)
                )

        if model_xml is None:
            for step_idx in range(states_array.shape[0]):
                reset_payload = {"states": states_array[step_idx]}
                _record_goal_for_step(step_idx, reset_payload)
        else:
            # Load per-demo model once; reloading XML every timestep is prohibitively slow.
            initial_payload = self._build_reset_payload(
                state=states_array[0],
                model_xml=model_xml,
            )
            _record_goal_for_step(0, initial_payload)
            for step_idx in range(1, states_array.shape[0]):
                reset_payload = {"states": states_array[step_idx]}
                _record_goal_for_step(step_idx, reset_payload)

        absolute_actions = np.concatenate(
            (action_goal_pos, action_goal_ori, action_gripper), axis=-1
        )
        return absolute_actions.reshape(actions_array.shape).astype(np.float32)
