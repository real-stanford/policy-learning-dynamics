from __future__ import annotations

import os
from dataclasses import dataclass
from multiprocessing.connection import Connection

import numpy as np

from envs.image_parallel.factory import create_image_env
from envs.image_parallel.spec import ImageEnvSpec


@dataclass(frozen=True)
class RLEnvSpec:
    image_spec: ImageEnvSpec
    max_episode_steps: int
    success_steps_before_termination: int
    aux_state_keys: tuple = ()


def _rgb_to_uint8(rgb: np.ndarray) -> np.ndarray:
    """RobomimicImageWrapper renders float32 = uint8 / 255 exactly; invert losslessly."""
    return np.rint(np.asarray(rgb, dtype=np.float32) * 255.0).astype(np.uint8)


class _EpisodeState:
    def __init__(self) -> None:
        self.steps = 0
        self.success_count = 0
        self.active = False


def rl_env_worker_loop(connection: Connection, spec: RLEnvSpec) -> None:
    aux_keys = tuple(str(k) for k in spec.aux_state_keys)
    assert aux_keys == tuple(getattr(spec.image_spec, "aux_obs_keys", ())), (
        "RLEnvSpec.aux_state_keys must match image_spec.aux_obs_keys "
        f"({aux_keys} vs {getattr(spec.image_spec, 'aux_obs_keys', ())})."
    )
    env = create_image_env(spec.image_spec)
    state = _EpisodeState()
    try:
        while True:
            message = connection.recv()
            command = str(message["command"])
            if command == "reset":
                seed = int(message["seed"])
                np.random.seed(seed)
                env.env.reset()
                obs = env.get_observation()
                state.steps = 0
                state.success_count = 0
                state.active = True
                reply = {
                    "states": np.asarray(obs["states"], dtype=np.float32),
                    "rgb": _rgb_to_uint8(obs["rgb"]),
                }
                if aux_keys:
                    reply["aux"] = np.asarray(obs["aux_states"], dtype=np.float32)
                connection.send(reply)
                continue
            if command == "step_chunk":
                assert state.active, "step_chunk before reset"
                actions = np.asarray(message["actions"], dtype=np.float32)
                assert actions.ndim == 2, actions.shape
                states_out, rgb_out, rewards_out, aux_out = [], [], [], []
                terminated = False
                truncated = False
                for action in actions:
                    obs, reward, _done, _info = env.step(action)
                    state.steps += 1
                    states_out.append(np.asarray(obs["states"], dtype=np.float32))
                    rgb_out.append(_rgb_to_uint8(obs["rgb"]))
                    if aux_keys:
                        aux_out.append(np.asarray(obs["aux_states"], dtype=np.float32))
                    success = float(reward) >= 1.0
                    if success:
                        state.success_count += 1
                    else:
                        state.success_count = 0
                    if state.success_count >= spec.success_steps_before_termination:
                        rewards_out.append(1.0)
                        terminated = True
                    else:
                        rewards_out.append(0.0)
                        if state.steps >= spec.max_episode_steps:
                            truncated = True
                    if terminated or truncated:
                        state.active = False
                        break
                reply = {
                    "states": np.stack(states_out, axis=0),
                    "rgb": np.stack(rgb_out, axis=0),
                    "rewards": np.asarray(rewards_out, dtype=np.float32),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                }
                if aux_keys:
                    reply["aux"] = np.stack(aux_out, axis=0)
                connection.send(reply)
                continue
            if command == "close":
                connection.send({"closed": True})
                break
            raise ValueError(f"Unsupported worker command '{command}'.")
    except EOFError:
        pass
    finally:
        try:
            env.close()
        except Exception:
            pass
        try:
            connection.close()
        except Exception:
            pass
    # Same rationale as envs/image_parallel/worker.py: skip interpreter teardown,
    # whose EGL context destructor can hang on headless nodes.
    os._exit(0)
