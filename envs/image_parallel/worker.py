from __future__ import annotations

import os
from multiprocessing.connection import Connection

import numpy as np

from envs.image_parallel.factory import create_image_env
from envs.image_parallel.spec import ImageEnvSpec


def image_env_worker_loop(connection: Connection, spec: ImageEnvSpec) -> None:
    env = create_image_env(spec)
    try:
        while True:
            message = connection.recv()
            command = str(message["command"])
            if command == "reset":
                seed = int(message["seed"])
                env.seed(seed)
                observation = env.reset()
                connection.send(
                    {
                        "observation": {
                            "states": np.asarray(
                                observation["states"], dtype=np.float32
                            ),
                            "rgb": np.asarray(observation["rgb"], dtype=np.float32),
                        }
                    }
                )
                continue
            if command == "step":
                action = np.asarray(message["action"], dtype=np.float32)
                observation, reward, done, _info = env.step(action)
                connection.send(
                    {
                        "observation": {
                            "states": np.asarray(
                                observation["states"], dtype=np.float32
                            ),
                            "rgb": np.asarray(observation["rgb"], dtype=np.float32),
                        },
                        "reward": float(reward),
                        "done": bool(done),
                    }
                )
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
    # Reached only on a clean exit (close ack or parent EOF) -- an exception raised
    # inside the try propagates past this and still crashes the worker, so the parent
    # detects real failures. os._exit(0) skips interpreter shutdown / GC, whose
    # robosuite EGL render-context __del__ can hang or raise EGL_NOT_INITIALIZED during
    # teardown on headless nodes; this exits 0 immediately instead of hanging until the
    # parent has to SIGTERM (-15) the worker.
    os._exit(0)
