from __future__ import annotations

import numpy as np

from envs.image_parallel.spec import ImageEnvSpec, ImageStepResult
from envs.image_parallel.worker import image_env_worker_loop
from envs.process_vector_env import ParallelProcessEnv


class ParallelImageEnv(ParallelProcessEnv):
    def __init__(
        self,
        *,
        spec: ImageEnvSpec,
        num_envs: int,
        start_method: str = "spawn",
    ) -> None:
        super().__init__(
            worker_target=image_env_worker_loop,
            spec=spec,
            num_envs=num_envs,
            start_method=start_method,
        )

    def reset(self, worker_index: int, seed: int) -> dict[str, np.ndarray]:
        message = super().reset(worker_index, seed)
        return {
            "states": np.asarray(message["states"], dtype=np.float32),
            "rgb": np.asarray(message["rgb"], dtype=np.float32),
        }

    def step_many(
        self, actions_by_worker: dict[int, np.ndarray]
    ) -> dict[int, ImageStepResult]:
        raw_results = super().step_many(actions_by_worker)
        results = {}
        for worker_index, message in raw_results.items():
            results[worker_index] = ImageStepResult(
                observation={
                    "states": np.asarray(
                        message["observation"]["states"], dtype=np.float32
                    ),
                    "rgb": np.asarray(message["observation"]["rgb"], dtype=np.float32),
                },
                reward=float(message["reward"]),
                done=bool(message["done"]),
            )
        return results
