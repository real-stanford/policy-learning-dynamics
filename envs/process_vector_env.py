from __future__ import annotations

import multiprocessing as mp
from collections.abc import Callable

import numpy as np


class ParallelProcessEnv:
    def __init__(
        self,
        *,
        worker_target: Callable,
        spec,
        num_envs: int,
        start_method: str = "spawn",
    ) -> None:
        assert num_envs > 0, f"num_envs must be positive, got {num_envs}."
        self.worker_target = worker_target
        self.spec = spec
        self.num_envs = int(num_envs)
        self.start_method = str(start_method)
        self._ctx = mp.get_context(self.start_method)
        self._parents = []
        self._workers = []
        self._closed = False

        for _ in range(self.num_envs):
            parent_conn, child_conn = self._ctx.Pipe()
            worker = self._ctx.Process(
                target=self.worker_target,
                args=(child_conn, self.spec),
            )
            worker.start()
            child_conn.close()
            self._parents.append(parent_conn)
            self._workers.append(worker)

    def __enter__(self) -> "ParallelProcessEnv":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def reset(self, worker_index: int, seed: int):
        self._send(worker_index, {"command": "reset", "seed": int(seed)})
        message = self._recv(worker_index)
        assert "observation" in message, "Worker reset response missing observation."
        return message["observation"]

    def step_many(self, actions_by_worker: dict[int, np.ndarray]) -> dict[int, dict]:
        results = {}
        for worker_index, action in actions_by_worker.items():
            payload = {
                "command": "step",
                "action": np.asarray(action, dtype=np.float32),
            }
            self._send(worker_index, payload)
        for worker_index in actions_by_worker:
            results[worker_index] = self._recv(worker_index)
        return results

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for parent in self._parents:
            if not parent.closed:
                parent.send({"command": "close"})
        for worker_index, parent in enumerate(self._parents):
            if parent.closed:
                continue
            message = self._recv(worker_index)
            assert bool(message["closed"]), (
                f"Expected close acknowledgement from worker {worker_index}."
            )
            parent.close()
        for worker_index, worker in enumerate(self._workers):
            # Graceful join: the worker has ack'd close and is tearing down its sim env.
            # mujoco/robosuite teardown can exceed a few seconds on busy/slow hosts, so give
            # it a generous window before force-killing.
            worker.join(timeout=30.0)
            if worker.is_alive():
                # Still hung after the grace period. Eval results were already collected, so
                # force-terminate as part of teardown. A worker WE just SIGTERM'd will have a
                # negative (-SIGTERM) exitcode by definition -- expected here, NOT a failure --
                # so do not assert clean exit on this path. (Previously this raised "Worker N
                # exited with code -15", crashing the rank and, under DDP, cascading SIGTERMs
                # to the other ranks -- but only on hosts slow enough to miss the join timeout.)
                worker.terminate()
                worker.join(timeout=5.0)
                continue
            # Worker exited on its own -> assert it did so cleanly (catches real crashes).
            assert worker.exitcode in {0, None}, (
                f"Worker {worker_index} exited with code {worker.exitcode}."
            )

    def _send(self, worker_index: int, payload: dict) -> None:
        assert not self._closed, "ParallelProcessEnv is already closed."
        self._parents[worker_index].send(payload)

    def _recv(self, worker_index: int) -> dict:
        try:
            message = self._parents[worker_index].recv()
        except EOFError as exc:
            worker = self._workers[worker_index]
            raise RuntimeError(
                f"Worker {worker_index} closed unexpectedly with exit code "
                f"{worker.exitcode}."
            ) from exc
        assert isinstance(message, dict), (
            f"Expected dict worker message, got {type(message)}."
        )
        return message
