from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class EpisodeResult:
    iteration: int
    rank: int
    env_index: int
    success: bool
    length: int
    truncated: bool


class RunningEpisodeStats:
    """Running statistics over the most recent ``window`` completed episodes
    (across all ranks; callers feed a globally ordered stream)."""

    def __init__(self, window: int) -> None:
        assert window >= 1
        self.window = int(window)
        self.recent: deque[EpisodeResult] = deque(maxlen=self.window)
        self.total_episodes = 0
        self.total_successes = 0

    def update(self, results: list[EpisodeResult]) -> None:
        for result in sorted(results, key=lambda r: (r.iteration, r.rank, r.env_index)):
            self.recent.append(result)
            self.total_episodes += 1
            self.total_successes += int(result.success)

    def summary(self) -> dict[str, float]:
        n = len(self.recent)
        if n == 0:
            return {
                "success_rate": 0.0,
                "episode_length": 0.0,
                "truncation_rate": 0.0,
                "window_episodes": 0.0,
                "total_episodes": float(self.total_episodes),
                "total_success_rate": 0.0,
            }
        return {
            "success_rate": sum(r.success for r in self.recent) / n,
            "episode_length": sum(r.length for r in self.recent) / n,
            "truncation_rate": sum(r.truncated for r in self.recent) / n,
            "window_episodes": float(n),
            "total_episodes": float(self.total_episodes),
            "total_success_rate": self.total_successes / max(1, self.total_episodes),
        }
