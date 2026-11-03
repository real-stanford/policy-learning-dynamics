from __future__ import annotations

import json
from pathlib import Path

import torch


def list_checkpoint_artifacts_for_reporting(output_dir: str | Path) -> list[str]:
    """Stable-ordered list of checkpoint-related files under ``output_dir/checkpoints``."""
    root = Path(output_dir).expanduser().resolve() / "checkpoints"
    if not root.is_dir():
        return []
    names: list[Path] = []
    for path in root.iterdir():
        if not path.is_file():
            continue
        if path.suffix == ".ckpt" or path.name == "checkpoint_state.json":
            names.append(path)
    names.sort(key=lambda p: p.name)
    return [str(p) for p in names]


def load_checkpoint_payload(path: str | Path, *, map_location, mmap: bool = False) -> dict:
    """Load a checkpoint payload. ``mmap=True`` memory-maps the file so tensor storages are
    read from disk only when accessed -- a caller that touches just a subset (e.g. only
    ``ema_state`` + ``normalizer_state``) never reads the tensors it skips: ``optimizer_state``
    always, and ``model_state`` whenever ``ema_state`` is present. (The pickled ``cfg`` is still
    fully materialized; mmap only defers tensor storages.) Defaults to ``False`` -> existing
    callers are byte-identical."""
    checkpoint_path = Path(path).expanduser().resolve()
    return torch.load(
        str(checkpoint_path),
        map_location=map_location,
        weights_only=False,
        mmap=mmap,
    )


class CheckpointManager:
    def __init__(self, output_dir: str | Path, cfg) -> None:
        self.output_dir = Path(output_dir)
        self.ckpt_dir = self.output_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.monitor_key = str(cfg.topk.monitor_key)
        self.mode = str(cfg.topk.mode)
        assert self.mode in {"min", "max"}, (
            f"Unsupported checkpoint mode '{self.mode}'."
        )
        self.save_last_ckpt = bool(cfg.save_last_ckpt)
        self.state_path = self.ckpt_dir / "checkpoint_state.json"
        self.best_score: float | None = None
        self.best_epoch: int | None = None
        if self.state_path.exists():
            with open(self.state_path, "r", encoding="utf-8") as handle:
                state = json.load(handle)
            best_score = state["best_score"]
            if best_score is not None:
                self.best_score = float(best_score)
            best_epoch = state["best_epoch"]
            if best_epoch is not None:
                self.best_epoch = int(best_epoch)

    @property
    def last_path(self) -> Path:
        return self.ckpt_dir / "last.ckpt"

    @property
    def best_path(self) -> Path:
        return self.ckpt_dir / "best.ckpt"

    def _is_better(self, score: float) -> bool:
        if self.best_score is None:
            return True
        if self.mode == "min":
            return score < self.best_score
        return score > self.best_score

    def save(self, *, payload: dict, metrics: dict, save_last: bool) -> dict:
        best_updated = False
        monitor_value = None
        if self.monitor_key in metrics:
            monitor_value = float(metrics[self.monitor_key])
            if self._is_better(monitor_value):
                # best.ckpt WRITE DISABLED (user decision, all training stacks):
                # early in training val_loss improves nearly every epoch, so this
                # line wrote the full multi-GB payload every epoch -- on /store
                # (~20-70 MB/s) the write dominated inter-epoch wall-clock.
                # Monitor bookkeeping (best_score/best_epoch) is kept for
                # logging/telemetry; only the file write is commented out.
                # torch.save(payload, self.best_path)
                self.best_score = monitor_value
                self.best_epoch = int(metrics["epoch"])
                best_updated = True

        if save_last and self.save_last_ckpt:
            assert payload is not None, "save_last requested without a payload."
            torch.save(payload, self.last_path)

        state = {
            "monitor_key": self.monitor_key,
            "mode": self.mode,
            "best_score": self.best_score,
            "best_epoch": self.best_epoch,
            "last_path": str(self.last_path) if self.last_path.exists() else None,
            "best_path": str(self.best_path) if self.best_path.exists() else None,
        }
        with open(self.state_path, "w", encoding="utf-8") as handle:
            json.dump(state, handle, indent=2, sort_keys=True)

        return {
            "best_updated": bool(best_updated),
            "best_path": str(self.best_path) if self.best_path.exists() else None,
            "last_path": str(self.last_path) if self.last_path.exists() else None,
            "monitor_value": monitor_value,
        }
