from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf


class WandbLogger:
    def __init__(self, cfg, full_cfg, output_dir: str | Path) -> None:
        self.cfg = cfg
        self.full_cfg = full_cfg
        self.output_dir = Path(output_dir)
        self._wandb = None
        self._run = None

    def __enter__(self):
        if not bool(self.cfg.enabled):
            return self

        import wandb

        resolved_cfg = OmegaConf.to_container(self.full_cfg, resolve=True)
        assert isinstance(resolved_cfg, dict), (
            "Resolved training config must be a dictionary."
        )
        self._wandb = wandb
        self._run = wandb.init(
            project=str(self.cfg.project),
            group=str(self.cfg.group),
            id=str(self.cfg.run_id),
            name=str(self.cfg.run_name),
            dir=str(self.output_dir),
            config=resolved_cfg,
            resume="allow",
        )
        assert self._run is not None, "wandb.init returned None."
        return self

    @property
    def run_url(self) -> str | None:
        """Public W&B run URL, or None if logging is disabled or init did not run."""
        if self._run is None:
            return None
        url = self._run.url
        return str(url) if url else None

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        if self._run is None:
            return
        self._run.finish()
        self._run = None
        self._wandb = None

    def log(self, payload: dict, *, step: int) -> None:
        if self._run is None:
            return
        assert self._wandb is not None, "wandb module is not initialized."
        self._wandb.log(payload, step=step)

    def update_summary(self, payload: dict) -> None:
        if self._run is None:
            return
        for key, value in payload.items():
            self._run.summary[key] = value
