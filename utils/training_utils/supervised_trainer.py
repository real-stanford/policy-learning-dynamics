"""Multi-GPU diffusion-policy trainer (behaviour cloning).

Training only writes checkpoints: ``checkpoints/epoch_XXXX.ckpt`` every ``save_every``
epochs (plus ``last.ckpt`` when ``checkpoint.save_last_ckpt`` is on) and the per-epoch
metrics to ``logs.json.txt`` / W&B. Closed-loop evaluation (pass@k) is a separate step:
``scripts/eval_checkpoints.py``.
"""

from __future__ import annotations

import copy
import json
import os
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path

import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import InitProcessGroupKwargs, set_seed
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.common.pytorch_util import dict_apply
from utils.training_utils.checkpointing import CheckpointManager, load_checkpoint_payload
from utils.training_utils.compile import (
    compile_policy_modules,
    load_compiled_state_dict,
    sanitize_compiled_state_dict,
    unwrap_compiled_module,
)
from utils.training_utils.ema import EMAConfig, EMAHandler
from utils.training_utils.factories import build_optimizer
from utils.training_utils.json_logger import JsonLogger
from utils.training_utils.lr_scheduler import get_scheduler
from utils.training_utils.wandb_logger import WandbLogger


class SupervisedTrainer:
    def __init__(self, cfg, policy, dataset, output_dir: str | Path) -> None:
        self.cfg = cfg
        self.policy = policy
        self.dataset = dataset
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.train_cfg = cfg.training.training
        self.wandb_cfg = cfg.training.logging.wandb
        self.optimizer = None
        self.lr_scheduler = None
        self.ema_handler = None
        self.checkpoint_manager = None
        self.global_step = 0
        self.epoch = 0

        ddp_kwargs = DistributedDataParallelKwargs(
            broadcast_buffers=bool(cfg.training.accelerate.ddp_broadcast_buffers),
            find_unused_parameters=bool(cfg.training.accelerate.ddp_find_unused_parameters),
            static_graph=bool(cfg.training.accelerate.ddp_static_graph),
        )
        pg_kwargs = InitProcessGroupKwargs(
            timeout=timedelta(seconds=int(cfg.training.accelerate.pg_timeout_seconds))
        )
        # split_batches is implemented manually in _build_dataloader (accelerate >=1.x
        # ignores this kwarg). Pass False so a future accelerate that honours it cannot
        # double-split the batch we already sized per GPU.
        self.accelerator = Accelerator(
            kwargs_handlers=[ddp_kwargs, pg_kwargs],
            split_batches=False,
        )

    # ------------------------------------------------------------------ helpers
    def _resolve_num_workers(self, configured) -> int:
        """Dataloader workers PER RANK. A negative value auto-sizes from the node's usable
        cores split across the ranks (one core of headroom, capped at 16)."""
        configured = int(configured)
        if configured >= 0:
            return configured
        try:
            cores = len(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            cores = os.cpu_count() or 8
        nproc = max(1, int(self.accelerator.num_processes))
        return max(1, min(16, cores // nproc - 1))

    def _build_dataloader(self, dataset, cfg_section) -> DataLoader:
        """accelerate.split_batches=true: ``batch_size`` is the GLOBAL batch, split across
        ranks, so the effective batch and the number of optimizer steps per epoch are
        identical for any num_processes. false: ``batch_size`` is per GPU."""
        num_workers = self._resolve_num_workers(cfg_section.num_workers)
        batch_size = int(cfg_section.batch_size)
        nproc = max(1, int(self.accelerator.num_processes))
        split = bool(self.cfg.training.accelerate.split_batches)
        assert not split or (batch_size % nproc == 0), (
            f"dataloader.batch_size={batch_size} must be divisible by num_processes="
            f"{nproc} when accelerate.split_batches=true (batch_size is the GLOBAL batch)."
        )
        per_gpu = (batch_size // nproc) if split else batch_size
        effective = batch_size if split else batch_size * nproc
        if bool(cfg_section.shuffle) and self.accelerator.is_main_process:
            print(
                f"[dataloader] effective(global) batch={effective} -> {per_gpu}/gpu x {nproc} "
                f"ranks | split_batches={split} | num_workers={num_workers}/rank",
                flush=True,
            )
        return DataLoader(
            dataset,
            batch_size=per_gpu,
            num_workers=num_workers,
            shuffle=bool(cfg_section.shuffle),
            pin_memory=bool(cfg_section.pin_memory),
            persistent_workers=bool(cfg_section.persistent_workers and num_workers > 0),
        )

    def _move_batch(self, batch: dict) -> dict:
        return dict_apply(
            batch,
            lambda x: x.to(device=self.accelerator.device, non_blocking=True),
        )

    def _base_policy(self):
        unwrapped = self.policy
        while isinstance(
            unwrapped,
            (torch.nn.parallel.DistributedDataParallel, torch.nn.DataParallel),
        ):
            unwrapped = unwrapped.module
        return unwrap_compiled_module(unwrapped)

    def _maybe_update_oa_mixer_epoch(self, steps_per_epoch: int) -> None:
        """Advance the joint generator mixer's noisy-fraction ramp (no-op without mixer)."""
        mixer = getattr(self._base_policy(), "oa_joint_mixer", None)
        if mixer is None:
            return
        mixer.set_steps_per_epoch(int(steps_per_epoch))
        prev = mixer.noisy_fraction()
        mixer.set_epoch(self.epoch)
        frac = mixer.noisy_fraction()
        if self.accelerator.is_main_process and prev <= 0.0 < frac:
            print(
                f"[oa_joint_mixer] noisy fraction starts ramping at epoch {self.epoch} "
                f"(frac={frac:.4f}, max={mixer.max_noisy_fraction}, "
                f"start_epochs={mixer.start_epochs}, ramp_epochs={mixer.ramp_epochs}, "
                f"ckpts={len(mixer.models)}, steps={mixer.num_inference_steps})",
                flush=True,
            )

    def _checkpoint_payload(self) -> dict:
        base_policy = self._base_policy()
        payload = {
            "cfg": copy.deepcopy(self.cfg),
            "model_state": sanitize_compiled_state_dict(base_policy.state_dict()),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.lr_scheduler.state_dict(),
            "normalizer_state": base_policy.normalizer.state_dict(),
            "epoch": self.epoch,
            "global_step": self.global_step,
        }
        if self.ema_handler is not None:
            payload["ema_state"] = sanitize_compiled_state_dict(
                self.ema_handler.ema_model.state_dict()
            )
        return payload

    def _try_resume(self) -> None:
        if not bool(self.train_cfg.resume):
            return
        last_path = self.checkpoint_manager.last_path
        if not last_path.exists():
            return
        payload = load_checkpoint_payload(last_path, map_location=self.accelerator.device)
        base_policy = self._base_policy()
        load_compiled_state_dict(base_policy, payload["model_state"])
        base_policy.normalizer.load_state_dict(payload["normalizer_state"])
        self.optimizer.load_state_dict(payload["optimizer_state"])
        self.lr_scheduler.load_state_dict(payload["scheduler_state"])
        self.epoch = int(payload["epoch"]) + 1
        self.global_step = int(payload["global_step"]) + 1
        if self.ema_handler is not None and "ema_state" in payload:
            load_compiled_state_dict(self.ema_handler.ema_model, payload["ema_state"])
        if self.accelerator.is_main_process:
            print(f"[resume] restored {last_path} at epoch {self.epoch}", flush=True)

    def _distributed_mean(self, value_sum: float, count: int) -> float:
        assert count >= 0, f"Count must be non-negative, got {count}."
        if count == 0:
            return 0.0
        stats = torch.tensor(
            [float(value_sum), float(count)],
            device=self.accelerator.device,
            dtype=torch.float64,
        )
        reduced = self.accelerator.reduce(stats, reduction="sum")
        total_count = max(1.0, float(reduced[1].item()))
        return float((reduced[0] / total_count).item())

    def _run_validation(self, eval_policy, val_loader) -> float | None:
        if len(val_loader) == 0:
            return None
        val_loss_sum = 0.0
        val_count = 0
        iterator = val_loader
        if self.accelerator.is_main_process:
            iterator = tqdm(
                val_loader,
                desc=f"Val epoch {self.epoch}",
                leave=False,
                mininterval=float(self.train_cfg.tqdm_interval_sec),
            )
        with torch.no_grad():
            for batch_idx, batch in enumerate(iterator):
                batch = self._move_batch(batch)
                val_loss = eval_policy(batch)
                val_loss_sum += float(val_loss.item())
                val_count += 1
                if self.train_cfg.max_val_steps is not None and (batch_idx + 1) >= int(
                    self.train_cfg.max_val_steps
                ):
                    break
        return self._distributed_mean(val_loss_sum, val_count)

    def _save_checkpoints(self, step_log: dict) -> None:
        """Main process only. Periodic epoch_XXXX.ckpt every save_every epochs; last.ckpt
        (resume point) on the same cadence when checkpoint.save_last_ckpt is on."""
        save_interval = int(self.train_cfg.save_every)
        is_final_epoch = self.epoch == (int(self.train_cfg.num_epochs) - 1)
        completed_epoch = int(self.epoch) + 1
        save_last = is_final_epoch or (self.epoch % save_interval) == 0
        save_periodic = (completed_epoch % save_interval) == 0
        payload = self._checkpoint_payload() if (save_last or save_periodic) else None
        self.checkpoint_manager.save(payload=payload, metrics=step_log, save_last=save_last)
        if save_periodic:
            periodic_path = self.output_dir / "checkpoints" / f"epoch_{completed_epoch:04d}.ckpt"
            torch.save(payload, periodic_path)

    # --------------------------------------------------------------------- run
    def run(self) -> dict:
        set_seed(int(self.train_cfg.seed), device_specific=True)

        train_dataset = self.dataset
        val_dataset = self.dataset.get_validation_dataset()
        assert len(train_dataset) > 0, "Training dataset is empty."
        train_loader = self._build_dataloader(train_dataset, self.cfg.training.dataloader)
        val_loader = self._build_dataloader(val_dataset, self.cfg.training.validation)

        normalizer = train_dataset.get_normalizer()
        self.policy.set_normalizer(normalizer)

        if bool(self.train_cfg.use_ema):
            ema_cfg = EMAConfig(
                update_after_step=int(self.cfg.training.ema.update_after_step),
                inv_gamma=float(self.cfg.training.ema.inv_gamma),
                power=float(self.cfg.training.ema.power),
                min_value=float(self.cfg.training.ema.min_value),
                max_value=float(self.cfg.training.ema.max_value),
            )
            self.ema_handler = EMAHandler(self.policy, ema_cfg)
            self.ema_handler.ema_model.set_normalizer(normalizer.copy())

        self.optimizer = build_optimizer(self.cfg, self.policy)
        self.policy = compile_policy_modules(self.policy, self.cfg.training.compile, "policy")
        self.policy, self.optimizer, train_loader, val_loader = self.accelerator.prepare(
            self.policy, self.optimizer, train_loader, val_loader
        )
        total_steps = max(
            1,
            (len(train_loader) * int(self.train_cfg.num_epochs))
            // int(self.train_cfg.gradient_accumulate_every),
        )
        self.lr_scheduler = get_scheduler(
            self.train_cfg.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=int(self.train_cfg.lr_warmup_steps),
            num_training_steps=total_steps,
        )
        self.checkpoint_manager = CheckpointManager(
            output_dir=self.output_dir, cfg=self.cfg.checkpoint
        )
        if self.ema_handler is not None:
            self.ema_handler.ema_model.to(self.accelerator.device)
        self._try_resume()

        train_sample_batch = None
        last_metrics: dict | None = None
        log_path = self.output_dir / "logs.json.txt"
        logger_context = (
            JsonLogger(str(log_path)) if self.accelerator.is_main_process else nullcontext(None)
        )
        wandb_context = (
            WandbLogger(self.wandb_cfg, self.cfg, self.output_dir)
            if self.accelerator.is_main_process
            else nullcontext(None)
        )

        with logger_context as json_logger, wandb_context as wandb_logger:
            outer_progress_context = nullcontext(None)
            if self.accelerator.is_main_process:
                outer_progress_context = tqdm(
                    total=int(self.train_cfg.num_epochs),
                    initial=int(self.epoch),
                    desc="Epochs",
                    dynamic_ncols=True,
                )

            with outer_progress_context as outer_progress:
                while self.epoch < int(self.train_cfg.num_epochs):
                    self._maybe_update_oa_mixer_epoch(len(train_loader))
                    self.policy.train()
                    train_loss_sum = 0.0
                    train_count = 0
                    step_log = {
                        "epoch": int(self.epoch),
                        "completed_epoch": int(self.epoch) + 1,
                        "global_step": int(self.global_step),
                    }

                    iterator = train_loader
                    if self.accelerator.is_main_process:
                        iterator = tqdm(
                            train_loader,
                            desc=f"Train epoch {self.epoch}",
                            leave=False,
                            mininterval=float(self.train_cfg.tqdm_interval_sec),
                        )

                    self.optimizer.zero_grad()
                    for batch_idx, batch in enumerate(iterator):
                        batch = self._move_batch(batch)
                        if train_sample_batch is None:
                            train_sample_batch = batch

                        raw_loss = self.policy(batch)
                        loss = raw_loss / int(self.train_cfg.gradient_accumulate_every)
                        self.accelerator.backward(loss)

                        should_step = (
                            (batch_idx + 1) % int(self.train_cfg.gradient_accumulate_every)
                        ) == 0 or (batch_idx + 1) == len(train_loader)
                        if should_step:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            self.lr_scheduler.step()
                            if self.ema_handler is not None:
                                self.ema_handler.step(self._base_policy())

                        raw_loss_cpu = float(raw_loss.item())
                        if self.accelerator.is_main_process:
                            iterator.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_loss_sum += raw_loss_cpu
                        train_count += 1
                        self.global_step += 1

                        if self.train_cfg.max_train_steps is not None and (
                            batch_idx + 1
                        ) >= int(self.train_cfg.max_train_steps):
                            break

                    step_log["train_loss"] = self._distributed_mean(train_loss_sum, train_count)
                    step_log["lr"] = float(self.lr_scheduler.get_last_lr()[0])

                    eval_policy = (
                        self.ema_handler.ema_model
                        if self.ema_handler is not None
                        else self._base_policy()
                    )
                    eval_policy.eval()

                    if len(val_dataset) > 0 and (self.epoch % int(self.train_cfg.val_every)) == 0:
                        val_loss = self._run_validation(eval_policy, val_loader)
                        if val_loss is not None:
                            step_log["val_loss"] = val_loss

                    if (
                        train_sample_batch is not None
                        and (self.epoch % int(self.train_cfg.sample_every)) == 0
                    ):
                        with torch.no_grad():
                            prediction = eval_policy.predict_action(train_sample_batch["obs"])
                            step_log["train_action_mse_error"] = float(
                                torch.nn.functional.mse_loss(
                                    prediction["action_pred"], train_sample_batch["action"]
                                ).item()
                            )

                    if self.accelerator.is_main_process:
                        self._save_checkpoints(step_log)
                        assert json_logger is not None
                        json_logger.log(step_log)
                        assert wandb_logger is not None
                        wandb_logger.log(step_log, step=int(step_log["completed_epoch"]))
                        if outer_progress is not None:
                            outer_progress.set_postfix(
                                train_loss=f"{step_log['train_loss']:.4f}",
                                val_loss=(
                                    f"{step_log['val_loss']:.4f}"
                                    if "val_loss" in step_log
                                    else "n/a"
                                ),
                                refresh=False,
                            )
                            outer_progress.update(1)

                    last_metrics = dict(step_log)
                    self.epoch += 1
                    self.accelerator.wait_for_everyone()

            assert last_metrics is not None, "Training loop completed without metrics."
            if self.accelerator.is_main_process:
                summary = {
                    "task_name": str(self.cfg.task.name),
                    "output_dir": str(self.output_dir),
                    "last_epoch": int(last_metrics["epoch"]),
                    "last_global_step": int(last_metrics["global_step"]),
                    "last_metrics": last_metrics,
                }
                with open(self.output_dir / "train_summary.json", "w", encoding="utf-8") as handle:
                    json.dump(summary, handle, indent=2, sort_keys=True)

        self.accelerator.wait_for_everyone()
        self.accelerator.end_training()
        return {"output_dir": str(self.output_dir), "last_metrics": last_metrics}
