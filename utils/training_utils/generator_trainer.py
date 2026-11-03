"""Multi-GPU trainer for a standalone generative model (the joint image generator).

Deliberately decoupled from ``SupervisedTrainer`` (which trains policies): plain
supervised loop over ``model(batch) -> loss``, validation loss on the held-out episode
split, EMA, dense ``epoch_XXXX.ckpt`` checkpointing, ``logs.json.txt`` and W&B logging.
"""

from __future__ import annotations

import copy
import json
from contextlib import nullcontext
from pathlib import Path

import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.common.pytorch_util import dict_apply
from utils.training_utils.checkpointing import CheckpointManager, load_checkpoint_payload
from utils.training_utils.compile import (
    load_compiled_state_dict,
    sanitize_compiled_state_dict,
    unwrap_compiled_module,
)
from utils.training_utils.ema import EMAConfig, EMAHandler
from utils.training_utils.json_logger import JsonLogger
from utils.training_utils.lr_scheduler import get_scheduler
from utils.training_utils.wandb_logger import WandbLogger


class GeneratorTrainer:
    def __init__(self, cfg, model, dataset, output_dir: str | Path) -> None:
        self.cfg = cfg
        self.model = model
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
        self._last_val_components: dict = {}

        ddp_kwargs = DistributedDataParallelKwargs(
            broadcast_buffers=bool(cfg.training.accelerate.ddp_broadcast_buffers),
            find_unused_parameters=bool(cfg.training.accelerate.ddp_find_unused_parameters),
            static_graph=bool(cfg.training.accelerate.ddp_static_graph),
        )
        # split_batches is implemented manually in _build_dataloader (accelerate >=1.x
        # ignores this kwarg). Pass False so a future accelerate that honours it cannot
        # double-split the batch we already sized per GPU.
        self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], split_batches=False)

    # ------------------------------------------------------------------ helpers
    def _build_dataloader(self, dataset, cfg_section) -> DataLoader:
        """accelerate.split_batches=true: ``batch_size`` is the GLOBAL batch, split across
        ranks (rank-agnostic training); false: ``batch_size`` is per GPU."""
        batch_size = int(cfg_section.batch_size)
        nproc = max(1, int(self.accelerator.num_processes))
        split = bool(self.cfg.training.accelerate.split_batches)
        assert not split or (batch_size % nproc == 0), (
            f"dataloader.batch_size={batch_size} must be divisible by num_processes={nproc} "
            f"when accelerate.split_batches=true (batch_size is the GLOBAL batch)."
        )
        per_gpu = (batch_size // nproc) if split else batch_size
        return DataLoader(
            dataset,
            batch_size=per_gpu,
            num_workers=int(cfg_section.num_workers),
            shuffle=bool(cfg_section.shuffle),
            pin_memory=bool(cfg_section.pin_memory),
            persistent_workers=bool(
                cfg_section.persistent_workers and int(cfg_section.num_workers) > 0
            ),
        )

    def _move_batch(self, batch: dict) -> dict:
        return dict_apply(
            batch, lambda x: x.to(device=self.accelerator.device, non_blocking=True)
        )

    def _base_model(self):
        return unwrap_compiled_module(self.accelerator.unwrap_model(self.model))

    def _distributed_mean(self, value_sum: float, count: int) -> float:
        if count == 0:
            return 0.0
        stats = torch.tensor(
            [float(value_sum), float(count)],
            device=self.accelerator.device,
            dtype=torch.float64,
        )
        reduced = self.accelerator.reduce(stats, reduction="sum")
        total = max(1.0, float(reduced[1].item()))
        return float((reduced[0] / total).item())

    def _checkpoint_payload(self) -> dict:
        base_model = self._base_model()
        payload = {
            "cfg": copy.deepcopy(self.cfg),
            "model_state": sanitize_compiled_state_dict(base_model.state_dict()),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.lr_scheduler.state_dict(),
            "normalizer_state": base_model.normalizer.state_dict(),
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
        base_model = self._base_model()
        load_compiled_state_dict(base_model, payload["model_state"])
        base_model.normalizer.load_state_dict(payload["normalizer_state"])
        self.optimizer.load_state_dict(payload["optimizer_state"])
        self.lr_scheduler.load_state_dict(payload["scheduler_state"])
        self.epoch = int(payload["epoch"]) + 1
        self.global_step = int(payload["global_step"]) + 1
        if self.ema_handler is not None and "ema_state" in payload:
            load_compiled_state_dict(self.ema_handler.ema_model, payload["ema_state"])

    def _run_validation(self, eval_model, val_loader) -> float | None:
        self._last_val_components = {}
        if len(val_loader) == 0:
            return None
        loss_sum = 0.0
        count = 0
        comp_sums: dict = {}
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
                loss = eval_model(batch)
                loss_sum += float(loss.item())
                count += 1
                # Per-modality components stashed by the model's compute_loss (if any).
                comps = getattr(eval_model, "_loss_components", None)
                if comps is not None:
                    for ck, cv in comps.items():
                        comp_sums[ck] = comp_sums.get(ck, 0.0) + float(cv.item())
                if self.train_cfg.max_val_steps is not None and (batch_idx + 1) >= int(
                    self.train_cfg.max_val_steps
                ):
                    break
        self._last_val_components = {
            ck: self._distributed_mean(cs, count) for ck, cs in comp_sums.items()
        }
        return self._distributed_mean(loss_sum, count)

    # --------------------------------------------------------------------- run
    def run(self) -> dict:
        set_seed(int(self.train_cfg.seed), device_specific=True)

        train_dataset = self.dataset
        val_dataset = self.dataset.get_validation_dataset()
        assert len(train_dataset) > 0, "Training dataset is empty."

        train_loader = self._build_dataloader(train_dataset, self.cfg.training.dataloader)
        val_loader = self._build_dataloader(val_dataset, self.cfg.training.validation)

        normalizer = train_dataset.get_normalizer()
        self.model.set_normalizer(normalizer)

        if bool(self.train_cfg.use_ema):
            ema_cfg = EMAConfig(
                update_after_step=int(self.cfg.training.ema.update_after_step),
                inv_gamma=float(self.cfg.training.ema.inv_gamma),
                power=float(self.cfg.training.ema.power),
                min_value=float(self.cfg.training.ema.min_value),
                max_value=float(self.cfg.training.ema.max_value),
            )
            self.ema_handler = EMAHandler(self.model, ema_cfg)
            self.ema_handler.ema_model.set_normalizer(normalizer.copy())

        opt_cfg = self.cfg.training.optimizer
        assert str(opt_cfg.type) == "adamw", f"Unsupported optimizer '{opt_cfg.type}'."
        self.optimizer = torch.optim.AdamW(
            params=self.model.parameters(),
            lr=float(opt_cfg.lr),
            betas=tuple(opt_cfg.betas),
            eps=float(opt_cfg.eps),
            weight_decay=float(opt_cfg.weight_decay),
        )

        self.model, self.optimizer, train_loader, val_loader = self.accelerator.prepare(
            self.model, self.optimizer, train_loader, val_loader
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

        log_path = self.output_dir / "logs.json.txt"
        logger_context = (
            JsonLogger(str(log_path)) if self.accelerator.is_main_process else nullcontext(None)
        )
        wandb_context = (
            WandbLogger(self.wandb_cfg, self.cfg, self.output_dir)
            if self.accelerator.is_main_process
            else nullcontext(None)
        )

        last_metrics: dict | None = None
        with logger_context as json_logger, wandb_context as wandb_logger:
            outer_ctx = nullcontext(None)
            if self.accelerator.is_main_process:
                outer_ctx = tqdm(
                    total=int(self.train_cfg.num_epochs),
                    initial=int(self.epoch),
                    desc="Epochs",
                    dynamic_ncols=True,
                )
            with outer_ctx as outer_progress:
                while self.epoch < int(self.train_cfg.num_epochs):
                    self.model.train()
                    train_loss_sum = 0.0
                    train_count = 0
                    comp_sums: dict = {}  # per-modality loss components
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
                        raw_loss = self.model(batch)
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
                                self.ema_handler.step(self._base_model())

                        raw_loss_cpu = float(raw_loss.item())
                        if self.accelerator.is_main_process:
                            iterator.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_loss_sum += raw_loss_cpu
                        train_count += 1
                        self.global_step += 1

                        comps = getattr(self._base_model(), "_loss_components", None)
                        if comps is not None:
                            for ck, cv in comps.items():
                                comp_sums[ck] = comp_sums.get(ck, 0.0) + float(cv.item())

                        if self.train_cfg.max_train_steps is not None and (
                            batch_idx + 1
                        ) >= int(self.train_cfg.max_train_steps):
                            break

                    step_log["train_loss"] = self._distributed_mean(train_loss_sum, train_count)
                    for ck, cs in comp_sums.items():
                        step_log[f"train_{ck}"] = self._distributed_mean(cs, train_count)
                    step_log["lr"] = float(self.lr_scheduler.get_last_lr()[0])

                    eval_model = (
                        self.ema_handler.ema_model
                        if self.ema_handler is not None
                        else self._base_model()
                    )
                    eval_model.eval()

                    if len(val_dataset) > 0 and (self.epoch % int(self.train_cfg.val_every)) == 0:
                        val_loss = self._run_validation(eval_model, val_loader)
                        if val_loss is not None:
                            step_log["val_loss"] = val_loss
                            for ck, cv in self._last_val_components.items():
                                step_log[f"val_{ck}"] = cv

                    save_interval = int(self.train_cfg.save_every)
                    is_final_epoch = self.epoch == (int(self.train_cfg.num_epochs) - 1)
                    if self.accelerator.is_main_process:
                        payload = self._checkpoint_payload()
                        self.checkpoint_manager.save(
                            payload=payload,
                            metrics=step_log,
                            save_last=is_final_epoch or (self.epoch % save_interval) == 0,
                        )
                        completed_epoch = int(self.epoch) + 1
                        if (completed_epoch % save_interval) == 0:
                            periodic_path = (
                                self.output_dir / "checkpoints" / f"epoch_{completed_epoch:04d}.ckpt"
                            )
                            torch.save(payload, periodic_path)

                        assert json_logger is not None
                        json_logger.log(step_log)
                        assert wandb_logger is not None
                        wandb_logger.log(step_log, step=int(step_log["completed_epoch"]))
                        if outer_progress is not None:
                            outer_progress.set_postfix(
                                train_loss=f"{step_log['train_loss']:.5f}",
                                val_loss=(
                                    f"{step_log['val_loss']:.5f}"
                                    if "val_loss" in step_log
                                    else "n/a"
                                ),
                                refresh=False,
                            )
                            outer_progress.update(1)

                    last_metrics = dict(step_log)
                    self.epoch += 1
                    self.accelerator.wait_for_everyone()

        assert last_metrics is not None, "Training completed without metrics."
        if self.accelerator.is_main_process:
            with open(self.output_dir / "train_summary.json", "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "task_name": str(self.cfg.task.name),
                        "output_dir": str(self.output_dir),
                        "last_metrics": last_metrics,
                    },
                    handle,
                    indent=2,
                    sort_keys=True,
                )
        self.accelerator.wait_for_everyone()
        self.accelerator.end_training()
        return last_metrics
