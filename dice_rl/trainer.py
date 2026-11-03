from __future__ import annotations

import json
import os
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.training_utils.json_logger import JsonLogger
from utils.training_utils.wandb_logger import WandbLogger

from dice_rl.actions import EnvActionConverter
from dice_rl.base_policy import FrozenDiffusionBasePolicy
from dice_rl.checkpoint import load_rl_checkpoint, save_rl_checkpoint
from dice_rl.expert_data import build_expert_npz, load_expert_buffer
from dice_rl.metrics import EpisodeResult, RunningEpisodeStats
from dice_rl.model import EXPLORATION_STRATEGIES, DistillResidualRLModel
from dice_rl.replay_buffer import ChunkReplayBuffer, MixedSampler, adaptive_expert_ratio
from dice_rl.scheduler import CosineAnnealingWarmupRestarts
from dice_rl.vector_env import RLEnvRunner


def _grad_norm(parameters) -> float:
    total = 0.0
    for p in parameters:
        if p.grad is not None:
            total += float(p.grad.detach().norm(2)) ** 2
    return total**0.5


class DiceRLTrainer:
    def __init__(self, cfg, output_dir: Path) -> None:
        self.cfg = cfg
        self.output_dir = Path(output_dir)
        # torch's NCCL heartbeat monitor (default 480 s) aborts a rank whose watchdog
        # sees no progress -- e.g. rank 1 idling in broadcast_object_list while rank 0
        # builds the expert cache. Align it with the 2 h collective timeout below.
        os.environ.setdefault("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", "7200")
        pg_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))
        self.accelerator = Accelerator(kwargs_handlers=[pg_kwargs])
        self.device = self.accelerator.device
        self.rank = int(self.accelerator.process_index)
        self.world_size = int(self.accelerator.num_processes)
        self.is_main = bool(self.accelerator.is_main_process)
        self.distributed = self.world_size > 1
        if self.distributed:
            assert dist.is_initialized()

        train_cfg = cfg.train
        env_cfg = cfg.env
        assert int(env_cfg.n_envs) % self.world_size == 0, (
            f"env.n_envs={env_cfg.n_envs} must be divisible by num_processes={self.world_size}."
        )
        assert int(train_cfg.batch_size) % self.world_size == 0, (
            f"train.batch_size={train_cfg.batch_size} must be divisible by "
            f"num_processes={self.world_size}."
        )
        self.n_envs_rank = int(env_cfg.n_envs) // self.world_size
        self.batch_rank = int(train_cfg.batch_size) // self.world_size
        self.seed = int(env_cfg.seed)
        torch.manual_seed(self.seed + 1000 * self.rank)
        np.random.seed(self.seed + 1000 * self.rank)

        # ---- frozen base policy (checkpoint cfg is the source of truth) ----
        bp_cfg = cfg.base_policy
        assert bp_cfg.checkpoint is not None, "base_policy.checkpoint is required."
        self.base = FrozenDiffusionBasePolicy(
            checkpoint_path=str(bp_cfg.checkpoint),
            device=self.device,
            num_inference_steps=int(bp_cfg.num_inference_steps),
            ddim_eta=float(bp_cfg.ddim_eta),
            state_key=str(bp_cfg.state_key),
            autocast=bp_cfg.autocast,
        )
        assert self.base.ddim_eta == 0.0, (
            "base_policy.ddim_eta must be 0: the residual actor conditions on z and "
            "needs pi_pre(s, z) to be deterministic."
        )
        self.pre_cfg = self.base.cfg
        self.horizon = self.base.horizon
        self.action_dim = self.base.action_dim
        self.gamma = float(train_cfg.gamma)

        m = cfg.model

        # ---- optional asymmetric critic (training-time critic on the simulator's
        # lowdim state; policy/actor keep acting from images unchanged) ----
        ac_cfg = getattr(m, "asymmetric_critic", None)
        asymmetric_critic = None
        self._aux_state_keys: tuple = ()
        self._aux_state_zarr = None
        self._aux_state_dim = 0
        if ac_cfg is not None and bool(ac_cfg.enabled):
            lowdim_task = str(ac_cfg.lowdim_task)
            task_yaml = (
                Path(__file__).resolve().parents[1] / "configs" / "task" / f"{lowdim_task}.yaml"
            )
            assert task_yaml.exists(), f"asymmetric_critic.lowdim_task: missing {task_yaml}"
            tcfg = OmegaConf.to_container(OmegaConf.load(task_yaml), resolve=False)
            state_keys = [str(k) for k in tcfg["observation"]["state_keys"]]
            state_dim = int(tcfg["observation"]["obs_dim"])
            state_zarr = getattr(ac_cfg, "state_zarr", None)
            if state_zarr is None:
                template = str(tcfg["data"]["zarr_path"])
                state_zarr = template.replace(
                    "${oc.env:MIMICGEN_DATA_ROOT}", os.environ["MIMICGEN_DATA_ROOT"]
                ).replace("${action.zarr_suffix}", str(self.pre_cfg.action.zarr_suffix))
            state_zarr = Path(str(state_zarr))
            assert state_zarr.exists(), f"asymmetric_critic lowdim zarr not found: {state_zarr}"
            self._aux_state_keys = tuple(state_keys)
            self._aux_state_zarr = state_zarr
            self._aux_state_dim = state_dim
            self.base.enable_state_features(state_dim)
            asymmetric_critic = {
                "state_dim": state_dim,
                "actor_on_state": bool(getattr(ac_cfg, "actor_on_state", False)),
            }

        # ---- DICE+ knobs: default off = byte-identical to the base recipe ----
        cns = getattr(m, "critic_next_state", None)
        self.critic_next_state_weight = (
            float(cns.weight) if cns is not None and bool(cns.enabled) else 0.0
        )
        eri = getattr(m, "expert_residual_imitation", None)
        self.expert_imitation_weight = (
            float(eri.weight) if eri is not None and bool(eri.enabled) else 0.0
        )
        self.with_next1 = self.critic_next_state_weight > 0

        # ---- model + DDP ----
        self.model = DistillResidualRLModel(
            base_policy=self.base,
            actor_hidden_dims=[int(v) for v in m.actor_hidden_dims],
            critic_hidden_dims=[int(v) for v in m.critic_hidden_dims],
            activation_type=str(m.activation_type),
            use_layernorm=bool(m.use_layernorm),
            critic_ensemble_size=int(m.critic_ensemble_size),
            q_depends_on_noise=bool(m.q_depends_on_noise),
            td_loss=str(m.td_loss),
            bc_loss_weight=float(m.bc_loss_weight),
            use_q_normalization=bool(m.use_q_normalization),
            num_multi_z_for_actor_loss=int(m.num_multi_z_for_actor_loss),
            num_next_noise_samples=int(m.num_next_noise_samples),
            condition_residual_on_base_action=bool(m.condition_residual_on_base_action),
            disable_q_loss_for_expert_data=bool(m.disable_q_loss_for_expert_data),
            disable_td_loss_for_expert_data=bool(m.disable_td_loss_for_expert_data),
            always_retain_bc_loss_for_expert_data=bool(m.always_retain_bc_loss_for_expert_data),
            use_soft_q_filtering=bool(m.use_soft_q_filtering),
            q_filtering_warmup_steps=int(m.q_filtering_warmup_steps),
            q_underestimation_threshold=float(m.q_underestimation_threshold),
            replay_flow_warmup_steps=int(m.replay_flow_warmup_steps),
            actor_zero_init_last_layer=bool(m.actor_zero_init_last_layer),
            clip_q=None if m.clip_q is None else tuple(m.clip_q),
            asymmetric_critic=asymmetric_critic,
            critic_next_state_weight=self.critic_next_state_weight,
            expert_imitation_weight=self.expert_imitation_weight,
        ).to(self.device)
        self.model.device = self.device
        if self.distributed:
            local = int(self.device.index) if self.device.type == "cuda" else None
            ids = [local] if local is not None else None
            self.model.actor_fwd = DDP(self.model.actor, device_ids=ids)
            self.model.critic_fwd = DDP(self.model.critic, device_ids=ids)
            # DDP broadcast rank-0 weights at construction; re-sync the target.
            self.model.target_critic.load_state_dict(self.model.critic.state_dict())

        # ---- optimizers / schedulers (dice-rl) ----
        self.actor_optimizer = torch.optim.Adam(
            self.model.actor.parameters(),
            lr=float(train_cfg.actor_lr),
            weight_decay=float(train_cfg.actor_weight_decay),
        )
        self.critic_optimizer = torch.optim.Adam(
            self.model.critic.parameters(),
            lr=float(train_cfg.critic_lr),
            weight_decay=float(train_cfg.critic_weight_decay),
        )
        self.actor_scheduler = None
        self.critic_scheduler = None
        if bool(train_cfg.use_lr_scheduler):
            sched = train_cfg.lr_scheduler
            self.actor_scheduler = CosineAnnealingWarmupRestarts(
                self.actor_optimizer,
                first_cycle_steps=int(sched.first_cycle_steps),
                cycle_mult=1.0,
                max_lr=float(train_cfg.actor_lr),
                min_lr=float(sched.min_lr),
                warmup_steps=int(sched.warmup_steps),
                gamma=1.0,
            )
            self.critic_scheduler = CosineAnnealingWarmupRestarts(
                self.critic_optimizer,
                first_cycle_steps=int(sched.first_cycle_steps),
                cycle_mult=1.0,
                max_lr=float(train_cfg.critic_lr),
                min_lr=float(sched.min_lr),
                warmup_steps=int(sched.warmup_steps),
                gamma=1.0,
            )

        # ---- buffers ----
        replay_cfg = cfg.replay
        self.online = ChunkReplayBuffer(
            capacity_frames=int(replay_cfg.capacity_frames),
            feature_dim=self.base.feature_dim,
            action_dim=self.action_dim,
            horizon=self.horizon,
            n_obs_steps=self.base.n_obs_steps,
            n_step=int(replay_cfg.n_step),
            gamma=self.gamma,
            bootstrap_on_truncation=bool(replay_cfg.bootstrap_on_truncation),
            data_source=0,
            device=self.device,
            obs_dtype=str(getattr(replay_cfg, "obs_dtype", "float32")),
            with_next1=self.with_next1,
        )
        self.expert = None
        if bool(cfg.expert.enabled):
            self.expert = self._build_expert_buffer()
        if asymmetric_critic is not None:
            # Fixed per-dim state standardization for the critic, from the expert
            # demos (identical on every rank -> DDP-consistent buffer update).
            if self.expert is not None and self.expert.frame_ptr > 0:
                state = self.expert.obs[: self.expert.frame_ptr, self.base.policy_feature_dim :]
                state = state.to(torch.float32)
                mu, sd = state.mean(dim=0), state.std(dim=0)
                self.model.critic.set_standardization(mu, sd)
                self.model.target_critic.set_standardization(mu, sd)
            elif self.is_main:
                print(
                    "[dice_rl] asymmetric_critic: expert disabled -> state standardization "
                    "left at identity.",
                    flush=True,
                )
        self.sampler = MixedSampler(
            self.online,
            self.expert,
            generator=torch.Generator(device=self.device).manual_seed(self.seed + 7 * self.rank),
        )

        # ---- envs ----
        max_steps = env_cfg.max_episode_steps
        if max_steps is None:
            max_steps = int(self.pre_cfg.task.evaluation.max_steps)
        self.max_episode_steps = int(max_steps)
        self.runner = RLEnvRunner(
            cfg=self.pre_cfg,
            num_envs=self.n_envs_rank,
            max_episode_steps=self.max_episode_steps,
            success_steps_before_termination=int(env_cfg.success_steps_before_termination),
            seed_start=self.seed + 1_000_000 * self.rank,
            encode=self._encode_np,
            action_converter=EnvActionConverter(
                action_pose_repr=self.base.action_pose_repr,
                rotation_rep=self.base.rotation_rep,
            ),
            n_obs_steps=self.base.n_obs_steps,
            aux_state_keys=self._aux_state_keys,
        )

        # ---- bookkeeping ----
        self.strategy = str(train_cfg.online_explore_strategy)
        assert self.strategy in EXPLORATION_STRATEGIES, self.strategy
        self.num_exploration_samples = int(train_cfg.num_exploration_samples)
        self.stats = RunningEpisodeStats(window=int(train_cfg.success_window))
        self._last_actor_metrics: dict[str, float] = {}
        self.iteration = 0
        self.global_env_steps = 0
        self.ckpt_dir = self.output_dir / "checkpoints"
        if bool(train_cfg.resume_from):
            payload = load_rl_checkpoint(
                Path(str(train_cfg.resume_from)),
                model=self.model,
                actor_optimizer=self.actor_optimizer,
                critic_optimizer=self.critic_optimizer,
                actor_scheduler=self.actor_scheduler,
                critic_scheduler=self.critic_scheduler,
                map_location=self.device,
            )
            self.iteration = int(payload["iteration"])
            self.global_env_steps = int(payload["env_steps"])
            # Every rank loads the same file (actor, critic AND target), so no resync.
        if self.is_main:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self._print_header()

    # ------------------------------------------------------------------ setup
    def _encode_np(self, states: np.ndarray, rgb: np.ndarray) -> torch.Tensor:
        return self.base.encode_frames(torch.from_numpy(states), torch.from_numpy(rgb))

    def _build_expert_buffer(self) -> ChunkReplayBuffer:
        ex = self.cfg.expert
        cache_dir = Path(str(ex.cache_dir))
        npz_path = None
        if self.is_main:
            npz_path = build_expert_npz(
                self.base,
                cache_dir=cache_dir,
                encode_batch_size=int(ex.encode_batch_size),
                no_success_policy=str(ex.no_success_demos),
                aux_state_zarr=self._aux_state_zarr,
                aux_state_dim=self._aux_state_dim if self._aux_state_zarr is not None else None,
            )
        if self.distributed:
            payload = [str(npz_path) if npz_path is not None else None]
            dist.broadcast_object_list(payload, src=0)
            npz_path = Path(payload[0])
        return load_expert_buffer(
            npz_path,
            base_policy=self.base,
            n_step=int(ex.n_step),
            gamma=self.gamma,
            device=self.device,
            bootstrap_chunks=(
                None if ex.bootstrap_chunks is None else int(ex.bootstrap_chunks)
            ),
            with_next1=self.with_next1,
        )

    def _print_header(self) -> None:
        print(f"Run directory: {self.output_dir}", flush=True)
        print(f"Checkpoint directory: {self.ckpt_dir}", flush=True)
        print(json.dumps({"base_policy": self.base.summary()}, indent=2), flush=True)
        print(
            f"world_size={self.world_size} envs/rank={self.n_envs_rank} "
            f"batch/rank={self.batch_rank} max_episode_steps={self.max_episode_steps} "
            f"online_capacity={self.online.capacity} "
            f"expert_chunks={0 if self.expert is None else len(self.expert)}",
            flush=True,
        )

    # -------------------------------------------------------------- collectives
    def _all_reduce_sum(self, value: float) -> float:
        if not self.distributed:
            return float(value)
        tensor = torch.tensor([float(value)], device=self.device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return float(tensor.item())

    def _all_reduce_min(self, value: float) -> float:
        if not self.distributed:
            return float(value)
        tensor = torch.tensor([float(value)], device=self.device)
        dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
        return float(tensor.item())

    def _all_reduce_mean_dict(self, values: dict[str, float]) -> dict[str, float]:
        if not self.distributed or not values:
            return values
        keys = sorted(values)
        tensor = torch.tensor([values[k] for k in keys], device=self.device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= self.world_size
        return {k: float(v) for k, v in zip(keys, tensor.tolist())}

    def _assert_replicas_in_sync(self) -> None:
        """All ranks must hold identical actor / critic / target weights (DDP
        gradient averaging + identical target updates). Cheap collective; run at
        checkpoint iterations so a divergence is caught early and loudly."""
        if not self.distributed:
            return
        for name, module in (
            ("actor", self.model.actor),
            ("critic", self.model.critic),
            ("target_critic", self.model.target_critic),
        ):
            flat = torch.cat([p.detach().reshape(-1) for p in module.parameters()])
            reference = flat.clone()
            dist.broadcast(reference, src=0)
            max_diff = float((flat - reference).abs().max())
            assert max_diff == 0.0, (
                f"rank {self.rank}: {name} parameters diverged from rank 0 "
                f"(max |diff| = {max_diff}) at iteration {self.iteration}."
            )

    def _gather_episode_results(self, local: list[EpisodeResult]) -> list[EpisodeResult]:
        if not self.distributed:
            return local
        gathered: list[list[EpisodeResult] | None] = [None] * self.world_size
        dist.all_gather_object(gathered, local)
        merged: list[EpisodeResult] = []
        for rows in gathered:
            assert rows is not None
            merged.extend(rows)
        return merged

    # ------------------------------------------------------------------ steps
    @torch.no_grad()
    def _collect(self) -> dict[str, float]:
        cond = self.runner.current_cond()
        actions, _noise = self.model.explore(
            cond,
            num_samples=self.num_exploration_samples,
            strategy=self.strategy,
            training_step=self.iteration,
        )
        raw = self.base.unnormalize(actions)
        finished = self.runner.step(actions, raw)
        chunks_added = 0
        for episode in finished:
            chunks_added += self.online.add_episode(episode)
        return {"episodes_finished": float(len(finished)), "chunks_added": float(chunks_added)}

    def _update(self) -> dict[str, float]:
        cfg = self.cfg.train
        model = self.model
        it = self.iteration
        expert_ratio = adaptive_expert_ratio(
            it,
            enabled=bool(self.cfg.expert.use_adaptive_ratio),
            fixed_ratio=float(self.cfg.expert.ratio),
            start=float(self.cfg.expert.adaptive_start),
            end=float(self.cfg.expert.adaptive_end),
            num_steps=int(self.cfg.expert.adaptive_steps),
        )
        metrics: dict[str, float] = {}
        update_actor = (it + 1) % int(cfg.actor_update_freq) == 0
        for _ in range(int(cfg.gradient_steps)):
            batch = self.sampler.sample(self.batch_rank, expert_ratio)
            # dice-rl regenerates the batch noise every update; the critic only
            # consumes it when model.q_depends_on_noise=true.
            batch_noise = model.sample_noise(batch.batch_size)
            q_over = None
            if bool(cfg.log_q_overestimation):
                with torch.no_grad():
                    q_pred = model._q_min(model.critic, batch.cond, batch_noise, batch.action)
                    q_over = q_pred - batch.mc_return
                    online = (batch.data_source == 0).to(torch.float32)
                    metrics["q_analysis/overestimation_mean"] = float(q_over.mean())
                    metrics["q_analysis/predicted_q_mean"] = float(q_pred.mean())
                    metrics["q_analysis/mc_return_mean"] = float(batch.mc_return.mean())
                    metrics["q_analysis/online_fraction"] = float(online.mean())
                    metrics["q_analysis/reward_mean"] = float(batch.reward.mean())
                    metrics["q_analysis/done_fraction"] = float(batch.done.mean())
            target = model.target_q(batch.next_cond, batch.reward, batch.done, batch.n_eff, self.gamma)

            if update_actor:
                actor_out = model.actor_loss(
                    batch.cond,
                    data_source=batch.data_source,
                    training_step=it,
                    q_overestimation=q_over,
                    demo_action=batch.action,  # only read when expert_imitation_weight > 0
                )
                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_out["actor_total"].backward()
                metrics["train/actor_grad_norm"] = _grad_norm(model.actor.parameters())
                if cfg.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.actor.parameters(), float(cfg.max_grad_norm))
                self.actor_optimizer.step()
                if self.actor_scheduler is not None:
                    self.actor_scheduler.step()
                metrics["train/actor_loss"] = float(actor_out["actor_total"].detach())
                for key in (
                    "actor_q_loss", "actor_bc_loss", "residual_rms", "q_advantage_mean",
                    "better_than_base_frac", "base_q_mean", "current_q_mean", "bc_filter_mean",
                ):
                    metrics[f"train/{key}"] = float(actor_out[key])
                if "actor_imitation_loss" in actor_out:
                    metrics["train/actor_imitation_loss"] = float(actor_out["actor_imitation_loss"])

            critic_out = model.critic_loss(
                batch.cond,
                batch.action,
                target,
                data_source=batch.data_source,
                noise=batch_noise,
                next1_cond=batch.next1_cond,  # None unless critic_next_state is on
            )
            self.critic_optimizer.zero_grad(set_to_none=True)
            critic_out["critic_total"].backward()
            metrics["train/critic_grad_norm"] = _grad_norm(model.critic.parameters())
            if cfg.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.critic.parameters(), float(cfg.max_grad_norm))
            self.critic_optimizer.step()
            if self.critic_scheduler is not None:
                self.critic_scheduler.step()
            for key in (
                "critic_loss", "critic_q_mean", "critic_q_std_across_ensemble",
                "target_q_mean", "target_q_max",
            ):
                metrics[f"train/{key}"] = float(critic_out[key])
            if "critic_next_loss" in critic_out:
                metrics["train/critic_next_loss"] = float(critic_out["critic_next_loss"])
            if (it + 1) % int(cfg.critic_target_update_freq) == 0:
                model.update_target(float(cfg.tau))
        metrics["train/expert_ratio"] = expert_ratio
        metrics["train/actor_lr"] = float(self.actor_optimizer.param_groups[0]["lr"])
        metrics["train/critic_lr"] = float(self.critic_optimizer.param_groups[0]["lr"])
        # Actor metrics are produced only on actor-update iterations
        # (actor_update_freq); carry the latest ones so every log row has them.
        if update_actor:
            self._last_actor_metrics = {k: metrics[k] for k in self._ACTOR_KEYS}
            if "train/actor_imitation_loss" in metrics:
                self._last_actor_metrics["train/actor_imitation_loss"] = metrics["train/actor_imitation_loss"]
        else:
            metrics.update(self._last_actor_metrics)
        return metrics

    _ACTOR_KEYS = (
        "train/actor_loss", "train/actor_grad_norm", "train/actor_q_loss",
        "train/actor_bc_loss", "train/residual_rms", "train/q_advantage_mean",
        "train/better_than_base_frac", "train/base_q_mean", "train/current_q_mean",
        "train/bc_filter_mean",
    )

    # -------------------------------------------------------------------- run
    def run(self) -> dict:
        cfg = self.cfg.train
        num_iters = int(cfg.num_train_iters)
        log_freq = int(cfg.log_freq)
        save_freq = int(cfg.save_freq)
        min_chunks = int(cfg.batch_size)  # dice-rl: updates start once the online buffer holds a batch
        self.runner.reset_all()
        log_path = self.output_dir / "logs.json.txt"
        logger_ctx = JsonLogger(str(log_path)) if self.is_main else _NullContext()
        wandb_ctx = (
            WandbLogger(self.cfg.logging.wandb, self.cfg, self.output_dir)
            if self.is_main
            else _NullContext()
        )
        last_update_metrics: dict[str, float] = {}
        timers = {"env": 0.0, "update": 0.0}
        t_start = time.time()
        with logger_ctx as json_logger, wandb_ctx as wandb_logger:
            while self.iteration < num_iters:
                t0 = time.time()
                collect_metrics = self._collect()
                timers["env"] += time.time() - t0

                # dice-rl gate (global online chunks >= batch) PLUS: every rank
                # samples its share from its OWN buffer, so all ranks must hold at
                # least one chunk. Both reductions are collective -> the decision
                # is identical on every rank (DDP steps stay in lockstep).
                online_chunks_global = self._all_reduce_sum(len(self.online))
                online_chunks_min_rank = self._all_reduce_min(len(self.online))
                do_update = (
                    online_chunks_global >= min_chunks
                    and online_chunks_min_rank >= 1
                    and self.iteration % int(cfg.update_freq) == 0
                )
                if do_update:
                    t0 = time.time()
                    last_update_metrics = self._update()
                    timers["update"] += time.time() - t0

                self.iteration += 1
                if self.iteration % log_freq == 0 or self.iteration == num_iters:
                    self._log(
                        json_logger,
                        wandb_logger,
                        collect_metrics,
                        last_update_metrics,
                        timers,
                        t_start,
                        did_update=do_update,
                    )
                if self.iteration % save_freq == 0 or self.iteration == num_iters:
                    self._assert_replicas_in_sync()
                    self._save(f"iter_{self.iteration:07d}.pt")
                    self._save("last.pt")
            summary = self.stats.summary()
        self.runner.close()
        if self.is_main:
            with open(self.output_dir / "summary.json", "w", encoding="utf-8") as handle:
                json.dump(
                    {"iterations": self.iteration, "env_steps": self.global_env_steps, **summary},
                    handle,
                    indent=2,
                    sort_keys=True,
                )
        return summary

    def _log(self, json_logger, wandb_logger, collect_metrics, update_metrics, timers, t_start, *, did_update) -> None:
        local_results = [
            EpisodeResult(
                iteration=self.iteration,
                rank=self.rank,
                env_index=index,
                success=o.success,
                length=o.length,
                truncated=o.truncated,
            )
            for index, o in enumerate(self.runner.pop_completed())
        ]
        self.stats.update(self._gather_episode_results(local_results))
        self.global_env_steps = int(self._all_reduce_sum(self.runner.total_env_steps))
        reduced = self._all_reduce_mean_dict(update_metrics)
        # buffer/* are PER-RANK means (each rank owns its own online buffer).
        buffer_stats = {f"buffer/{k}": v for k, v in self.online.stats().items()}
        buffer_stats = self._all_reduce_mean_dict(buffer_stats)
        if not self.is_main:
            return
        payload = {
            "iteration": self.iteration,
            "env_steps": self.global_env_steps,
            "wall_time_s": time.time() - t_start,
            "time/env_s_per_iter": timers["env"] / max(1, self.iteration),
            "time/update_s_per_iter": timers["update"] / max(1, self.iteration),
            "updates_active": float(did_update),
            **{f"train/{k}": v for k, v in self.stats.summary().items()},
            **buffer_stats,
            **reduced,
        }
        json_logger.log(payload)
        wandb_logger.log(payload, step=self.global_env_steps)
        s = self.stats.summary()
        msg = (
            f"[it {self.iteration}] env_steps={self.global_env_steps} "
            f"succ@{self.stats.window}={s['success_rate']:.3f} "
            f"eps={int(s['total_episodes'])} len={s['episode_length']:.0f} "
            f"buf={int(len(self.online))}"
        )
        if "train/critic_loss" in reduced:
            msg += (
                f" critic={reduced['train/critic_loss']:.4f} "
                f"q={reduced.get('train/current_q_mean', float('nan')):.3f} "
                f"res={reduced.get('train/residual_rms', float('nan')):.4f} "
                f"bc={reduced.get('train/actor_bc_loss', float('nan')):.5f}"
            )
        msg += f" env={payload['time/env_s_per_iter']:.2f}s upd={payload['time/update_s_per_iter']:.2f}s"
        print(msg, flush=True)

    def _save(self, name: str) -> None:
        if not self.is_main:
            return
        save_rl_checkpoint(
            self.ckpt_dir / name,
            cfg=self.cfg,
            model=self.model,
            actor_optimizer=self.actor_optimizer,
            critic_optimizer=self.critic_optimizer,
            actor_scheduler=self.actor_scheduler,
            critic_scheduler=self.critic_scheduler,
            iteration=self.iteration,
            env_steps=self.global_env_steps,
            base_policy_summary=self.base.summary(),
        )


class _NullContext:
    def __enter__(self):
        return None

    def __exit__(self, *args) -> None:
        return None
