"""End-to-end smoke check of a downloaded release subset: one task, one GPU, a few minutes.

    source set_env.sh
    python -m scripts.release.download_release --groups bc_pretraining_data joint_generator_ckpts --tasks hammer_cleanup_d1
    python -m scripts.release.download_release --groups bc_pretraining_ckpts --tasks hammer_cleanup_d1 --epochs 200 800
    python -m scripts.release.download_release --groups joint_aug_ckpts --tasks hammer_cleanup_d1 --epochs 600
    CUDA_VISIBLE_DEVICES=0 python -m scripts.release.smoke_check --task hammer_cleanup_d1

Exercises every code path the five experiments use, at toy sizes, on the downloaded files:
  1. pass@k rollouts of a BC checkpoint and of a distillation checkpoint (experiment 1 / 3 eval)
  2. held-out split, a 32-state x 32-draw bank, the error cache of two checkpoints and the CRB analysis (experiment 4)
  3. two tiny BC epochs with the generated-sample mixer drawing from the released generator (experiment 3)
  4. a DICE-RL --smoke run on the BC checkpoint with the asymmetric critic (experiment 5; reads the hdf5 and
     the lowdim zarr)
Outputs go to --out (default $OUTPUTS_ROOT/release_smoke/<task>); training runs to $TRAINING_ROOT as usual.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from scripts.release.manifest import (
    AUG_GEN_INDICES,
    GROUP_AUG_CKPTS,
    GROUP_BC_CKPTS,
    GROUP_GEN_CKPTS,
    PRETRAINED_ROOT,
    TASKS,
    ckpt_name,
    task_dir,
)


def run(step: str, cmd: list[str]) -> None:
    print(f"\n[smoke] === {step}\n[smoke] {' '.join(cmd)}", flush=True)
    t0 = time.time()
    subprocess.run(cmd, check=True)
    print(f"[smoke] {step}: ok ({time.time() - t0:.0f}s)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, choices=list(TASKS))
    ap.add_argument("--ckpt_root", default=PRETRAINED_ROOT)
    ap.add_argument("--bc_epoch", type=int, default=200, help="BC checkpoint for the rollout, bank and DICE-RL steps")
    ap.add_argument("--cache_epochs", type=int, nargs=2, default=[200, 800], help="BC checkpoints for the error cache")
    ap.add_argument("--aug_epoch", type=int, default=600, help="distillation checkpoint for the rollout step")
    ap.add_argument("--out", default=None)
    ap.add_argument("--skip", nargs="*", default=[], choices=["eval", "crb", "mixer", "dice"])
    args = ap.parse_args()

    py = sys.executable
    root = Path(args.ckpt_root).expanduser().resolve()
    bc = root / GROUP_BC_CKPTS / task_dir(args.task)
    gen = root / GROUP_GEN_CKPTS / task_dir(args.task)
    aug = root / GROUP_AUG_CKPTS / task_dir(args.task)
    if args.out is None:
        assert "OUTPUTS_ROOT" in os.environ, "source set_env.sh or pass --out"
        out = Path(os.environ["OUTPUTS_ROOT"]) / "release_smoke" / args.task
    else:
        out = Path(args.out)
    out = out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    bc_ckpt = bc / "checkpoints" / ckpt_name(args.bc_epoch)
    aug_ckpt = aug / "checkpoints" / ckpt_name(args.aug_epoch)
    for p in (bc_ckpt, aug_ckpt, bc / ".hydra" / "config.yaml"):
        assert p.is_file(), f"missing {p}"
    gen_indices = list(AUG_GEN_INDICES[args.task])
    for e in gen_indices:
        assert (gen / "checkpoints" / ckpt_name(e)).is_file(), gen / "checkpoints" / ckpt_name(e)
    eval_args = ["--pass_at_k", "2", "--num_episodes", "4", "--num_envs_per_rank", "2",
                 "--scheduler", "ddim", "--num_inference_steps", "10", "--eta", "0.0"]

    if "eval" not in args.skip:
        for name, ckpt in (("bc", bc_ckpt), ("aug", aug_ckpt)):
            d = out / f"eval_{name}_{ckpt.stem}"
            run(f"rollouts of {name} {ckpt.stem}",
                [py, "-m", "scripts.eval_checkpoints", "--checkpoints", str(ckpt), "--output_dir", str(d)] + eval_args)
            s = json.load(open(d / ckpt.stem / "summary.json"))
            print(f"[smoke] {name} {ckpt.stem}: pass@2 {s['success_rate_mean']:.2f}, per pass {s['per_pass_success_rates']}")

    if "crb" not in args.skip:
        run("held-out split", [py, "-m", "scripts.crb.make_heldout_split", "--run", str(bc), "--out_dir", str(out / "split")])
        run("bank (32 states x 32 draws)",
            [py, "-m", "scripts.crb.build_eval_bank", "--ckpt", str(bc_ckpt), "--episodes_json", str(out / "split" / "heldout.json"),
             "--out", str(out / "bank"), "--n_windows", "32", "--k_draws", "32"])
        run("error cache", [py, "-m", "scripts.crb.cache_errors", "--run", str(bc), "--bank", str(out / "bank"),
                            "--epochs", *map(str, args.cache_epochs), "--out", str(out / "offline_errors"), "--sanity"])
        run("CRB analysis", [py, "-m", "scripts.crb.analyze", "--cache_dir", str(out / "offline_errors"),
                             "--out_dir", str(out / "crb"), "--title", f"{args.task} smoke"])
        print(f"[smoke] selected epoch (2 candidates): {json.load(open(out / 'crb' / 'meta.json'))['selected_epoch']}")

    if "mixer" not in args.skip:
        oa = "training.jointaug_w_oa_joint_model"
        run("BC --smoke with the generated-sample mixer",
            [py, "-m", "scripts.training.pretrain_image", "--smoke", f"task=mimicgen/{task_dir(args.task)}",
             "action=absolute_rotation6d", "dataset.processing.in_memory=false", "training.dataloader.batch_size=16",
             "training.validation.batch_size=16", "training.logging.wandb.enabled=false",
             f"{oa}.enabled=true", f"{oa}.oa_model_dir={gen / 'checkpoints'}",
             f"{oa}.ckpt_indices=[{','.join(map(str, gen_indices))}]", f"{oa}.max_noisy_fraction=0.5",
             f"{oa}.start_epochs=0", f"{oa}.ramp_epochs=1", f"{oa}.num_inference_steps=10",
             f"{oa}.periodic_resampling=true", f"{oa}.refresh_interval=1", "runtime.exp_suffix=release_smoke"])

    if "dice" not in args.skip:
        run("DICE-RL --smoke",
            [py, "-m", "scripts.dice_rl.finetune", "--smoke", f"base_policy.checkpoint={bc_ckpt}",
             "model.asymmetric_critic.enabled=true", f"model.asymmetric_critic.lowdim_task=mimicgen/{args.task}_lowdim",
             "model.asymmetric_critic.actor_on_state=true", "model.critic_next_state.enabled=true",
             "logging.wandb.enabled=false", "runtime.exp_suffix=release_smoke"])

    print(f"\n[smoke] all steps passed for {args.task}; outputs under {out}", flush=True)


if __name__ == "__main__":
    main()
