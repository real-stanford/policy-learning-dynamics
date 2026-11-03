# Beyond Validation Loss: Coverage and Reliability Dynamics in Robot Policy Pretraining

[[Paper](https://arxiv.org/abs/2611.01565)]&nbsp;&nbsp;[[Blog post](https://zhanyisun.github.io/robot-policy-learning-dynamics/)]&nbsp;&nbsp;[[Datasets and Checkpoints](https://huggingface.co/datasets/wintermelontree/robot_policy_overfitting_study)]

This repository contains the code, data and checkpoints behind the MimicGen experiments in the
paper. It covers four things: training image-based diffusion policies with behavior cloning and
tracking pass@k as training progresses, a joint generator over images, proprioception and
actions whose samples are mixed into policy training, a coverage-reliability criterion that
picks a checkpoint from held-out prediction errors without running any rollouts, and residual
RL finetuning from different checkpoints.

We use five MimicGen tasks: `threading_d2`, `square_d2`, `mug_cleanup_d1`,
`hammer_cleanup_d1` and `three_piece_assembly_d1`. All policies take images as input and
output absolute end-effector actions with a rotation-6d block. Each task comes with 1000
MimicGen demonstrations. We train on 200 of them and keep the remaining 800 as a held-out set.

- [Setup](#setup)
- [Data and checkpoints](#data-and-checkpoints)
- [1. Behavior cloning and pass@k eval](#1-behavior-cloning-and-passk-eval)
- [2. Joint generator](#2-joint-generator)
- [3. Behavior cloning with generated samples](#3-behavior-cloning-with-generated-samples)
- [4. Coverage-reliability checkpoint selection](#4-coverage-reliability-checkpoint-selection)
- [5. Residual RL finetuning](#5-residual-rl-finetuning)
- [Starting from raw MimicGen data](#starting-from-raw-mimicgen-data)
- [Layout and conventions](#layout-and-conventions)

## Setup

```bash
./install_deps.sh            # conda env "data": python 3.10, torch 2.9.1, diffusers 0.11.1; clones the
                             # robosuite / robomimic / mimicgen forks into third_party/ and installs them
conda activate data
source set_env.sh            # REPO_ROOT, OUTPUTS_ROOT=./outputs, MIMICGEN_DATA_ROOT, TRAINING_ROOT, EGL rendering, PYTHONPATH
python -m unittest discover -s tests -t . -p 'test_*.py'
```

Run everything from the repo root after sourcing the environment. Rollouts render offscreen
with EGL (`MUJOCO_GL=egl`), and each process picks its GPU through `MUJOCO_EGL_DEVICE_ID`.
Hammer Cleanup needs `third_party/robosuite-task-zoo`, which `set_env.sh` adds to `PYTHONPATH`.

## Data and checkpoints

All data and checkpoints are hosted on the Hugging Face Hub at
[`wintermelontree/robot_policy_overfitting_study`](https://huggingface.co/datasets/wintermelontree/robot_policy_overfitting_study).
The repo has four folders:

| folder | per task | size |
|---|---|---|
| `bc_pretraining_data/` | `<task>.hdf5` contains the 1000 MimicGen demos and is needed for every rollout and for the residual RL rewards. `<task>_image_abs.zarr.tar` is the training data of every policy and generator. `<task>_lowdim_abs.zarr.tar` is the simulator-state version of the same demos. | 37 GB total |
| `bc_pretraining_ckpts/<task>_image/` | the behavior cloning run: its config and the checkpoints at epochs 50, 100, ..., 800 | 33 GB per task |
| `joint_generator_ckpts/<task>_image/` | the joint generator run: its config and the checkpoints the jointaug distillation run sampled from | 0.4 to 1.7 GB per task |
| `joint_aug_ckpts/<task>_image/` | the jointaug distillation run: its config, the checkpoints at epochs 50 to 800, and its closed-loop evaluation results | 33 GB per task |

Each checkpoint stores the training config, the model and EMA weights (all evaluation uses
the EMA weights), the action and state normalizer, and the epoch and step counters.

You can download only what you need. The script is incremental, so running it again with more groups
or tasks adds to what is already on disk:

```bash
# all five tasks, everything (~370 GB)
python -m scripts.release.download_release

# data for one task
python -m scripts.release.download_release --groups bc_pretraining_data --tasks threading_d2

# every checkpoint group for one task
python -m scripts.release.download_release --groups bc_pretraining_ckpts joint_generator_ckpts joint_aug_ckpts --tasks hammer_cleanup_d1

# just the two base checkpoints the residual RL examples start from
python -m scripts.release.download_release --groups bc_pretraining_ckpts --tasks hammer_cleanup_d1 --epochs 250 800
```

After downloading, the files are laid out as follows:

```
$MIMICGEN_DATA_ROOT/core/            <task>.hdf5, <task>_image_abs.zarr, <task>_lowdim_abs.zarr   (= outputs/data/mimicgen/core)
pretrained_ckpts/bc_pretraining_ckpts/<task>_image/{.hydra/config.yaml, checkpoints/epoch_XXXX.ckpt}
pretrained_ckpts/joint_generator_ckpts/<task>_image/{.hydra/config.yaml, checkpoints/epoch_XXXX.ckpt}
pretrained_ckpts/joint_aug_ckpts/<task>_image/{.hydra/config.yaml, checkpoints/epoch_XXXX.ckpt, evals/ or eval_ddim_.../}
```

Each `pretrained_ckpts/<group>/<task>_image/` folder has the same structure as a run directory,
so you can pass it to any command below that expects one. Runs you train yourself are written
to `outputs/training/<task>_image/<datetime>_<exp_suffix>/` with the same structure.

There are three ways to use the release:

1. **Download the checkpoints** and go straight to the evaluation and analysis commands. You
   still need `bc_pretraining_data` for the task, because rollouts build the simulator from the
   hdf5 file, and the held-out banks and the residual RL expert buffer read the zarrs.
2. **Download the data and train the models yourself.** The training commands below reproduce
   each released run from its config.
3. **Start from the raw MimicGen release** and convert it yourself, as described in
   [Starting from raw MimicGen data](#starting-from-raw-mimicgen-data).

## 1. Behavior cloning and pass@k eval

We train a diffusion policy on 200 demonstrations. The policy is a ConditionalUnet1D over
8-step action chunks with a ResNet-18 image encoder, trained with 100 diffusion timesteps and
deployed with DDIM-10. Training only writes checkpoints. Evaluation is a separate step.

<details>
<summary>Training commands (one per task; 2 GPUs)</summary>

```bash
# threading_d2  ->  bc_pretraining_ckpts/threading_d2_image
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --multi_gpu --num_processes 2 --num_machines 1 --mixed_precision no --dynamo_backend no -m scripts.training.pretrain_image task=mimicgen/threading_d2_image action=absolute_rotation6d dataset.processing.max_train_episodes=200 training.dataloader.batch_size=512 checkpoint.save_last_ckpt=true runtime.exp_suffix=bc

# square_d2  ->  bc_pretraining_ckpts/square_d2_image
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --multi_gpu --num_processes 2 --num_machines 1 --mixed_precision no --dynamo_backend no -m scripts.training.pretrain_image task=mimicgen/square_d2_image action=absolute_rotation6d dataset.processing.max_train_episodes=200 training.dataloader.batch_size=512 checkpoint.save_last_ckpt=true runtime.exp_suffix=bc

# mug_cleanup_d1  ->  bc_pretraining_ckpts/mug_cleanup_d1_image
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --multi_gpu --num_processes 2 --num_machines 1 --mixed_precision no --dynamo_backend no -m scripts.training.pretrain_image task=mimicgen/mug_cleanup_d1_image action=absolute_rotation6d dataset.processing.max_train_episodes=200 training.dataloader.batch_size=512 checkpoint.save_last_ckpt=true runtime.exp_suffix=bc

# hammer_cleanup_d1  ->  bc_pretraining_ckpts/hammer_cleanup_d1_image
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --multi_gpu --num_processes 2 --num_machines 1 --mixed_precision no --dynamo_backend no -m scripts.training.pretrain_image task=mimicgen/hammer_cleanup_d1_image action=absolute_rotation6d dataset.processing.max_train_episodes=200 training.dataloader.batch_size=512 checkpoint.save_last_ckpt=true runtime.exp_suffix=bc

# three_piece_assembly_d1  ->  bc_pretraining_ckpts/three_piece_assembly_d1_image (1000 epochs; epochs 50..800 released)
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --multi_gpu --num_processes 2 --num_machines 1 --mixed_precision no --dynamo_backend no -m scripts.training.pretrain_image task=mimicgen/three_piece_assembly_d1_image action=absolute_rotation6d dataset.processing.max_train_episodes=200 training.dataloader.batch_size=512 task.training.num_epochs=1000 checkpoint.save_last_ckpt=true runtime.exp_suffix=bc
```

A run resumes from `checkpoints/last.ckpt` if you relaunch it with the same `runtime.run_name`.
</details>

The evaluation computes pass@10 on 500 held-out environment seeds (100000 to 100499) for every
checkpoint of a run, using the DDIM sampler with EMA weights. The command
below uses 4 GPUs with 24 parallel environments each. `--min_epoch` and `--max_epoch` specify
which checkpoints are evaluated.

```bash
# released run
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --multi_gpu --num_processes 4 --num_machines 1 --mixed_precision no --dynamo_backend no -m scripts.eval_checkpoints --run_dir pretrained_ckpts/bc_pretraining_ckpts/threading_d2_image --pass_at_k 10 --num_episodes 500 --num_envs_per_rank 24 --scheduler ddim --num_inference_steps 10 --eta 0.0

```

Results are written to `<run>/eval_ddim_num_steps_10_eta_0.0_pass_10/epoch_XXXX/summary.json`.
There, `success_rate_mean` is pass@10, i.e. whether any of the 10 attempts on a seed succeeded,
and `per_pass_success_rates` lists the 10 single-attempt success rates, whose mean is pass@1.
The per-episode rows of attempt 0 are in `summary.json` and those of attempts 1 to 9 in
`pass_<a>/summary.json`. To get pass@k for any k <= 10 with the unbiased estimator (averaged
over all k-subsets of the attempts rather than the first k), run `scripts/crb/analyze.py
--eval_dir`, which writes `passk.csv`.

## 2. Joint generator

The generator is a joint diffusion model over the image, the proprioceptive state and the
action chunk. It combines a dual-stream convolutional UNet with an MM-DiT, uses v-prediction
with 1000 diffusion steps, and is trained on the same 200-demo split as the policy (the mixer
in experiment 3 checks that this is the case).

<details>
<summary>Training commands (one per task; 2 GPUs)</summary>

```bash
# mug_cleanup_d1  ->  joint_generator_ckpts/mug_cleanup_d1_image
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --multi_gpu --num_processes 2 --num_machines 1 --mixed_precision bf16 --dynamo_backend no -m scripts.training.joint_generator_image task=mimicgen/mug_cleanup_d1_image action=absolute_rotation6d dataset.processing.max_train_episodes=200 training.dataloader.batch_size=128 training.training.gradient_accumulate_every=2 training.training.save_every=25 runtime.exp_suffix=joint_gen_img

# square_d2  ->  joint_generator_ckpts/square_d2_image
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --multi_gpu --num_processes 2 --num_machines 1 --mixed_precision bf16 --dynamo_backend no -m scripts.training.joint_generator_image task=mimicgen/square_d2_image action=absolute_rotation6d dataset.processing.max_train_episodes=200 training.dataloader.batch_size=256 training.training.gradient_accumulate_every=2 training.training.save_every=50 runtime.exp_suffix=joint_gen_img

# threading_d2  ->  joint_generator_ckpts/threading_d2_image
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --multi_gpu --num_processes 2 --num_machines 1 --mixed_precision bf16 --dynamo_backend no -m scripts.training.joint_generator_image task=mimicgen/threading_d2_image action=absolute_rotation6d dataset.processing.max_train_episodes=200 training.dataloader.batch_size=256 training.training.gradient_accumulate_every=2 training.training.save_every=50 runtime.exp_suffix=joint_gen_img

# hammer_cleanup_d1  ->  joint_generator_ckpts/hammer_cleanup_d1_image
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --multi_gpu --num_processes 2 --num_machines 1 --mixed_precision bf16 --dynamo_backend no -m scripts.training.joint_generator_image task=mimicgen/hammer_cleanup_d1_image action=absolute_rotation6d dataset.processing.max_train_episodes=200 training.dataloader.batch_size=256 training.training.gradient_accumulate_every=2 training.training.save_every=50 runtime.exp_suffix=joint_gen_img

# three_piece_assembly_d1  ->  joint_generator_ckpts/three_piece_assembly_d1_image
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --multi_gpu --num_processes 2 --num_machines 1 --mixed_precision bf16 --dynamo_backend no -m scripts.training.joint_generator_image task=mimicgen/three_piece_assembly_d1_image action=absolute_rotation6d dataset.processing.max_train_episodes=200 training.dataloader.batch_size=256 training.training.gradient_accumulate_every=2 training.training.save_every=50 runtime.exp_suffix=joint_gen_img
```
</details>

We decide which generator checkpoints to distill from with a nearest-neighbor ratio. For each
checkpoint we draw 1024 action chunks from the generator and compare their nearest-neighbor L2
distance (in normalized action space) to the generator's 200 training demos against the
distance to an equally sized held-out set drawn from the other 800 demos. `gamma_0` is the
first checkpoint whose ratio drops below 1.

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.eval_joint_gen_nn_ratio --gen_run outputs/training/mug_cleanup_d1_image_joint_gen_img/2026.06.29_14.46.20_joint_gen_img --policy_ckpt pretrained_ckpts/bc_pretraining_ckpts/mug_cleanup_d1_image/checkpoints/epoch_0200.ckpt --in_memory
#   -> <gen_run>/analysis/nn_ratio/summary.json (ratios per epoch) and selection.json (gamma_0 and the mixture)
```

`scripts/eval_joint_generator_image.py --ckpt <file>` reports the sample quality of a single
checkpoint: a feature-space nearest-neighbor ratio, the marginals and an image grid.

## 3. Behavior cloning with generated samples

This is the BC recipe from experiment 1 with the `training.jointaug_w_oa_joint_model.*`
options turned on, which mix generated samples into each training batch. The commands below
point at the released generators under `pretrained_ckpts/joint_generator_ckpts/`. To use your
own generator run instead, point `oa_model_dir` at `<gen_run>/checkpoints` and set
`ckpt_indices` to the epochs you selected.

<details>
<summary>Training commands (one per task; 2 GPUs)</summary>

```bash
# mug_cleanup_d1  ->  joint_aug_ckpts/mug_cleanup_d1_image
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --multi_gpu --num_processes 2 --num_machines 1 --mixed_precision no --dynamo_backend no -m scripts.training.pretrain_image task=mimicgen/mug_cleanup_d1_image action=absolute_rotation6d dataset.processing.max_train_episodes=200 training.dataloader.batch_size=512 checkpoint.save_last_ckpt=true training.jointaug_w_oa_joint_model.enabled=true training.jointaug_w_oa_joint_model.oa_model_dir=pretrained_ckpts/joint_generator_ckpts/mug_cleanup_d1_image/checkpoints training.jointaug_w_oa_joint_model.ckpt_indices=[100,200,300,400,500] training.jointaug_w_oa_joint_model.max_noisy_fraction=0.5 training.jointaug_w_oa_joint_model.start_epochs=75 training.jointaug_w_oa_joint_model.ramp_epochs=300 training.jointaug_w_oa_joint_model.num_inference_steps=100 training.jointaug_w_oa_joint_model.periodic_resampling=true training.jointaug_w_oa_joint_model.refresh_interval=10 runtime.exp_suffix=bc_oa_joint_aug

# square_d2  ->  joint_aug_ckpts/square_d2_image
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --multi_gpu --num_processes 2 --num_machines 1 --mixed_precision no --dynamo_backend no -m scripts.training.pretrain_image task=mimicgen/square_d2_image action=absolute_rotation6d dataset.processing.max_train_episodes=200 training.dataloader.batch_size=512 checkpoint.save_last_ckpt=true training.jointaug_w_oa_joint_model.enabled=true training.jointaug_w_oa_joint_model.oa_model_dir=pretrained_ckpts/joint_generator_ckpts/square_d2_image/checkpoints training.jointaug_w_oa_joint_model.ckpt_indices=[200] training.jointaug_w_oa_joint_model.max_noisy_fraction=0.5 training.jointaug_w_oa_joint_model.start_epochs=0 training.jointaug_w_oa_joint_model.ramp_epochs=300 training.jointaug_w_oa_joint_model.num_inference_steps=100 training.jointaug_w_oa_joint_model.periodic_resampling=true training.jointaug_w_oa_joint_model.refresh_interval=15 runtime.exp_suffix=bc_oa_joint_aug

# threading_d2  ->  joint_aug_ckpts/threading_d2_image
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --multi_gpu --num_processes 2 --num_machines 1 --mixed_precision no --dynamo_backend no -m scripts.training.pretrain_image task=mimicgen/threading_d2_image action=absolute_rotation6d dataset.processing.max_train_episodes=200 training.dataloader.batch_size=512 checkpoint.save_last_ckpt=true training.jointaug_w_oa_joint_model.enabled=true training.jointaug_w_oa_joint_model.oa_model_dir=pretrained_ckpts/joint_generator_ckpts/threading_d2_image/checkpoints training.jointaug_w_oa_joint_model.ckpt_indices=[200] training.jointaug_w_oa_joint_model.max_noisy_fraction=0.5 training.jointaug_w_oa_joint_model.start_epochs=0 training.jointaug_w_oa_joint_model.ramp_epochs=300 training.jointaug_w_oa_joint_model.num_inference_steps=100 training.jointaug_w_oa_joint_model.periodic_resampling=true training.jointaug_w_oa_joint_model.refresh_interval=15 runtime.exp_suffix=bc_oa_joint_aug

# hammer_cleanup_d1  ->  joint_aug_ckpts/hammer_cleanup_d1_image
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --multi_gpu --num_processes 2 --num_machines 1 --mixed_precision no --dynamo_backend no -m scripts.training.pretrain_image task=mimicgen/hammer_cleanup_d1_image action=absolute_rotation6d dataset.processing.max_train_episodes=200 training.dataloader.batch_size=512 checkpoint.save_last_ckpt=true training.jointaug_w_oa_joint_model.enabled=true training.jointaug_w_oa_joint_model.oa_model_dir=pretrained_ckpts/joint_generator_ckpts/hammer_cleanup_d1_image/checkpoints training.jointaug_w_oa_joint_model.ckpt_indices=[100,200,300,400,500] training.jointaug_w_oa_joint_model.max_noisy_fraction=0.5 training.jointaug_w_oa_joint_model.start_epochs=75 training.jointaug_w_oa_joint_model.ramp_epochs=300 training.jointaug_w_oa_joint_model.num_inference_steps=100 training.jointaug_w_oa_joint_model.periodic_resampling=true training.jointaug_w_oa_joint_model.refresh_interval=15 runtime.exp_suffix=bc_oa_joint_aug

# three_piece_assembly_d1  ->  joint_aug_ckpts/three_piece_assembly_d1_image 
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --multi_gpu --num_processes 2 --num_machines 1 --mixed_precision no --dynamo_backend no -m scripts.training.pretrain_image task=mimicgen/three_piece_assembly_d1_image action=absolute_rotation6d dataset.processing.max_train_episodes=200 training.dataloader.batch_size=512 checkpoint.save_last_ckpt=true training.jointaug_w_oa_joint_model.enabled=true training.jointaug_w_oa_joint_model.oa_model_dir=pretrained_ckpts/joint_generator_ckpts/three_piece_assembly_d1_image/checkpoints training.jointaug_w_oa_joint_model.ckpt_indices=[100,200,300,400,500] training.jointaug_w_oa_joint_model.max_noisy_fraction=0.5 training.jointaug_w_oa_joint_model.start_epochs=75 training.jointaug_w_oa_joint_model.ramp_epochs=300 training.jointaug_w_oa_joint_model.num_inference_steps=100 training.jointaug_w_oa_joint_model.periodic_resampling=true training.jointaug_w_oa_joint_model.refresh_interval=15 runtime.exp_suffix=bc_oa_joint_aug
```

Once the mixer becomes active, the trainer prints
`[oa_joint_mixer] noisy fraction starts ramping at epoch ...`. Evaluate the new run with the
command from experiment 1.
</details>

For each task, `joint_aug_ckpts` contains the `.hydra/config.yaml` of the run that the command
above reproduces, its checkpoints, and the closed-loop evaluation results of that run
(`evals/`, or `eval_ddim_num_steps_10_eta_0.0_pass_10/` for hammer).

### Choosing the generator checkpoints

The commands above use the generator checkpoints from the runs in the paper. If you train
your own generator (experiment 2), you can select its checkpoints with the nearest-neighbor ratio.
`scripts/eval_joint_gen_nn_ratio.py` scores every saved checkpoint of the run and writes
`selection.json`, which names `gamma_0`, the first checkpoint whose ratio drops below 1.

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.eval_joint_gen_nn_ratio --gen_run outputs/training/hammer_cleanup_d1_image_joint_gen_img/<run> --policy_ckpt pretrained_ckpts/bc_pretraining_ckpts/hammer_cleanup_d1_image/checkpoints/epoch_0200.ckpt --in_memory
# then: ...oa_model_dir=outputs/training/hammer_cleanup_d1_image_joint_gen_img/<run>/checkpoints ...ckpt_indices=[<the epochs in selection.json>]
```

In our experience the distillation runs are not sensitive to the exact set of checkpoints.
Changing how many checkpoints are mixed, or which epochs near `gamma_0` they come from, moved
pass@k by less than the run-to-run spread, as long as the earliest checkpoint stays close to
`gamma_0` rather than deep into the generator's own overfitting.

## 4. Coverage-reliability checkpoint selection

This experiment picks a checkpoint of a BC run from held-out action-prediction errors alone,
without running any rollouts. The full procedure is described in `CRB_SELECTION.md`. In short,
we take N held-out windows and, for every checkpoint, we draw K = 64 action chunks per window from
a fixed set of initial noises. For each window we compute the error quantiles at 19
percentiles. We then normalize each percentile in log space to [0, 1] across checkpoints,
average the lower half of the percentiles into a coverage risk and the upper half into a
reliability risk, and select the checkpoint with the smallest maximum of the two. Closed-loop
results come in only afterwards, as an oracle reference in the figure.

The walkthrough below runs threading on the released BC run. A run you trained yourself works
the same way if you substitute its directory for
`pretrained_ckpts/bc_pretraining_ckpts/threading_d2_image`.

```bash
# 4a. the run's 200 training demos and the 800 held-out ones (recomputed from the run's config)
python -m scripts.crb.make_heldout_split --run pretrained_ckpts/bc_pretraining_ckpts/threading_d2_image --out_dir outputs/probes/threading_d2_image/split

# 4b. the evaluation bank: N held-out windows (seed 20260824) and K = 64 fixed initial noises per window (seed 222)
CUDA_VISIBLE_DEVICES=0 python -m scripts.crb.build_eval_bank --ckpt pretrained_ckpts/bc_pretraining_ckpts/threading_d2_image/checkpoints/epoch_0050.ckpt --episodes_json outputs/probes/threading_d2_image/split/heldout.json --out outputs/probes/threading_d2_image/eval_bank_16384_k64 --n_windows 16384 --k_draws 64 --in_memory

# 4c. per-checkpoint error cache: 64 DDIM-10 samples per bank state from the fixed noise, EMA weights,
#     per-element MSE of the raw action chunk against the demo chunk; every checkpoint of the run
#     (--epochs 50 100 ... restricts it, e.g. to shard over GPUs). -> <run>/analysis/crb/offline_errors/epoch_XXXX.npz
CUDA_VISIBLE_DEVICES=0 python -m scripts.crb.cache_errors --run pretrained_ckpts/bc_pretraining_ckpts/threading_d2_image --bank outputs/probes/threading_d2_image/eval_bank_16384_k64 --sanity

# 4d. quantiles, best-of-k, the two risks and the selected checkpoint (CPU, offline)
#     -> <run>/analysis/crb/{quantiles_full.csv,quantiles.csv,bok.csv,crb.csv,meta.json,surface.npz}
#     Add --eval_dir <run>/eval_ddim_num_steps_10_eta_0.0_pass_32 (a 32-attempt eval from experiment 1) to also
#     write passk.csv and mark the closed-loop oracle in the figure; the selection itself never uses it.
python -m scripts.crb.analyze --cache_dir pretrained_ckpts/bc_pretraining_ckpts/threading_d2_image/analysis/crb/offline_errors --out_dir pretrained_ckpts/bc_pretraining_ckpts/threading_d2_image/analysis/crb --title "Threading (D2), 200 demos"

# 4e. sampled action clouds at a few held-out states for the first and last checkpoint (figure panel b)
CUDA_VISIBLE_DEVICES=0 python -m scripts.crb.dump_action_clouds --run pretrained_ckpts/bc_pretraining_ckpts/threading_d2_image --bank outputs/probes/threading_d2_image/eval_bank_16384_k64 --epochs 50 800 --select

# 4f. the figure: quantile clouds, action clouds, the two risks and the selected checkpoint
python -m scripts.crb.plot_quantile_clouds_crb --analysis_dir pretrained_ckpts/bc_pretraining_ckpts/threading_d2_image/analysis/crb --samples pretrained_ckpts/bc_pretraining_ckpts/threading_d2_image/analysis/crb/qualitative/samples.npz --out outputs/vis/fig_quantile_clouds_crb
```

## 5. Residual RL finetuning

Residual RL finetuning (`dice_rl/`) trains a residual actor on top of a frozen diffusion-policy
checkpoint. The three hammer example commands below start from BC epoch 250
and 800 with the expert residual-imitation term enabled, and from epoch 800 of the distillation
run with that term disabled. Each run uses 2 GPUs. `env.n_envs` and `train.batch_size` are
global values that get split across GPUs.

<details>
<summary>DICE-RL finetuning commands</summary>

```bash
# base = BC epoch 250
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --multi_gpu --num_processes 2 --num_machines 1 --mixed_precision no --dynamo_backend no -m scripts.dice_rl.finetune base_policy.checkpoint=pretrained_ckpts/bc_pretraining_ckpts/hammer_cleanup_d1_image/checkpoints/epoch_0250.ckpt env.n_envs=64 train.gradient_steps=40 model.use_q_normalization=true model.bc_loss_weight=200 model.condition_residual_on_base_action=true model.asymmetric_critic.enabled=true model.asymmetric_critic.lowdim_task=mimicgen/hammer_cleanup_d1_lowdim model.asymmetric_critic.actor_on_state=true model.critic_next_state.enabled=true model.expert_residual_imitation.enabled=true runtime.exp_suffix=hammer200img_bc_ep250_e64_ASYM_actorstate_nextstate_eri

# base = BC epoch 800
CUDA_VISIBLE_DEVICES=2,3 accelerate launch --multi_gpu --num_processes 2 --num_machines 1 --mixed_precision no --dynamo_backend no -m scripts.dice_rl.finetune base_policy.checkpoint=pretrained_ckpts/bc_pretraining_ckpts/hammer_cleanup_d1_image/checkpoints/epoch_0800.ckpt env.n_envs=64 train.gradient_steps=40 model.use_q_normalization=true model.bc_loss_weight=200 model.condition_residual_on_base_action=true model.asymmetric_critic.enabled=true model.asymmetric_critic.lowdim_task=mimicgen/hammer_cleanup_d1_lowdim model.asymmetric_critic.actor_on_state=true model.critic_next_state.enabled=true model.expert_residual_imitation.enabled=true runtime.exp_suffix=hammer200img_bc_ep800_e64_ASYM_actorstate_nextstate_eri

# base = epoch 800 of the distillation run (experiment 3)
CUDA_VISIBLE_DEVICES=4,5 accelerate launch --multi_gpu --num_processes 2 --num_machines 1 --mixed_precision no --dynamo_backend no -m scripts.dice_rl.finetune base_policy.checkpoint=pretrained_ckpts/joint_aug_ckpts/hammer_cleanup_d1_image/checkpoints/epoch_0800.ckpt env.n_envs=64 train.gradient_steps=40 model.use_q_normalization=true model.bc_loss_weight=200 model.condition_residual_on_base_action=true model.asymmetric_critic.enabled=true model.asymmetric_critic.lowdim_task=mimicgen/hammer_cleanup_d1_lowdim model.asymmetric_critic.actor_on_state=true model.critic_next_state.enabled=true model.expert_residual_imitation.enabled=false runtime.exp_suffix=hammer200img_aug_ep800_e64_ASYM_actorstate_nextstate
```

## Starting from raw MimicGen data

If you would rather start from the raw MimicGen release than from our zarrs, the following
commands download the demonstrations and convert them:

```bash
# the 1000-demo hdf5 of each task from the MimicGen Hugging Face release -> $MIMICGEN_DATA_ROOT/core/<task>.hdf5
python -m scripts.dataset.download_mimicgen --tasks threading_d2 square_d2 mug_cleanup_d1 hammer_cleanup_d1 three_piece_assembly_d1

# image zarr with absolute rotation-6d actions (what the policies and generators train on) -> <task>_image_abs.zarr
python -m scripts.dataset.convert_mimicgen_to_zarr --task threading_d2 --modality image --action-pose-repr absolute --rotation-rep rotation_6d
python -m scripts.dataset.convert_mimicgen_to_zarr --task square_d2 --modality image --action-pose-repr absolute --rotation-rep rotation_6d
python -m scripts.dataset.convert_mimicgen_to_zarr --task mug_cleanup_d1 --modality image --action-pose-repr absolute --rotation-rep rotation_6d
python -m scripts.dataset.convert_mimicgen_to_zarr --task hammer_cleanup_d1 --modality image --action-pose-repr absolute --rotation-rep rotation_6d
python -m scripts.dataset.convert_mimicgen_to_zarr --task three_piece_assembly_d1 --modality image --action-pose-repr absolute --rotation-rep rotation_6d

# lowdim zarr of the same demos (simulator state; the residual RL critic reads it) -> <task>_lowdim_abs.zarr
python -m scripts.dataset.convert_mimicgen_to_zarr --task hammer_cleanup_d1 --modality lowdim --action-pose-repr absolute --rotation-rep rotation_6d
```

The absolute-action conversion replays the controller in simulation, so it needs the EGL setup
from `set_env.sh`. The files produced here are exactly what `bc_pretraining_data` ships.

## Layout

```
configs/                Hydra configs: diffusion_image.yaml (BC), joint_generator_image.yaml, dice_rl/finetune.yaml,
                        task/mimicgen/<task>_{image,lowdim}.yaml, model/, training/, dataset/, action/
datasets/               CanonicalImageDataset (zarr windows, seeded train/val split), MimicGen hdf5 -> zarr conversion
envs/                   task registry; parallel image environments (subprocess workers, EGL rendering)
models/                 ConditionalUnet1D, DiffusionUnetImagePolicy, ResNet observation encoder, JointImageOADiTModel
utils/training_utils/   factories, SupervisedTrainer (BC), GeneratorTrainer, evaluation (closed-loop rollouts),
                        deployed_policy (checkpoint loading + sampler for offline probes), EMA, checkpointing
utils/oa_joint_image_mixer.py   generated-sample mixer attached to the policy for experiment 3
dice_rl/                residual RL finetuning on top of a frozen diffusion-policy checkpoint
scripts/training/       pretrain_image.py, joint_generator_image.py
scripts/eval_checkpoints.py     pass@k closed-loop evaluation of every checkpoint of a run
scripts/eval_joint_gen_nn_ratio.py   generator checkpoint selection (nearest-neighbor ratio)
scripts/crb/            held-out bank, error cache, quantile / best-of-k / pass@k analysis, CRB selection, figure
scripts/dice_rl/        finetune.py
scripts/dataset/        download_mimicgen.py, convert_mimicgen_to_zarr.py
scripts/release/        download_release.py (Hub -> local layout); stage_release.py / upload_release.py built the release
tests/                  unit tests (synthetic data, CPU)
```

## License

The code and the released checkpoints are under the MIT License (see `LICENSE`). The
demonstrations in `bc_pretraining_data` are MimicGen data and remain under the
[MimicGen license](https://github.com/NVlabs/mimicgen).
