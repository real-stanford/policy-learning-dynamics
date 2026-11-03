---
license: mit
task_categories:
- robotics
tags:
- robotics
- imitation-learning
- diffusion-policy
- mimicgen
- robosuite
pretty_name: Robot policy overfitting study
size_categories:
- 100B<n<1T
---

# Robot policy overfitting study: data and checkpoints

Companion release for the `robot_policy_overfitting_study` code. Five MimicGen manipulation
tasks (`mug_cleanup_d1`, `square_d2`, `threading_d2`, `hammer_cleanup_d1`,
`three_piece_assembly_d1`), image-based diffusion policies, absolute end-effector actions with
a rotation-6d block.

| folder | content |
|---|---|
| `bc_pretraining_data/` | per task: `<task>.hdf5` (the 1000 MimicGen demos: environment metadata for rollouts and rewards), `<task>_image_abs.zarr.tar` (the image dataset every policy and generator was trained on), `<task>_lowdim_abs.zarr.tar` (simulator-state version of the same demos, read by the DICE-RL critic) |
| `bc_pretraining_ckpts/<task>_image/` | behaviour-cloning run: `.hydra/config.yaml` and `checkpoints/epoch_0050.ckpt` ... `epoch_0800.ckpt` (16 per task) |
| `joint_generator_ckpts/<task>_image/` | joint (image, state, action) generator run: config and the checkpoints the distillation runs sampled from |
| `joint_aug_ckpts/<task>_image/` | behaviour cloning with generated-sample distillation: config, 16 checkpoints (15 for three-piece), and the closed-loop eval results of the run |

Each checkpoint holds `cfg` (the resolved training config, with data paths written as
`${oc.env:MIMICGEN_DATA_ROOT}/core/...`), `model_state`, `ema_state` (the weights used for
every evaluation), `normalizer_state`, `epoch` and `global_step`. Optimizer state was dropped.
`release_manifest.json` lists every file with the run it came from.

The code repo's README explains how to download a subset (`scripts/release/download_release.py`)
and rerun the five experiments.

License: the checkpoints and configs are released under the MIT License, like the code. The
demonstrations in `bc_pretraining_data/` are MimicGen data and remain under the
[MimicGen license](https://github.com/NVlabs/mimicgen).
