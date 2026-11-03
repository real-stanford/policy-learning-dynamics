"""Single source of truth for the Hugging Face release of this repo's data and checkpoints.

Everything that scripts/release/{stage_release,upload_release,download_release}.py and the
README refer to is defined here: the release repo, its folder layout, the five tasks, the
research-repo runs each artifact comes from, and which checkpoints of each run are shipped.

Release layout (one dataset repo, HF_REPO_ID):

    bc_pretraining_data/<task>.hdf5                    raw MimicGen demos (1000 per task): environment
                                                       metadata for rollouts + rewards for DICE-RL
    bc_pretraining_data/<task>_image_abs.zarr.tar      image zarr, absolute rotation-6d actions (BC / generator)
    bc_pretraining_data/<task>_lowdim_abs.zarr.tar     lowdim zarr of the same demos (DICE-RL asymmetric critic)
    bc_pretraining_ckpts/<task>_image/                 .hydra/config.yaml + checkpoints/epoch_0050..0800.ckpt
    joint_generator_ckpts/<task>_image/                .hydra/config.yaml + checkpoints/<epochs the joint-aug run used>
    joint_aug_ckpts/<task>_image/                      .hydra/config.yaml + checkpoints/ + the run's eval results

Shipped checkpoints keep cfg / model_state / ema_state / normalizer_state / epoch / global_step
and drop optimizer_state / scheduler_state (see stage_release.py). Their stored config has the
research machine's absolute data paths replaced by ${oc.env:MIMICGEN_DATA_ROOT}/..., so it
resolves against the user's own set_env.sh.

Downloaded, the artifacts land in (download_release.py):

    $MIMICGEN_DATA_ROOT/core/<task>.hdf5, <task>_image_abs.zarr, <task>_lowdim_abs.zarr
    pretrained_ckpts/<group>/<task>_image/...          (group = one of the three *_ckpts folders)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

HF_REPO_ID = "wintermelontree/robot_policy_overfitting_study"
HF_REPO_TYPE = "dataset"

# Research repo (the archive the release is assembled from).
RESEARCH_OUTPUTS = Path("/store/real/zhanyis/offline_learning_data_study/outputs")
RESEARCH_TRAINING = RESEARCH_OUTPUTS / "training"
RESEARCH_MIMICGEN = RESEARCH_OUTPUTS / "data" / "mimicgen"

# Strings substituted into the checkpoints' stored configs.
RELEASE_MIMICGEN_PREFIX = "${oc.env:MIMICGEN_DATA_ROOT}"
RELEASE_TRAINING_ROOT = "${oc.env:TRAINING_ROOT}"
PRETRAINED_ROOT = "pretrained_ckpts"      # download root for the three checkpoint groups, relative to the repo root

GROUP_BC_DATA = "bc_pretraining_data"
GROUP_BC_CKPTS = "bc_pretraining_ckpts"
GROUP_GEN_CKPTS = "joint_generator_ckpts"
GROUP_AUG_CKPTS = "joint_aug_ckpts"
CKPT_GROUPS = (GROUP_BC_CKPTS, GROUP_GEN_CKPTS, GROUP_AUG_CKPTS)
ALL_GROUPS = (GROUP_BC_DATA,) + CKPT_GROUPS

TASKS = ("mug_cleanup_d1", "square_d2", "threading_d2", "hammer_cleanup_d1", "three_piece_assembly_d1")

BC_EPOCHS = tuple(range(50, 801, 50))


@dataclass(frozen=True)
class RunSource:
    """A research-repo run shipped as one `<task>_image/` folder.

    run:             the run directory whose .hydra/config.yaml (and eval results) are shipped
    epochs:          checkpoints/epoch_XXXX.ckpt to ship
    checkpoints_run: run directory supplying the checkpoints when it is not `run` (three-piece
                     joint-aug: the 07/15 run's checkpoints were not kept; the 08/01 run of the
                     same recipe with ckpt_indices=[200] supplies them)
    eval_subdir:     eval results directory inside `run` to ship verbatim (None: none)
    """

    run: Path
    epochs: tuple[int, ...]
    checkpoints_run: Path | None = None
    eval_subdir: str | None = None

    @property
    def checkpoints_dir(self) -> Path:
        return (self.run if self.checkpoints_run is None else self.checkpoints_run) / "checkpoints"


BC_RUNS: dict[str, RunSource] = {
    "mug_cleanup_d1": RunSource(RESEARCH_TRAINING / "mug_cleanup_d1_image/2026.08.20_13.08.54_abs_pass_at_k", BC_EPOCHS),
    "square_d2": RunSource(RESEARCH_TRAINING / "square_d2_image/2026.09.02_16.05.36_abs_pass_at_k", BC_EPOCHS),
    "threading_d2": RunSource(RESEARCH_TRAINING / "threading_d2_image/2026.07.26_18.47.23_abs_pass_at_k", BC_EPOCHS),
    "hammer_cleanup_d1": RunSource(RESEARCH_TRAINING / "hammer_cleanup_d1_image/2026.08.28_09.31.06_abs_pass_at_k", BC_EPOCHS),
    "three_piece_assembly_d1": RunSource(
        RESEARCH_TRAINING / "three_piece_assembly_d1_image/2026.08.29_21.14.56_abs_pass_at_k", BC_EPOCHS
    ),  # trained 1000 epochs; epochs 50..800 shipped
}

# Generator checkpoints the joint-aug runs consumed (`ckpt_indices`, in the order the runs listed them).
AUG_GEN_INDICES: dict[str, tuple[int, ...]] = {
    "mug_cleanup_d1": (200, 300, 400, 500, 100),
    "square_d2": (200,),
    "threading_d2": (200,),
    "hammer_cleanup_d1": (200, 300, 400, 500, 100),
    "three_piece_assembly_d1": (200, 300, 400, 500, 100),
}

GEN_RUNS: dict[str, RunSource] = {
    "mug_cleanup_d1": RunSource(
        RESEARCH_TRAINING / "mug_cleanup_d1_image_joint_gen_img/2026.06.29_14.46.20_joint_gen_img",
        tuple(sorted(AUG_GEN_INDICES["mug_cleanup_d1"])),
    ),
    "square_d2": RunSource(
        RESEARCH_TRAINING / "square_d2_image_joint_gen_img/2026.07.04_01.21.27_joint_gen_img",
        tuple(sorted(AUG_GEN_INDICES["square_d2"])),
    ),
    "threading_d2": RunSource(
        RESEARCH_TRAINING / "threading_d2_image_joint_gen_img/2026.07.04_01.22.03_joint_gen_img",
        tuple(sorted(AUG_GEN_INDICES["threading_d2"])),
    ),
    "hammer_cleanup_d1": RunSource(
        RESEARCH_TRAINING / "hammer_cleanup_d1_image_joint_gen_img/2026.07.04_01.36.02_joint_gen_img",
        tuple(sorted(AUG_GEN_INDICES["hammer_cleanup_d1"])),
    ),
    "three_piece_assembly_d1": RunSource(
        RESEARCH_TRAINING / "three_piece_assembly_d1_image_joint_gen_img/2026.07.12_22.44.11_joint_gen_img",
        tuple(sorted(AUG_GEN_INDICES["three_piece_assembly_d1"])),
    ),
}

AUG_RUNS: dict[str, RunSource] = {
    "mug_cleanup_d1": RunSource(
        RESEARCH_TRAINING / "mug_cleanup_d1_image/2026.07.05_19.55.19_abs_pass_at_k", BC_EPOCHS, eval_subdir="evals"
    ),  # saved every 25 epochs; every 50 shipped
    "square_d2": RunSource(
        RESEARCH_TRAINING / "square_d2_image/2026.07.31_00.04.32_abs_pass_at_k", BC_EPOCHS, eval_subdir="evals"
    ),
    "threading_d2": RunSource(
        RESEARCH_TRAINING / "threading_d2_image/2026.08.01_18.20.06_abs_pass_at_k", BC_EPOCHS, eval_subdir="evals"
    ),
    "hammer_cleanup_d1": RunSource(
        RESEARCH_TRAINING / "hammer_cleanup_d1_image/2026.10.27_00.48.01_bc_oa_joint_aug",
        BC_EPOCHS,
        eval_subdir="eval_ddim_num_steps_10_eta_0.0_pass_10",
    ),
    "three_piece_assembly_d1": RunSource(
        RESEARCH_TRAINING / "three_piece_assembly_d1_image/2026.07.15_00.28.35_abs_pass_at_k",
        tuple(range(50, 751, 50)),
        checkpoints_run=RESEARCH_TRAINING / "three_piece_assembly_d1_image/2026.08.01_21.53.32_abs_pass_at_k",
        eval_subdir="evals",
    ),
}

RUNS_BY_GROUP: dict[str, dict[str, RunSource]] = {
    GROUP_BC_CKPTS: BC_RUNS,
    GROUP_GEN_CKPTS: GEN_RUNS,
    GROUP_AUG_CKPTS: AUG_RUNS,
}


def task_dir(task: str) -> str:
    """Folder name of a task inside every checkpoint group (= the run's task.name)."""
    assert task in TASKS, task
    return f"{task}_image"


def ckpt_name(epoch: int) -> str:
    return f"epoch_{epoch:04d}.ckpt"


def data_files(task: str) -> dict[str, str]:
    """File names under $MIMICGEN_DATA_ROOT/core (zarrs are shipped as <name>.tar)."""
    assert task in TASKS, task
    return {
        "hdf5": f"{task}.hdf5",
        "image_zarr": f"{task}_image_abs.zarr",
        "lowdim_zarr": f"{task}_lowdim_abs.zarr",
    }


def release_generator_dir(task: str) -> str:
    """`training.jointaug_w_oa_joint_model.oa_model_dir` after download (relative to the repo root)."""
    return f"{PRETRAINED_ROOT}/{GROUP_GEN_CKPTS}/{task_dir(task)}/checkpoints"


def _check() -> None:
    for task in TASKS:
        assert GEN_RUNS[task].epochs == tuple(sorted(AUG_GEN_INDICES[task])), task
        for group in CKPT_GROUPS:
            src = RUNS_BY_GROUP[group][task]
            assert src.run.name.startswith("2026."), src.run
            assert len(src.epochs) == len(set(src.epochs)) and src.epochs == tuple(sorted(src.epochs)), (group, task)


_check()
