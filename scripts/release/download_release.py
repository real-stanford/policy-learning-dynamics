"""Download data and/or checkpoints of the Hugging Face release into the layout the commands expect.

    source set_env.sh
    python -m scripts.release.download_release [--groups ...] [--tasks ...] [--epochs ...]

Groups (scripts/release/manifest.py):
    bc_pretraining_data     <task>.hdf5 + <task>_image_abs.zarr + <task>_lowdim_abs.zarr
                            -> $MIMICGEN_DATA_ROOT/core/   (--data_root overrides; zarr tars are extracted
                            there and deleted; an existing <name>.zarr is left untouched)
    bc_pretraining_ckpts    -> pretrained_ckpts/bc_pretraining_ckpts/<task>_image/{.hydra/config.yaml, checkpoints/}
    joint_generator_ckpts   -> pretrained_ckpts/joint_generator_ckpts/<task>_image/...
    joint_aug_ckpts         -> pretrained_ckpts/joint_aug_ckpts/<task>_image/...   (+ the run's eval results)
                            (--ckpt_root overrides pretrained_ckpts; relative to the current directory)

--epochs restricts the checkpoint groups to those epochs (e.g. --epochs 200 800 for the DICE-RL
examples); configs and eval results are always fetched. Re-running is incremental: files already
present with the right size are skipped by huggingface_hub.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

from huggingface_hub import snapshot_download

from scripts.release.manifest import (
    ALL_GROUPS,
    CKPT_GROUPS,
    GROUP_BC_DATA,
    HF_REPO_ID,
    HF_REPO_TYPE,
    PRETRAINED_ROOT,
    RUNS_BY_GROUP,
    TASKS,
    ckpt_name,
    data_files,
    task_dir,
)


def ckpt_patterns(group: str, task: str, epochs: list[int] | None) -> list[str]:
    d = f"{group}/{task_dir(task)}"
    shipped = RUNS_BY_GROUP[group][task].epochs
    if epochs is None:
        wanted = list(shipped)
    else:
        missing = sorted(set(epochs) - set(shipped))
        assert not missing, f"{group}/{task}: epochs {missing} are not part of the release (shipped: {list(shipped)})"
        wanted = sorted(set(epochs))
    patterns = [f"{d}/config.yaml", f"{d}/checkpoints_source_config.yaml", f"{d}/.hydra/*", f"{d}/eval*/*"]
    patterns += [f"{d}/checkpoints/{ckpt_name(e)}" for e in wanted]
    return patterns


def download_checkpoints(groups: list[str], tasks: list[str], epochs: list[int] | None, ckpt_root: Path) -> None:
    patterns = [p for g in groups for t in tasks for p in ckpt_patterns(g, t, epochs)]
    ckpt_root.mkdir(parents=True, exist_ok=True)
    print(f"[download] {len(patterns)} patterns -> {ckpt_root}", flush=True)
    snapshot_download(repo_id=HF_REPO_ID, repo_type=HF_REPO_TYPE, allow_patterns=patterns, local_dir=str(ckpt_root))
    for g in groups:
        for t in tasks:
            run = ckpt_root / g / task_dir(t)
            n = len(list((run / "checkpoints").glob("epoch_*.ckpt")))
            assert (run / ".hydra" / "config.yaml").is_file(), run
            print(f"[download] {run}: {n} checkpoints", flush=True)


def download_data(tasks: list[str], data_root: Path) -> None:
    core = data_root / "core"
    core.mkdir(parents=True, exist_ok=True)
    scratch = data_root / "_release_download"
    patterns = []
    for t in tasks:
        names = data_files(t)
        patterns.append(f"{GROUP_BC_DATA}/{names['hdf5']}")
        for key in ("image_zarr", "lowdim_zarr"):
            if not (core / names[key]).exists():
                patterns.append(f"{GROUP_BC_DATA}/{names[key]}.tar")
            else:
                print(f"[download] {core / names[key]} exists; not re-downloaded", flush=True)
    print(f"[download] {len(patterns)} files -> {scratch}", flush=True)
    snapshot_download(repo_id=HF_REPO_ID, repo_type=HF_REPO_TYPE, allow_patterns=patterns, local_dir=str(scratch))
    src_dir = scratch / GROUP_BC_DATA
    for t in tasks:
        names = data_files(t)
        hdf5 = core / names["hdf5"]
        if not hdf5.exists():
            shutil.move(str(src_dir / names["hdf5"]), str(hdf5))
        print(f"[download] {hdf5}  {hdf5.stat().st_size / 2**30:.2f} GiB", flush=True)
        for key in ("image_zarr", "lowdim_zarr"):
            zarr_dir = core / names[key]
            tar = src_dir / f"{names[key]}.tar"
            if zarr_dir.exists():
                continue
            assert tar.is_file(), tar
            subprocess.run(["tar", "-xf", str(tar), "-C", str(core)], check=True)
            assert (zarr_dir / ".zgroup").is_file() or (zarr_dir / ".zarray").is_file(), f"extraction failed: {zarr_dir}"
            tar.unlink()
            print(f"[download] {zarr_dir} extracted", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--groups", nargs="+", default=list(ALL_GROUPS), choices=list(ALL_GROUPS))
    ap.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    ap.add_argument("--epochs", nargs="+", type=int, default=None, help="checkpoint epochs to fetch (default: all shipped)")
    ap.add_argument("--data_root", default=None, help="default: $MIMICGEN_DATA_ROOT (set_env.sh)")
    ap.add_argument("--ckpt_root", default=PRETRAINED_ROOT)
    args = ap.parse_args()

    ckpt_groups = [g for g in args.groups if g in CKPT_GROUPS]
    if ckpt_groups:
        download_checkpoints(ckpt_groups, args.tasks, args.epochs, Path(args.ckpt_root).expanduser().resolve())
    if GROUP_BC_DATA in args.groups:
        if args.data_root is None:
            assert "MIMICGEN_DATA_ROOT" in os.environ, "source set_env.sh or pass --data_root"
            data_root = Path(os.environ["MIMICGEN_DATA_ROOT"])
        else:
            data_root = Path(args.data_root)
        download_data(args.tasks, data_root.expanduser().resolve())


if __name__ == "__main__":
    main()
