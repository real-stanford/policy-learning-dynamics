#!/usr/bin/env python
"""Download MimicGen datasets from Hugging Face."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import hf_hub_download

from envs.task_registry import get_family_data_root, get_task_entry, list_task_names

HF_MIMICGEN_REPO_ID = "amandlek/mimicgen_datasets"


def _download_file(entry, destination: Path, overwrite: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    hf_hub_download(
        repo_id=HF_MIMICGEN_REPO_ID,
        repo_type="dataset",
        subfolder=entry.dataset_type,
        filename=entry.artifact_filenames["hdf5"],
        local_dir=destination.parent.parent,
        force_download=overwrite,
    )
    assert destination.exists(), f"Expected downloaded file at '{destination}'."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Base MimicGen data root. Defaults to MIMICGEN_DATA_ROOT.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["stack_d1"],
        choices=list(list_task_names("mimicgen")) + ["all"],
        help="MimicGen tasks to download. Use 'all' for all locally supported tasks.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Redownload files even if they already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned output paths without downloading.",
    )
    args = parser.parse_args()

    data_root = get_family_data_root("mimicgen", args.data_root)
    requested_tasks = (
        list(list_task_names("mimicgen")) if "all" in args.tasks else args.tasks
    )

    tasks_to_download = []
    plan = {}
    for task_name in requested_tasks:
        entry = get_task_entry("mimicgen", task_name)
        destination = entry.artifact_path(data_root, "hdf5")
        plan[task_name] = {"hdf5": str(destination)}
        if args.overwrite or not destination.exists():
            tasks_to_download.append(task_name)

    if not args.dry_run:
        for task_name in tasks_to_download:
            entry = get_task_entry("mimicgen", task_name)
            destination = entry.artifact_path(data_root, "hdf5")
            _download_file(entry, destination, overwrite=args.overwrite)

    print(json.dumps({"data_root": str(data_root), "tasks": plan}, indent=2))


if __name__ == "__main__":
    main()
