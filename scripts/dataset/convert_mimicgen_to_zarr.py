#!/usr/bin/env python
"""Convert MimicGen HDF5 datasets into the canonical zarr schema."""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import validate_canonical_zarr
from datasets.mimicgen_conversion import convert_mimicgen_hdf5_to_zarr
from envs.task_registry import get_family_data_root, get_task_entry, list_task_names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=list(list_task_names("mimicgen")),
        help="MimicGen task name from the local registry.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Optional input HDF5 path override.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output zarr path override.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Base MimicGen data root. Defaults to MIMICGEN_DATA_ROOT.",
    )
    parser.add_argument(
        "--modality",
        type=str,
        choices=("lowdim", "image"),
        required=True,
        help="Canonical output modality.",
    )
    parser.add_argument(
        "--action-pose-repr",
        type=str,
        default="delta",
        choices=("delta", "absolute"),
        help="Action pose representation to store (source datasets are delta; optional absolute conversion).",
    )
    parser.add_argument(
        "--rotation-rep",
        type=str,
        default="axis_angle",
        choices=("axis_angle", "rotation_6d"),
    )
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument(
        "--truncate",
        action="store_true",
        help=(
            "Truncate every trajectory to a single terminal success: cut at the "
            "first success step and store rewards as 0,...,0,1. Demos that never "
            "trip the reward checker keep their full length with the last step "
            "forced to success. Adds a per-step 'rewards' field. Without this flag "
            "the full trajectory is kept and the full per-step rewards are stored. "
            "When set and --output is not given, '_trunc' is appended to the name."
        ),
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=None,
        help=(
            "Worker processes for episode-level conversion. "
            "Default: auto (min(8, os.cpu_count()))."
        ),
    )
    args = parser.parse_args()
    data_root = get_family_data_root("mimicgen", args.data_root)
    entry = get_task_entry("mimicgen", args.task)
    input_path = (
        Path(args.input) if args.input else entry.artifact_path(data_root, "hdf5")
    )
    if args.output:
        output_path = Path(args.output)
    else:
        artifact = "image_zarr" if args.modality == "image" else "lowdim_zarr"
        output_path = entry.artifact_path(data_root, artifact)
        suffix = output_path.suffix
        stem = output_path.stem
        if args.action_pose_repr == "absolute":
            stem = stem + "_abs"
        if args.truncate:
            stem = stem + "_trunc"
        output_path = output_path.with_name(stem + suffix)

    output_path = convert_mimicgen_hdf5_to_zarr(
        input_path,
        output_path,
        task_name=args.task,
        modality=args.modality,
        action_pose_repr=args.action_pose_repr,
        rotation_rep=args.rotation_rep,
        max_episodes=args.max_episodes,
        jobs=args.jobs,
        truncate=args.truncate,
    )
    errors = validate_canonical_zarr(output_path)
    if errors:
        raise SystemExit("Schema validation failed:\n- " + "\n- ".join(errors))
    print(output_path)


if __name__ == "__main__":
    main()
