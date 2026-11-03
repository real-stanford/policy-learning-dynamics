"""Upload the staged release folder (scripts/release/stage_release.py) to the Hugging Face Hub.

    python scripts/release/upload_release.py --staging <staging_dir> [--groups ...] [--tasks ...] [--dry_run]

Runs the `hf` CLI (huggingface_hub >= 1.0; `--hf_bin` points at it) once per `<group>/<task>_image`
folder and once per data file, so every commit stays well under the Hub's per-commit limits and a
failed call can simply be re-run (unchanged files are skipped by the Hub). The dataset card
(scripts/release/hf_dataset_card.md) is uploaded first, the release manifest last.
Authentication: `hf auth login` or HF_TOKEN in the environment.

Plain script (no package imports) so it runs from the venv that holds the recent `hf` CLI.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.release.manifest import (  # noqa: E402
    ALL_GROUPS,
    CKPT_GROUPS,
    GROUP_BC_DATA,
    HF_REPO_ID,
    HF_REPO_TYPE,
    TASKS,
    data_files,
    task_dir,
)

CARD = Path(__file__).with_name("hf_dataset_card.md")


def hf(hf_bin: str, args: list[str], dry_run: bool) -> None:
    cmd = [hf_bin] + args
    print("[upload] " + " ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def upload_path(hf_bin: str, local: Path, path_in_repo: str, message: str, dry_run: bool) -> None:
    assert local.exists(), local
    hf(hf_bin, ["upload", HF_REPO_ID, str(local), path_in_repo, "--repo-type", HF_REPO_TYPE, "--no-private",
                "--exclude", "*.tmp", "--commit-message", message], dry_run)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--staging", required=True)
    ap.add_argument("--groups", nargs="+", default=list(ALL_GROUPS), choices=list(ALL_GROUPS))
    ap.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    ap.add_argument("--hf_bin", default="hf")
    ap.add_argument("--skip_card", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()
    root = Path(args.staging).expanduser().resolve()
    manifest = json.load(open(root / "release_manifest.json"))

    if not args.skip_card:
        upload_path(args.hf_bin, CARD, "README.md", "dataset card", args.dry_run)
    for group in args.groups:
        for task in args.tasks:
            if group == GROUP_BC_DATA:
                names = data_files(task)
                files = [names["hdf5"], f"{names['image_zarr']}.tar", f"{names['lowdim_zarr']}.tar"]
                for f in files:
                    rel = f"{GROUP_BC_DATA}/{f}"
                    assert rel in manifest[GROUP_BC_DATA][task], f"{rel} not staged"
                    upload_path(args.hf_bin, root / rel, rel, rel, args.dry_run)
            else:
                assert group in CKPT_GROUPS, group
                rel = f"{group}/{task_dir(task)}"
                assert task_dir(task) in manifest[group], f"{rel} not staged"
                upload_path(args.hf_bin, root / rel, rel, rel, args.dry_run)
    upload_path(args.hf_bin, root / "release_manifest.json", "release_manifest.json", "release manifest", args.dry_run)


if __name__ == "__main__":
    main()
