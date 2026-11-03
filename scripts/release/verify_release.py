"""Verify every staged checkpoint of a task against its research-repo source (release_manifest.json).

    python -m scripts.release.verify_release --staging <staging_dir> --tasks hammer_cleanup_d1 [...]

Runs scripts/release/verify_staged_checkpoint.py per checkpoint (bit-identical tensors, only the
path keys rewritten) and prints one OK/FAIL line per file plus a final summary; exit code 1 on any failure.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from scripts.release.manifest import CKPT_GROUPS, TASKS, task_dir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--staging", required=True)
    ap.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    ap.add_argument("--groups", nargs="+", default=list(CKPT_GROUPS), choices=list(CKPT_GROUPS))
    args = ap.parse_args()
    root = Path(args.staging).expanduser().resolve()
    manifest = json.load(open(root / "release_manifest.json"))
    n_ok, failures = 0, []
    for group in args.groups:
        for task in args.tasks:
            files = manifest[group][task_dir(task)]["files"]
            for rel, info in files.items():
                if not rel.endswith(".ckpt"):
                    continue
                r = subprocess.run(
                    [sys.executable, "-m", "scripts.release.verify_staged_checkpoint", "--src", info["source"], "--dst", str(root / rel)],
                    capture_output=True, text=True,
                )
                if r.returncode == 0:
                    n_ok += 1
                    print(f"OK   {rel}", flush=True)
                else:
                    failures.append(rel)
                    print(f"FAIL {rel}\n{r.stdout[-600:]}\n{r.stderr[-600:]}", flush=True)
    print(f"VERIFY-DONE {n_ok} ok, {len(failures)} failed", flush=True)
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
