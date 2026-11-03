"""Check a staged (stripped, path-rewritten) checkpoint against its research-repo source.

    python -m scripts.release.verify_staged_checkpoint --src <research .ckpt> --dst <staged .ckpt>

Asserts: the staged payload has exactly the kept keys; every tensor of model_state / ema_state /
normalizer_state is bit-identical (same dtype, shape, values); epoch / global_step equal; the
stored config differs from the source only in the rewritten path keys and the renamed mixer block,
and resolves back to the source's absolute paths when MIMICGEN_DATA_ROOT / TRAINING_ROOT point at
the research repo.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from scripts.release.manifest import RESEARCH_MIMICGEN, RESEARCH_TRAINING
from scripts.release.stage_release import KEEP_KEYS, LEGACY_OA_KEY, OA_KEY

REWRITTEN_KEYS = {
    "task.data.hdf5_path",
    "task.data.zarr_path",
    "dataset.source.hdf5_path",
    "dataset.source.zarr_path",
    "task.evaluation.dataset_path",
    "runtime.training_root",
    f"training.{OA_KEY}.oa_model_dir",
}


def flatten(obj, prefix="") -> dict:
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            out.update(flatten(v, f"{prefix}.{k}" if prefix else str(k)))
        return out
    if isinstance(obj, list):
        out = {}
        for i, v in enumerate(obj):
            out.update(flatten(v, f"{prefix}[{i}]"))
        return out
    return {prefix: obj}


def same_state(a, b, name: str) -> int:
    assert type(a) is type(b), (name, type(a), type(b))
    if isinstance(a, torch.Tensor):
        assert a.dtype == b.dtype and a.shape == b.shape, (name, a.dtype, b.dtype, a.shape, b.shape)
        assert torch.equal(a, b), f"{name}: tensor values differ"
        return 1
    if isinstance(a, np.ndarray):
        assert a.dtype == b.dtype and a.shape == b.shape, (name, a.dtype, b.dtype, a.shape, b.shape)
        assert np.array_equal(a, b), f"{name}: array values differ"
        return 1
    if isinstance(a, dict):
        assert list(a) == list(b), (name, "key order/set differs")
        return sum(same_state(a[k], b[k], f"{name}.{k}") for k in a)
    if isinstance(a, (list, tuple)):
        assert len(a) == len(b), name
        return sum(same_state(x, y, f"{name}[{i}]") for i, (x, y) in enumerate(zip(a, b)))
    assert a == b, (name, a, b)
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    args = ap.parse_args()
    src = torch.load(args.src, map_location="cpu", weights_only=False, mmap=True)
    dst = torch.load(args.dst, map_location="cpu", weights_only=False, mmap=True)
    assert tuple(dst) == KEEP_KEYS, tuple(dst)

    n = 0
    for key in ("model_state", "ema_state", "normalizer_state"):
        n += same_state(src[key], dst[key], key)
    assert src["epoch"] == dst["epoch"] and src["global_step"] == dst["global_step"]
    print(f"[verify] {n} tensors bit-identical; epoch {dst['epoch']} step {dst['global_step']}")

    fs = flatten(OmegaConf.to_container(src["cfg"], resolve=False, throw_on_missing=False))
    fd = flatten(OmegaConf.to_container(dst["cfg"], resolve=False, throw_on_missing=False))
    n_legacy = sum(LEGACY_OA_KEY in k for k in fs)
    fs = {k.replace(f"training.{LEGACY_OA_KEY}.", f"training.{OA_KEY}."): v for k, v in fs.items()}   # the rename
    assert not any(LEGACY_OA_KEY in k for k in fd), "legacy mixer key survived in the staged config"
    assert set(fs) == set(fd), set(fs) ^ set(fd)
    if n_legacy:
        print(f"[verify] mixer block renamed {LEGACY_OA_KEY} -> {OA_KEY} ({n_legacy} keys)")
    changed = {k for k in fs if fs[k] != fd[k]}
    assert changed <= REWRITTEN_KEYS, changed - REWRITTEN_KEYS
    for k in sorted(changed):
        print(f"[verify] rewritten {k}: {fs[k]!r} -> {fd[k]!r}")

    # resolution round-trip against the research repo
    os.environ["MIMICGEN_DATA_ROOT"] = str(RESEARCH_MIMICGEN)
    os.environ["TRAINING_ROOT"] = str(RESEARCH_TRAINING)
    cfg = dst["cfg"]
    for k in sorted(changed - {f"training.{OA_KEY}.oa_model_dir"}):
        resolved = str(OmegaConf.select(cfg, k))
        assert resolved == fs[k], (k, resolved, fs[k])
    print(f"[verify] {len(changed)} rewritten keys resolve back to the source paths; "
          f"{Path(args.dst).stat().st_size / 2**30:.2f} GiB vs {Path(args.src).stat().st_size / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
