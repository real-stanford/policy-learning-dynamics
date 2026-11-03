"""Assemble the Hugging Face release folder from the research repo's runs (scripts/release/manifest.py).

    python -m scripts.release.stage_release --out <staging_dir> [--groups ...] [--tasks ...] [--dry_run]

Per checkpoint (resumable: an existing destination file is skipped):
  * keep cfg / model_state / ema_state / normalizer_state / epoch / global_step, drop
    optimizer_state / scheduler_state (4.2 GB -> 2.1 GB; nothing in the released experiments
    resumes training from these files);
  * in the stored cfg replace the research machine's absolute data paths
    (/store/.../outputs/data/mimicgen/...) by ${oc.env:MIMICGEN_DATA_ROOT}/..., runtime.training_root
    by ${oc.env:TRAINING_ROOT}, rename the mixer block from its research-repo name
    (context_smoothing_w_oa_joint_model) to training.jointaug_w_oa_joint_model, and point its
    oa_model_dir at the released generator folder. Tensors are copied unchanged
    (scripts/release/verify_staged_checkpoint.py checks this bit for bit).
Per run: .hydra/config.yaml (also as config.yaml; same block rename) and the eval results directory are copied.
Per task's data: the hdf5 is copied; each zarr becomes one uncompressed tar (`tar -xf` recreates
<name>.zarr) because the five image zarrs hold ~150k chunk files, far above the Hub's file limits.
A release_manifest.json at the staging root records every shipped file and its source (merged from
per-(group, task) fragments under .manifest/, so several stage processes can run side by side).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf

from scripts.release.manifest import (
    ALL_GROUPS,
    CKPT_GROUPS,
    GROUP_BC_DATA,
    RELEASE_MIMICGEN_PREFIX,
    RELEASE_TRAINING_ROOT,
    RESEARCH_MIMICGEN,
    RESEARCH_TRAINING,
    RUNS_BY_GROUP,
    TASKS,
    RunSource,
    ckpt_name,
    data_files,
    release_generator_dir,
    task_dir,
)

KEEP_KEYS = ("cfg", "model_state", "ema_state", "normalizer_state", "epoch", "global_step")
DROP_KEYS = ("optimizer_state", "scheduler_state")
OA_KEY = "jointaug_w_oa_joint_model"
LEGACY_OA_KEY = "context_smoothing_w_oa_joint_model"   # the block's name in the research repo (every source checkpoint)


# ------------------------------------------------------------------ stored-config rewrite
def rewrite_strings(obj, research_data: str, research_training: str):
    """Recursively replace research-machine paths in a plain container."""
    if isinstance(obj, dict):
        return {k: rewrite_strings(v, research_data, research_training) for k, v in obj.items()}
    if isinstance(obj, list):
        return [rewrite_strings(v, research_data, research_training) for v in obj]
    if isinstance(obj, str):
        if obj == research_training:
            return RELEASE_TRAINING_ROOT
        if obj.startswith(research_data + "/"):
            return RELEASE_MIMICGEN_PREFIX + obj[len(research_data):]
        return obj
    return obj


def _walk_strings(obj, prefix=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk_strings(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _walk_strings(v, f"{prefix}[{i}]")
    elif isinstance(obj, str):
        yield prefix, obj


def release_cfg(cfg: DictConfig, task: str) -> DictConfig:
    """The checkpoint's stored config with release paths; everything else identical."""
    assert isinstance(cfg, DictConfig), type(cfg)
    raw = OmegaConf.to_container(cfg, resolve=False, throw_on_missing=False)
    new = rewrite_strings(raw, str(RESEARCH_MIMICGEN), str(RESEARCH_TRAINING))
    if LEGACY_OA_KEY in new["training"]:
        assert OA_KEY not in new["training"], "both the legacy and the new mixer key present"
        new["training"][OA_KEY] = new["training"].pop(LEGACY_OA_KEY)
    if OA_KEY in new["training"]:
        new["training"][OA_KEY]["oa_model_dir"] = release_generator_dir(task)
    assert new["runtime"]["training_root"] == RELEASE_TRAINING_ROOT, new["runtime"]["training_root"]
    for key, s in _walk_strings(new):
        assert LEGACY_OA_KEY not in key, key
        assert str(RESEARCH_MIMICGEN) not in s, (key, s)
        if key != "runtime.output_dir":   # kept verbatim as provenance; no loader reads it
            assert not s.startswith("/store/"), (key, s)
    return OmegaConf.create(new)


def strip_checkpoint(src: Path, dst: Path, task: str) -> None:
    payload = torch.load(str(src), map_location="cpu", weights_only=False, mmap=True)
    assert set(payload) == set(KEEP_KEYS) | set(DROP_KEYS), (src, sorted(payload))
    out = {k: payload[k] for k in KEEP_KEYS}
    out["cfg"] = release_cfg(payload["cfg"], task)
    tmp = dst.with_name(dst.name + ".tmp")
    torch.save(out, str(tmp))
    os.replace(tmp, dst)


# ------------------------------------------------------------------ staging
def _copy_file(src: Path, dst: Path) -> None:
    assert src.is_file(), src
    tmp = dst.with_name(dst.name + ".tmp")
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)


def _copy_config(src: Path, dst: Path) -> None:
    """The run's Hydra config, verbatim except for the mixer block's name (see LEGACY_OA_KEY)."""
    assert src.is_file(), src
    text = src.read_text().replace(LEGACY_OA_KEY, OA_KEY)
    assert LEGACY_OA_KEY not in text
    tmp = dst.with_name(dst.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, dst)


def stage_run(group: str, task: str, src: RunSource, out_root: Path, dry_run: bool, manifest: dict) -> None:
    dst_run = out_root / group / task_dir(task)
    entry = {
        "source_run": str(src.run),
        "checkpoints_from": str(src.checkpoints_dir.parent),
        "epochs": list(src.epochs),
        "eval_subdir": src.eval_subdir,
        "files": {},
    }
    plan: list[tuple[str, Path, Path]] = []
    plan.append(("config", src.run / ".hydra" / "config.yaml", dst_run / ".hydra" / "config.yaml"))
    plan.append(("config", src.run / ".hydra" / "config.yaml", dst_run / "config.yaml"))
    if src.checkpoints_run is not None:
        plan.append(("config", src.checkpoints_run / ".hydra" / "config.yaml", dst_run / "checkpoints_source_config.yaml"))
    for epoch in src.epochs:
        plan.append(("ckpt", src.checkpoints_dir / ckpt_name(epoch), dst_run / "checkpoints" / ckpt_name(epoch)))
    for kind, s, d in plan:
        assert s.is_file(), f"missing source {s}"
    if src.eval_subdir is not None:
        assert (src.run / src.eval_subdir).is_dir(), src.run / src.eval_subdir

    for kind, s, d in plan:
        rel = str(d.relative_to(out_root))
        if d.exists():
            entry["files"][rel] = {"bytes": d.stat().st_size, "source": str(s), "status": "kept"}
            continue
        if dry_run:
            entry["files"][rel] = {"bytes": None, "source": str(s), "status": "planned"}
            print(f"[stage] would write {rel}  <- {s}", flush=True)
            continue
        d.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        if kind == "ckpt":
            strip_checkpoint(s, d, task)
        else:
            _copy_config(s, d)
        entry["files"][rel] = {"bytes": d.stat().st_size, "source": str(s), "status": "written"}
        print(f"[stage] {rel}  {d.stat().st_size / 2**30:.2f} GiB  {time.time() - t0:.0f}s", flush=True)

    if src.eval_subdir is not None:
        s_dir, d_dir = src.run / src.eval_subdir, dst_run / src.eval_subdir
        if dry_run:
            print(f"[stage] would copy {s_dir} -> {d_dir.relative_to(out_root)}", flush=True)
        else:
            shutil.copytree(s_dir, d_dir, dirs_exist_ok=True)
            n = sum(1 for p in d_dir.rglob("*") if p.is_file())
            entry["files"][str(d_dir.relative_to(out_root))] = {"n_files": n, "source": str(s_dir), "status": "copied"}
            print(f"[stage] {d_dir.relative_to(out_root)}: {n} files copied", flush=True)
    manifest.setdefault(group, {})[task_dir(task)] = entry


def stage_data(task: str, out_root: Path, dry_run: bool, manifest: dict) -> None:
    core = RESEARCH_MIMICGEN / "core"
    dst_dir = out_root / GROUP_BC_DATA
    names = data_files(task)
    entry = manifest.setdefault(GROUP_BC_DATA, {}).setdefault(task, {})
    jobs = [("copy", core / names["hdf5"], dst_dir / names["hdf5"])]
    for key in ("image_zarr", "lowdim_zarr"):
        jobs.append(("tar", core / names[key], dst_dir / f"{names[key]}.tar"))
    for kind, s, d in jobs:
        assert s.exists(), f"missing source {s}"
        rel = str(d.relative_to(out_root))
        if d.exists():
            entry[rel] = {"bytes": d.stat().st_size, "source": str(s), "status": "kept"}
            continue
        if dry_run:
            entry[rel] = {"bytes": None, "source": str(s), "status": "planned"}
            print(f"[stage] would {kind} {s} -> {rel}", flush=True)
            continue
        d.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        if kind == "copy":
            _copy_file(s, d)
        else:
            tmp = d.with_name(d.name + ".tmp")
            # uncompressed (zarr chunks are already compressed), sorted member order, root member = <name>.zarr
            subprocess.run(["tar", "--sort=name", "-cf", str(tmp), "-C", str(s.parent), s.name], check=True)
            os.replace(tmp, d)
        entry[rel] = {"bytes": d.stat().st_size, "source": str(s), "status": "written"}
        print(f"[stage] {rel}  {d.stat().st_size / 2**30:.2f} GiB  {time.time() - t0:.0f}s", flush=True)


FRAGMENT_DIR = ".manifest"   # one JSON per (group, task); several stage processes may run side by side


def _write_json(obj, path: Path) -> None:
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    json.dump(obj, open(tmp, "w"), indent=1, sort_keys=True)
    os.replace(tmp, path)


def write_fragment(out_root: Path, group: str, key: str, entry: dict) -> None:
    d = out_root / FRAGMENT_DIR
    d.mkdir(parents=True, exist_ok=True)
    _write_json(entry, d / f"{group}__{key}.json")


def merge_manifest(out_root: Path) -> dict:
    """release_manifest.json = the union of every fragment present (a rerun refreshes its own fragments)."""
    manifest: dict = {}
    for frag in sorted((out_root / FRAGMENT_DIR).glob("*__*.json")):
        group, key = frag.stem.split("__", 1)
        manifest.setdefault(group, {})[key] = json.load(open(frag))
    _write_json(manifest, out_root / "release_manifest.json")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="staging root (becomes the Hub repo's content)")
    ap.add_argument("--groups", nargs="+", default=list(ALL_GROUPS), choices=list(ALL_GROUPS))
    ap.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    ap.add_argument("--dry_run", action="store_true", help="print the plan, write nothing")
    args = ap.parse_args()

    out_root = Path(args.out).expanduser().resolve()
    if not args.dry_run:
        out_root.mkdir(parents=True, exist_ok=True)
    for group in args.groups:
        for task in args.tasks:
            print(f"[stage] === {group} / {task}", flush=True)
            manifest: dict = {}
            if group == GROUP_BC_DATA:
                stage_data(task, out_root, args.dry_run, manifest)
                key = task
            else:
                assert group in CKPT_GROUPS, group
                stage_run(group, task, RUNS_BY_GROUP[group][task], out_root, args.dry_run, manifest)
                key = task_dir(task)
            if not args.dry_run:
                write_fragment(out_root, group, key, manifest[group][key])
    if args.dry_run:
        return
    manifest = merge_manifest(out_root)
    total = sum(
        f["bytes"] for g in manifest.values() for t in g.values() for f in t.get("files", t).values()
        if isinstance(f, dict) and f.get("bytes")
    )
    print(f"[stage] done; {total / 2**30:.1f} GiB in the manifest under {out_root}", flush=True)


if __name__ == "__main__":
    main()
