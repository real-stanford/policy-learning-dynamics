"""Training / held-out demonstration split of a behaviour-cloning run.

Recomputes the run's seeded episode split exactly as CanonicalImageDataset does
(get_val_mask -> apply_train_ratio -> downsample_mask with the run's val_ratio / ratio /
max_train_episodes / seed) and writes
    <out_dir>/train.json     the training episodes (e.g. 200 of 1000)
    <out_dir>/heldout.json   every other episode (the trainer's own 2% validation
                             episodes included; they are not treated specially)
    <out_dir>/meta.json      split parameters + sha1 fingerprints
When the run directory has the trainer's __init__.log, the recomputed list is cross-checked
against the "[data-split] train episodes" line logged at training time.

    python -m scripts.crb.make_heldout_split --run outputs/training/threading_d2_image/<run> \
        --out_dir outputs/probes/threading_d2_image/split
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import zarr
from omegaconf import OmegaConf

from datasets.sampler import apply_train_ratio, downsample_mask, get_val_mask
from utils.training_utils.runtime import register_training_resolvers


def sha(idx) -> str:
    return hashlib.sha1(np.asarray(sorted(idx), dtype=np.int64).tobytes()).hexdigest()[:12]


def count_episodes(zarr_path: str) -> int:
    root = zarr.open(str(Path(zarr_path).expanduser()), mode="r")
    return len(list(root["data"].keys()))


def compute_split(n_episodes: int, *, val_ratio: float, ratio: float | None, max_train_episodes: int | None,
                  seed: int) -> tuple[list[int], list[int]]:
    val_mask = get_val_mask(n_episodes, val_ratio=val_ratio, seed=seed)
    train_mask = apply_train_ratio(~val_mask, ratio=ratio, seed=seed)
    train_mask = downsample_mask(train_mask, max_train_episodes, seed=seed)
    train = sorted(np.flatnonzero(train_mask).tolist())
    heldout = sorted(set(range(n_episodes)) - set(train))
    return train, heldout


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run dir with .hydra/config.yaml")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_episodes", type=int, default=None, help="default: count the episodes of the run's zarr")
    args = ap.parse_args()
    run = Path(args.run)
    register_training_resolvers()
    # resolve only the dataset subtree: the full run config also carries Hydra-only
    # resolvers (${now:...}) that are not available outside a Hydra launch
    cfg = OmegaConf.load(run / ".hydra" / "config.yaml")
    dataset_cfg = OmegaConf.to_container(cfg.dataset, resolve=True)
    proc = dataset_cfg["processing"]
    zarr_path = str(dataset_cfg["source"]["zarr_path"])
    n_episodes = int(args.n_episodes) if args.n_episodes is not None else count_episodes(zarr_path)
    assert proc.get("episode_include_path", None) is None, "run used an explicit episode list; nothing to recompute"
    params = {
        "val_ratio": float(proc["val_ratio"]), "ratio": None if proc.get("ratio") is None else float(proc["ratio"]),
        "max_train_episodes": None if proc["max_train_episodes"] is None else int(proc["max_train_episodes"]),
        "seed": int(proc["seed"]),
    }
    train, heldout = compute_split(n_episodes, **params)

    log_path = run / "__init__.log"
    logged = None
    if log_path.exists():
        m = re.search(r"\[data-split\] train episodes: (\[[^\]]*\])", log_path.read_text())
        if m:
            logged = sorted(ast.literal_eval(m.group(1)))
            assert logged == train, "recomputed train split != the split logged at training time"

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"run": str(run), "zarr": zarr_path, "n_episodes": n_episodes, **params,
            "n_train": len(train), "n_heldout": len(heldout),
            "sha1": {"train": sha(train), "heldout": sha(heldout)},
            "cross_checked_against_log": logged is not None}
    if (out / "meta.json").exists():
        old = json.load(open(out / "meta.json"))
        assert old["sha1"] == meta["sha1"], f"existing split at {out} DIFFERS; refusing to overwrite"
        print(f"[split] {out} already exists and matches.")
        return
    json.dump(train, open(out / "train.json", "w"))
    json.dump(heldout, open(out / "heldout.json", "w"))
    json.dump(meta, open(out / "meta.json", "w"), indent=1)
    print(f"[split] {len(train)} train / {len(heldout)} held-out episodes "
          f"(sha1 train={meta['sha1']['train']}, log check={logged is not None}) -> {out}")


if __name__ == "__main__":
    main()
