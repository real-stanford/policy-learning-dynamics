"""Build the fixed held-out evaluation bank for the checkpoint-selection probes.

Every training-style window over the held-out episodes is enumerated exactly as the
training sampler does (CanonicalImageDataset with the run's horizon / padding / n_obs_steps,
restricted to --episodes_json); N windows are drawn uniformly without replacement with
numpy.random.default_rng(20260824) and persisted together with K fixed initial-noise draws:
    rgb      (N, n_obs, 6, 84, 84) uint8   policy observation (lossless: dataset rgb is uint8/255)
    states   (N, n_obs, 9)         float32 RAW proprio state
    actions  (N, H, Da)            float32 RAW demo action chunk a*
    z_bok    (N, K, H, Da)         float32 init noise, torch.Generator().manual_seed(222)
    window_indices (N,)            rows of the enumerated window list
plus bank_meta.json. The same bank (states AND noise) is reused for every checkpoint, so all
comparisons across checkpoints are paired. Idempotent: refuses to overwrite a bank with a
different fingerprint.

    python -m scripts.crb.build_eval_bank --ckpt <run>/checkpoints/epoch_0050.ckpt \
        --episodes_json outputs/probes/<task>/split/heldout.json \
        --out outputs/probes/<task>/eval_bank_8192_k64 --n_windows 8192 --k_draws 64
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from datasets.canonical_zarr_dataset import CanonicalImageDataset
from utils.training_utils.checkpointing import load_checkpoint_payload
from utils.training_utils.runtime import register_training_resolvers

WINDOW_SEED = 20260824      # window selection
Z_BOK_SEED = 222            # init-noise bank


def build_bank_dataset(ckpt: str, episodes_json: str, in_memory: bool = False) -> CanonicalImageDataset:
    """All valid windows over the listed episodes, with the checkpoint's window geometry."""
    register_training_resolvers()
    cfg = load_checkpoint_payload(ckpt, map_location="cpu")["cfg"]
    samp, proc = cfg.dataset.sampling, cfg.dataset.processing
    return CanonicalImageDataset(
        zarr_path=str(cfg.dataset.source.zarr_path),
        horizon=int(samp.horizon), pad_before=int(samp.pad_before), pad_after=int(samp.pad_after),
        n_obs_steps=int(samp.n_obs_steps), seed=int(proc.seed), val_ratio=0.0, ratio=1.0,
        max_train_episodes=None, action_pose_repr=str(proc.action_pose_repr),
        episode_include_path=str(episodes_json), in_memory=in_memory,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="any checkpoint of the run (supplies zarr path + window geometry)")
    ap.add_argument("--episodes_json", required=True, help="held-out episode list (scripts/crb/make_heldout_split.py)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_windows", type=int, default=8192)
    ap.add_argument("--k_draws", type=int, default=64)
    ap.add_argument("--window_indices", type=str, default=None,
                    help="JSON list of explicit window rows (overrides the seeded draw; smoke/paired checks)")
    ap.add_argument("--noise_from", type=str, default=None,
                    help="bank.npz of an existing bank over the same windows: reuse its z_bok rows (first k_draws) "
                         "for the selected window rows instead of drawing fresh noise (paired checks)")
    ap.add_argument("--in_memory", action="store_true", help="copy the zarr into RAM first (faster window reads)")
    args = ap.parse_args()

    episodes_json = str(Path(args.episodes_json).expanduser().resolve())
    ds = build_bank_dataset(args.ckpt, episodes_json, in_memory=args.in_memory)
    n_all = len(ds)
    print(f"[bank] {Path(episodes_json).name}: {n_all} valid windows", flush=True)

    if args.window_indices is not None:
        widx = np.asarray(sorted(int(i) for i in json.load(open(args.window_indices))), dtype=np.int64)
        assert np.unique(widx).size == widx.size and widx.min() >= 0 and widx.max() < n_all
    else:
        assert args.n_windows <= n_all
        rng = np.random.default_rng(WINDOW_SEED)
        widx = np.sort(rng.choice(n_all, size=args.n_windows, replace=False))

    N, K = int(widx.size), int(args.k_draws)
    it0 = ds[int(widx[0])]
    H, Da = it0["action"].shape
    n_obs = int(ds.n_obs_steps)
    rgb = np.zeros((N, n_obs, *it0["obs"]["rgb"].shape[1:]), dtype=np.uint8)
    states = np.zeros((N, n_obs, it0["obs"]["states"].shape[-1]), dtype=np.float32)
    actions = np.zeros((N, H, Da), dtype=np.float32)
    for j, i in enumerate(widx):
        it = ds[int(i)]
        r = (it["obs"]["rgb"].numpy() * 255.0).round()
        assert np.allclose(r / 255.0, it["obs"]["rgb"].numpy()), "rgb not uint8-derived"
        rgb[j] = r.astype(np.uint8)
        states[j] = it["obs"]["states"].numpy()
        actions[j] = it["action"].numpy()
        if j % 2000 == 0:
            print(f"[bank] {j}/{N} windows", flush=True)

    if args.noise_from is not None:
        src = np.load(args.noise_from)
        src_rows = {int(w): r for r, w in enumerate(src["window_indices"].tolist())}
        missing = [int(w) for w in widx if int(w) not in src_rows]
        assert not missing, f"--noise_from bank lacks windows {missing[:5]}..."
        assert src["z_bok"].shape[1] >= K, (src["z_bok"].shape, K)
        z_bok = np.stack([src["z_bok"][src_rows[int(w)], :K] for w in widx]).astype(np.float32)
    else:
        z_bok = torch.randn((N, K, H, Da), generator=torch.Generator().manual_seed(Z_BOK_SEED)).numpy().astype(np.float32)
    content_sha = hashlib.sha1(widx.tobytes() + actions.tobytes() + z_bok.tobytes()).hexdigest()[:12]
    meta = {
        "n_windows": N, "bok_draws": K, "horizon": int(H), "action_dim": int(Da), "n_obs_steps": n_obs,
        "window_seed": None if args.window_indices else WINDOW_SEED,
        "z_bok_seed": None if args.noise_from else Z_BOK_SEED, "noise_from": args.noise_from,
        "n_valid_windows": n_all, "episodes_json": episodes_json, "ckpt": str(args.ckpt),
        "content_sha1": content_sha,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "bank_meta.json").exists():
        old = json.load(open(out / "bank_meta.json"))
        assert old["content_sha1"] == content_sha, "existing bank DIFFERS; refusing to overwrite."
        print("[bank] existing bank matches (idempotent).")
        return
    np.savez(out / "bank.npz", window_indices=widx, rgb=rgb, states=states, actions=actions, z_bok=z_bok)
    json.dump(meta, open(out / "bank_meta.json", "w"), indent=1)
    print(f"[bank] wrote {out} (N={N}, K={K}, content_sha1={content_sha})", flush=True)


if __name__ == "__main__":
    main()
