"""Per-checkpoint held-out action-error cache e[tau, i, m] for checkpoint selection.

For every checkpoint tau of the run and every bank state i, K action chunks are generated
with the DEPLOYED sampler (EMA weights, DDIM num_inference_steps / eta as flagged, the policy's
own conditional_sample loop) from the bank's fixed per-(state, draw) init noise z_bok, and
scored against the demo chunk:
    e[i, m] = mean over all H x Da chunk elements of (a_raw[i, m] - a*_raw[i])^2
(RAW / unnormalized actions, full chunk, per-element mean). TF32 is disabled.

Writes <out>/epoch_XXXX.npz: errors (N, K) float32, demo_id (N,) zarr episode index,
timestep (N,) window start within the demo, state_id (N,) bank window row; plus meta.json.
Shardable: --epochs picks the checkpoints for this invocation; each epoch file is written
atomically and skipped if present.

    CUDA_VISIBLE_DEVICES=0 python -m scripts.crb.cache_errors --run outputs/training/<task>/<run> \
        --bank outputs/probes/<task>/eval_bank_8192_k64 --epochs 50 100 150 200
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch

from scripts.crb.build_eval_bank import build_bank_dataset
from utils.training_utils.deployed_policy import (
    configure_determinism, ddim_record, encode_obs, load_deployed_policy,
)


def epochs_in(run_dir: Path) -> list[int]:
    eps = []
    for f in (run_dir / "checkpoints").glob("epoch_*.ckpt"):
        m = re.fullmatch(r"epoch_(\d+)\.ckpt", f.name)
        if m:
            eps.append(int(m.group(1)))
    return sorted(eps)


def window_meta(bank_meta: dict, widx: np.ndarray, run: Path) -> tuple[np.ndarray, np.ndarray]:
    """(demo_id, timestep) of the bank rows via the same window enumeration the bank used."""
    ckpt = sorted((run / "checkpoints").glob("epoch_*.ckpt"))[0]
    ds = build_bank_dataset(str(ckpt), bank_meta["episodes_json"])
    assert len(ds) == int(bank_meta["n_valid_windows"]), (len(ds), bank_meta["n_valid_windows"])
    idx = ds.indices[widx]      # rows: (episode_idx, buffer_start, buffer_end, sample_start, sample_end)
    return idx[:, 0].astype(np.int64), idx[:, 1].astype(np.int64)


@torch.no_grad()
def sanity_check_sampler(policy, rgb_u8, states, z, device) -> None:
    """The recording loop must be bitwise identical to the deployed conditional_sample."""
    gc = encode_obs(policy, rgb_u8, states)
    a = ddim_record(policy, gc, z)
    b = ddim_record(policy, gc, z)
    for s, (xa, xb) in enumerate(zip(a, b)):
        assert torch.equal(xa, xb), f"nonzero divergence between identical runs at step {s}"
    g = torch.Generator(device=device).manual_seed(1234)
    z2 = torch.randn(z.shape, dtype=z.dtype, device=device, generator=g)
    ours = ddim_record(policy, gc, z2)[-1]
    g2 = torch.Generator(device=device).manual_seed(1234)
    cond = torch.zeros(z.shape, dtype=z.dtype, device=device)
    theirs = policy.conditional_sample(condition_data=cond, condition_mask=torch.zeros_like(cond, dtype=torch.bool),
                                       global_cond=gc, generator=g2)
    assert torch.equal(ours, theirs), "ddim_record != policy.conditional_sample"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--epochs", type=int, nargs="*", default=None, help="default: every checkpoint of the run")
    ap.add_argument("--out", default=None, help="default <run>/analysis/crb/offline_errors")
    ap.add_argument("--num_inference_steps", type=int, default=10)
    ap.add_argument("--eta", type=float, default=0.0)
    ap.add_argument("--windows_per_batch", type=int, default=8, help="states per forward batch (each x K draws)")
    ap.add_argument("--limit_states", type=int, default=None, help="debug: only the first N bank states (not cached)")
    ap.add_argument("--limit_k", type=int, default=None, help="debug: only the first M draws (not cached)")
    ap.add_argument("--sanity", action="store_true")
    args = ap.parse_args()

    run = Path(args.run)
    assert run.is_dir(), f"--run is not a directory: {run}"
    out = Path(args.out) if args.out else run / "analysis" / "crb" / "offline_errors"
    configure_determinism()
    device = torch.device("cuda")
    out.mkdir(parents=True, exist_ok=True)

    bank_dir = Path(args.bank)
    bank_meta = json.load(open(bank_dir / "bank_meta.json"))
    bank = np.load(bank_dir / "bank.npz")
    N_full, K_full = bank["z_bok"].shape[:2]
    N = N_full if args.limit_states is None else min(args.limit_states, N_full)
    K = K_full if args.limit_k is None else min(args.limit_k, K_full)
    rgb = torch.from_numpy(bank["rgb"][:N])
    sts = torch.from_numpy(bank["states"][:N])
    astar = torch.from_numpy(bank["actions"][:N])
    z_bok = torch.from_numpy(bank["z_bok"][:N, :K])
    widx = bank["window_indices"]
    demo_id, timestep = window_meta(bank_meta, widx, run)
    print(f"[cache] bank {bank_dir.name}: N={N} K={K} covering {len(np.unique(demo_id))} demos", flush=True)

    epochs = args.epochs or epochs_in(run)
    full = args.limit_states is None and args.limit_k is None
    cache_meta = {
        "run": str(run), "bank": str(bank_dir), "bank_sha": bank_meta["content_sha1"],
        "episodes_json": bank_meta["episodes_json"], "n_states": int(N), "K": int(K),
        "num_inference_steps": args.num_inference_steps, "eta": args.eta, "weights": "ema_state",
        "error_metric": "per-element MSE mean over the full RAW action chunk vs the RAW demo chunk",
        "noise": f"bank z_bok row i, draw m: fixed across checkpoints (z_bok_seed={bank_meta['z_bok_seed']})",
    }
    if full:
        mp = out / "meta.json"
        if mp.exists():
            old = json.load(open(mp))
            for key in ("bank_sha", "n_states", "K", "num_inference_steps", "eta", "weights"):
                assert old[key] == cache_meta[key], f"protocol mismatch vs existing cache: {key} {old[key]} != {cache_meta[key]}"
        else:
            json.dump(cache_meta, open(mp, "w"), indent=1)

    for ep in epochs:
        out_f = out / f"epoch_{ep:04d}.npz"
        if full and out_f.exists():
            print(f"[cache] ep {ep}: exists, skipping", flush=True)
            continue
        t0 = time.time()
        policy = load_deployed_policy(str(run / "checkpoints" / f"epoch_{ep:04d}.ckpt"),
                                      args.num_inference_steps, args.eta, device)
        assert int(astar.shape[1]) == int(policy.horizon), (int(astar.shape[1]), int(policy.horizon))
        assert int(rgb.shape[1]) == int(policy.n_obs_steps), (int(rgb.shape[1]), int(policy.n_obs_steps))
        if args.sanity:
            sanity_check_sampler(policy, rgb[:2].to(device), sts[:2].to(device), z_bok[:2, 0].to(device), device)
            print(f"[cache] ep {ep}: sampler sanity check passed", flush=True)
        errors = torch.zeros(N, K, dtype=torch.float32)
        W = args.windows_per_batch
        for s0 in range(0, N, W):
            sl = slice(s0, min(s0 + W, N))
            n_w = sl.stop - sl.start
            gc = encode_obs(policy, rgb[sl].to(device), sts[sl].to(device)).repeat_interleave(K, dim=0)
            z = z_bok[sl].to(device).reshape(n_w * K, *z_bok.shape[2:])
            act = policy.normalizer["action"].unnormalize(ddim_record(policy, gc, z)[-1]).reshape(n_w, K, *astar.shape[1:])
            tgt = astar[sl].to(device).unsqueeze(1)
            errors[sl] = (act - tgt).pow(2).reshape(n_w, K, -1).mean(-1).cpu()
            if s0 and s0 % (W * 64) < W:
                r = (time.time() - t0) / sl.stop
                print(f"[cache] ep {ep}: {sl.stop}/{N} ({r * 1000:.0f} ms/state, ETA {(N - sl.stop) * r / 60:.1f} min)", flush=True)
        if full:
            tmp = out_f.with_suffix(".tmp.npz")
            np.savez(tmp, errors=errors.numpy(), demo_id=demo_id, timestep=timestep, state_id=widx[:N].astype(np.int64))
            tmp.rename(out_f)
        else:
            np.savez(out / f"epoch_{ep:04d}_limit.npz", errors=errors.numpy(), demo_id=demo_id[:N],
                     timestep=timestep[:N], state_id=widx[:N].astype(np.int64))
        print(f"[cache] ep {ep}: done in {(time.time() - t0) / 60:.1f} min | B_1={float(errors.mean()):.5f} "
              f"B_{K}(min)={float(errors.min(dim=1).values.mean()):.5f}", flush=True)
        del policy
        torch.cuda.empty_cache()
    print(f"[cache] complete -> {out}", flush=True)


if __name__ == "__main__":
    main()
