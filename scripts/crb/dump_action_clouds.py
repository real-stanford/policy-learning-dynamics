"""Sampled action chunks at a few held-out states for the qualitative action-cloud panel.

Same bank, same per-(state, draw) init noise, same deployed sampler (EMA, DDIM-10, eta 0) and
the same RAW-space error as scripts/crb/cache_errors.py; where the run has a cache for an
epoch, the regenerated errors must reproduce it (asserted).

State selection (--select, from the run's error cache at the first and last epoch, the rule
used for the paper's states): candidates are states whose early best-of-K error is in the
bottom 20% of states, whose late/early best-of-K ratio is >= 2, whose 5%-95% error spread
contracts from early to late, and whose early draws contain no blow-up (max draw error below
the 90th percentile over states); 6 states at the 30/40/50/60/70/80th percentile of the
ratio among candidates, at most one state per demo. Writes <qdir>/selected_states.json.

Output: <qdir>/samples.npz with actions (S, E, K, H, Da) raw, demo (S, H, Da) raw,
errors (S, E, K), epochs (E,), rows (S,), demo_id (S,), timestep (S,).

    CUDA_VISIBLE_DEVICES=0 python -m scripts.crb.dump_action_clouds --run <run> --bank <bank> \
        --epochs 50 800 --select
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from utils.training_utils.deployed_policy import (
    configure_determinism, ddim_record, encode_obs, load_deployed_policy,
)

TARGET_PERCENTILES = (30, 40, 50, 60, 70, 80)


def select_states(cache_dir: Path, early: int, late: int, n_states: int = 6) -> dict:
    ze, zl = np.load(cache_dir / f"epoch_{early:04d}.npz"), np.load(cache_dir / f"epoch_{late:04d}.npz")
    Ee, El = ze["errors"].astype(np.float64), zl["errors"].astype(np.float64)
    demo_id, timestep, state_id = ze["demo_id"], ze["timestep"], ze["state_id"]
    e_min, l_min = Ee.min(1), El.min(1)
    ratio = l_min / e_min
    spread_e = np.quantile(Ee, 0.95, axis=1) - np.quantile(Ee, 0.05, axis=1)
    spread_l = np.quantile(El, 0.95, axis=1) - np.quantile(El, 0.05, axis=1)
    e_max = Ee.max(1)
    cand = np.flatnonzero(
        (e_min <= np.quantile(e_min, 0.20)) & (ratio >= 2.0) & (spread_l < spread_e)
        & (e_max < np.quantile(e_max, 0.90))
    )
    assert cand.size >= n_states, f"only {cand.size} candidate states"
    order = cand[np.argsort(ratio[cand])]
    chosen, used_demos = [], set()
    for p in TARGET_PERCENTILES[:n_states]:
        target = int(round(p / 100 * (order.size - 1)))
        for step in range(order.size):
            for j in (target + step, target - step):
                if 0 <= j < order.size and int(demo_id[order[j]]) not in used_demos:
                    break
            else:
                continue
            break
        row = int(order[j])
        used_demos.add(int(demo_id[row]))
        chosen.append({
            "state_id": int(state_id[row]), "row": row, "demo_id": int(demo_id[row]), "timestep": int(timestep[row]),
            "early_min": float(e_min[row]), "late_min": float(l_min[row]), "ratio": float(ratio[row]),
            "early_spread": float(spread_e[row]), "late_spread": float(spread_l[row]), "early_max_draw_err": float(e_max[row]),
        })
    return {"epochs": [early, late], "n_candidates": int(cand.size), "states": chosen}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--epochs", type=int, nargs="+", required=True, help="checkpoint epochs (early ... late)")
    ap.add_argument("--qdir", default=None, help="default <run>/analysis/crb/qualitative")
    ap.add_argument("--cache_dir", default=None, help="default <run>/analysis/crb/offline_errors")
    ap.add_argument("--select", action="store_true", help="(re)select the states from the cache; else read selected_states.json")
    ap.add_argument("--num_inference_steps", type=int, default=10)
    ap.add_argument("--eta", type=float, default=0.0)
    args = ap.parse_args()
    run = Path(args.run)
    qdir = Path(args.qdir) if args.qdir else run / "analysis" / "crb" / "qualitative"
    cache_dir = Path(args.cache_dir) if args.cache_dir else run / "analysis" / "crb" / "offline_errors"
    qdir.mkdir(parents=True, exist_ok=True)
    if args.select:
        sel = select_states(cache_dir, args.epochs[0], args.epochs[-1])
        json.dump(sel, open(qdir / "selected_states.json", "w"), indent=1)
    sel = json.load(open(qdir / "selected_states.json"))
    rows = np.array([s["row"] for s in sel["states"]], dtype=np.int64)

    configure_determinism()
    device = torch.device("cuda")
    bank = np.load(Path(args.bank) / "bank.npz")
    rgb = torch.from_numpy(bank["rgb"][rows]).to(device)
    sts = torch.from_numpy(bank["states"][rows]).to(device)
    astar = torch.from_numpy(bank["actions"][rows]).to(device)
    z = torch.from_numpy(bank["z_bok"][rows]).to(device)
    S, K = z.shape[:2]

    acts, errs = [], []
    for ep in args.epochs:
        policy = load_deployed_policy(str(run / "checkpoints" / f"epoch_{ep:04d}.ckpt"), args.num_inference_steps, args.eta, device)
        gc = encode_obs(policy, rgb, sts).repeat_interleave(K, dim=0)
        act = policy.normalizer["action"].unnormalize(ddim_record(policy, gc, z.reshape(S * K, *z.shape[2:]))[-1]).reshape(S, K, *astar.shape[1:])
        err = (act - astar.unsqueeze(1)).pow(2).reshape(S, K, -1).mean(-1)
        cache = cache_dir / f"epoch_{ep:04d}.npz"
        note = "no cache for this checkpoint"
        if cache.exists():
            cached = np.load(cache)["errors"][rows].astype(np.float32)
            diff = float(np.abs(err.cpu().numpy() - cached).max())
            tol = max(5e-5, 2e-4 * float(np.abs(cached).max()))
            assert diff < tol, f"ep {ep}: regenerated errors differ from the cache by {diff:.2e} (tol {tol:.2e})"
            note = f"max |err - cached| = {diff:.1e}"
        print(f"[dump] ep {ep}: {S} states x {K} draws | {note} | best-of-{K}: "
              f"{np.round(err.min(1).values.cpu().numpy(), 5)}", flush=True)
        acts.append(act.cpu().numpy())
        errs.append(err.cpu().numpy())

    out = qdir / "samples.npz"
    np.savez(out, actions=np.stack(acts, 1).astype(np.float32), demo=astar.cpu().numpy().astype(np.float32),
             errors=np.stack(errs, 1).astype(np.float32), epochs=np.array(args.epochs), rows=rows,
             demo_id=np.array([s["demo_id"] for s in sel["states"]]), timestep=np.array([s["timestep"] for s in sel["states"]]))
    print(f"[dump] -> {out}")


if __name__ == "__main__":
    main()
