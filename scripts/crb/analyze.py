"""Coverage-Reliability Balance (CRB) checkpoint selection from a held-out error cache.

Inputs (per run):
  <cache_dir>/epoch_XXXX.npz   errors (N, K): RAW action-chunk MSE of K sampled chunks per
                               held-out state (scripts/crb/cache_errors.py)
  <eval_dir>/epoch_XXXX/       closed-loop pass@K eval (scripts/eval_checkpoints.py), used only
                               as the oracle reference; optional

Per checkpoint tau the errors are sorted within each state, the 19 within-state percentiles
u = 0.05, ..., 0.95 are averaged over states (Qbar_tau(u)), log Qbar is min-max normalized per
percentile across the checkpoints, and
    R_cov(tau) = mean_{u < 0.5}  r_tau(u),   R_rel(tau) = mean_{u >= 0.5} r_tau(u),
    selected   = argmin_tau max{R_cov(tau), R_rel(tau)}   (earliest epoch on ties).
Best-of-k B_k and pass@k P_k (k in K_GRID) are exact U-statistics over the K samples /
attempts (never "first k of K").

Outputs (<out_dir>): quantiles_full.csv (19 percentiles), quantiles.csv (every other one),
bok.csv, passk.csv, crb.csv, meta.json, surface.npz (arrays for the figure scripts).

    python -m scripts.crb.analyze --cache_dir <run>/analysis/crb/offline_errors \
        --eval_dir <run>/eval_ddim_num_steps_10_eta_0.0_pass_32 --out_dir <run>/analysis/crb \
        --title "Threading (D2), 200 demos"
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from math import lgamma
from pathlib import Path

import numpy as np

U_GRID = np.round(np.arange(0.05, 0.951, 0.05), 2)   # 19 within-state percentiles
K_GRID = [1, 2, 4, 8, 16, 32]                         # best-of-k / pass@k grid
K_SHOW = [1, 4, 16, 32]                               # best-of-k columns of bok.csv
U_SHOW = [j for j in range(len(U_GRID)) if j % 2 == 0]   # 10 percentiles of quantiles.csv


# --------------------------------------------------------------- estimators
def ustat_min_weights(K: int, k: int) -> np.ndarray:
    """w_r = C(K-r, k-1) / C(K, k) for r = 1..K (0 beyond K-k+1): the probability that
    the r-th smallest of K samples is the minimum of a uniformly chosen k-subset."""
    assert 1 <= k <= K, (K, k)
    lc_Kk = lgamma(K + 1) - lgamma(k + 1) - lgamma(K - k + 1)
    w = np.zeros(K)
    for r in range(1, K - k + 2):
        lc = lgamma(K - r + 1) - lgamma(k) - lgamma(K - r - k + 2)
        w[r - 1] = np.exp(lc - lc_Kk)
    assert abs(w.sum() - 1.0) < 1e-9, (K, k, w.sum())
    return w


def b_ki(errors_sorted: np.ndarray, ks: list[int]) -> np.ndarray:
    """(N, K) ascending-sorted errors -> (N, len(ks)) expected minimum over a random k-subset."""
    K = errors_sorted.shape[1]
    W = np.stack([ustat_min_weights(K, k) for k in ks])   # (nk, K)
    return errors_sorted @ W.T


def pass_ustat(s: np.ndarray, N: int, ks) -> np.ndarray:
    """p_hat_{i,k} = 1 - C(N - s_i, k) / C(N, k) for success counts s (n,) out of N -> (n, len(ks))."""
    s = np.asarray(s)
    ks = np.asarray(ks)
    out = np.zeros((len(s), len(ks)))
    for j, k in enumerate(ks):
        assert 1 <= k <= N, (k, N)
        num = np.zeros(len(s))
        valid = (N - s) >= k
        ns = (N - s)[valid].astype(float)
        lg = np.vectorize(lgamma)(ns + 1) - lgamma(k + 1) - np.vectorize(lgamma)(ns - k + 1)
        lg -= lgamma(N + 1) - lgamma(k + 1) - lgamma(N - k + 1)
        num[valid] = np.exp(lg)
        out[:, j] = 1.0 - num
    return out


def crb_select(Qbar: np.ndarray, u: np.ndarray = U_GRID) -> dict:
    """Qbar (T, len(u)) -> R_cov, R_rel, R_max (T,) and i_sel (argmin R_max, first on ties)."""
    assert Qbar.ndim == 2 and Qbar.shape[1] == len(u) and Qbar.shape[0] >= 2
    assert np.all(Qbar > 0), "errors must be positive for the log normalization"
    logQ = np.log(Qbar)
    span = logQ.max(0) - logQ.min(0)
    assert np.all(span > 0), "a percentile is constant across checkpoints"
    r = (logQ - logQ.min(0)) / span
    R_cov = r[:, u < 0.5].mean(1)
    R_rel = r[:, u >= 0.5].mean(1)
    R_max = np.maximum(R_cov, R_rel)
    return {"R_cov": R_cov, "R_rel": R_rel, "R_max": R_max, "i_sel": int(np.argmin(R_max))}


def oracle_index(P: np.ndarray) -> int:
    """Closed-loop analogue of the selector on the pass@k table P (T, nk): minimax over k of the
    relative regret (P*_k - P_k(tau)) / (P*_k - P_k^worst). Reference only."""
    hi, lo = P.max(0, keepdims=True), P.min(0, keepdims=True)
    r_on = (hi - P) / (hi - lo)
    return int(np.argmin(r_on.max(1)))


# --------------------------------------------------------------- loading
def epochs_of(cache_dir: Path, max_epoch: int) -> list[int]:
    eps = []
    for p in cache_dir.glob("epoch_*.npz"):
        m = re.fullmatch(r"epoch_(\d+)\.npz", p.name)
        if m and int(m.group(1)) <= max_epoch:
            eps.append(int(m.group(1)))
    return sorted(eps)


def _episode_rows(d: Path) -> list[dict]:
    """Per-episode rows of one attempt directory: the research repo wrote episodes.json,
    scripts/eval_checkpoints.py writes summary.json with an "episodes" list."""
    if (d / "episodes.json").exists():
        return json.load(open(d / "episodes.json"))
    summary = json.load(open(d / "summary.json"))
    if "episodes" in summary:
        return summary["episodes"]
    return summary["seed_summaries"][0]["episodes"]


def load_pass_counts(eval_dir: Path, max_epoch: int) -> dict[int, tuple[np.ndarray, int]]:
    """{epoch: (success counts (n_cfg,), n_attempts)}: attempt 0 is the epoch dir itself,
    attempts >= 1 the pass_<a> subdirectories; every attempt must cover the same seeds."""
    out = {}
    for ed in sorted(eval_dir.glob("epoch_*")):
        if not ed.is_dir():
            continue
        e = int(ed.name.split("_")[1])
        if e > max_epoch:
            continue
        attempt_dirs = [ed] + sorted(
            (p for p in ed.glob("pass_*") if p.is_dir()),
            key=lambda p: int(p.name.split("_")[1]),
        )
        seeds0, mats = None, []
        for d in attempt_dirs:
            rows = _episode_rows(d)
            seeds = [int(x["seed"]) for x in rows]
            if seeds0 is None:
                seeds0 = seeds
            assert seeds == seeds0, f"seed mismatch in {d}"
            mats.append([float(x["success"]) for x in rows])
        M = np.array(mats).T                                   # (n_cfg, n_attempts)
        out[e] = (M.sum(1).astype(int), int(M.shape[1]))
    return out


def load_cache(cache_dir: Path, eps: list[int]) -> dict[int, np.ndarray]:
    """{epoch: (N, K) errors sorted ascending per state, float64}."""
    E = {}
    for e in eps:
        E[e] = np.sort(np.load(cache_dir / f"epoch_{e:04d}.npz")["errors"].astype(np.float64), axis=1)
    shapes = {E[e].shape for e in eps}
    assert len(shapes) == 1, f"inconsistent cache shapes {shapes}"
    return E


# --------------------------------------------------------------- analysis
def run_analysis(cache_dir: Path, eval_dir: Path | None, out_dir: Path, title: str, max_epoch: int) -> dict:
    eps = epochs_of(cache_dir, max_epoch)
    assert len(eps) >= 2, f"need >= 2 cached checkpoints in {cache_dir}, found {eps}"
    E = load_cache(cache_dir, eps)
    N, K = E[eps[0]].shape
    Qbar = np.stack([np.quantile(E[e], U_GRID, axis=1).mean(1) for e in eps])   # (T, 19)
    B = np.stack([b_ki(E[e], K_GRID).mean(0) for e in eps])                     # (T, 6)
    sel = crb_select(Qbar)
    i_sel = sel["i_sel"]

    P, n_attempts, i_or = None, None, None
    if eval_dir is not None:
        pc = load_pass_counts(eval_dir, max_epoch)
        missing = [e for e in eps if e not in pc]
        assert not missing, f"eval epochs {sorted(pc)} do not cover cached epochs {missing}"
        natt = {pc[e][1] for e in eps}
        assert len(natt) == 1, f"attempt count differs across epochs: {natt}"
        n_attempts = natt.pop()
        assert n_attempts >= max(K_GRID), f"pass@{max(K_GRID)} needs >= {max(K_GRID)} attempts, got {n_attempts}"
        P = np.stack([pass_ustat(pc[e][0], n_attempts, K_GRID).mean(0) for e in eps])   # (T, 6)
        i_or = oracle_index(P)

    out_dir.mkdir(parents=True, exist_ok=True)

    def write_csv(name: str, header: list[str], rows) -> None:
        with open(out_dir / name, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)

    write_csv("bok.csv", ["epoch"] + [f"B_k{k}" for k in K_SHOW],
              [[e] + [repr(float(B[t, K_GRID.index(k)])) for k in K_SHOW] for t, e in enumerate(eps)])
    write_csv("quantiles.csv", ["epoch"] + [f"Q_u{U_GRID[j]:.2f}" for j in U_SHOW],
              [[e] + [repr(float(Qbar[t, j])) for j in U_SHOW] for t, e in enumerate(eps)])
    write_csv("quantiles_full.csv", ["epoch"] + [f"Q_u{u:.2f}" for u in U_GRID],
              [[e] + [repr(float(v)) for v in Qbar[t]] for t, e in enumerate(eps)])
    write_csv("crb.csv", ["epoch", "R_cov", "R_rel", "R_max"],
              [[e, repr(float(sel["R_cov"][t])), repr(float(sel["R_rel"][t])), repr(float(sel["R_max"][t]))]
               for t, e in enumerate(eps)])
    if P is not None:
        write_csv("passk.csv", ["epoch"] + [f"pass_k{k}" for k in K_GRID],
                  [[e] + [repr(float(v)) for v in P[t]] for t, e in enumerate(eps)])
    meta = {
        "title": title, "error_cache": str(cache_dir), "eval": None if eval_dir is None else str(eval_dir),
        "max_epoch": int(max_epoch), "n_states": int(N), "K": int(K), "n_attempts": n_attempts,
        "n_checkpoints": int(len(eps)), "epochs": [int(e) for e in eps],
        "selected_epoch": int(eps[i_sel]), "R_max_selected": float(sel["R_max"][i_sel]),
        "R_cov_selected": float(sel["R_cov"][i_sel]), "R_rel_selected": float(sel["R_rel"][i_sel]),
        "oracle_epoch": None if i_or is None else int(eps[i_or]),
        "k_grid": list(K_GRID), "k_shown": list(K_SHOW), "u_grid": [float(u) for u in U_GRID],
        "percentiles_shown": [float(U_GRID[j]) for j in U_SHOW],
    }
    json.dump(meta, open(out_dir / "meta.json", "w"), indent=1)
    np.savez(out_dir / "surface.npz", epochs=np.array(eps), u=U_GRID, Qbar=Qbar, B=B, k_grid=np.array(K_GRID),
             R_cov=sel["R_cov"], R_rel=sel["R_rel"], R_max=sel["R_max"], i_sel=i_sel,
             P=(np.zeros((0, len(K_GRID))) if P is None else P), i_or=(-1 if i_or is None else i_or))
    print(f"[crb] {title}: {len(eps)} checkpoints, N={N}, K={K} | selected ep {eps[i_sel]} "
          f"(R_max {sel['R_max'][i_sel]:.3f}) | oracle ep {meta['oracle_epoch']} -> {out_dir}", flush=True)
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--eval_dir", default=None, help="pass@K eval dir (oracle reference); omit to skip")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--title", default="")
    ap.add_argument("--max_epoch", type=int, default=800)
    args = ap.parse_args()
    run_analysis(Path(args.cache_dir), None if args.eval_dir is None else Path(args.eval_dir),
                 Path(args.out_dir), args.title, args.max_epoch)


if __name__ == "__main__":
    main()
