"""Generator checkpoint selection by action nearest-neighbour ratio (manuscript Eq. gen_nn_ratio).

For each saved generator checkpoint G_gamma, draw joint samples (o~, a~) with the deployed
sampler (DDIM, num_inference_steps as in the distillation mixer, fixed seed) and compare each
generated action chunk, in the generator's NORMALIZED action space, with two equally sized
sets of expert chunks: the demonstrations the generator trained on, D_train, and a held-out
set D_held of the same task that trained neither the generator nor the policy. With
d(a~, D) = min_{a' in D} ||a~ - a'||_2,

    rho(gamma) = E[d(a~, D_train)] / E[d(a~, D_held)]

rho = 1: generated actions are as close to the training as to the held-out demonstrations.
gamma_0 = the first saved checkpoint with rho < 1 (generated samples first become closer to
the training data than to held-out data); the distillation mixture is gamma_0 and the four
subsequent checkpoints spaced 100 epochs apart. Written to <gen_run>/analysis/nn_ratio/selection.json.

Also reported per checkpoint (the richer diagnostic of the original study): the NN ratio of
generated samples to the training windows relative to REAL held-out windows,
mean gen-NN / mean held-out-NN, in three spaces: the generator-normalized action chunk
("act"), the generator-normalized proprio state ("state") and the canonical policy's
conditioning vector on the generated rgb ("obs"); plus the replica fraction
(gen NN < 5% of the held-out NN median).

Held-out windows come from the complement of the generator's training episodes in its own
zarr (default), or from an existing eval bank (--bank, bank.npz with rgb/states/actions).

    CUDA_VISIBLE_DEVICES=0 python -m scripts.eval_joint_gen_nn_ratio \
        --gen_run outputs/training/mug_cleanup_d1_image_joint_gen_img/<run> \
        --policy_ckpt outputs/training/mug_cleanup_d1_image/<run>/checkpoints/epoch_0200.ckpt
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch

from datasets.canonical_zarr_dataset import CanonicalImageDataset
from utils.oa_joint_image_mixer import load_generator_checkpoint
from utils.training_utils.deployed_policy import (
    configure_determinism,
    encode_obs,
    load_deployed_policy,
)
from utils.training_utils.runtime import register_training_resolvers

SPACES = ("act", "state", "obs")
MIXTURE_SIZE = 5
MIXTURE_SPACING = 100


def epochs_of(run: Path) -> list[int]:
    out = []
    for p in sorted((run / "checkpoints").glob("epoch_*.ckpt")):
        m = re.match(r"epoch_(\d+)\.ckpt", p.name)
        if m:
            out.append(int(m.group(1)))
    return out


@torch.no_grad()
def nn_dist(query: torch.Tensor, bank: torch.Tensor, exclude_self: bool = False, chunk: int = 2048) -> torch.Tensor:
    """L2 distance of each query row to its nearest bank row (GPU, chunked)."""
    out = []
    for s in range(0, query.shape[0], chunk):
        d = torch.cdist(query[s : s + chunk], bank)
        if exclude_self:  # query == bank[s:s+chunk]
            idx = torch.arange(d.shape[0], device=d.device)
            d[idx, idx + s] = float("inf")
        out.append(d.min(dim=1).values)
    return torch.cat(out)


@torch.no_grad()
def embed(gen, policy, rgb_u8: torch.Tensor, states: torch.Tensor, action: torch.Tensor, device) -> dict:
    """rgb_u8 (n, n_obs, 6, 84, 84) uint8; states (n, n_obs, sd); action (n, H, Da) raw
    -> {"act": (n, H*Da), "state": (n, sd), "obs": (n, cond_dim)} on CPU."""
    act = gen.normalizer["action"].normalize(action.to(device)).reshape(action.shape[0], -1)
    st = gen.normalizer["states"].normalize(states.to(device)).reshape(states.shape[0], -1)
    obs = encode_obs(policy, rgb_u8.to(device), states.to(device).float())
    return {"act": act.float().cpu(), "state": st.float().cpu(), "obs": obs.float().cpu()}


def dataset_like(cfg, episode_include_path: str | None, in_memory: bool) -> CanonicalImageDataset:
    """The generator's window geometry over (a) its own train split or (b) an explicit episode list."""
    s_cfg, p_cfg = cfg.dataset.sampling, cfg.dataset.processing
    common = dict(
        zarr_path=str(cfg.dataset.source.zarr_path),
        horizon=int(s_cfg.horizon),
        pad_before=int(s_cfg.pad_before),
        pad_after=int(s_cfg.pad_after),
        n_obs_steps=int(s_cfg.n_obs_steps),
        action_pose_repr=str(p_cfg.action_pose_repr),
        in_memory=in_memory,
    )
    if episode_include_path is None:
        return CanonicalImageDataset(
            seed=int(p_cfg.seed),
            val_ratio=float(p_cfg.val_ratio),
            ratio=float(p_cfg.ratio),
            max_train_episodes=p_cfg.max_train_episodes,
            **common,
        )
    return CanonicalImageDataset(
        seed=int(p_cfg.seed),
        val_ratio=0.0,
        ratio=None,
        max_train_episodes=None,
        episode_include_path=episode_include_path,
        **common,
    )


@torch.no_grad()
def window_features(ds: CanonicalImageDataset, idx: np.ndarray, gen, policy, device, tag: str) -> dict:
    t0 = time.time()
    feats = {k: [] for k in SPACES}
    for s in range(0, len(idx), 256):
        items = [ds[int(i)] for i in idx[s : s + 256]]
        rgb = torch.stack([it["obs"]["rgb"] for it in items])  # (n, n_obs, 6, 84, 84) float [0,1]
        rgb_u8 = (rgb * 255.0).round().clamp(0, 255).to(torch.uint8)
        st = torch.stack([it["obs"]["states"] for it in items])
        ac = torch.stack([it["action"] for it in items])
        f = embed(gen, policy, rgb_u8, st, ac, device)
        for k in SPACES:
            feats[k].append(f[k])
        if s % 2048 == 0:
            print(f"[nn] {tag} windows {s + len(items)}/{len(idx)} ({time.time() - t0:.0f}s)", flush=True)
    return {k: torch.cat(v).to(device) for k, v in feats.items()}


def select_mixture(epochs: list[int], ratio: dict[int, float]) -> dict:
    """gamma_0 = first checkpoint with ratio < 1; mixture = gamma_0 + 100 * {0, 1, 2, 3, 4}."""
    below = [e for e in epochs if ratio[e] < 1.0]
    assert below, "no checkpoint has ratio < 1; train the generator longer or save more checkpoints"
    gamma_0 = below[0]
    selected = [gamma_0 + MIXTURE_SPACING * i for i in range(MIXTURE_SIZE)]
    missing = [e for e in selected if e not in epochs]
    return {
        "gamma_0": gamma_0,
        "selected_epochs": selected,
        "missing_checkpoints": missing,
        "ratio": {str(e): ratio[e] for e in epochs},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen_run", required=True)
    ap.add_argument("--policy_ckpt", required=True, help="canonical BC policy ckpt (its EMA encoder defines the obs space)")
    ap.add_argument("--bank", default=None, help="optional held-out bank dir (bank.npz: rgb/states/actions); default: draw held-out windows from the complement episodes of the generator's zarr")
    ap.add_argument("--epochs", type=int, nargs="*", default=None, help="generator epochs (default: all checkpoints)")
    ap.add_argument("--out", default=None, help="default <gen_run>/analysis/nn_ratio")
    ap.add_argument("--n_gen", type=int, default=1024, help="generated samples per checkpoint (diagnostic ratios)")
    ap.add_argument("--n_rho", type=int, default=1024, help="generated chunks used for rho (first n_rho of n_gen; default = all 1024)")
    ap.add_argument("--n_set", type=int, default=8192, help="size of D_train and D_held (equal; random windows, seed 0)")
    ap.add_argument("--n_train", type=int, default=None, help="override the training-set size (default n_set; the original study used 8192 train / 4096 held-out)")
    ap.add_argument("--select_on", choices=["rho", "act"], default="rho",
                    help="ratio that defines gamma_0: rho = manuscript definition (the n_rho generated chunks, equal-size "
                         "D_train/D_held); act = original study's act ratio (all n_gen samples, mean gen-NN / mean held-out-NN). "
                         "Both selections are always written to selection.json; this picks the one printed as the mixture")
    ap.add_argument("--num_inference_steps", type=int, default=100, help="as in the distillation mixer")
    ap.add_argument("--in_memory", action="store_true", help="load the zarr into RAM (faster window reads)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    assert args.n_rho <= args.n_gen

    register_training_resolvers()
    configure_determinism()
    device = torch.device("cuda")
    run = Path(args.gen_run)
    out = Path(args.out) if args.out else run / "analysis" / "nn_ratio"
    out.mkdir(parents=True, exist_ok=True)
    epochs = args.epochs or epochs_of(run)
    assert epochs, f"no checkpoints under {run}"

    # ---- policy (obs space) and the generator's own training split ----
    policy = load_deployed_policy(args.policy_ckpt, 10, 0.0, device)
    gen0, cfg = load_generator_checkpoint(run / f"checkpoints/epoch_{epochs[0]:04d}.ckpt", map_location="cpu")
    gen0.to(device)
    ds_train = dataset_like(cfg, None, args.in_memory)
    train_eps = np.flatnonzero(ds_train.train_mask).tolist()
    heldout_eps = sorted(set(range(len(ds_train.train_mask))) - set(train_eps))
    print(f"[nn] generator train split: {len(train_eps)} episodes, {len(ds_train)} windows; "
          f"held-out pool: {len(heldout_eps)} episodes", flush=True)

    rng = np.random.default_rng(args.seed)
    n_set = min(args.n_set, len(ds_train))
    n_train = n_set if args.n_train is None else min(int(args.n_train), len(ds_train))
    train_idx = np.sort(rng.choice(len(ds_train), size=n_train, replace=False))
    bank = window_features(ds_train, train_idx, gen0, policy, device, "train")

    # ---- held-out windows (same size as the training set) ----
    if args.bank is None:
        heldout_json = out / "heldout_episodes.json"
        json.dump(heldout_eps, open(heldout_json, "w"))
        ds_held = dataset_like(cfg, str(heldout_json), args.in_memory)
        assert len(ds_held) >= n_set, f"held-out pool has only {len(ds_held)} windows < {n_set}"
        held_idx = np.sort(rng.choice(len(ds_held), size=n_set, replace=False))
        held = window_features(ds_held, held_idx, gen0, policy, device, "held-out")
        heldout_src = f"complement episodes of {cfg.dataset.source.zarr_path}"
    else:
        b = np.load(Path(args.bank) / "bank.npz")
        assert b["rgb"].shape[0] >= n_set, f"bank has only {b['rgb'].shape[0]} windows < {n_set}"
        held = embed(gen0, policy, torch.from_numpy(b["rgb"][:n_set]), torch.from_numpy(b["states"][:n_set]),
                     torch.from_numpy(b["actions"][:n_set]), device)
        held = {k: v.to(device) for k, v in held.items()}
        heldout_src = str(args.bank)

    ref_nn = {k: nn_dist(held[k], bank[k]).cpu().numpy() for k in SPACES}
    own_nn = {k: nn_dist(bank[k], bank[k], exclude_self=True).cpu().numpy() for k in SPACES}
    summary = {
        "gen_run": str(run), "policy_ckpt": args.policy_ckpt, "heldout_source": heldout_src,
        "n_set": int(n_set), "n_train": int(n_train), "n_gen": args.n_gen, "n_rho": args.n_rho,
        "num_inference_steps": args.num_inference_steps, "seed": args.seed,
        "ref_nn_mean": {k: float(ref_nn[k].mean()) for k in SPACES},
        "ref_nn_median": {k: float(np.median(ref_nn[k])) for k in SPACES},
        "train_own_nn_median": {k: float(np.median(own_nn[k])) for k in SPACES},
        "epochs": {},
    }
    np.savez(out / "reference.npz", **{f"ref_nn_{k}": ref_nn[k] for k in SPACES},
             **{f"own_nn_{k}": own_nn[k] for k in SPACES}, train_idx=train_idx)
    print("[nn] reference NN (held-out -> train), mean: " +
          " ".join(f"{k}={ref_nn[k].mean():.4f}" for k in SPACES), flush=True)

    # ---- per checkpoint ----
    rho: dict[int, float] = {}
    for ep in epochs:
        f_out = out / f"epoch_{ep:04d}.npz"
        if f_out.exists():
            z = np.load(f_out)
            summary["epochs"][str(ep)] = {k: float(z[f"gen_nn_{k}"].mean() / ref_nn[k].mean()) for k in SPACES}
            rho[ep] = float(z["rho"])
            print(f"[nn] ep {ep}: exists, skipping (rho={rho[ep]:.4f})", flush=True)
            continue
        t0 = time.time()
        gen, _ = load_generator_checkpoint(run / f"checkpoints/epoch_{ep:04d}.ckpt", map_location="cpu")
        gen.to(device)
        g_rng = torch.Generator(device=device).manual_seed(args.seed)
        parts = {k: [] for k in SPACES}
        for s in range(0, args.n_gen, 256):
            n = min(256, args.n_gen - s)
            g = gen.sample(n, device=device, generator=g_rng, num_inference_steps=args.num_inference_steps)
            rgb_u8 = (g["obs"]["rgb"] * 255.0).round().clamp(0, 255).to(torch.uint8)
            f = embed(gen, policy, rgb_u8, g["obs"]["states"], g["action"], device)
            for k in SPACES:
                parts[k].append(f[k])
        gen_feats = {k: torch.cat(parts[k]).to(device) for k in SPACES}
        gen_nn = {k: nn_dist(gen_feats[k], bank[k]).cpu().numpy() for k in SPACES}
        # manuscript rho: first n_rho generated NORMALIZED action chunks vs D_train and D_held
        a_rho = gen_feats["act"][: args.n_rho]
        d_train = nn_dist(a_rho, bank["act"]).cpu().numpy()
        d_held = nn_dist(a_rho, held["act"]).cpu().numpy()
        rho[ep] = float(d_train.mean() / d_held.mean())
        np.savez(f_out, **{f"gen_nn_{k}": gen_nn[k] for k in SPACES},
                 rho=rho[ep], rho_d_train=d_train, rho_d_held=d_held)
        rat = {k: float(gen_nn[k].mean() / ref_nn[k].mean()) for k in SPACES}
        rep = {k: float((gen_nn[k] < 0.05 * np.median(ref_nn[k])).mean()) for k in SPACES}
        summary["epochs"][str(ep)] = rat
        print(f"[nn] ep {ep}: rho={rho[ep]:.4f} | NN ratio " + " ".join(f"{k}={rat[k]:.3f}" for k in SPACES) +
              " | replica% " + " ".join(f"{k}={100 * rep[k]:.2f}" for k in SPACES) +
              f" ({time.time() - t0:.0f}s)", flush=True)
        json.dump(summary, open(out / "summary.json", "w"), indent=1)
    json.dump(summary, open(out / "summary.json", "w"), indent=1)

    act_ratio = {e: float(summary["epochs"][str(e)]["act"]) for e in epochs}
    selection = {
        "select_on": args.select_on,
        "selection_rho": select_mixture(epochs, rho),
        "selection_act": select_mixture(epochs, act_ratio),
    }
    chosen = selection[f"selection_{args.select_on}"]
    assert not chosen["missing_checkpoints"], (
        f"mixture checkpoints {chosen['missing_checkpoints']} are not saved (have {epochs})"
    )
    selection["gamma_0"] = chosen["gamma_0"]
    selection["selected_epochs"] = chosen["selected_epochs"]
    json.dump(selection, open(out / "selection.json", "w"), indent=1)
    for key in ("rho", "act"):
        sel = selection[f"selection_{key}"]
        print(f"[nn] {key}: gamma_0 = {sel['gamma_0']} -> mixture {sel['selected_epochs']}"
              + ("  (selected)" if key == args.select_on else "")
              + (f"  [not saved: {sel['missing_checkpoints']}]" if sel["missing_checkpoints"] else ""))
    print(f"[nn] -> {out}")


if __name__ == "__main__":
    main()
