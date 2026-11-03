"""Sanity-check a joint generator checkpoint: are its samples realistic yet novel?

Loads a JointImageOADiTModel checkpoint, generates joint samples, and compares them with
the generator's TRAINING windows:
  1. Feature-space NN ratio of generated images to the train bank in a frozen random-init
     ResNet18 feature space: median(gen NN) / median(data own-NN), plus duplicate fraction.
  2. Per-channel pixel marginals (mean/std of each of the 6 rgb channels).
  3. State / action marginals in the generator's normalized space (|delta std|).
  4. A grid of sampled images (both cameras) for visual inspection.

    CUDA_VISIBLE_DEVICES=0 python -m scripts.eval_joint_generator_image \
        --ckpt outputs/training/<task>_joint_gen_img/<run>/checkpoints/epoch_0100.ckpt
"""

from __future__ import annotations

import argparse
import pathlib

import matplotlib
import numpy as np
import torch
import torch.nn as nn

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from scripts.training.joint_generator_image import build_joint_image_dataset  # noqa: E402
from utils.oa_joint_image_mixer import load_generator_checkpoint  # noqa: E402


def _nn_dist(query: torch.Tensor, bank: torch.Tensor, exclude_self: bool) -> torch.Tensor:
    d = torch.cdist(query, bank)
    if exclude_self:
        d = d + torch.eye(d.shape[0], d.shape[1], device=d.device) * 1e12
    return d.min(dim=1).values


def _build_feature_extractor(in_ch: int, device) -> nn.Module:
    """Frozen random-init ResNet18 (conv1 adapted to `in_ch` channels, fc removed) -> 512-d
    features: a fixed random projection, i.e. a stable feature space for the NN ratio."""
    from torchvision.models import resnet18

    net = resnet18(weights=None)
    net.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
    net.fc = nn.Identity()
    for p in net.parameters():
        p.requires_grad_(False)
    return net.to(device).eval()


@torch.no_grad()
def _collect_bank(model, feat, dataset, n_items: int, device, chunk: int = 64):
    rng = np.random.default_rng(0)
    idx = rng.choice(len(dataset), size=min(n_items, len(dataset)), replace=False)
    feats, rgb_chan, states, acts = [], [], [], []
    for s in range(0, len(idx), chunk):
        items = [dataset[int(i)] for i in idx[s : s + chunk]]
        rgb = torch.stack([it["obs"]["rgb"][0] for it in items], 0).to(device)  # (n,6,84,84) [0,1]
        feats.append(feat(rgb * 2 - 1).cpu())  # feature on [-1,1]
        rgb_chan.append(rgb.reshape(rgb.shape[0], 6, -1).cpu())
        st = torch.stack([it["obs"]["states"] for it in items], 0)  # (n,1,9)
        ac = torch.stack([it["action"] for it in items], 0)         # (n,8,10)
        states.append(model.normalizer["states"].normalize(st.to(device)).reshape(st.shape[0], -1).cpu())
        acts.append(model.normalizer["action"].normalize(ac.to(device)).reshape(ac.shape[0], -1).cpu())
    return torch.cat(feats, 0), torch.cat(rgb_chan, 0), torch.cat(states, 0), torch.cat(acts, 0)


def _save_grid(rgb: torch.Tensor, path: pathlib.Path, n: int = 6) -> None:
    """rgb: (B,6,84,84) in [0,1]; show cam0 (ch0:3) and cam1 (ch3:6) for n samples."""
    n = min(n, rgb.shape[0])
    fig, axes = plt.subplots(2, n, figsize=(2 * n, 4))
    for i in range(n):
        for cam in range(2):
            img = rgb[i, cam * 3 : cam * 3 + 3].permute(1, 2, 0).clamp(0, 1).numpy()
            ax = axes[cam, i] if n > 1 else axes[cam]
            ax.imshow(img)
            ax.axis("off")
    fig.suptitle("sampled images (rows = 2 cameras)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n_gen", type=int, default=1000)
    ap.add_argument("--n_bank", type=int, default=3000)
    ap.add_argument("--num_inference_steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda")
    model, cfg = load_generator_checkpoint(pathlib.Path(args.ckpt), map_location="cpu")
    model.to(device)
    dataset = build_joint_image_dataset(cfg)
    feat = _build_feature_extractor(model.rgb_shape[0], device)
    print(
        f"ckpt={pathlib.Path(args.ckpt).name} | rgb={model.rgb_shape} "
        f"action_dim={model.action_dim} | train_windows={len(dataset)}",
        flush=True,
    )

    bank_feat, bank_rgb, bank_state, bank_act = _collect_bank(model, feat, dataset, args.n_bank, device)

    gen = torch.Generator(device=device).manual_seed(args.seed)
    g = model.sample(args.n_gen, device=device, generator=gen, num_inference_steps=args.num_inference_steps)
    grgb = g["obs"]["rgb"][:, 0]  # (n,6,84,84) [0,1]
    gen_feat = feat(grgb.to(device) * 2 - 1).cpu()
    gen_rgb = grgb.reshape(args.n_gen, 6, -1).cpu()
    gen_state = model.normalizer["states"].normalize(g["obs"]["states"]).reshape(args.n_gen, -1).cpu()
    gen_act = model.normalizer["action"].normalize(g["action"]).reshape(args.n_gen, -1).cpu()

    n_probe = min(args.n_gen, bank_feat.shape[0])
    data_nn = _nn_dist(bank_feat[:n_probe], bank_feat, exclude_self=True)
    gen_nn = _nn_dist(gen_feat, bank_feat, exclude_self=False)
    dm, gm = float(data_nn.median()), float(gen_nn.median())
    print("\n[feature-space realism] (frozen random ResNet18)")
    print(f"  data own-NN median = {dm:.3f} | gen NN median = {gm:.3f}")
    print(f"  NN ratio (gen/data) = {gm / max(dm, 1e-9):.2f}  (realistic ~[1.0,1.5])")
    print(f"  duplicate fraction  = {float((gen_nn < 0.05 * dm).float().mean()):.1%}")

    print("\n[pixel marginals] per rgb channel (mean / std), gen vs data")
    for c in range(6):
        gmu, gsd = float(gen_rgb[:, c].mean()), float(gen_rgb[:, c].std())
        bmu, bsd = float(bank_rgb[:, c].mean()), float(bank_rgb[:, c].std())
        print(f"  ch{c}: gen({gmu:.3f},{gsd:.3f}) data({bmu:.3f},{bsd:.3f}) "
              f"dmean={abs(gmu - bmu):.3f} dstd={abs(gsd - bsd):.3f}")

    print("\n[low-dim marginals] generated vs train (normalized; |dstd|<0.1 = good)")
    for name, gen_b, bank_b in (("states", gen_state, bank_state), ("action", gen_act, bank_act)):
        dstd = (gen_b.std(0) - bank_b.std(0)).abs()
        print(f"  {name}: max|dstd|={float(dstd.max()):.3f} mean|dstd|={float(dstd.mean()):.3f}")

    out_png = pathlib.Path(args.ckpt).parent.parent / f"samples_{pathlib.Path(args.ckpt).stem}.png"
    _save_grid(grgb.cpu(), out_png)
    print(f"\nwrote image grid -> {out_png}")


if __name__ == "__main__":
    main()
