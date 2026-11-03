"""Paper figure: (a) within-state error quantiles vs epoch, (b) the policy's action samples
at one held-out state for the early and late checkpoints, (c) the Coverage-Reliability
Balance with the selected checkpoint. Without --samples, the two-panel (a)+(c) version.

Inputs: <analysis_dir>/surface.npz (+ meta.json) from scripts/crb/analyze.py and, for (b),
samples.npz from scripts/crb/dump_action_clouds.py.

    python -m scripts.crb.plot_quantile_clouds_crb --analysis_dir <run>/analysis/crb \
        --samples <run>/analysis/crb/qualitative/samples.npz --out <dir>/fig_quantile_clouds_crb
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from scripts.crb.fig_style import (  # noqa: E402
    COV_COLOR, EARLY_COLOR, LATE_COLOR, REL_COLOR, SEL_COLOR, U_CMAP,
    apply_grid, caption, log_fmt, use_paper_style,
)

DOT = 4.0
MIN_ASPECT, MAX_ASPECT = 0.45, 1.0


def load_metric(analysis_dir: Path) -> dict:
    z = np.load(analysis_dir / "surface.npz")
    d = {k: z[k] for k in ("epochs", "u", "Qbar", "R_cov", "R_rel", "R_max", "P")}
    d["i_sel"] = int(z["i_sel"])
    d["i_or"] = int(z["i_or"])
    return d


# --------------------------------------------------------------- panel (a)
def draw_quantiles(ax, fig, d: dict, cbar_box=(0.56, 0.86, 0.40, 0.06)) -> None:
    eps, U, Qbar = d["epochs"], d["u"], d["Qbar"]
    for j, u in enumerate(U):
        if j % 2:
            continue
        ax.plot(eps, Qbar[:, j], "-", lw=1.0, color=U_CMAP(j / (len(U) - 1)))
    ax.set_yscale("log")
    log_fmt(ax.yaxis)
    ax.set_xlabel("training epoch", labelpad=1.2)
    ax.set_ylabel("error quantile $\\bar{Q}_\\tau(u)$", labelpad=0.5)
    cax = ax.inset_axes(list(cbar_box))
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=U_CMAP, norm=plt.Normalize(U[0], U[-1])),
                      cax=cax, orientation="horizontal", ticks=[0.05, 0.5, 0.95])
    cb.outline.set_visible(False)
    cax.tick_params(labelsize=5.6, length=1.5, pad=1)
    cax.set_title("percentile $u$", fontsize=6, pad=1.5)


# --------------------------------------------------------------- panel (c)
def draw_crb(ax, d: dict, legend_fontsize=None, legend_colspacing: float = 0.5,
             vline_ymax: float = 1.0, legend_alpha: float = 0.55) -> None:
    eps, R_cov, R_rel, M, i_sel, i_or = (d[k] for k in ("epochs", "R_cov", "R_rel", "R_max", "i_sel", "i_or"))
    t_sel = eps[i_sel]
    ax.plot(eps, R_cov, "-", color=COV_COLOR, lw=1.7, zorder=3, label="$r_{\\mathrm{cov}|H}$")
    ax.plot(eps, R_rel, "-", color=REL_COLOR, lw=1.7, zorder=3, label="$r_{\\mathrm{rel}|H}$")
    ax.plot(eps, M, ls=(0, (1, 1.4)), color="#55505e", lw=1.3, zorder=4,
            label="$\\max\\{r_{\\mathrm{cov}|H}, r_{\\mathrm{rel}|H}\\}$")
    star, = ax.plot(t_sel, M[i_sel], "*", color=SEL_COLOR, ms=9, zorder=6,
                    markeredgecolor="white", markeredgewidth=0.6)
    vline = ax.axvline(t_sel, ymax=vline_ymax, color=SEL_COLOR, lw=0.8, ls=(0, (4, 2)), alpha=0.55, zorder=2)
    handles_extra, labels_extra = [], []
    if i_or >= 0:
        ring, = ax.plot(eps[i_or], M[i_or], "o", ms=9, mfc="none", mec="#e3a857", mew=1.1, zorder=5)
        handles_extra, labels_extra = [ring], ["oracle"]
    ax.set_ylim(0, 1.5)
    ax.set_yticks([0, 0.5, 1.0])
    step = 200
    ticks = [t for t in range(step, int(eps[-1]), step)]
    if ticks and int(eps[-1]) - ticks[-1] < 0.6 * step:
        ticks.pop()
    ax.set_xticks(ticks + [float(eps[-1])])
    ax.set_xticklabels([str(t) for t in ticks] + ["$H$"])
    ax.set_xlabel("training epoch", labelpad=1.2)
    ax.set_ylabel("normalized risk", labelpad=0.5)
    ax.annotate(f"ep {t_sel}", xy=(t_sel, 0.95), xytext=(3, 0), textcoords="offset points",
                fontsize=7, fontweight="bold", color=SEL_COLOR, ha="left", va="top")
    handles, labels = ax.get_legend_handles_labels()
    blank = matplotlib.patches.Patch(alpha=0)
    handles = [handles[0], handles[1]] + handles_extra + [handles[2], (vline, star), blank]
    labels = [labels[0], labels[1]] + labels_extra + [labels[2], "selected", ""]
    ax.legend(handles, labels, handlelength=0.9, ncol=2, columnspacing=legend_colspacing, loc="upper right",
              frameon=True, framealpha=legend_alpha, edgecolor="none", markerscale=0.8, borderpad=0.25,
              labelspacing=0.25, handletextpad=0.4, fontsize=legend_fontsize,
              handler_map={tuple: matplotlib.legend_handler.HandlerTuple(ndivide=None)})


# --------------------------------------------------------------- panel (b)
def pca2(X: np.ndarray) -> np.ndarray:
    mu = X.mean(0)
    _, _, vt = np.linalg.svd(X - mu, full_matrices=False)
    return (X - mu) @ vt[:2].T


def embed(A, demo, err, s: int, ie: int, il: int):
    """Shared 2-d PCA embedding of one state's early / late draws and the expert chunk."""
    K = A.shape[2]
    X = np.concatenate([A[s, ie].reshape(K, -1), A[s, il].reshape(K, -1), demo[s].reshape(1, -1)])
    P = pca2(X)
    return P[:K], P[K:2 * K], P[2 * K], int(err[s, ie].argmin()), int(err[s, il].argmin())


def shared_lims(*Ps, pad=0.10):
    Q = np.concatenate([np.atleast_2d(p) for p in Ps])
    lo, hi = Q.min(0), Q.max(0)
    c = (lo + hi) / 2
    xr, yr = (hi - lo) * (1 + pad)
    xr = max(xr, 1e-9)
    yr = max(yr, 1e-9)
    if yr / xr < MIN_ASPECT:
        yr = MIN_ASPECT * xr
    elif yr / xr > MAX_ASPECT:
        xr = yr / MAX_ASPECT
    return ((c[0] - xr / 2, c[0] + xr / 2), (c[1] - yr / 2, c[1] + yr / 2))


def pane(ax, P, star, best, color, mse: float, K: int, lims, dot_size: float = DOT,
         ring_color=None, ring_lw: float = 1.0) -> None:
    ax.scatter(P[:, 0], P[:, 1], s=dot_size, color=color, alpha=0.65, lw=0, zorder=3)
    ax.scatter(P[best, 0], P[best, 1], s=34, facecolors="none",
               edgecolors=color if ring_color is None else ring_color, lw=ring_lw, zorder=6)
    ax.scatter(star[0], star[1], marker="*", s=64, color="black", zorder=7, edgecolors="white", linewidths=0.5)
    ax.set_xlim(*lims[0])
    ax.set_ylim(*lims[1])
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_color("#c9c4c6")
        sp.set_linewidth(0.6)
    ax.text(0.5, -0.06, f"best-of-{K} MSE $\\mathbf{{{mse * 1e3:.2f}\\times10^{{-3}}}}$", transform=ax.transAxes,
            fontsize=5.8, va="top", ha="center", color="#33303a")


def legend_handles(e_early: int, e_late: int, K: int):
    return [Line2D([], [], marker="o", ls="", color=EARLY_COLOR, ms=3.5, label=f"ep{e_early} sample"),
            Line2D([], [], marker="o", ls="", color=LATE_COLOR, ms=3.5, label=f"ep{e_late} sample"),
            Line2D([], [], marker="o", ls="", mfc="none", mec="#55505e", ms=5.5, mew=1.0, label=f"best-of-{K} sample"),
            Line2D([], [], marker="*", ls="", color="black", ms=7, label="expert action")]


def darker(c, f: float = 0.72):
    r, g, b = to_rgb(c)
    return (r * f, g * f, b * f)


# --------------------------------------------------------------- figures
def figure_three_panel(d: dict, samples: Path, state: int, width: float):
    z = np.load(samples)
    A, demo, err, epochs = z["actions"], z["demo"], z["errors"], [int(e) for e in z["epochs"]]
    S, E, K = err.shape
    ie, il, s = 0, E - 1, state
    e_early, e_late = epochs[ie], epochs[il]

    fig = plt.figure(figsize=(width, 1.55))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 0.78, 0.78, 0.52, 1], wspace=0.30,
                          left=0.06, right=0.995, top=0.97, bottom=0.34)
    ax_a, ax_b1, ax_b2, ax_leg, ax_c = (fig.add_subplot(gs[i]) for i in range(5))
    ax_leg.set_axis_off()

    draw_quantiles(ax_a, fig, d)
    ax_a.set_xticks([e_early, 200, 400, 600, e_late])
    for lab in ax_a.get_xticklabels():
        v = int(float(lab.get_position()[0]))
        if v == e_early:
            lab.set_color(EARLY_COLOR)
            lab.set_fontweight("bold")
        elif v == e_late:
            lab.set_color(LATE_COLOR)
            lab.set_fontweight("bold")
    ax_a.set_ylabel("error quantile $\\bar{Q}_\\tau(u)$", labelpad=1.0)
    apply_grid(ax_a)
    caption(ax_a, "(a) Validation action-MSE quantiles")

    Pe, Pl, Ps, be, bl = embed(A, demo, err, s, ie, il)
    lims = shared_lims(Pe, Pl, Ps)
    pane(ax_b1, Pe, Ps, be, EARLY_COLOR, float(err[s, ie].min()), K, lims,
         dot_size=7.0, ring_color=darker(EARLY_COLOR), ring_lw=1.3)
    pane(ax_b2, Pl, Ps, bl, LATE_COLOR, float(err[s, il].min()), K, lims,
         dot_size=7.0, ring_color=darker(LATE_COLOR), ring_lw=1.3)
    for ax, t, c in ((ax_b1, f"early ckpt (ep{e_early})", EARLY_COLOR), (ax_b2, f"late ckpt (ep{e_late})", LATE_COLOR)):
        ax.set_aspect("auto")
        ax.texts[-1].set_position((0.5, -0.21))
        ax.texts[-1].set_fontsize(7.0)
        ax.text(0.5, -0.05, t, transform=ax.transAxes, ha="center", va="top", fontsize=7, color=c)
    ax_leg.legend(handles=legend_handles(e_early, e_late, K), loc="center left", ncol=1, fontsize=6.8,
                  markerscale=1.15, frameon=False, handletextpad=0.3, labelspacing=0.5, borderaxespad=0.0,
                  bbox_to_anchor=(-0.55, 0.5))
    caption(ax_b1, "(b) Action samples at a representative held-out state", dy=-0.40)
    ax_b1.texts[-1].set_x(1.40)

    draw_crb(ax_c, d, legend_fontsize=5.9, legend_colspacing=0.7, vline_ymax=0.66, legend_alpha=0.25)
    ax_c.set_ylabel("normalized risk", labelpad=1.0)
    apply_grid(ax_c)
    caption(ax_c, "(c) Coverage–Reliability Balance")
    return fig


def figure_two_panel(d: dict):
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(3.45, 1.55))
    fig.subplots_adjust(wspace=0.5, left=0.15, right=0.99, top=0.97, bottom=0.34)
    draw_quantiles(ax_a, fig, d)
    caption(ax_a, "(a) Full error quantiles")
    draw_crb(ax_b, d, legend_fontsize=5.6, legend_colspacing=0.6)
    caption(ax_b, "(b) Coverage–Reliability Balance")
    for ax in (ax_a, ax_b):
        apply_grid(ax)
    ax_a.yaxis.set_label_coords(-0.3, 0.5)
    ax_b.yaxis.set_label_coords(-0.24, 0.5)
    return fig


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--analysis_dir", required=True, help="scripts/crb/analyze.py output dir")
    ap.add_argument("--samples", default=None, help="samples.npz (panel b); omit for the two-panel figure")
    ap.add_argument("--state", type=int, default=0, help="state index within samples.npz")
    ap.add_argument("--width", type=float, default=7.2, help="three-panel figure width (in)")
    ap.add_argument("--out", required=True, help="output path stem (.png and .pdf are written)")
    args = ap.parse_args()

    use_paper_style()
    plt.rcParams.update({"font.size": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 6.5,
                         "ytick.labelsize": 6.5, "legend.fontsize": 6.0})
    d = load_metric(Path(args.analysis_dir))
    fig = figure_three_panel(d, Path(args.samples), args.state, args.width) if args.samples else figure_two_panel(d)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}")
    plt.close(fig)
    oracle = "" if d["i_or"] < 0 else f", oracle ep {d['epochs'][d['i_or']]}"
    print(f"[crb-fig] selected ep {d['epochs'][d['i_sel']]}{oracle} -> {out}.png|pdf")


if __name__ == "__main__":
    main()
