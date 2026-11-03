"""Shared figure style for the checkpoint-selection figures (paper conventions)."""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

K_GRID = [1, 2, 4, 8, 16, 32]
K_CMAP = LinearSegmentedColormap.from_list(
    "kviolet", ["#c3b8dd", "#a493c6", "#8571ae", "#67509c", "#463569"])
COV_COLOR = "#df9d9a"          # muted rose  (coverage / favorable tail)
REL_COLOR = "#8aa8c8"          # muted slate blue (reliability tail)
SEL_COLOR = "#67509c"          # violet accents (selection)
EARLY_COLOR, LATE_COLOR = "#E08262", "#5E93CE"
U_CMAP = LinearSegmentedColormap.from_list(
    "udiv", ["#c96b66", "#df9d9a", "#c9c4c6", "#8aa8c8", "#4a6f96"])


def k_color(i: int, n: int = len(K_GRID)):
    return K_CMAP(i / (n - 1))


def use_paper_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["P052", "Palatino", "TeX Gyre Pagella", "CMU Serif"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "P052", "mathtext.it": "P052:italic",
        "mathtext.bf": "P052:bold",
        "axes.unicode_minus": False,
        "font.size": 7.5,
        "axes.titlesize": 8,
        "axes.labelsize": 6.5,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "lines.solid_capstyle": "round",
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def apply_grid(ax, minor: bool = True) -> None:
    from matplotlib.ticker import AutoMinorLocator
    for axis in (ax.xaxis, ax.yaxis):
        if axis.get_scale() == "linear":
            axis.set_minor_locator(AutoMinorLocator(2))
    ax.grid(True, which="major", ls="-", lw=0.5, alpha=0.22)
    if minor:
        ax.grid(True, which="minor", ls="-", lw=0.4, alpha=0.10)
    ax.tick_params(which="minor", length=0)
    ax.set_axisbelow(True)


def log_fmt(ax_axis) -> None:
    ax_axis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda v, _: f"$10^{{{int(round(np.log10(v)))}}}$"))


def caption(ax, text: str, dy: float = -0.40) -> None:
    ax.text(0.5, dy, text, transform=ax.transAxes, ha="center", va="top",
            fontsize=7.8, fontweight="bold")
