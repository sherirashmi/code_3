"""Small drawing kit for the presentation architecture figures (thesis Computer Modern look).

Everything is drawn on one axes whose data coordinates are 0..W by 0..H, so a figure is just a
list of calls placing panels, neurons, boxes and arrows.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle

import utils.plot_style  # noqa: F401  (thesis fonts)

W, H = 100.0, 56.0
BLUE, GREEN, ORANGE, YELLOW, PURPLE, GRAY, PINK = "#e3ecf8", "#e2f1e2", "#fbe5d6", "#fdf3c9", "#ece3f4", "#ececec", "#fbe3e8"
EDGE = "#333333"
OUT = Path(__file__).resolve().parent


def new_figure(width_in: float = 16.0):
    fig = plt.figure(figsize=(width_in, width_in * H / W))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    return fig, ax


def panel(ax, x, y, w, h, fc="white", ec=EDGE, ls="-", lw=1.1, r=1.6, z=1, alpha=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec,
                                ls=ls, lw=lw, zorder=z, alpha=alpha))


def text(ax, x, y, s, size=9.5, ha="center", va="center", color="black", z=10, **kw):
    return ax.text(x, y, s, fontsize=size, ha=ha, va=va, color=color, zorder=z, **kw)


def titled_panel(ax, x, y, w, h, title, fc, ls="-", size=11, title_dy=1.5, **kw):
    panel(ax, x, y, w, h, fc=fc, ls=ls, **kw)
    text(ax, x + w / 2, y + h - title_dy, title, size=size)


def box(ax, x, y, w, h, s, fc=GRAY, size=9, ec=EDGE, lw=1.0, r=1.0, z=3):
    panel(ax, x, y, w, h, fc=fc, ec=ec, lw=lw, r=r, z=z)
    text(ax, x + w / 2, y + h / 2, s, size=size, linespacing=1.35)


def neuron(ax, x, y, label="", r=0.95, fc="#ededed", size=8.5, z=5):
    ax.add_patch(Circle((x, y), r, fc=fc, ec=EDGE, lw=0.9, zorder=z))
    if label:
        text(ax, x, y, label, size=size, z=z + 1)


def dots(ax, x, y, size=9):
    text(ax, x, y, r"$\vdots$", size=size)


def mlp(ax, xs, layers, label=r"$\sigma$", r=0.95, link_alpha=0.55, size=8.5):
    """Draw fully connected layers; ``layers`` = list (per column) of y-lists (None = '...' gap)."""
    pos = []
    for x, ys in zip(xs, layers):
        pos.append([(x, y) for y in ys if y is not None])
    for a, b in zip(pos[:-1], pos[1:]):
        for (x0, y0) in a:
            for (x1, y1) in b:
                ax.plot([x0 + r, x1 - r], [y0, y1], color="#444444", lw=0.5, alpha=link_alpha, zorder=2)
    for x, ys in zip(xs, layers):
        for y in ys:
            if y is None:
                continue
            neuron(ax, x, y, label, r=r, size=size)
    return pos


def arrow(ax, pts, color="black", lw=1.2, ls="-", head=True, z=6, style="-|>", ms=9):
    xs, ys = zip(*pts)
    if len(pts) > 2:
        ax.plot(xs[:-1], ys[:-1], color=color, lw=lw, ls=ls, zorder=z, solid_capstyle="butt")
    ax.annotate("", xy=pts[-1], xytext=pts[-2], zorder=z,
                arrowprops=dict(arrowstyle=style if head else "-", color=color, lw=lw, ls=ls,
                                shrinkA=0, shrinkB=0, mutation_scale=ms))


def circle_op(ax, x, y, s, r=1.2, fc="#ededed", size=11):
    ax.add_patch(Circle((x, y), r, fc=fc, ec=EDGE, lw=1.0, zorder=5))
    text(ax, x, y, s, size=size, z=6)


def diamond(ax, x, y, w, h, s, fc="#f7f0c8", size=10):
    ax.add_patch(Polygon([(x - w / 2, y), (x, y + h / 2), (x + w / 2, y), (x, y - h / 2)], closed=True,
                         fc=fc, ec=EDGE, lw=1.0, zorder=4))
    text(ax, x, y, s, size=size, z=6, linespacing=1.2)


def spectrum_inset(fig, ax, x, y, w, h, curve, color="#2a78d6", fill=True, xlabel="", ylabel="", dashed=None):
    """Tiny ERP-vs-frequency axes placed in diagram coordinates."""
    ia = fig.add_axes([x / W, y / H, w / W, h / H])
    f = np.linspace(10, 160, curve.size)
    ia.plot(f, curve, color=color, lw=1.4)
    if dashed is not None:
        ia.plot(f, dashed, color="#c2412c", lw=1.1, ls="--")
    if fill:
        ia.fill_between(f, curve, curve.min() - 3, color=color, alpha=0.12, lw=0)
    ia.set_xlim(10, 160)
    ia.set_ylim(curve.min() - 3, curve.max() + 6)
    ia.set_xticks([])
    ia.set_yticks([])
    for s in ("top", "right"):
        ia.spines[s].set_visible(False)
    ia.set_facecolor("none")
    if xlabel:
        ia.set_xlabel(xlabel, fontsize=8, labelpad=1)
    if ylabel:
        ia.set_ylabel(ylabel, fontsize=8, labelpad=1)
    return ia


def sample_erp(index: int = 21) -> np.ndarray:
    """A real ERP spectrum (dB) from the 10k dataset, used purely as an illustration."""
    from utils.support import load_dataset

    payload = load_dataset(str(ROOT / "datasets" / "dataset_erp_ft.pth"))
    return np.asarray(payload["responses"], dtype=np.float64)[index, :, 0]


def save(fig, name: str):
    for ext, kw in (("png", dict(dpi=220)), ("pdf", {}), ("svg", {})):
        fig.savefig(OUT / f"{name}.{ext}", facecolor="white", **kw)
    plt.close(fig)
    print("saved", OUT / f"{name}.png")
