"""General top-to-bottom block diagrams: input, architecture, output.
Forward: one design -> one ERP (one-to-one).  Inverse: one target ERP -> several possible designs (many designs share one ERP).
Light colours, thesis fonts, plate and ERP icons (the ERP curve is the real example in inverse_example.npz).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle

import utils.plot_style  # noqa: F401

ERP = np.load(ROOT / "presentation_figures" / "inverse_example.npz")["target"]
LX, LY = 1.4, 0.5
FWD, INV = "#3d6a94", "#e8632b"
TITLE, SUB = "#1c3550", "#4a5a6a"
C_PURPLE = ("#ebe2f5", "#b39ad1")
ICON_BG = "#dbe8f6"
# illustrative resonator positions (x, y) in metres
DESIGNS = [[(0.30, 0.15), (0.75, 0.38), (1.15, 0.22)], [(0.25, 0.38), (0.60, 0.12), (1.05, 0.30)], [(0.45, 0.25), (0.90, 0.42), (1.25, 0.12)]]


def icon_plate(ax, x0, y0, w, h, pts):
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0,rounding_size=1.0", fc=ICON_BG, ec="#8fb4d9", lw=1.4, zorder=3))
    pw = 0.80 * w
    ph = pw * LY / LX
    px0, py0 = x0 + (w - pw) / 2, y0 + (h - ph) / 2
    ax.add_patch(Rectangle((px0, py0), pw, ph, fc="white", ec="black", lw=2.0, zorder=4))
    for x, y in pts:
        ax.plot([px0 + x / LX * pw], [py0 + y / LY * ph], "o", color="black", ms=8, zorder=5)


def icon_erp(ax, x0, y0, w, h):
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0,rounding_size=1.0", fc=ICON_BG, ec="#8fb4d9", lw=1.4, zorder=3))
    a0, b0, a1, b1 = x0 + 0.12 * w, y0 + 0.14 * h, x0 + 0.92 * w, y0 + 0.90 * h
    ax.plot([a0, a0, a1], [b1, b0, b0], color="black", lw=1.4, zorder=4, solid_capstyle="butt")
    t = np.linspace(0, 1, ERP.size)
    v = (ERP - ERP.min()) / (ERP.max() - ERP.min())
    ax.plot(a0 + 0.02 * w + t * (a1 - a0 - 0.04 * w), b0 + 0.04 * h + v * (b1 - b0 - 0.10 * h), color="black", lw=2.4, zorder=5)


def arrow(ax, p0, p1, color, ls="-"):
    ax.annotate("", xy=p1, xytext=p0, zorder=2,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=2.2, ls=ls, shrinkA=0, shrinkB=0, mutation_scale=20))


def make(name, title, input_kind, output_kind, arrow_color, ls, arch_sub, tag):
    W, H = 50.0, 78.0
    fig = plt.figure(figsize=(6.4, 6.4 * H / W))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    ax.text(W / 2, H - 3.0, title, fontsize=18, ha="center", va="center", color=TITLE)
    cx = W / 2

    def place(kind, y0, label):
        if kind == "plate":
            icon_plate(ax, cx - 9, y0, 18, 13, DESIGNS[0])
        elif kind == "erp":
            icon_erp(ax, cx - 9, y0, 18, 13)
        elif kind == "plates":
            for i, d in enumerate(DESIGNS):
                icon_plate(ax, cx - 22 + i * 15.5, y0, 13, 10, d)
        ax.text(cx, y0 - 3.2, label, fontsize=13, ha="center", va="center", color=TITLE)

    labels = {"plate": "Resonator configuration", "erp": "ERP spectrum"}
    # input
    y_in = H - 20.0
    place(input_kind, y_in, "Input: " + labels[input_kind].lower() if False else labels[input_kind])
    ax.text(cx + 12, y_in + 11.5, "Input", fontsize=12, color=SUB, ha="left", va="center")
    # architecture
    ya0, ya1 = 28.0, 40.0
    ax.add_patch(FancyBboxPatch((cx - 14, ya0), 28, ya1 - ya0, boxstyle="round,pad=0,rounding_size=1.6", fc=C_PURPLE[0], ec=C_PURPLE[1], lw=1.8, zorder=3))
    ax.text(cx, (ya0 + ya1) / 2 + 1.3, "Architecture", fontsize=18, ha="center", va="center", color=TITLE, zorder=4)
    ax.text(cx, (ya0 + ya1) / 2 - 2.4, arch_sub, fontsize=11, ha="center", va="center", color=SUB, zorder=4)
    # output
    y_out = 6.0 if output_kind != "plates" else 7.0
    place(output_kind, y_out, "Possible designs" if output_kind == "plates" else labels[output_kind])
    if output_kind == "plates":
        ax.text(2.0, y_out + 13.0, "Output", fontsize=12, color=SUB, ha="left", va="center")
    else:
        ax.text(cx + 12, y_out + 11.5, "Output", fontsize=12, color=SUB, ha="left", va="center")
    # arrows
    arrow(ax, (cx, y_in - 6.0), (cx, ya1 + 0.3), arrow_color, ls)
    if output_kind == "plates":
        for i in range(3):
            arrow(ax, (cx + (i - 1) * 8, ya0 - 0.3), (cx - 22 + i * 15.5 + 6.5, y_out + 10.4), arrow_color, ls)
    else:
        arrow(ax, (cx, ya0 - 0.3), (cx, y_out + 13.4), arrow_color, ls)
    ax.text(W / 2, 0.9, tag, fontsize=12.5, ha="center", va="center", color=arrow_color)
    for ext, kw in (("png", dict(dpi=220)), ("pdf", {}), ("svg", {})):
        fig.savefig(ROOT / "presentation_figures" / f"{name}.{ext}", facecolor="white", **kw)
    plt.close(fig)
    print("saved", name)


make("forward_general_block_diagram", "Forward", "plate", "erp", FWD, "-", "design $\\rightarrow$ ERP", "one design gives one ERP  (one-to-one)")
make("inverse_general_block_diagram", "Inverse", "erp", "plates", INV, "--", "ERP $\\rightarrow$ design", "many designs share one ERP  (many-to-one)")


# ---------------------------------------------------------------------------------------------------------------------
# Invertible: one architecture used in both directions (design <-> ERP), forward arrows down, inverse arrows up
# ---------------------------------------------------------------------------------------------------------------------
def make_invertible(name="invertible_general_block_diagram"):
    W, H = 50.0, 78.0
    fig = plt.figure(figsize=(6.4, 6.4 * H / W))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    cx = W / 2
    ax.text(cx, H - 3.0, "Invertible", fontsize=18, ha="center", va="center", color=TITLE)
    y_top, y_arch0, y_arch1, y_bot = H - 19.0, 28.0, 40.0, 6.0
    icon_plate(ax, cx - 9, y_top, 18, 13, DESIGNS[0])
    ax.text(cx, y_top - 3.2, "Resonator configuration", fontsize=13, ha="center", va="center", color=TITLE)
    ax.text(cx + 12, y_top + 11.5, "Input / output", fontsize=12, color=SUB, ha="left", va="center")
    ax.add_patch(FancyBboxPatch((cx - 14, y_arch0), 28, y_arch1 - y_arch0, boxstyle="round,pad=0,rounding_size=1.6", fc=C_PURPLE[0], ec=C_PURPLE[1], lw=1.8, zorder=3))
    ax.text(cx, (y_arch0 + y_arch1) / 2 + 1.3, "Architecture", fontsize=18, ha="center", va="center", color=TITLE, zorder=4)
    ax.text(cx, (y_arch0 + y_arch1) / 2 - 2.4, "design $\\leftrightarrow$ ERP", fontsize=11, ha="center", va="center", color=SUB, zorder=4)
    icon_erp(ax, cx - 9, y_bot, 18, 13)
    ax.text(cx, y_bot - 3.2, "ERP spectrum", fontsize=13, ha="center", va="center", color=TITLE)
    ax.text(cx + 12, y_bot + 11.5, "Input / output", fontsize=12, color=SUB, ha="left", va="center")
    dx = 3.0
    arrow(ax, (cx - dx, y_top - 6.0), (cx - dx, y_arch1 + 0.3), FWD)      # forward: design -> architecture
    arrow(ax, (cx - dx, y_arch0 - 0.3), (cx - dx, y_bot + 13.4), FWD)     # forward: architecture -> ERP
    arrow(ax, (cx + dx, y_bot + 14.2), (cx + dx, y_arch0 - 0.3), INV, "--")  # inverse: ERP -> architecture
    arrow(ax, (cx + dx, y_arch1 + 0.3), (cx + dx, y_top - 6.0), INV, "--")   # inverse: architecture -> design
    ax.text(cx - dx - 1.2, (y_top - 6.0 + y_arch1) / 2, "forward", fontsize=11.5, color=FWD, ha="right", va="center")
    ax.text(cx + dx + 1.2, (y_top - 6.0 + y_arch1) / 2, "inverse", fontsize=11.5, color=INV, ha="left", va="center")
    ax.text(W / 2, 0.9, "one architecture, both directions", fontsize=12.5, ha="center", va="center", color=SUB)
    for ext, kw in (("png", dict(dpi=220)), ("pdf", {}), ("svg", {})):
        fig.savefig(ROOT / "presentation_figures" / f"{name}.{ext}", facecolor="white", **kw)
    plt.close(fig)
    print("saved", name)


make_invertible()
