"""Plate with a driving force and three sprung-mass resonators (line-art schematic, thesis fonts)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, Rectangle

import utils.plot_style  # noqa: F401

W_, S = 100.0, np.array([36.0, 20.0])  # x-edge length and the oblique y-edge vector (screen units)
LXY = (1.4, 0.5)
fig = plt.figure(figsize=(13, 5.1))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(-14, 150)
ax.set_ylim(-13, 45)
ax.set_aspect("equal")
ax.axis("off")
BLK = "black"


def P(x, y):  # plate coordinates (m) -> screen
    return np.array([x / LXY[0] * W_, 0.0]) + y / LXY[1] * S


# ---- plate: top surface, thickness strips -------------------------------------------------------------
t = 1.7
top = [P(0, 0), P(1.4, 0), P(1.4, 0.5), P(0, 0.5)]
ax.add_patch(Polygon(top, closed=True, fc="#f3f6fa", ec=BLK, lw=1.5, zorder=1))
ax.add_patch(Polygon([P(0, 0), P(1.4, 0), P(1.4, 0) + [0, -t], P(0, 0) + [0, -t]], closed=True, fc="white", ec=BLK, lw=1.5, zorder=1))
ax.add_patch(Polygon([P(1.4, 0), P(1.4, 0.5), P(1.4, 0.5) + [0, -t], P(1.4, 0) + [0, -t]], closed=True, fc="white", ec=BLK, lw=1.5, zorder=1))

# ---- axes and dimensions ------------------------------------------------------------------------------
O = P(0, 0)


def arrow(a, b, lw=1.3):
    ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="-|>", color=BLK, lw=lw, shrinkA=0, shrinkB=0, mutation_scale=12), zorder=6)


arrow(O, O + [0, 30])
ax.text(O[0] - 2.2, O[1] + 30, "$z$", fontsize=21, ha="right", va="center")
arrow(P(0, 0.5), P(0, 0.5) + S * 0.22)
ax.text(P(0, 0.5)[0] + S[0] * 0.22 + 2.0, P(0, 0.5)[1] + S[1] * 0.22 + 1.0, "$y$", fontsize=21, ha="left", va="center")
arrow(P(1.4, 0), P(1.4, 0) + [12, 0])
ax.text(P(1.4, 0)[0] + 14.0, P(1.4, 0)[1], "$x$", fontsize=21, ha="left", va="center")
ax.text(50, -t - 4.6, "$L_x$", fontsize=21, ha="center", va="center")
mid = (P(0, 0) + P(0, 0.5)) / 2
ax.text(mid[0] - 7.0, mid[1] + 1.2, "$L_y$", fontsize=21, ha="right", va="center")
ax.plot([O[0] - 8.5, O[0] - 0.8], [O[1], O[1]], color=BLK, lw=1.0, zorder=6)
ax.plot([O[0] - 8.5, O[0] - 0.8], [O[1] - t, O[1] - t], color=BLK, lw=1.0, zorder=6)
ax.text(O[0] - 10.0, O[1] - t / 2, "$h$", fontsize=19, ha="right", va="center")

# ---- driving force ------------------------------------------------------------------------------------------
xd, yd = 0.865, 0.309
pd = P(xd, yd)
arrow(pd + [0, 20], pd + [0, 0.8], lw=1.6)
ax.plot(*pd, "o", color=BLK, ms=3.5, zorder=7)
ax.text(pd[0] + 2.4, pd[1] + 19.5, "$f_d$", fontsize=22, ha="left", va="center")
ax.text(pd[0] + 0.5, pd[1] - 4.2, "$(x_d,y_d)$", fontsize=16, ha="center", va="center")


# ---- sprung-mass resonators ----------------------------------------------------------------------------------
def resonator(i, x, y):
    p = P(x, y)
    hs, mw, mh, dd = 9.5, 12.5, 5.6, np.array([2.8, 1.8])
    dx, sx = p[0] - 2.4, p[0] + 2.4
    ax.plot([dx, sx], [p[1], p[1]], color=BLK, lw=1.3, zorder=7)                       # base on the plate
    ax.plot([p[0]], [p[1]], "o", color=BLK, ms=3.0, zorder=8)
    ax.plot([dx, dx], [p[1], p[1] + 2.6], color=BLK, lw=1.3, zorder=7)                  # damper: cylinder + piston rod
    ax.add_patch(Rectangle((dx - 1.2, p[1] + 2.6), 2.4, 3.0, fc="white", ec=BLK, lw=1.3, zorder=7))
    ax.plot([dx, dx], [p[1] + 4.2, p[1] + hs], color=BLK, lw=1.3, zorder=7)
    ys = np.linspace(p[1], p[1] + hs, 8)                                                # spring
    zig = np.array([sx, sx - 1.3, sx + 1.3, sx - 1.3, sx + 1.3, sx - 1.3, sx + 1.3, sx])
    ax.plot(zig, ys, color=BLK, lw=1.3, zorder=7)
    x0, y0 = p[0] - mw / 2, p[1] + hs                                                   # mass block
    ax.add_patch(Rectangle((x0, y0), mw, mh, fc="white", ec=BLK, lw=1.4, zorder=7))
    ax.add_patch(Polygon([(x0, y0 + mh), (x0 + mw, y0 + mh), (x0 + mw + dd[0], y0 + mh + dd[1]), (x0 + dd[0], y0 + mh + dd[1])], closed=True, fc="white", ec=BLK, lw=1.4, zorder=7))
    ax.add_patch(Polygon([(x0 + mw, y0), (x0 + mw + dd[0], y0 + dd[1]), (x0 + mw + dd[0], y0 + mh + dd[1]), (x0 + mw, y0 + mh)], closed=True, fc="white", ec=BLK, lw=1.4, zorder=7))
    ax.text(p[0], y0 + mh / 2, f"$m_{i}$", fontsize=16, ha="center", va="center", zorder=8)
    ax.text(dx - 2.0, p[1] + hs * 0.42, f"$c_{i}$", fontsize=13, ha="right", va="center", zorder=8)
    ax.text(sx + 2.8, p[1] + hs * 0.42, f"$k_{i}$", fontsize=13, ha="left", va="center", zorder=8)
    ax.text(p[0], p[1] - 3.6, f"$(x_{i},y_{i})$", fontsize=13.5, ha="center", va="center", zorder=8)


for i, (x, y) in enumerate([(0.22, 0.17), (0.55, 0.31), (1.22, 0.14)], start=1):
    resonator(i, x, y)

for ext, kw in (("png", dict(dpi=220)), ("pdf", {}), ("svg", {})):
    fig.savefig(ROOT / "presentation_figures" / f"plate_three_resonators.{ext}", facecolor="white", **kw)
print("saved")
