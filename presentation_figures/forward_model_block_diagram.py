"""Simple block diagram of the forward model (supervisor's layout, our blocks, thesis fonts).

Resonators r_i = (m, k, f_t, x, y) -> resonator encoder; frequency f -> frequency encoder; both -> resonance query;
the three feature streams -> core architecture (the block that changes between DON, DNO, FNO, DCO, GNO, STO, SIREN,
WNO, LNO; NN is a plain MLP) -> decoder (local refinement + output head) -> ERP y(f).
Facts: erp_forward/scripts/*.py (forward() of each operator), checked against the 10 trained checkpoints.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

import utils.plot_style  # noqa: F401  (Computer Modern text and maths)

import numpy as np

from utils.support import load_dataset

_payload = load_dataset(str(ROOT / "datasets" / "erp" / "3res" / "10k" / "dataset_erp_ft.pth"))
SAMPLE = 21  # one real configuration and its ERP, shown as icons
ERP = np.asarray(_payload["responses"], dtype=np.float64)[SAMPLE, :, 0]
CFG = np.asarray(_payload["configuration_features"], dtype=np.float64)[SAMPLE]  # (3, 5): m, k, f_t, x, y
LX, LY = 1.4, 0.5  # plate size (m)

W, H = 133.0, 45.0
BLUE, ACCENT = "#3d6a94", "#e8632b"        # arrow colour, accent
TITLE, SUB = "#1c3550", "#4a5a6a"           # text on the light blocks
# light block colours: (fill, edge)
C_ENC, C_QRY, C_FRQ = ("#d9e8f7", "#8fb4d9"), ("#e1f1de", "#97c791"), ("#d6eef0", "#86c3c8")
C_CORE, C_DEC = ("#fde7d3", ACCENT), ("#ebe2f5", "#b39ad1")

fig = plt.figure(figsize=(16, 16 * H / W))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")


def block(x0, x1, y0, y1, title, sub, color, accent=False, tsize=17, ssize=9.6):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=1.6", fc=color[0],
                                ec=color[1], lw=3.0 if accent else 1.6, ls="--" if accent else "-", zorder=3))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    ax.text(cx, cy + 1.6, title, color=TITLE, fontsize=tsize, ha="center", va="center", zorder=4)
    ax.text(cx, cy - 2.2, sub, color=SUB, fontsize=ssize, ha="center", va="center", zorder=4, linespacing=1.35)


ICON_BG = "#dbe8f6"


def icon_erp(x0, y0, w, h):
    """Minimalistic ERP-vs-frequency icon: black curve on two plain axes, light-blue panel."""
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0,rounding_size=1.0", fc=ICON_BG, ec="#8fb4d9", lw=1.4, zorder=3))
    ax0, ay0, ax1, ay1 = x0 + 0.12 * w, y0 + 0.14 * h, x0 + 0.92 * w, y0 + 0.90 * h
    ax.plot([ax0, ax0, ax1], [ay1, ay0, ay0], color="black", lw=1.4, zorder=4, solid_capstyle="butt")
    t = np.linspace(0.0, 1.0, ERP.size)
    v = (ERP - ERP.min()) / (ERP.max() - ERP.min())
    ax.plot(ax0 + 0.02 * w + t * (ax1 - ax0 - 0.04 * w), ay0 + 0.04 * h + v * (ay1 - ay0 - 0.10 * h), color="black", lw=2.6, zorder=5,
            solid_joinstyle="round")


def icon_plate(x0, y0, w, h):
    """Minimalistic resonator configuration: a plate (outline) with one dot per resonator."""
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0,rounding_size=1.0", fc=ICON_BG, ec="#8fb4d9", lw=1.4, zorder=3))
    pw = 0.80 * w
    ph = pw * LY / LX
    px0, py0 = x0 + (w - pw) / 2, y0 + (h - ph) / 2
    ax.add_patch(plt.Rectangle((px0, py0), pw, ph, fc="white", ec="black", lw=2.0, zorder=4))
    for x, y in CFG[:, 3:5]:
        ax.plot([px0 + x / LX * pw], [py0 + y / LY * ph], "o", color="black", ms=9, zorder=5)


def arrow(pts, color=BLUE, lw=1.8, ls="-", head=True):
    xs, ys = zip(*pts)
    if len(pts) > 2:
        ax.plot(xs[:-1], ys[:-1], color=color, lw=lw, ls=ls, zorder=2, solid_capstyle="butt")
    ax.annotate("", xy=pts[-1], xytext=pts[-2], zorder=2,
                arrowprops=dict(arrowstyle="-|>" if head else "-", color=color, lw=lw, ls=ls, shrinkA=0, shrinkB=0,
                                mutation_scale=16))


# vertical layout (base 5 leaves room for the footer)
Y_ENC, Y_Q, Y_F = (33.0, 42.5), (19.0, 28.5), (5.0, 14.5)
ENC_X, CORE_X, DEC_X = (17.0, 46.0), (65.0, 90.0), (97.0, 118.0)
mid = lambda y: (y[0] + y[1]) / 2

# ---- blocks ------------------------------------------------------------------------------------------
block(*ENC_X, *Y_ENC, "Resonator encoder", "set + $f_t$-sorted encoders\n(GNO: graph nodes, STO: tokens)", C_ENC)
block(*ENC_X, *Y_Q, "Resonance query", "$\\delta=f-f_t$, $|\\delta|$, $\\delta^2$ per resonator,\npooled over the resonators", C_QRY)
block(*ENC_X, *Y_F, "Frequency encoder", "MLP embedding of $f$\n(or the frequency itself)", C_FRQ)
block(*CORE_X, 12.0, 36.0, "Core architecture", "architecture-specific:\nDON, DNO, FNO, DCO, GNO,\nSTO, SIREN, WNO, LNO\n(NN: plain MLP)", C_CORE, accent=True, tsize=18, ssize=10.2)
block(*DEC_X, 15.5, 32.5, "Decoder", "local refinement along $f$,\noutput head", C_DEC)
ax.text((CORE_X[0] + CORE_X[1]) / 2, 37.4, "the only block that changes", color=ACCENT, fontsize=10.5, ha="center", va="center")

# ---- inputs ------------------------------------------------------------------------------------------
icon_plate(0.8, mid(Y_ENC) - 4.0, 10.6, 8.0)
ax.text(6.1, mid(Y_ENC) - 8.2, "Resonators\n$(m,k,f_t,x,y)$\n$i=1,2,3$", fontsize=12.5, ha="center", va="center", linespacing=1.4)
ax.text(0.8, mid(Y_F), "Frequency\n$f$\n301 points", fontsize=13, ha="left", va="center", linespacing=1.4)
arrow([(12.6, mid(Y_ENC)), (ENC_X[0], mid(Y_ENC))])
arrow([(12.6, mid(Y_F)), (ENC_X[0], mid(Y_F))])
# both inputs also feed the resonance query
arrow([(14.6, mid(Y_ENC)), (14.6, mid(Y_Q) + 1.8), (ENC_X[0], mid(Y_Q) + 1.8)], lw=1.4)
arrow([(14.6, mid(Y_F)), (14.6, mid(Y_Q) - 1.8), (ENC_X[0], mid(Y_Q) - 1.8)], lw=1.4)
ax.plot([14.6], [mid(Y_ENC)], "o", color=BLUE, ms=5, zorder=5)
ax.plot([14.6], [mid(Y_F)], "o", color=BLUE, ms=5, zorder=5)

# ---- encoders -> core -> decoder -> output --------------------------------------------------------------
cy = 24.0
for y, ty in ((Y_ENC, cy + 4.5), (Y_Q, cy), (Y_F, cy - 4.5)):
    arrow([(ENC_X[1], mid(y)), (CORE_X[0], ty)])
arrow([(CORE_X[1], cy), (DEC_X[0], cy)])
arrow([(DEC_X[1], cy), (121.0, cy)])
icon_erp(121.6, cy - 4.5, 10.6, 9.0)
ax.text(126.9, cy + 7.2, r"ERP $\hat y(f)$", fontsize=15, ha="center", va="center")

# ---- footer ------------------------------------------------------------------------------------------
ax.text(66.0, 3.0, "Frequency encoder: MLP embedding in DON (trunk), DNO, DCO, GNO and STO; the raw frequency enters FNO, WNO, LNO, SIREN and NN.  "
                   "Resonance query: inside the core for GNO and STO, absent in DON and NN.", fontsize=9.0, ha="center", va="center", color="#444444")
ax.text(66.0, 1.2, "Decoder: local refinement is part of the core blocks in FNO and WNO and absent in NN.  Each resonator also carries 120 plate-mode features; "
                   "ERP values are normalised.", fontsize=9.0, ha="center", va="center", color="#444444")

for ext, kw in (("png", dict(dpi=220)), ("pdf", {}), ("svg", {})):
    fig.savefig(ROOT / "presentation_figures" / f"forward_model_block_diagram.{ext}", facecolor="white", **kw)
print("saved")
