"""Minimal block diagrams of (1) the Invertible DeepONet (Q8 / Q64) and (2) the invertible coupling-flow family
(iFNO, iDCO, iGNO, iDNO, iWNO, iLNO, iSIREN, iSTO).  Light colours, thesis fonts, plate and ERP icons.

Icons: the real 2-resonator example in inverse_example.npz (a held-out target spectrum and its true design).
Facts: erp_invertible_deeponet/scripts/model.py, erp_invertible/scripts/{common,coupling,ifno}.py.
iDON parameters: Q8 938,665, Q64 1,384,257 (checkpoints).
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

EX = np.load(ROOT / "presentation_figures" / "inverse_example.npz")
ERP, CFG = EX["target"], EX["true_cfg"]  # (301,), (2, 5): m, k, f_t, x, y
LX, LY = 1.4, 0.5

FWD, INV = "#3d6a94", "#e8632b"
TITLE, SUB = "#1c3550", "#4a5a6a"
C_BLUE, C_GREEN, C_PEACH, C_PURPLE, C_TEAL = ("#d9e8f7", "#8fb4d9"), ("#e1f1de", "#97c791"), ("#fde7d3", "#eba46f"), ("#ebe2f5", "#b39ad1"), ("#d6eef0", "#86c3c8")
ICON_BG = "#dbe8f6"


class Canvas:
    def __init__(self, W, H, name):
        self.W, self.H, self.name = W, H, name
        self.fig = plt.figure(figsize=(16, 16 * H / W))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, W)
        self.ax.set_ylim(0, H)
        self.ax.axis("off")

    def block(self, x0, x1, y0, y1, title, sub, color, tsize=17, ssize=10.5, dashed=False, lw=1.6, sub_dy=-2.4):
        self.ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=1.6", fc=color[0], ec=color[1],
                                         lw=3.0 if dashed else lw, ls="--" if dashed else "-", zorder=3))
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        self.ax.text(cx, cy + (1.5 if sub else 0), title, color=TITLE, fontsize=tsize, ha="center", va="center", zorder=4)
        if sub:
            self.ax.text(cx, cy + sub_dy, sub, color=SUB, fontsize=ssize, ha="center", va="center", zorder=4, linespacing=1.35)

    def arrow(self, pts, color=FWD, lw=1.9, ls="-"):
        xs, ys = zip(*pts)
        if len(pts) > 2:
            self.ax.plot(xs[:-1], ys[:-1], color=color, lw=lw, ls=ls, zorder=2, solid_capstyle="butt")
        self.ax.annotate("", xy=pts[-1], xytext=pts[-2], zorder=2,
                         arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, ls=ls, shrinkA=0, shrinkB=0, mutation_scale=16))

    def text(self, x, y, s, size=12, color="black", ha="center", va="center", **kw):
        return self.ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va, zorder=6, **kw)

    def icon_erp(self, x0, y0, w, h):
        ax = self.ax
        ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0,rounding_size=1.0", fc=ICON_BG, ec="#8fb4d9", lw=1.4, zorder=3))
        a0, b0, a1, b1 = x0 + 0.12 * w, y0 + 0.14 * h, x0 + 0.92 * w, y0 + 0.90 * h
        ax.plot([a0, a0, a1], [b1, b0, b0], color="black", lw=1.4, zorder=4, solid_capstyle="butt")
        t = np.linspace(0, 1, ERP.size)
        v = (ERP - ERP.min()) / (ERP.max() - ERP.min())
        ax.plot(a0 + 0.02 * w + t * (a1 - a0 - 0.04 * w), b0 + 0.04 * h + v * (b1 - b0 - 0.10 * h), color="black", lw=2.6, zorder=5,
                solid_joinstyle="round")

    def icon_plate(self, x0, y0, w, h):
        ax = self.ax
        ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0,rounding_size=1.0", fc=ICON_BG, ec="#8fb4d9", lw=1.4, zorder=3))
        pw = 0.80 * w
        ph = pw * LY / LX
        px0, py0 = x0 + (w - pw) / 2, y0 + (h - ph) / 2
        ax.add_patch(Rectangle((px0, py0), pw, ph, fc="white", ec="black", lw=2.0, zorder=4))
        for x, y in CFG[:, 3:5]:
            ax.plot([px0 + x / LX * pw], [py0 + y / LY * ph], "o", color="black", ms=9, zorder=5)

    def save(self):
        for ext, kw in (("png", dict(dpi=220)), ("pdf", {}), ("svg", {})):
            self.fig.savefig(ROOT / "presentation_figures" / f"{self.name}.{ext}", facecolor="white", **kw)
        plt.close(self.fig)
        print("saved", self.name)


# ============================================================================================================
# 1. Invertible DeepONet (Q8, Q64): DeepONet layout, a branch (invertible) and a trunk, combined by a dot product
# ============================================================================================================
c = Canvas(124.0, 52.0, "idon_block_diagram")
by, ty = 34.0, 11.0  # branch row, trunk row
c.text(62.0, 49.0, "Invertible DeepONet (Q8, Q64)", size=19, color=TITLE)
# group frames
c.ax.add_patch(FancyBboxPatch((20.0, 24.0), 56.0, 21.0, boxstyle="round,pad=0,rounding_size=1.6", fc="none", ec=C_BLUE[1], lw=2.2, ls="--", zorder=1))
c.text(48.0, 42.5, "Branch network (invertible)", size=14, color=TITLE)
c.ax.add_patch(FancyBboxPatch((20.0, 2.5), 56.0, 17.5, boxstyle="round,pad=0,rounding_size=1.6", fc="none", ec=C_GREEN[1], lw=2.2, ls="--", zorder=1))
c.text(48.0, 17.5, "Trunk network", size=14, color=TITLE)
# branch row: design -> T -> b
c.icon_plate(2.0, by - 5.0, 13.0, 10.0)
c.text(8.5, by - 8.6, "Design $\\mathbf{a}$\n$(m,f_t,x,y)\\times2$", size=11.5, linespacing=1.3)
c.block(25.0, 52.0, by - 6.0, by + 5.0, "RealNVP $T$", "bijection on $\\mathbb{R}^Q$, pad with zeros", C_BLUE, tsize=16, ssize=9.5)
c.block(60.0, 71.0, by - 6.0, by + 5.0, "$\\mathbf{b}$", "", C_BLUE, tsize=17)
c.text(65.5, by - 8.2, "$Q$ coefficients", size=10.5, color=SUB)
c.text(38.5, by - 8.2, "Q8: $Q=D=8$   $|$   Q64: $Q=64$", size=10, color=SUB)
# trunk row: f -> trunk -> Psi
c.text(8.5, ty, "Frequency\n$f$", size=13, linespacing=1.3)
c.block(25.0, 52.0, ty - 5.0, ty + 4.0, "Trunk net", "dense layers", C_GREEN, tsize=16, ssize=9.5)
c.block(60.0, 71.0, ty - 5.0, ty + 4.0, "$\\Psi(f)$", "", C_GREEN, tsize=17)
c.text(65.5, ty - 7.3, "basis, QR-orthonormal", size=10, color=SUB)
# combine and output
cx, cr = 88.0, 3.3
c.ax.add_patch(plt.Circle((cx, by - 0.5), cr, fc="white", ec=TITLE, lw=2.0, zorder=4))
c.ax.add_patch(plt.Circle((cx, by - 0.5), 0.5, fc=TITLE, ec=TITLE, zorder=5))
c.text(cx, by + 5.2, "$\\Psi\\mathbf{b}+\\psi_0$", size=13, color=TITLE)
c.icon_erp(108.0, by - 5.5, 13.0, 10.0)
c.text(114.5, by + 6.4, "ERP $\\hat y(f)$", size=13)
yf, yi = by + 0.8, by - 3.0
c.arrow([(15.0, yf), (25.0, yf)])
c.arrow([(25.0, yi), (15.0, yi)], color=INV, ls="--")
c.arrow([(52.0, yf), (60.0, yf)])
c.arrow([(60.0, yi), (52.0, yi)], color=INV, ls="--")
c.arrow([(71.0, by - 0.5), (cx - cr, by - 0.5)])
c.arrow([(cx - cr, by - 3.4), (71.0, by - 3.4)], color=INV, ls="--")
c.arrow([(cx + cr, by - 0.5), (108.0, by - 0.5)])
c.arrow([(108.0, by - 3.4), (cx + cr, by - 3.4)], color=INV, ls="--")
c.arrow([(15.0, ty), (25.0, ty)])
c.arrow([(52.0, ty), (60.0, ty)])
c.arrow([(71.0, ty), (cx, ty), (cx, by - 0.5 - cr)])
c.text(cx + 1.4, 22.0, "$\\Psi$ is shared by both directions", size=10.5, color=SUB, ha="left")
c.text(98.0, by - 9.2, "inverse: $\\mathbf{b}^*=\\Psi^\\top(y-\\psi_0)/F$", size=11.5, color=INV)
# key
c.arrow([(88.5, 9.0), (94.5, 9.0)])
c.text(95.5, 9.0, "forward: design $\\rightarrow$ ERP", size=11, color=FWD, ha="left")
c.arrow([(94.5, 5.8), (88.5, 5.8)], color=INV, ls="--")
c.text(95.5, 5.8, "inverse: ERP $\\rightarrow$ designs", size=11, color=INV, ha="left")
c.save()

# ============================================================================================================
# 2. Invertible coupling-flow family (iFNO and its variants)
# ============================================================================================================
d = Canvas(124.0, 48.0, "ifno_family_block_diagram")
d.text(62.0, 45.4, "Invertible operators: iFNO and its variants", size=19, color=TITLE)
FY, IY = 34.0, 11.5
d.icon_plate(2.0, FY - 5.0, 13.0, 10.0)
d.text(8.5, FY + 7.4, "Design $\\mathbf{a}$", size=12.5)
d.icon_erp(109.0, FY - 5.0, 13.0, 10.0)
d.text(115.5, FY + 7.4, "ERP $\\hat y(f)$", size=12.5)
d.icon_erp(109.0, IY - 5.0, 13.0, 10.0)
d.text(115.5, IY - 8.0, "target ERP $y(f)$", size=12)
d.icon_plate(2.0, IY - 5.0, 13.0, 10.0)
d.text(8.5, IY - 8.0, "designs $\\hat{\\mathbf{a}}$", size=12)
d.block(24.0, 42.0, FY - 5.5, FY + 5.5, "Lift $P$", "design, $f$ $\\rightarrow$ latent", C_BLUE, tsize=16, ssize=10)
d.block(24.0, 42.0, IY - 5.5, IY + 5.5, "Readout $Q'$", "latent $\\rightarrow$ design", C_PEACH, tsize=16, ssize=10)
d.block(82.0, 100.0, FY - 5.5, FY + 5.5, "Readout $Q$", "latent $\\rightarrow$ ERP", C_BLUE, tsize=16, ssize=10)
d.block(82.0, 100.0, IY - 5.5, IY + 5.5, "Lift $P'$", "ERP $\\rightarrow$ latent", C_PEACH, tsize=16, ssize=10)
# coupling stack (one block shared by both rows) with the gate slot
d.block(50.0, 74.0, 4.0, 42.5, "", "", C_PURPLE)
d.text(62.0, 40.0, "Coupling stack", size=16, color=TITLE)
d.text(62.0, 37.2, "same weights both ways", size=10, color=SUB)
d.block(52.0, 72.0, 15.5, 33.0, "Gate $L$", "changes with the model:\niFNO, iDCO, iGNO, iDNO,\niWNO, iLNO, iSIREN, iSTO", C_PEACH, tsize=15, ssize=10, dashed=True, sub_dy=-3.2)
d.arrow([(15.0, FY), (24.0, FY)])
d.arrow([(42.0, FY), (50.0, FY)])
d.arrow([(74.0, FY), (82.0, FY)])
d.arrow([(100.0, FY), (109.0, FY)])
d.arrow([(109.0, IY), (100.0, IY)], color=INV, ls="--")
d.arrow([(82.0, IY), (74.0, IY)], color=INV, ls="--")
d.arrow([(50.0, IY), (42.0, IY)], color=INV, ls="--")
d.arrow([(24.0, IY), (15.0, IY)], color=INV, ls="--")
d.text(62.0, 1.7, "Only the coupling stack is an exact bijection; a $\\beta$-VAE on the design estimate gives several candidate designs.", size=10.5, color="#444444")
d.save()
