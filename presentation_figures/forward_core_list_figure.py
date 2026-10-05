"""Core architecture of each forward operator, in the style of the supervisor's list: architecture -> core operation.

Core operations read from the forward() of each operator (erp_forward/scripts/*.py); the colours group the operations
by type (combination, transform, relational, periodic).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

import utils.plot_style  # noqa: F401

COMB, TRANS, REL, PER = ("#dbe8f6", "#8fb4d9"), ("#e1f1de", "#97c791"), ("#fde7d3", "#eba46f"), ("#fdf3c9", "#e0c25a")
LEFT = ("#eef2f7", "#9aa9ba")
TITLE, SUB = "#1c3550", "#4a5a6a"

ROWS = [  # (architecture, core operation, detail, colour group)
    ("DeepONet (DON)", "Inner product", "branch coefficients $\\cdot$ trunk basis", COMB),
    ("Deep Neural Operator (DNO)", "Modulation (FiLM)", "scale and shift of residual blocks", COMB),
    ("Deep Cat Operator (DCO)", "Concatenation", "branch, trunk, query $\\rightarrow$ residual MLP", COMB),
    ("Fourier Neural Operator (FNO)", "Fourier transform", "spectral convolution along $f$", TRANS),
    ("Wavelet Neural Operator (WNO)", "Wavelet transform", "3-level Haar wavelet blocks", TRANS),
    ("Laplace Neural Operator (LNO)", "Poles and residues", "Laplace-domain rational basis", TRANS),
    ("Graph Neural Operator (GNO)", "Graph network", "message passing, attention over resonators", REL),
    ("Set Transformer Operator (STO)", "Attention", "frequencies attend to resonator tokens", REL),
    ("SIREN Neural Operator (SIREN)", "Sine layers", "modulated by the design context", PER),
    ("Plain Neural Network (NN)", "Dense layers", "one MLP on the flattened design and $f$", COMB),
]
W, H = 66.0, 80.0
fig = plt.figure(figsize=(8.6, 8.6 * H / W))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")


def box(x0, x1, y0, y1, color):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=1.1", fc=color[0],
                                ec=color[1], lw=1.6, zorder=3))


ax.text(W / 2, H - 2.6, "Core architecture of each forward operator", fontsize=16, ha="center", va="center", color=TITLE)
RH, GAP, TOP = 5.6, 1.5, H - 6.0
for i, (name, core, detail, color) in enumerate(ROWS):
    y1 = TOP - i * (RH + GAP)
    y0 = y1 - RH
    box(1.5, 28.0, y0, y1, LEFT)
    box(36.0, 64.5, y0, y1, color)
    ax.text(14.75, (y0 + y1) / 2, name, fontsize=12.2, ha="center", va="center", color=TITLE, zorder=4)
    ax.text(50.25, (y0 + y1) / 2 + 0.95, core, fontsize=15.5, ha="center", va="center", color=TITLE, zorder=4)
    ax.text(50.25, (y0 + y1) / 2 - 1.55, detail, fontsize=9.2, ha="center", va="center", color=SUB, zorder=4)
    ax.annotate("", xy=(36.0, (y0 + y1) / 2), xytext=(28.0, (y0 + y1) / 2), zorder=2,
                arrowprops=dict(arrowstyle="-|>", color="#3d6a94", lw=1.6, shrinkA=0, shrinkB=0, mutation_scale=15))
# colour key
x = 3.0
for label, c in (("combination", COMB), ("transform", TRANS), ("relational", REL), ("periodic", PER)):
    ax.add_patch(FancyBboxPatch((x, 1.0), 2.4, 1.5, boxstyle="round,pad=0,rounding_size=0.4", fc=c[0], ec=c[1], lw=1.2))
    ax.text(x + 3.2, 1.75, label, fontsize=9.5, ha="left", va="center", color=SUB)
    x += 16.0
for ext, kw in (("png", dict(dpi=220)), ("pdf", {}), ("svg", {})):
    fig.savefig(ROOT / "presentation_figures" / f"forward_core_architectures.{ext}", facecolor="white", **kw)
print("saved")
