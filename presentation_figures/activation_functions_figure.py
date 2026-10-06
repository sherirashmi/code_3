"""Activation functions used in the forward operators (erp_forward/scripts, DEFAULT_MODEL_CONFIG) as curves: ReLU (NN), GELU (FNO, WNO, STO), SiLU (DNO, DCO, LNO, GNO and the
shared refinement conv), Tanh (DON, shared encoders, FiLM gates), sine (SIREN, omega_0 = 20), softplus (LNO poles) and softmax (attention weights, DON term weights).
Output: activation_functions.png/pdf/svg
"""
import sys
from math import erf, sqrt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import utils.plot_style  # noqa: F401

OUT = Path(__file__).resolve().parent
INK, MUTED, GRID = "#1f2933", "#6b7785", "#e3e7ec"
BLUE, ORANGE, GREEN, RED, VIOLET, TEAL = "#2a78d6", "#eb6834", "#1baf7a", "#e34948", "#4a3aa7", "#008300"

x = np.linspace(-4, 4, 801)
phi = 0.5 * (1.0 + np.vectorize(erf)(x / sqrt(2.0)))
sig = 1.0 / (1.0 + np.exp(-x))
z = np.linspace(-0.5, 0.5, 1001)

PANELS = [  # name, formula, models, colour, x, y, xlim, ylim
    ("ReLU", r"$\max(0,\,x)$", "NN", VIOLET, x, np.maximum(0, x), (-4, 4), (-1.2, 4.2)),
    ("GELU", r"$x\,\Phi(x)$", "FNO, WNO, STO", ORANGE, x, x * phi, (-4, 4), (-1.2, 4.2)),
    ("SiLU", r"$x\,/\,(1+e^{-x})$", "DNO, DCO, LNO, GNO, refinement conv", BLUE, x, x * sig, (-4, 4), (-1.2, 4.2)),
    ("Tanh", r"$(e^{x}-e^{-x})\,/\,(e^{x}+e^{-x})$", "DON, shared encoders, FiLM gates", GREEN, x, np.tanh(x), (-4, 4), (-1.5, 1.5)),
    ("Sine (SIREN)", r"$\sin(\omega_0 z),\ \omega_0=20$", "SIREN", RED, z, np.sin(20 * z), (-0.5, 0.5), (-1.5, 1.5)),
    ("Softplus", r"$\ln(1+e^{x})$", "LNO (damping, frequency $>0$)", TEAL, x, np.log1p(np.exp(x)), (-4, 4), (-1.2, 4.2)),
]

fig, axes = plt.subplots(2, 4, figsize=(15.5, 9.2))
fig.subplots_adjust(left=0.04, right=0.99, top=0.92, bottom=0.06, wspace=0.28, hspace=0.85)
for ax, (name, formula, models, colour, xs, ys, xl, yl) in zip(axes.flat, PANELS):
    ax.axhline(0, color=MUTED, lw=0.9)
    ax.axvline(0, color=MUTED, lw=0.9)
    ax.plot(xs, ys, color=colour, lw=3.0, solid_capstyle="round")
    ax.set_xlim(xl)
    ax.set_ylim(yl)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=11)
    ax.set_xlabel("z" if name.startswith("Sine") else "x", color=MUTED, fontsize=13)
    ax.set_title(name, fontsize=20, fontweight="bold", color=colour, pad=40)
    ax.text(0.5, 1.045, formula, transform=ax.transAxes, ha="center", va="bottom", fontsize=15, color=INK)
    ax.text(0.5, -0.24, models, transform=ax.transAxes, ha="center", va="top", fontsize=12.5, color=MUTED)

# softmax: scores -> weights (attention, DON term weights)
ax = axes[1, 2]
scores = np.array([2.0, 0.5, -1.0])
w = np.exp(scores) / np.exp(scores).sum()
pos = np.arange(3)
ax.bar(pos - 0.2, scores, width=0.38, color="#c9ced6", label="scores $s_i$")
ax.bar(pos + 0.2, w, width=0.38, color=VIOLET, label="weights")
for p, v in zip(pos + 0.2, w):
    ax.text(p, v + 0.08, f"{v:.2f}", ha="center", fontsize=11, color=INK)
ax.axhline(0, color=MUTED, lw=0.9)
ax.set_xticks(pos, ["1", "2", "3"])
ax.set_ylim(-1.5, 2.9)
ax.grid(True, axis="y", color=GRID, lw=0.8)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(MUTED)
ax.tick_params(colors=MUTED, labelsize=11)
ax.set_xlabel("item", color=MUTED, fontsize=13)
ax.legend(frameon=False, fontsize=11, loc="upper right")
ax.set_title("Softmax", fontsize=20, fontweight="bold", color=VIOLET, pad=40)
ax.text(0.5, 1.045, r"$e^{s_i}\,/\,\Sigma_j\,e^{s_j}$", transform=ax.transAxes, ha="center", va="bottom", fontsize=15, color=INK)
ax.text(0.5, -0.24, "attention in GNO, STO; DON term weights", transform=ax.transAxes, ha="center", va="top", fontsize=12.5, color=MUTED)

# last panel: the smooth-ReLU family together
ax = axes[1, 3]
ax.axhline(0, color=MUTED, lw=0.9)
ax.axvline(0, color=MUTED, lw=0.9)
ax.plot(x, np.maximum(0, x), color=VIOLET, lw=2.4, label="ReLU")
ax.plot(x, x * phi, color=ORANGE, lw=2.4, label="GELU")
ax.plot(x, x * sig, color=BLUE, lw=2.4, ls="--", label="SiLU")
ax.plot(x, np.log1p(np.exp(x)), color=TEAL, lw=2.0, label="Softplus")
ax.set_xlim(-4, 4)
ax.set_ylim(-1.2, 4.2)
ax.grid(True, color=GRID, lw=0.8)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(MUTED)
ax.tick_params(colors=MUTED, labelsize=11)
ax.set_xlabel("x", color=MUTED, fontsize=13)
ax.legend(frameon=False, fontsize=11, loc="upper left")
ax.set_title("Compared", fontsize=20, fontweight="bold", color=INK, pad=40)
ax.text(0.5, 1.045, "the ReLU family", transform=ax.transAxes, ha="center", va="bottom", fontsize=15, color=INK)
ax.text(0.5, -0.24, "GELU and SiLU are smooth ReLUs", transform=ax.transAxes, ha="center", va="top", fontsize=12.5, color=MUTED)

for ext in ("png", "pdf", "svg"):
    fig.savefig(OUT / f"activation_functions.{ext}", dpi=200 if ext == "png" else None, facecolor="white")
print("saved", OUT / "activation_functions.png")
