"""Gate network L of each invertible coupling-flow operator, in the style of forward_core_architectures.

Facts: erp_invertible/scripts/{ifno,idco,igno,idno,iwno,ilno,isiren,isto}.py (gate layer passed to InvertibleCouplingStack):
iFNO FourierLayer1d (48 modes, FFT padded by 8), iDCO DCOGateLayer1d, iGNO GNOGateLayer1d, iDNO DNOGateLayer1d,
iWNO MultiLevelHaarWaveletBlock1d (3 levels), iLNO LNOGateLayer1d (8 learned poles), iSIREN SineGateLayer1d, iSTO FrequencyMixer.
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

TRANS, MLPG, PER, CONV = ("#e1f1de", "#97c791"), ("#dbe8f6", "#8fb4d9"), ("#fdf3c9", "#e0c25a"), ("#d6eef0", "#86c3c8")
LEFT = ("#eef2f7", "#9aa9ba")
TITLE, SUB = "#1c3550", "#4a5a6a"

ROWS = [  # (model, gate network L, detail, colour group)
    ("iFNO (Fourier)", "Fourier layer", "spectral convolution (48 modes) + Conv1d", TRANS),
    ("iWNO (Wavelet)", "Wavelet transform", "3-level Haar wavelet block", TRANS),
    ("iLNO (Laplace)", "Poles and residues", "8 learned poles + pointwise transform", TRANS),
    ("iDCO (Deep Cat)", "Residual MLP", "pointwise residual block + local refinement", MLPG),
    ("iGNO (Graph)", "Pointwise MLP", "plain MLP + local refinement", MLPG),
    ("iDNO (Deep Neural)", "Modulation (FiLM)", "FiLM from a frequency embedding + refinement", MLPG),
    ("iSIREN (SIREN)", "Sine layer", "pointwise sine layer + local refinement", PER),
    ("iSTO (Set Transformer)", "Frequency mixer", "local + dilated Conv1d", CONV),
]
W, H = 66.0, 66.0
fig = plt.figure(figsize=(8.6, 8.6 * H / W))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")


def box(x0, x1, y0, y1, color):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=1.1", fc=color[0], ec=color[1], lw=1.6, zorder=3))


ax.text(W / 2, H - 2.6, r"Gate network $L$ of each invertible operator", fontsize=16, ha="center", va="center", color=TITLE)
RH, GAP, TOP = 5.6, 1.5, H - 6.0
for i, (name, gate, detail, color) in enumerate(ROWS):
    y1 = TOP - i * (RH + GAP)
    y0 = y1 - RH
    box(1.5, 28.0, y0, y1, LEFT)
    box(36.0, 64.5, y0, y1, color)
    ax.text(14.75, (y0 + y1) / 2, name, fontsize=12.6, ha="center", va="center", color=TITLE, zorder=4)
    ax.text(50.25, (y0 + y1) / 2 + 0.95, gate, fontsize=15.5, ha="center", va="center", color=TITLE, zorder=4)
    ax.text(50.25, (y0 + y1) / 2 - 1.55, detail, fontsize=9.0, ha="center", va="center", color=SUB, zorder=4)
    ax.annotate("", xy=(36.0, (y0 + y1) / 2), xytext=(28.0, (y0 + y1) / 2), zorder=2,
                arrowprops=dict(arrowstyle="-|>", color="#3d6a94", lw=1.6, shrinkA=0, shrinkB=0, mutation_scale=15))
x = 3.0
for label, c in (("transform", TRANS), ("pointwise MLP", MLPG), ("periodic", PER), ("convolution", CONV)):
    ax.add_patch(FancyBboxPatch((x, 2.3), 2.4, 1.5, boxstyle="round,pad=0,rounding_size=0.4", fc=c[0], ec=c[1], lw=1.2))
    ax.text(x + 3.2, 3.05, label, fontsize=9.5, ha="left", va="center", color=SUB)
    x += 16.0
ax.text(W / 2, 0.9, "Every gate acts on the latent along $f$ and does not see the design;  coupling: $v_1'=v_1\\odot S(L(v_2))$,  $S(x)=e^{2\\tanh(ax)}$.",
        fontsize=8.4, ha="center", va="center", color="#444444")
for ext, kw in (("png", dict(dpi=220)), ("pdf", {}), ("svg", {})):
    fig.savefig(ROOT / "presentation_figures" / f"invertible_gate_networks.{ext}", facecolor="white", **kw)
print("saved")
