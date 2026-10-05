"""Table of the forward operators: core operation and test errors of the latest 200-epoch checkpoints in erp_forward/models/100k
(*_sorted_phys, GNO_phys, STO_phys, NN_perm), read from erp_forward/plots/models/100k/ALL_MODELS/forward_models_metrics.csv
(100k dataset, 10,000 test spectra). Columns: RMSE, R^2, correlation (Pearson r over all points), RMSE at the true resonance peaks.
"""
import csv
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
LEFT, CELL = ("#eef2f7", "#9aa9ba"), ("#ffffff", "#c3ccd6")
TITLE, SUB = "#1c3550", "#4a5a6a"
CSV = ROOT / "erp_forward" / "plots" / "models" / "100k" / "ALL_MODELS" / "forward_models_metrics.csv"

ROWS = [  # (name in the csv, label, core operation, colour group)
    ("DeepONet", "DeepONet (DON)", "Inner product", COMB),
    ("Deep Neural Operator", "Deep Neural Operator (DNO)", "Modulation (FiLM)", COMB),
    ("Deep Cat Operator", "Deep Cat Operator (DCO)", "Concatenation", COMB),
    ("Fourier Neural Operator", "Fourier Neural Operator (FNO)", "Fourier transform", TRANS),
    ("Wavelet Neural Operator", "Wavelet Neural Operator (WNO)", "Wavelet transform", TRANS),
    ("Laplace Neural Operator", "Laplace Neural Operator (LNO)", "Poles and residues", TRANS),
    ("Graph Neural Operator", "Graph Neural Operator (GNO)", "Graph network", REL),
    ("Set Transformer Operator", "Set Transformer Operator (STO)", "Attention", REL),
    ("SIREN Neural Operator", "SIREN Neural Operator (SIREN)", "Sine layers", PER),
    ("Plain Neural Network", "Plain Neural Network (NN)", "Dense layers", COMB),
]
COLUMNS = [  # (csv column, header, format, better = lower?)
    ("RMSE (dB)", "RMSE (dB)", "{:.2f}", True),
    ("R2", "$R^2$", "{:.3f}", False),
    ("Pearson r (all points)", "Correlation", "{:.3f}", False),
    ("RMSE at all peaks (dB)", "Peak RMSE (dB)", "{:.2f}", True),
]
data = {r["Model"]: r for r in csv.DictReader(open(CSV))}
best = {c: (min if low else max)(float(data[n][c]) for n, *_ in ROWS) for c, _, _, low in COLUMNS}

W, RH, GAP = 134.0, 5.6, 1.4
H = 8.5 + 1.2 + len(ROWS) * (RH + GAP) + 4.5
fig = plt.figure(figsize=(12.6, 12.6 * H / W))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")


def box(x0, x1, y0, y1, color, lw=1.4):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=1.0", fc=color[0], ec=color[1], lw=lw, zorder=3))


ax.text(W / 2, H - 2.4, "Forward operators: core operation and test error", fontsize=16, ha="center", va="center", color=TITLE)
X_ARCH, X_CORE, X_MET, MW, MGAP = (1.5, 36.0), (38.0, 64.0), 66.0, 15.0, 1.6
y_head = H - 6.2
ax.text((X_ARCH[0] + X_ARCH[1]) / 2, y_head, "Architecture", fontsize=12, ha="center", va="center", color=SUB)
ax.text((X_CORE[0] + X_CORE[1]) / 2, y_head, "Core operation", fontsize=12, ha="center", va="center", color=SUB)
for j, (_, header, _, _) in enumerate(COLUMNS):
    ax.text(X_MET + j * (MW + MGAP) + MW / 2, y_head, header, fontsize=12, ha="center", va="center", color=SUB)
top = y_head - 1.8
for i, (key, label, core, color) in enumerate(ROWS):
    y1 = top - i * (RH + GAP)
    y0 = y1 - RH
    ym = (y0 + y1) / 2
    box(X_ARCH[0], X_ARCH[1], y0, y1, LEFT)
    box(X_CORE[0], X_CORE[1], y0, y1, color)
    ax.text((X_ARCH[0] + X_ARCH[1]) / 2, ym, label, fontsize=11.6, ha="center", va="center", color=TITLE, zorder=4)
    ax.text((X_CORE[0] + X_CORE[1]) / 2, ym, core, fontsize=13, ha="center", va="center", color=TITLE, zorder=4)
    for j, (col, _, fmt, _) in enumerate(COLUMNS):
        x0 = X_MET + j * (MW + MGAP)
        value = float(data[key][col])
        box(x0, x0 + MW, y0, y1, TRANS if abs(value - best[col]) < 1e-9 else CELL)
        ax.text(x0 + MW / 2, ym, fmt.format(value), fontsize=13.5, ha="center", va="center", color=TITLE, zorder=4)
ax.text(W / 2, 2.0, "Latest 200-epoch models, 100k dataset (10,000 test spectra). Correlation: Pearson $r$ over all points. "
        "Peak RMSE: error at the resonance peaks of the true ERP. Best value of each column in green.",
        fontsize=9.6, ha="center", va="center", color=SUB)
for ext, kw in (("png", dict(dpi=220)), ("pdf", {}), ("svg", {})):
    fig.savefig(ROOT / "presentation_figures" / f"forward_metrics_table.{ext}", facecolor="white", **kw)
print("saved")
