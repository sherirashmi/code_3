"""Training and validation loss of the forward operators (100k dataset, 200 epochs): the histories stored in the checkpoints of
erp_forward/models/100k (*_sorted_phys, GNO_phys, STO_phys, NN_perm). Two panels, y axes "Training loss" / "Validation loss", no titles,
one shared legend and the loss equation underneath. Colours and dashed lines as in erp_forward/plots/models/100k/ALL_MODELS.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import utils.plot_style  # noqa: F401

MODELS = [  # label, checkpoint, colour, dashed
    ("Deep Cat Operator", "dco_sorted_phys", "#2a78d6", False),
    ("Deep Neural Operator", "dno_sorted_phys", "#eb6834", False),
    ("DeepONet", "don_sorted_phys", "#1baf7a", False),
    ("Fourier Neural Operator", "fno_sorted_phys", "#eda100", False),
    ("Graph Neural Operator", "gno_phys", "#e87ba4", False),
    ("Laplace Neural Operator", "lno_sorted_phys", "#008300", False),
    ("Plain Neural Network", "nn_perm", "#4a3aa7", False),
    ("SIREN Neural Operator", "siren_sorted_phys", "#e34948", False),
    ("Wavelet Neural Operator", "wno_sorted_phys", "#1baf7a", True),
    ("Set Transformer Operator", "sto_phys", "#4a3aa7", True),
]
hist = {}
for label, name, _, _ in MODELS:
    ck = torch.load(ROOT / "erp_forward" / "models" / "100k" / f"{name}.pth", map_location="cpu", weights_only=False)
    hist[label] = {k: np.asarray(ck["history"][k], dtype=float) for k in ("train", "val")}

fig, axes = plt.subplots(1, 2, figsize=(16, 7.0))
for ax, key, ylabel in zip(axes, ("train", "val"), ("Training loss", "Validation loss")):
    for label, _, colour, dashed in MODELS:
        y = hist[label][key]
        ax.plot(np.arange(1, y.size + 1), y, color=colour, lw=2.0, ls="--" if dashed else "-", label=label)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 200)
    ax.grid(True, which="both", alpha=0.3)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, fontsize=11, bbox_to_anchor=(0.5, 0.105))
fig.text(0.5, 0.055, r"$\mathcal{L}=\mathrm{MSE}+0.5\,\mathrm{MSE}_{\Delta}+0.05\sum_{\mathrm{peaks}}\mathrm{SE}$", ha="center", va="center", fontsize=15)
fig.text(0.5, 0.008, r"MSE: mean squared error of the normalized spectrum; $\mathrm{MSE}_{\Delta}$: the same on first differences (slope); "
         "SE: squared error at the true peak frequencies, summed.", ha="center", va="center", fontsize=9.5, color="#555555")
fig.subplots_adjust(left=0.065, right=0.99, top=0.97, bottom=0.30, wspace=0.18)
for ext, kw in (("png", dict(dpi=220)), ("pdf", {}), ("svg", {})):
    fig.savefig(ROOT / "presentation_figures" / f"forward_loss_curves.{ext}", facecolor="white", **kw)
print("saved")
