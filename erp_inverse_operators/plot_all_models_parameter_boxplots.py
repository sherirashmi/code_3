"""Combined predicted-vs-true parameter box plot for ALL 7 inverse models
(same idea as surrogate_inverse_parameter_recovery.png, extended across
every model): for each of m/k/f_t/x/y, bins the x-axis across that field's
COMPLETE physical generation range and box-plots the PREDICTED values
falling in each true-value bin.

One sample per target (no "best of N" cherry-picking, no solver needed --
picking a best sample would bias the box toward looking better than the
model's raw output distribution actually is, and this needs many test
targets for meaningful bins, which the solver-based selection used
elsewhere in this package would make expensive).
"""
from __future__ import annotations

import sys
sys.path.insert(0, "/home/user/code_3")

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from erp_inverse_operators.common import denormalize_design, prepare_inverse_data
from erp_inverse_operators.evaluate import MODEL_BUILDERS, load_inverse_model
from erp_inverse_operators.registry import NUM_RES
from erp_inverse_operators.surrogate_inverse import _CONFIG_FIELDS, _DESIGN_PHYSICAL_BOUNDS
from utils.support import device

OUT_DIR = Path("erp_inverse_operators/plots")
NUM_TEST_EXAMPLES = 3000
NUM_BINS = 8

dataset, loaders = prepare_inverse_data(
    num_configurations=100000, batch_size=64,
    dataset_file=[
        "datasets/dataset_erp_ft_100k_part1.pth",
        "datasets/dataset_erp_ft_100k_part2.pth",
    ],
    seed=727,
)
norm = dataset.norm_params

test_spectrum, test_design = [], []
for spectrum, design in loaders["test"]:
    test_spectrum.append(spectrum)
    test_design.append(design)
    if sum(s.shape[0] for s in test_spectrum) >= NUM_TEST_EXAMPLES:
        break
test_spectrum = torch.cat(test_spectrum, dim=0)[:NUM_TEST_EXAMPLES]
test_design = torch.cat(test_design, dim=0)[:NUM_TEST_EXAMPLES]
n = test_spectrum.shape[0]

true_physical = denormalize_design(test_design.numpy().reshape(n, -1), NUM_RES, norm)  # (n, num_res, 5)

names = list(MODEL_BUILDERS)
predictions = {}
for name in names:
    print(f"Predicting with {name} ...")
    model, _ = load_inverse_model(name)
    model.eval()
    with torch.no_grad():
        result = model.sample(test_spectrum.to(device), num_samples=1)
    flat = result[0] if isinstance(result, tuple) else result
    flat = flat[:, 0, :].cpu()  # (n, D) -- single sample per target
    predictions[name] = denormalize_design(flat.numpy(), NUM_RES, norm)  # (n, num_res, 5)

fig, axes = plt.subplots(
    len(_CONFIG_FIELDS), len(names), figsize=(3.6 * len(names), 3.6 * len(_CONFIG_FIELDS)), squeeze=False
)
for row, field in enumerate(_CONFIG_FIELDS):
    lo, hi = _DESIGN_PHYSICAL_BOUNDS[field]
    edges = np.linspace(lo, hi, NUM_BINS + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    true_vals = true_physical[..., row].ravel()

    for col, name in enumerate(names):
        pred_vals = predictions[name][..., row].ravel()
        ax = axes[row, col]

        box_data, positions = [], []
        for b in range(NUM_BINS):
            in_bin = (true_vals >= edges[b]) & (
                true_vals < edges[b + 1] if b < NUM_BINS - 1 else true_vals <= edges[b + 1]
            )
            values = pred_vals[in_bin]
            if values.size == 0:
                continue
            box_data.append(values)
            positions.append(centers[b])

        width = 0.7 * (edges[1] - edges[0])
        if box_data:
            ax.boxplot(
                box_data, positions=positions, widths=width, showfliers=False,
                patch_artist=True, boxprops=dict(facecolor=f"C{col}", alpha=0.6),
                medianprops=dict(color="black"), whiskerprops=dict(lw=0.8), capprops=dict(lw=0.8),
            )
        ax.plot([lo, hi], [lo, hi], "k--", lw=1.0)
        ax.set_xlim(lo, hi)
        ax.set_xticks([lo, (lo + hi) / 2, hi])
        ax.set_xticklabels([f"{lo:.2g}", f"{(lo + hi) / 2:.2g}", f"{hi:.2g}"], fontsize=7)
        if row == 0:
            ax.set_title(name, fontsize=11)
        if col == 0:
            ax.set_ylabel(f"{field}\npredicted", fontsize=9)
        if row == len(_CONFIG_FIELDS) - 1:
            ax.set_xlabel("true (complete range)", fontsize=8)
        ax.grid(alpha=0.3)

fig.suptitle(
    f"Predicted-value spread per true-value bin, all 7 models ({n} test targets x {NUM_RES} "
    f"resonators, {NUM_BINS} bins across each field's complete physical range, 1 sample/target, "
    "dashed = predicted=true)",
    fontsize=13,
)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(OUT_DIR / "all_models_parameter_boxplots.png", dpi=150)
plt.close(fig)
print(f"Saved {OUT_DIR / 'all_models_parameter_boxplots.png'}")
