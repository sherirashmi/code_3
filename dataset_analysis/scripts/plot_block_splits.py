"""Training, validation and test data of every model-bank block, one figure per f_t.

Each block (``10k_2res_fixed_m0.2_ft<f_t>_18modes``: 2 identical resonators,
m = 0.2 kg, positions by Latin hypercube) is split 80 / 10 / 10 exactly as for
training its model. Per block the figure shows

* top: the ERP of each split in the 40-120 Hz band the models see (median and
  25-75 % range), with the block's tuning frequency;
* middle: 25 random configurations of each split on the plate (the two
  resonators of one configuration joined by a line);
* bottom: per-split distributions of the resonator spacing, the distance from
  the excitation force to the nearest resonator, and the distance (4-D, cm) to
  the nearest TRAINING configuration (for training samples: to another one).

Saves ``dataset_analysis/<block tag>/plots/split_overview.png``.

Run from the repository root:  python dataset_analysis/scripts/plot_block_splits.py
"""
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle

import utils.plot_style  # noqa: F401
from erp_forward_operators.neural_operator_utils import _configuration_features, _split_ids
from utils.erp_dataset import FIXED_BLOCK_FREQUENCIES, fixed_block_tag
from utils.physics import Lx, Ly, edge_margin, xf, yf
from utils.plot_style import save_figure

common = importlib.import_module("2_res_erp_inverse_models.common")
sort_by_x = importlib.import_module("2_res_erp_inverse_models.design_space").sort_by_x

SPLITS = (("train", "Training", "#2a78d6"), ("val", "Validation", "#1f9e74"), ("test", "Test", "#eb6834"))
NUM_SHOWN = 25


def nearest_train_cm(query: np.ndarray, train: np.ndarray, exclude_self: bool = False) -> np.ndarray:
    q, t = torch.from_numpy(query).float(), torch.from_numpy(train).float()
    out = []
    for start in range(0, len(q), 2000):
        d = torch.cdist(q[start:start + 2000], t)
        if exclude_self:
            d[torch.arange(d.shape[0]), torch.arange(start, start + d.shape[0])] = float("inf")
        out.append(d.min(dim=1).values.numpy())
    return 100 * np.concatenate(out)


def draw_plate(ax):
    ax.add_patch(Rectangle((0, 0), Lx, Ly, facecolor="#f4f4f1", ec="black", lw=1.0, zorder=0))
    ax.add_patch(Rectangle((edge_margin, edge_margin), Lx - 2 * edge_margin, Ly - 2 * edge_margin, fill=False,
                           ec="#b5b5ae", ls="--", lw=0.6, zorder=1))
    ax.plot([xf], [yf], marker="*", color="#35d0ff", ms=12, mec="black", mew=0.7, ls="", zorder=6)
    ax.set_xlim(-0.02, Lx + 0.02)
    ax.set_ylim(-0.02, Ly + 0.02)
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(0, Lx + 1e-9, 0.2))
    ax.set_yticks([0, 0.25, 0.5])
    ax.tick_params(labelsize=8)
    ax.set_xlabel("$x$ (m)", fontsize=9)


def block_figure(f_t: float, rng) -> Path:
    tag = fixed_block_tag(f_t)
    dataset, _ = common.prepare_data(tag)
    splits = _split_ids(dataset)
    configs = sort_by_x(_configuration_features(dataset).astype(np.float64))
    erp = np.asarray(dataset.responses, dtype=np.float64)[:, :, 0]
    freq = np.asarray(dataset.frequency_values, dtype=np.float64)
    band = dataset.band_mask
    pos = configs[..., 3:5].reshape(len(configs), -1)  # [x1, y1, x2, y2]
    k = float(configs[0, 0, 1])

    fig = plt.figure(figsize=(16, 11.5))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 0.62, 0.9], hspace=0.3, wspace=0.22,
                          top=0.92, bottom=0.06, left=0.06, right=0.98)

    # ---- ERP per split ------------------------------------------------------------------------
    ax = fig.add_subplot(gs[0, :])
    for key, label, color in SPLITS:
        e = erp[splits[key]][:, band]
        q1, med, q3 = np.percentile(e, [25, 50, 75], axis=0)
        if key == "train":
            ax.fill_between(freq[band], q1, q3, color=color, alpha=0.18, lw=0)
        else:
            ax.plot(freq[band], q1, color=color, lw=0.9, ls="--")
            ax.plot(freq[band], q3, color=color, lw=0.9, ls="--")
        rng_label = "shaded" if key == "train" else "dashed"
        ax.plot(freq[band], med, color=color, lw=1.6,
                label=f"{label} ({len(splits[key]):,} configurations): median, 25--75 % range ({rng_label})")
    ax.axvline(f_t, color="#c2412c", ls="--", lw=1.0, label=f"Tuning frequency $f_t = {f_t:g}$ Hz")
    ax.set_xlim(freq[band][0], freq[band][-1])
    ax.set_xlabel("Frequency $f$ (Hz)")
    ax.set_ylabel("ERP (dB re 1 pW)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.62, 1.0))
    ax.set_title("ERP of each split in the band the model sees (40--120 Hz)")

    # ---- sample configurations per split -----------------------------------------------------------
    for col, (key, label, color) in enumerate(SPLITS):
        ax = fig.add_subplot(gs[1, col])
        draw_plate(ax)
        ids = rng.choice(splits[key], NUM_SHOWN, replace=False)
        for cid in ids:
            c = configs[cid]
            ax.plot(c[:, 3], c[:, 4], "-", color=color, lw=0.8, alpha=0.6, zorder=2)
            ax.plot(c[:, 3], c[:, 4], "o", color=color, ms=4.5, mec="black", mew=0.4, zorder=3)
        ax.set_title(f"{label}: {NUM_SHOWN} random configurations", fontsize=10)
        if col == 0:
            ax.set_ylabel("$y$ (m)", fontsize=9)

    # ---- configuration-level statistics -----------------------------------------------------------
    spacing = 100 * np.hypot(configs[:, 0, 3] - configs[:, 1, 3], configs[:, 0, 4] - configs[:, 1, 4])
    force = 100 * np.hypot(configs[..., 3] - xf, configs[..., 4] - yf).min(axis=1)
    train_pos = pos[splits["train"]]
    nearest = {"train": nearest_train_cm(train_pos, train_pos, exclude_self=True),
               "val": nearest_train_cm(pos[splits["val"]], train_pos),
               "test": nearest_train_cm(pos[splits["test"]], train_pos)}
    panels = (
        ("Resonator spacing (cm)", {k_: spacing[splits[k_]] for k_, _, _ in SPLITS}),
        ("Excitation force to the nearest resonator (cm)", {k_: force[splits[k_]] for k_, _, _ in SPLITS}),
        ("Distance to the nearest training configuration (cm)", nearest),
    )
    for col, (name, data) in enumerate(panels):
        ax = fig.add_subplot(gs[2, col])
        hi = np.percentile(np.concatenate(list(data.values())), 99.5)
        bins = np.linspace(0, hi, 40)
        for key, label, color in SPLITS:
            extra = " (to another)" if col == 2 and key == "train" else ""
            ax.hist(data[key], bins=bins, density=True, histtype="step", lw=1.7, color=color,
                    label=f"{label}{extra}: median {np.median(data[key]):.1f}")
        ax.set_xlabel(name, fontsize=9.5)
        ax.set_ylabel("Probability density" if col == 0 else "", fontsize=9)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7.8, loc="best")

    fig.suptitle(f"Block $f_t = {f_t:g}$ Hz: training, validation and test data "
                 f"(10,000 configurations, 2 resonators, $m = 0.2$ kg, $k = {k / 1000:.1f}$ kN/m)", fontsize=13, y=0.975)
    out = ROOT / "dataset_analysis" / tag / "plots" / "split_overview.png"
    save_figure(fig, out)
    return out


def main():
    rng = np.random.default_rng(11)
    for f_t in FIXED_BLOCK_FREQUENCIES:
        block_figure(f_t, rng)


if __name__ == "__main__":
    main()
