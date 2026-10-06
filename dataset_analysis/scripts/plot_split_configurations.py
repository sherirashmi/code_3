"""How the three resonators are combined in each sample, per data split.

Figure 1 (``split_sample_configurations.png``): four random configurations from
each of the training, validation and test data. Each panel is the plate with
the configuration's resonators joined into a triangle; colour = tuning
frequency, marker size = mass. Every validation / test panel also shows its
NEAREST TRAINING CONFIGURATION (hollow markers, dashed triangle) -- how far a
"new" sample really is from what the models were trained on.

Figure 2 (``split_configuration_statistics.png``): distributions of properties
of whole configurations, overlaid for the three splits: closest and widest
resonator spacing, closest pair of tuning frequencies, lowest tuning frequency,
distance from the excitation force to the nearest resonator, total mass, and
the distance to the nearest training configuration.

Uses the split stored in the forward-operator checkpoints.

Run from the repository root:
    python dataset_analysis/scripts/plot_split_configurations.py [dataset tag]
"""
import sys
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import Normalize
from matplotlib.patches import Polygon, Rectangle

import utils.plot_style  # noqa: F401
from utils.erp_dataset import DATASETS
from utils.physics import Lx, Ly, edge_margin, fmax, fmin, m_max, m_min, xf, yf
from utils.plot_style import save_figure
from utils.support import load_dataset

TAG = sys.argv[1] if len(sys.argv) > 1 else "100k"
CHECKPOINT = ROOT / "erp_forward" / "models" / TAG / "dno.pth"
OUT_DIR = ROOT / "dataset_analysis" / TAG / "plots"
SPLITS = (("train", "Training", "#2a78d6"), ("val", "Validation", "#1f9e74"), ("test", "Test", "#eb6834"))
CMAP = plt.get_cmap("viridis")
FT_NORM = Normalize(fmin, fmax)
NUM_SHOWN = 4


def load_split():
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)["preprocessing_state"]
    files = state.get("dataset_file") or DATASETS[TAG]["files"]
    configs = np.concatenate([np.asarray(load_dataset(f)["configuration_features"], dtype=np.float64) for f in files])
    configs = configs[np.asarray(state["selected_source_ids"], dtype=np.int64)]
    splits = {k: np.asarray(v, dtype=np.int64) for k, v in state["split_configuration_ids"].items()}
    return configs, splits


def design_features(configs: np.ndarray) -> np.ndarray:
    """Order-free comparison vector: resonators sorted by f_t, [m, f_t, x, y] scaled to [0, 1]."""
    order = np.argsort(configs[..., 2], axis=-1)
    c = np.take_along_axis(configs, order[..., None], axis=-2)
    scaled = np.stack([(c[..., 0] - m_min) / (m_max - m_min), (c[..., 2] - fmin) / (fmax - fmin),
                       c[..., 3] / Lx, c[..., 4] / Ly], axis=-1)
    return scaled.reshape(len(configs), -1)


def nearest_training(query: np.ndarray, train: np.ndarray, chunk: int = 2000):
    """Index into ``train`` and distance of the nearest training configuration (design_features space)."""
    t = torch.from_numpy(train).float()
    idx, dist = [], []
    for start in range(0, len(query), chunk):
        d = torch.cdist(torch.from_numpy(query[start:start + chunk]).float(), t)
        v, i = d.min(dim=1)
        idx.append(i.numpy())
        dist.append(v.numpy())
    return np.concatenate(idx), np.concatenate(dist)


def draw_plate(ax):
    ax.add_patch(Rectangle((0, 0), Lx, Ly, facecolor="#f4f4f1", ec="black", lw=1.0, zorder=0))
    ax.add_patch(Rectangle((edge_margin, edge_margin), Lx - 2 * edge_margin, Ly - 2 * edge_margin, fill=False,
                           ec="#b5b5ae", ls="--", lw=0.6, zorder=1))
    ax.plot([xf], [yf], marker="*", color="#35d0ff", ms=11, mec="black", mew=0.7, ls="", zorder=6)
    ax.set_xlim(-0.02, Lx + 0.02)
    ax.set_ylim(-0.02, Ly + 0.02)
    ax.set_aspect("equal")
    ax.set_xticks([0, 0.7, 1.4])
    ax.set_yticks([0, 0.25, 0.5])
    ax.tick_params(labelsize=7)


def marker_size(m):
    return 40 + 260 * (np.asarray(m) - m_min) / (m_max - m_min)


def draw_configuration(ax, config, hollow=False):
    xy = config[:, 3:5]
    if len(config) >= 3:
        ax.add_patch(Polygon(xy, closed=True, fill=not hollow, fc="#9a9a94" if not hollow else "none", alpha=0.18 if not hollow else 1.0,
                             ec="#555555", ls="--" if hollow else "-", lw=0.8, zorder=2))
    colors = CMAP(FT_NORM(config[:, 2]))
    if hollow:
        ax.scatter(xy[:, 0], xy[:, 1], s=marker_size(config[:, 0]), facecolors="none", edgecolors=colors, linewidths=1.6, zorder=4)
    else:
        ax.scatter(xy[:, 0], xy[:, 1], s=marker_size(config[:, 0]), c=colors, edgecolors="black", linewidths=0.7, zorder=5)


def pair_stats(configs):
    pairs = list(combinations(range(configs.shape[1]), 2))
    dist = np.stack([np.hypot(*(configs[:, i, 3:5] - configs[:, j, 3:5]).T) for i, j in pairs], axis=1)
    dft = np.stack([np.abs(configs[:, i, 2] - configs[:, j, 2]) for i, j in pairs], axis=1)
    force = np.hypot(configs[..., 3] - xf, configs[..., 4] - yf).min(axis=1)
    return {
        "Closest resonator spacing (cm)": 100 * dist.min(axis=1),
        "Widest resonator spacing (cm)": 100 * dist.max(axis=1),
        r"Closest pair of tuning frequencies, $\min|\Delta f_t|$ (Hz)": dft.min(axis=1),
        r"Lowest tuning frequency $f_t$ (Hz)": configs[..., 2].min(axis=1),
        "Excitation force to nearest resonator (cm)": 100 * force,
        "Total resonator mass (kg)": configs[..., 0].sum(axis=1),
    }


def main():
    configs, splits = load_split()
    feats = design_features(configs)
    train_feats = feats[splits["train"]]
    rng = np.random.default_rng(7)

    # ---- figure 1: sample configurations ----------------------------------------------
    fig, axes = plt.subplots(3, NUM_SHOWN, figsize=(4.3 * NUM_SHOWN, 7.6))
    for row, (key, label, _) in enumerate(SPLITS):
        ids = rng.choice(splits[key], NUM_SHOWN, replace=False)
        if key != "train":
            near_idx, near_dist = nearest_training(feats[ids], train_feats)
        for col, cid in enumerate(ids):
            ax = axes[row, col]
            draw_plate(ax)
            config = configs[cid]
            if key != "train":
                draw_configuration(ax, configs[splits["train"][near_idx[col]]], hollow=True)
            draw_configuration(ax, config)
            text = ", ".join(f"{f:.0f}" for f in np.sort(config[:, 2]))
            title = f"{label} sample {cid + 1:,}: $f_t$ = {text} Hz"
            if key != "train":
                title += f"\nnearest training configuration at distance {near_dist[col]:.2f}"
            ax.set_title(title, fontsize=8.5)
            if col == 0:
                ax.set_ylabel(f"{label}\n$y$ (m)", fontsize=10)
            if row == 2:
                ax.set_xlabel("$x$ (m)", fontsize=9)
    sm = plt.cm.ScalarMappable(norm=FT_NORM, cmap=CMAP)
    cbar = fig.colorbar(sm, cax=fig.add_axes([0.925, 0.16, 0.011, 0.68]), orientation="vertical")
    cbar.set_label("Tuning frequency $f_t$ (Hz)")
    handles = [
        plt.scatter([], [], s=marker_size(m_min), c="0.5", edgecolors="black", label=f"Resonator, $m = {m_min:g}$ kg"),
        plt.scatter([], [], s=marker_size(m_max), c="0.5", edgecolors="black", label=f"Resonator, $m = {m_max:g}$ kg"),
        plt.scatter([], [], s=marker_size(0.5), facecolors="none", edgecolors="0.3", linewidths=1.6,
                    label="Nearest training configuration (hollow, dashed)"),
        plt.Line2D([], [], marker="*", color="#35d0ff", mec="black", ls="", ms=11, label="Excitation force"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(0.47, -0.01))
    fig.suptitle(f"Resonator combinations in random samples of the training, validation and test data "
                 f"(dataset {TAG}; marker size = mass)", fontsize=12)
    fig.subplots_adjust(left=0.05, right=0.91, top=0.9, bottom=0.1, hspace=0.45, wspace=0.08)
    save_figure(fig, OUT_DIR / "split_sample_configurations.png")

    # ---- figure 2: configuration-level statistics -----------------------------------------
    stats = {key: pair_stats(configs[ids]) for key, ids in splits.items()}
    probe = rng.choice(splits["train"], 10000, replace=False)
    near = {}
    for key in ("val", "test"):
        near[key] = nearest_training(feats[splits[key]], train_feats)[1]
    # training -> other training configurations (leave-one-out: drop the self-match)
    t = torch.from_numpy(train_feats).float()
    pos = {cid: i for i, cid in enumerate(splits["train"])}
    loo = []
    for start in range(0, len(probe), 2000):
        chunk = probe[start:start + 2000]
        d = torch.cdist(torch.from_numpy(feats[chunk]).float(), t)
        d[torch.arange(len(chunk)), torch.tensor([pos[c] for c in chunk])] = float("inf")
        loo.append(d.min(dim=1).values.numpy())
    near["train"] = np.concatenate(loo)

    names = list(stats["train"]) + ["Distance to the nearest training configuration"]
    fig, axes = plt.subplots(2, 4, figsize=(18, 7.8))
    for ax, name in zip(axes.ravel(), names):
        data = {k: (near[k] if name.startswith("Distance to the nearest") else stats[k][name]) for k, _, _ in SPLITS}
        lo, hi = np.percentile(np.concatenate(list(data.values())), [0, 99.5])
        bins = np.linspace(lo, hi, 50)
        for key, label, color in SPLITS:
            extra = " (other training samples)" if name.startswith("Distance to the nearest") and key == "train" else ""
            ax.hist(data[key], bins=bins, density=True, histtype="step", lw=1.7, color=color,
                    label=f"{label}{extra}: median {np.median(data[key]):.2f}")
        ax.set_xlabel(name, fontsize=9.5)
        ax.set_ylabel("Probability density", fontsize=9)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7.5)
    axes.ravel()[-1].set_visible(False)
    fig.suptitle(f"Configuration-level properties of the training, validation and test data (dataset {TAG}); "
                 "distance in scaled $[m, f_t, x, y]$ space, resonators ordered by $f_t$", fontsize=12)
    fig.tight_layout()
    save_figure(fig, OUT_DIR / "split_configuration_statistics.png")


if __name__ == "__main__":
    main()
