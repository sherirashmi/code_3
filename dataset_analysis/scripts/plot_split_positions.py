"""Resonator positions of the training, validation and test data on the plate.

Uses the exact split the forward operators were trained with (stored in their
checkpoints), so the three figures show what each model saw during training,
what chose its best epoch, and what it was scored on. Every resonator of every
configuration is one point.

Run from the repository root (default: the 100k forward-model dataset):
    python dataset_analysis/scripts/plot_split_positions.py [dataset tag]
"""
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

import utils.plot_style  # noqa: F401  (thesis LaTeX / Computer Modern look)
from utils.erp_dataset import DATASETS
from utils.physics import Lx, Ly, edge_margin, xf, yf
from utils.plot_style import save_figure
from utils.support import load_dataset

TAG = sys.argv[1] if len(sys.argv) > 1 else "100k"
CHECKPOINT = ROOT / "erp_forward" / "models" / TAG / "dno.pth"
SPLITS = (("train", "Training data", "#2a78d6"), ("val", "Validation data", "#1f9e74"), ("test", "Test data", "#eb6834"))
OUT_DIR = ROOT / "dataset_analysis" / TAG / "plots"


def main():
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)["preprocessing_state"]
    selected = np.asarray(state["selected_source_ids"], dtype=np.int64)
    files = state.get("dataset_file") or DATASETS[TAG]["files"]
    configs = np.concatenate([np.asarray(load_dataset(f)["configuration_features"], dtype=np.float64) for f in files])
    configs = configs[selected]  # split ids are local to the selected configurations
    num_res = configs.shape[1]

    for key, label, color in SPLITS:
        ids = np.asarray(state["split_configuration_ids"][key], dtype=np.int64)
        xy = configs[ids][..., 3:5].reshape(-1, 2)
        fig, ax = plt.subplots(figsize=(12, 5.0))
        ax.add_patch(Rectangle((0, 0), Lx, Ly, facecolor="#f4f4f1", ec="black", lw=1.2, zorder=0))
        ax.add_patch(Rectangle((edge_margin, edge_margin), Lx - 2 * edge_margin, Ly - 2 * edge_margin, fill=False,
                               ec="#9a9a94", ls="--", lw=0.8, zorder=1))
        size = 0.25 if len(xy) > 100_000 else 0.8
        ax.scatter(xy[:, 0], xy[:, 1], s=size, color=color, alpha=0.35, edgecolors="none", rasterized=True, zorder=2)
        ax.scatter([], [], s=25, color=color, label=f"Resonator positions ({len(xy):,})")
        ax.plot([xf], [yf], marker="*", color="#35d0ff", ms=17, mec="black", mew=0.9, ls="", zorder=5, label="Excitation force")
        ax.set_xlim(-0.02, Lx + 0.02)
        ax.set_ylim(-0.02, Ly + 0.02)
        ax.set_aspect("equal")
        ax.set_xticks(np.arange(0, Lx + 1e-9, 0.2))
        ax.set_yticks(np.arange(0, Ly + 1e-9, 0.1))
        ax.set_xlabel("Position $x$ (m)")
        ax.set_ylabel("Position $y$ (m)")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)
        ax.set_title(f"{label}: {len(ids):,} configurations $\\times$ {num_res} resonators "
                     f"(dataset {TAG}, plate {Lx:g} m $\\times$ {Ly:g} m)")
        fig.tight_layout()
        save_figure(fig, OUT_DIR / f"resonator_positions_{key}.png")


if __name__ == "__main__":
    main()
