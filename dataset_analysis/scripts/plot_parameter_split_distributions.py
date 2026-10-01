"""Per-resonator-parameter distribution across the train/val/test split.

For each of the 5 physical resonator parameters (m, k, f_t, x, y), overlays
the train/validation/test subsets (one color each) on the same histogram --
a sanity check that the random configuration-level split used everywhere in
this project (see erp_forward_operators.neural_operator_utils._split_ids)
doesn't accidentally skew any parameter's distribution toward one split.
Values are pooled across all num_res resonators per configuration (the
"parameter" is m, k, f_t, x, or y in general, not "resonator #2's mass"
specifically).
"""
import sys

sys.path.insert(0, "/home/user/code_3")

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from erp_forward_operators.neural_operator_utils import (
    _configuration_features,
    _split_ids,
    prepare_operator_data,
)

DATASET_100K = [
    "datasets/dataset_erp_ft_100k_part1.pth",
    "datasets/dataset_erp_ft_100k_part2.pth",
]
OUT_DIR = Path("dataset_analysis/plots")

FIELDS = ["m", "k", "f_t", "x", "y"]
UNITS = {"m": "kg", "k": "N/m", "f_t": "Hz", "x": "m", "y": "m"}
SPLIT_COLORS = {"train": "#2a78d6", "val": "#e08214", "test": "#2ca02c"}
NUM_BINS = 60


def main():
    dataset, _ = prepare_operator_data(
        num_configurations=100000, batch_size=64,
        dataset_file=DATASET_100K, regenerate_dataset=False, seed=727,
    )
    splits = _split_ids(dataset)
    features = _configuration_features(dataset)  # (N, num_res, 5) in [m,k,f_t,x,y] order
    print(
        f"Loaded {features.shape[0]} configurations x {features.shape[1]} resonators. "
        f"Split sizes: " + ", ".join(f"{name}={ids.size}" for name, ids in splits.items())
    )

    fig, axes = plt.subplots(1, len(FIELDS), figsize=(4.2 * len(FIELDS), 4.5))
    for col, field in enumerate(FIELDS):
        ax = axes[col]
        values = {
            name: features[ids, :, col].ravel()  # pool across all resonators
            for name, ids in splits.items()
        }
        all_vals = np.concatenate(list(values.values()))
        bins = np.linspace(all_vals.min(), all_vals.max(), NUM_BINS + 1)

        for name in ("train", "val", "test"):
            ax.hist(
                values[name], bins=bins, density=True, histtype="step", lw=2,
                color=SPLIT_COLORS[name], label=f"{name} (n={values[name].size:,})",
            )
        ax.set_xlabel(f"{field} ({UNITS[field]})")
        ax.set_ylabel("Density")
        ax.set_title(field)
        ax.grid(alpha=0.3)
        if col == 0:
            ax.legend(fontsize=8)

    fig.suptitle(
        "Resonator-parameter distribution by split (pooled across all resonators) -- "
        "overlapping curves confirm the split doesn't skew any parameter's coverage",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "parameter_split_distributions.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
