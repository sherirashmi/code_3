"""Per-frequency ERP statistics (mean, std, min, max) across the full
100k-configuration dataset, over the complete frequency range.
"""
import sys

sys.path.insert(0, "/home/user/code_3")

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from utils.erp_dataset import ERPDataset

DATASET_100K = [
    "datasets/dataset_erp_ft_100k_part1.pth",
    "datasets/dataset_erp_ft_100k_part2.pth",
]
OUT_DIR = Path("dataset_analysis/100k/plots")
STATS_DIR = Path("dataset_analysis/100k/stats")

BLUE = "#2a78d6"


def main():
    dataset = ERPDataset().load_shards(DATASET_100K)
    freq_hz = dataset.frequency_values  # (n_freq,)
    responses = dataset.responses[:, :, 0]  # (N, n_freq), physical dB
    n_configs = responses.shape[0]
    print(f"Loaded {n_configs} configurations, {freq_hz.size} frequency points "
          f"({freq_hz.min():.1f}-{freq_hz.max():.1f} Hz)")

    mean = responses.mean(axis=0)
    std = responses.std(axis=0)
    q25 = np.percentile(responses, 25, axis=0)
    q75 = np.percentile(responses, 75, axis=0)
    vmin = responses.min(axis=0)
    vmax = responses.max(axis=0)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.fill_between(freq_hz, vmin, vmax, color=BLUE, alpha=0.15, label="Min-max range")
    ax.fill_between(freq_hz, q25, q75, color=BLUE, alpha=0.25, label="25th-75th percentile")
    ax.fill_between(freq_hz, mean - std, mean + std, color=BLUE, alpha=0.35, label="Mean +/- 1 std dev")
    ax.plot(freq_hz, mean, color="#0b0b0b", lw=1.8, label="Mean")

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("ERP (dB)")
    ax.set_xlim(freq_hz.min(), freq_hz.max())
    ax.set_title(f"ERP value distribution across full frequency range ({n_configs:,} configurations)")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "erp_frequency_statistics.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")

    # Companion CSV with the raw per-frequency numbers, for anyone who
    # wants exact values rather than reading them off the plot.
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = STATS_DIR / "erp_frequency_statistics.csv"
    header = "frequency_hz,mean_db,std_db,min_db,q25_db,q75_db,max_db"
    rows = np.column_stack([freq_hz, mean, std, vmin, q25, q75, vmax])
    np.savetxt(csv_path, rows, delimiter=",", header=header, comments="", fmt="%.4f")
    print(f"Saved {csv_path}")
    print("DONE")


if __name__ == "__main__":
    main()
