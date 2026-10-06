"""ERP distribution at each of the 301 frequency points of a dataset (panel (a) of dataset_analysis/<tag>/plots/erp_frequency_boxplot.png),
drawn alone from the stored statistics (dataset_analysis/<tag>/stats/erp_frequency_boxplot.csv): median, interquartile range,
Tukey whiskers (1.5 x IQR), min-max.   Usage: python presentation_figures/erp_frequency_band_plot.py [tag]   (default 100k)
Output: presentation_figures/erp_frequency_band_plot[_<tag>].{png,pdf,svg}
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import utils.physics as physics
from utils.erp_dataset import DATASETS
import utils.plot_style  # noqa: F401

BLUE, MODE_COLOR = "#2a78d6", "#9a9a94"
TAG = sys.argv[1] if len(sys.argv) > 1 else "100k"
spec = DATASETS[TAG]
physics.set_modal_resolution(*spec["modal_resolution"], verbose=False)
NUM = {1: "one", 2: "two", 3: "three"}[int(spec["num_res"])]
N_CONFIG = int(spec["num_configurations"])
d = np.genfromtxt(ROOT / "dataset_analysis" / TAG / "stats" / "erp_frequency_boxplot.csv", delimiter=",", names=True)
freq, vmin, lo, q1, med, q3, hi, vmax = (d[k] for k in ("frequency_hz", "min_db", "lower_whisker_db", "q1_db", "median_db",
                                                       "q3_db", "upper_whisker_db", "max_db"))
modes = [f for f in physics.omega_n / (2 * np.pi) if freq.min() <= f <= freq.max()]

fig, ax = plt.subplots(figsize=(13, 5.0))
for j, f in enumerate(modes):
    ax.axvline(f, color=MODE_COLOR, ls="--", lw=0.9, zorder=0, label="Bare-plate natural frequencies" if j == 0 else None)
ax.fill_between(freq, vmin, vmax, color=BLUE, alpha=0.10, lw=0, label="Minimum to maximum")
ax.fill_between(freq, lo, hi, color=BLUE, alpha=0.22, lw=0, label=r"Whiskers ($1.5\times$IQR)")
ax.fill_between(freq, q1, q3, color=BLUE, alpha=0.45, lw=0, label="Interquartile range (25th to 75th percentile)")
ax.plot(freq, med, color="black", lw=1.6, label="Median")
ax.set_xlim(freq.min(), freq.max())
pad = 0.04 * (vmax.max() - vmin.min())
ax.set_ylim(vmin.min() - 0.2 * (vmax.max() - vmin.min()), vmax.max() + pad)
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("ERP (dB)")
ax.grid(alpha=0.3)
ax.legend(loc="lower right", fontsize=9, ncol=2)
ax.set_title(f"ERP distribution at each of the {freq.size} frequency points")
fig.suptitle(f"ERP at each frequency over {N_CONFIG:,} configurations: {NUM} resonators with varying mass, tuning frequency and position", fontsize=13)
fig.tight_layout()
for ext, kw in (("png", dict(dpi=220)), ("pdf", {}), ("svg", {})):
    fig.savefig(ROOT / "presentation_figures" / f"erp_frequency_band_plot{'' if TAG == '100k' else '_' + TAG}.{ext}", facecolor="white", **kw)
print("saved")
