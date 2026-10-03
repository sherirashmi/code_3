"""Box-plot statistics of the ERP at every frequency point of a dataset.

(a) All frequency points: median, interquartile range (box), Tukey whiskers
    (1.5 x IQR) and the min-max range, drawn as bands.
(b) Classic box plots at every 5 Hz (outliers beyond the whiskers summarised by
    the min / max markers instead of being drawn individually).

Also writes the per-frequency numbers as CSV.

Run from the repository root (default: the fixed-resonator dataset):
    python dataset_analysis/scripts/plot_erp_frequency_boxplot.py [dataset tag]
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import utils.physics as physics
from utils.erp_dataset import DATASETS, ERPDataset
from utils.plot_style import save_figure

TAG = sys.argv[1] if len(sys.argv) > 1 else "100k_2res_fixed_m0.2_ft72_18modes"
OUT_DIR = ROOT / "dataset_analysis" / "plots"
BOX_STEP_HZ = 5.0

BLUE = "#2a78d6"
MODE_COLOR = "#9a9a94"
TUNING_COLOR = "#c2412c"
FREQ_LABEL = "Frequency (Hz)"
ERP_LABEL = "ERP (dB)"


def main():
    spec = DATASETS[TAG]
    physics.set_modal_resolution(*spec["modal_resolution"], verbose=False)
    dataset = ERPDataset().load_shards(spec["files"])
    freq = np.asarray(dataset.frequency_values, dtype=np.float64)
    erp = np.asarray(dataset.responses[:, :, 0], dtype=np.float64)  # (N, n_freq) in dB
    n = erp.shape[0]
    print(f"Loaded {n:,} configurations, {freq.size} frequency points")

    q1, med, q3 = np.percentile(erp, [25, 50, 75], axis=0)
    iqr = q3 - q1
    vmin, vmax = erp.min(axis=0), erp.max(axis=0)
    # Tukey whiskers: furthest data point within 1.5 IQR of the box.
    lo_whisk = np.where(erp >= q1 - 1.5 * iqr, erp, np.inf).min(axis=0)
    hi_whisk = np.where(erp <= q3 + 1.5 * iqr, erp, -np.inf).max(axis=0)

    modes = [f for f in physics.omega_n / (2 * np.pi) if freq.min() <= f <= freq.max()]
    fixed = spec.get("fixed_resonator")

    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(13, 9.5), sharex=True)
    for ax in (ax_a, ax_b):
        for j, f in enumerate(modes):
            ax.axvline(f, color=MODE_COLOR, ls="--", lw=0.9, zorder=0,
                       label="Bare-plate natural frequencies" if j == 0 else None)
        if fixed is not None:
            ax.axvline(fixed[1], color=TUNING_COLOR, ls="-.", lw=1.3, zorder=0,
                       label=f"Resonator tuning frequency $f_t = {fixed[1]:.0f}$ Hz")
        ax.grid(alpha=0.3)
        ax.set_ylabel(ERP_LABEL)

    # (a) every frequency point as bands
    ax_a.fill_between(freq, vmin, vmax, color=BLUE, alpha=0.10, lw=0, label="Minimum to maximum")
    ax_a.fill_between(freq, lo_whisk, hi_whisk, color=BLUE, alpha=0.22, lw=0, label=r"Whiskers ($1.5\times$IQR)")
    ax_a.fill_between(freq, q1, q3, color=BLUE, alpha=0.45, lw=0, label="Interquartile range (25th--75th percentile)")
    ax_a.plot(freq, med, color="black", lw=1.6, label="Median")
    ax_a.set_title(f"(a) ERP distribution at each of the {freq.size} frequency points")
    ax_a.legend(loc="lower right", fontsize=9, ncol=2)

    # (b) classic box plots every BOX_STEP_HZ
    picks = np.unique([int(np.argmin(np.abs(freq - f))) for f in np.arange(freq.min(), freq.max() + 1e-9, BOX_STEP_HZ)])
    ax_b.boxplot(
        [erp[:, i] for i in picks], positions=freq[picks], widths=BOX_STEP_HZ * 0.6, whis=1.5,
        showfliers=False, patch_artist=True, manage_ticks=False,
        boxprops=dict(facecolor=BLUE, alpha=0.55, edgecolor="black", lw=0.7),
        medianprops=dict(color="black", lw=1.4), whiskerprops=dict(lw=0.8), capprops=dict(lw=0.8),
    )
    ax_b.plot(freq[picks], vmin[picks], "v", color="#555555", ms=3.5, ls="", label="Minimum")
    ax_b.plot(freq[picks], vmax[picks], "^", color="#555555", ms=3.5, ls="", label="Maximum")
    ax_b.set_title(f"(b) Box plots every {BOX_STEP_HZ:.0f} Hz (box: interquartile range, line: median, "
                   r"whiskers: $1.5\times$IQR)")
    ax_b.set_xlabel(FREQ_LABEL)
    ax_b.set_xlim(freq.min() - BOX_STEP_HZ, freq.max() + BOX_STEP_HZ)
    ax_b.legend(loc="lower right", fontsize=9, ncol=2)

    ymin, ymax = vmin.min(), vmax.max()
    pad = 0.04 * (ymax - ymin)
    ax_a.set_ylim(ymin - pad, ymax + pad)
    ax_b.set_ylim(ymin - pad, ymax + pad)
    what = (f"two identical resonators ($m = {fixed[0]:g}$ kg, $f_t = {fixed[1]:.0f}$ Hz), only the positions varied"
            if fixed is not None else spec.get("label", TAG))
    fig.suptitle(f"ERP at each frequency over {n:,} configurations: {what}", fontsize=13)
    fig.tight_layout()
    save_figure(fig, OUT_DIR / f"erp_frequency_boxplot_{TAG}.png")

    csv_path = OUT_DIR / f"erp_frequency_boxplot_{TAG}.csv"
    rows = np.column_stack([freq, vmin, lo_whisk, q1, med, q3, hi_whisk, vmax])
    np.savetxt(csv_path, rows, delimiter=",", fmt="%.4f", comments="",
               header="frequency_hz,min_db,lower_whisker_db,q1_db,median_db,q3_db,upper_whisker_db,max_db")
    print(f"Saved {csv_path}")


if __name__ == "__main__":
    main()
