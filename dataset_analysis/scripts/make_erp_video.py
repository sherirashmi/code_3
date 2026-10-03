import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from gif_common import (ERP_COLOR, ERP_LABEL, FREQ_LABEL, GIF_DPI, M_RES, OUT_DIR, TUNING_COLOR, configuration_row,
                        style_axes)
from utils.erp_dataset import configuration_to_resonators
from utils.solver import compute_erp_spectrum

# Three resonators with fixed positions and tuning frequencies (within the dataset's sampling bounds).
RESONATORS = [(45.0, 0.30, 0.15), (90.0, 0.70, 0.30), (130.0, 1.10, 0.40)]  # (tuning frequency, x, y)
configuration = np.array([configuration_row(ft, x, y) for ft, x, y in RESONATORS])
resonators = configuration_to_resonators(configuration)

freqs_fine = np.arange(10.0, 160.0 + 1e-9, 0.25)
erp = compute_erp_spectrum(resonators, frequencies=freqs_fine)
print(f"Computed {freqs_fine.size} points, ERP range [{erp.min():.2f}, {erp.max():.2f}] dB")

fig, ax = plt.subplots(figsize=(9.5, 5.4))
style_axes(ax)
ax.set_xlim(freqs_fine.min(), freqs_fine.max())
pad = 0.08 * (erp.max() - erp.min())
ax.set_ylim(erp.min() - pad, erp.max() + pad)
ax.set_xlabel(FREQ_LABEL)
ax.set_ylabel(ERP_LABEL)
for k, (ft, _, _) in enumerate(RESONATORS):
    ax.axvline(ft, color=TUNING_COLOR, ls="--", lw=1.1, alpha=0.6,
               label="Resonator tuning frequencies" if k == 0 else None)

title = ax.set_title("")
(line,) = ax.plot([], [], color=ERP_COLOR, lw=1.9)
(marker,) = ax.plot([], [], "o", color="#e08a1e", ms=7, mec="black", mew=0.6)
sweep_line = ax.axvline(freqs_fine[0], color="#e08a1e", lw=1, alpha=0.6)
ax.legend(loc="upper left", fontsize=9)

fig.suptitle(f"ERP of a plate with three resonators (mass {M_RES:g} kg each)", y=0.985, fontsize=13)
details = "   ".join(f"Resonator {i + 1}: tuning frequency {ft:.0f} Hz, position ({x:.2f} m, {y:.2f} m)"
                     for i, (ft, x, y) in enumerate(RESONATORS))
fig.text(0.5, 0.915, details, ha="center", fontsize=8.5, color="#3d3c39")
fig.subplots_adjust(top=0.84, left=0.11, right=0.97, bottom=0.12)


def update(i):
    line.set_data(freqs_fine[: i + 1], erp[: i + 1])
    marker.set_data([freqs_fine[i]], [erp[i]])
    sweep_line.set_xdata([freqs_fine[i], freqs_fine[i]])
    title.set_text(f"Frequency sweep: {freqs_fine[i]:6.2f} Hz, ERP = {erp[i]:6.2f} dB")
    return line, marker, sweep_line, title


fps = 40
anim = animation.FuncAnimation(fig, update, frames=freqs_fine.size, blit=True, interval=1000 / fps)
out_path = OUT_DIR / "erp_sweep_3_resonators.gif"
anim.save(out_path, writer=animation.PillowWriter(fps=fps), dpi=GIF_DPI)
print("Saved:", out_path)
