import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import time

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from gif_common import (ERP_COLOR, ERP_LABEL, FREQ_LABEL, GIF_DPI, M_RES, MODE_COLOR, OUT_DIR, TUNING_COLOR,
                        configuration_row, position_text, style_axes)
from utils.erp_dataset import configuration_to_resonators
from utils.physics import freqs, omega_n
from utils.solver import compute_erp_spectrum

# One resonator at a fixed position; only its tuning frequency changes from frame to frame.
X0, Y0 = 0.70, 0.25
tuning_values = np.arange(10.0, 160.0 + 1e-9, 1.0)  # 151 frames, 1 Hz steps
bare_erp = compute_erp_spectrum([], frequencies=freqs)

t0 = time.time()
spectra = np.empty((tuning_values.size, freqs.size))
for i, ft in enumerate(tuning_values):
    spectra[i] = compute_erp_spectrum(
        configuration_to_resonators(np.array([configuration_row(ft, X0, Y0)])), frequencies=freqs)
print(f"Computed {tuning_values.size} full spectra in {time.time() - t0:.1f}s")

plate_modes_in_range = [f for f in (omega_n / (2 * np.pi)) if freqs.min() <= f <= freqs.max()]
y_min, y_max = min(spectra.min(), bare_erp.min()), max(spectra.max(), bare_erp.max())
pad = 0.08 * (y_max - y_min)

fig, ax = plt.subplots(figsize=(9.5, 5.4))
style_axes(ax)
ax.set_xlim(freqs.min(), freqs.max())
ax.set_ylim(y_min - pad, y_max + pad)
ax.set_xlabel(FREQ_LABEL)
ax.set_ylabel(ERP_LABEL)
ax.plot(freqs, bare_erp, color="#6b6a66", lw=1.3, ls=":", label="Bare plate (no resonator)")
for k, f in enumerate(plate_modes_in_range):
    ax.axvline(f, color=MODE_COLOR, ls="--", lw=1, zorder=0, label="Plate natural frequencies" if k == 0 else None)
(line,) = ax.plot([], [], color=ERP_COLOR, lw=1.9, label="Plate with one resonator")
tuning_marker = ax.axvline(tuning_values[0], color=TUNING_COLOR, lw=1.6, alpha=0.85, label="Resonator tuning frequency")
title = ax.set_title("")
ax.legend(loc="upper left", fontsize=9)
fig.suptitle(f"Sweeping the tuning frequency of one resonator ({M_RES:g} kg, {position_text(X0, Y0)})",
             y=0.985, fontsize=13)
fig.subplots_adjust(top=0.86, left=0.11, right=0.97, bottom=0.12)


def update(i):
    line.set_data(freqs, spectra[i])
    tuning_marker.set_xdata([tuning_values[i], tuning_values[i]])
    title.set_text(f"Resonator tuning frequency: {tuning_values[i]:5.1f} Hz")
    return line, tuning_marker, title


fps = 15
anim = animation.FuncAnimation(fig, update, frames=tuning_values.size, blit=True, interval=1000 / fps)
out_path = OUT_DIR / "erp_ft_sweep_1_resonator.gif"
anim.save(out_path, writer=animation.PillowWriter(fps=fps), dpi=GIF_DPI)
print("Saved:", out_path)
