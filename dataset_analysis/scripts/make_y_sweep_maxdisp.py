import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import time

import matplotlib.animation as animation
import numpy as np

from gif_common import (ERP_COLOR, ERP_LABEL, FREQ_LABEL, GIF_DPI, M_RES, MODE_COLOR, OUT_DIR, TUNING_COLOR,
                        PlateHeatmap, configuration_row, displacement_db, finish, make_two_panel_figure)
from utils.erp_dataset import configuration_to_resonators
from utils.physics import Lx, Ly, X_grid, freqs, omega_n
from utils.solver import compute_displacement, compute_erp_spectrum

TUNING_FREQUENCY = 90.0

# Fix the position along the plate length at the plate's own maximum-displacement point
# (bare-plate forced response at the tuning frequency): the true antinode line.
bare = np.abs(compute_displacement([], frequencies=TUNING_FREQUENCY)[..., 0])
iy, ix = np.unravel_index(np.argmax(bare), bare.shape)
X_FIXED = float(X_grid[iy, ix])
print(f"Fixed x = {X_FIXED:.4f} m (maximum-displacement point at {TUNING_FREQUENCY:g} Hz)")

plate_modes_in_range = [f for f in (omega_n / (2 * np.pi)) if freqs.min() <= f <= freqs.max()]
y_values = np.linspace(0.05, Ly - 0.05, 121)
n = y_values.size

spectra = np.empty((n, freqs.size))
fields = None
t0 = time.time()
for i, y in enumerate(y_values):
    resonators = configuration_to_resonators(np.array([configuration_row(TUNING_FREQUENCY, X_FIXED, y)]))
    spectra[i] = compute_erp_spectrum(resonators, frequencies=freqs)
    field = np.abs(compute_displacement(resonators, frequencies=TUNING_FREQUENCY)[..., 0])
    if fields is None:
        fields = np.empty((n, *field.shape))
    fields[i] = field
print(f"Computed {n} frames in {time.time() - t0:.1f}s")

fields_db = np.array([displacement_db(f, f.max()) for f in fields])
erp_min, erp_max = spectra.min(), spectra.max()
pad = 0.08 * (erp_max - erp_min)

fig, ax_erp, ax_field, cax = make_two_panel_figure()
ax_erp.set_xlim(freqs.min(), freqs.max())
ax_erp.set_ylim(erp_min - pad, erp_max + pad)
ax_erp.set_xlabel(FREQ_LABEL)
ax_erp.set_ylabel(ERP_LABEL)
for k, f in enumerate(plate_modes_in_range):
    ax_erp.axvline(f, color=MODE_COLOR, ls="--", lw=1, zorder=0, label="Plate natural frequencies" if k == 0 else None)
ax_erp.axvline(TUNING_FREQUENCY, color=TUNING_COLOR, ls="--", lw=1.4, alpha=0.8,
               label=f"Resonator tuning frequency ({TUNING_FREQUENCY:g} Hz)")
(erp_line,) = ax_erp.plot([], [], color=ERP_COLOR, lw=1.9)
ax_erp.legend(loc="upper left", fontsize=9)
erp_title = ax_erp.set_title("")

heat = PlateHeatmap(ax_field, cax, fields_db[0])
ax_field.axvline(X_FIXED, color="white", lw=1.0, ls=":", alpha=0.8)
field_title = ax_field.set_title("")
finish(fig, f"Moving one resonator across the plate width along the maximum-displacement line: "
            f"tuning frequency {TUNING_FREQUENCY:g} Hz, mass {M_RES:g} kg")


def update(i):
    y = y_values[i]
    erp_line.set_data(freqs, spectra[i])
    erp_title.set_text(f"ERP spectrum, resonator at position ({X_FIXED:.2f} m, {y:.2f} m)")
    heat.update(fields_db[i], [(X_FIXED, y)])
    field_title.set_text(f"Plate displacement at {TUNING_FREQUENCY:g} Hz (dotted line: maximum-displacement line)")
    return erp_line,


fps = 15
anim = animation.FuncAnimation(fig, update, frames=n, blit=False, interval=1000 / fps)
out_path = OUT_DIR / "erp_y_sweep_maxdisp.gif"
anim.save(out_path, writer=animation.PillowWriter(fps=fps), dpi=GIF_DPI)
print("Saved:", out_path)
