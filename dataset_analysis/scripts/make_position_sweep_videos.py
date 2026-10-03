import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import time

import matplotlib.animation as animation
import numpy as np

from gif_common import (ERP_COLOR, ERP_LABEL, FREQ_LABEL, GIF_DPI, M_RES, MODE_COLOR, OUT_DIR, TUNING_COLOR,
                        PlateHeatmap, configuration_row, displacement_um, finish, make_two_panel_figure)
from utils.erp_dataset import configuration_to_resonators
from utils.physics import Lx, Ly, freqs, omega_n
from utils.solver import compute_displacement, compute_erp_spectrum

TUNING_FREQUENCY = 90.0  # resonator tuning frequency (Hz), held fixed for both sweeps
plate_modes_in_range = [f for f in (omega_n / (2 * np.pi)) if freqs.min() <= f <= freqs.max()]


def build_sweep(sweep_axis, fixed_value, sweep_values, out_path):
    n = sweep_values.size
    spectra = np.empty((n, freqs.size))
    fields = None
    t0 = time.time()
    positions = []
    for i, v in enumerate(sweep_values):
        x, y = (v, fixed_value) if sweep_axis == "x" else (fixed_value, v)
        positions.append((x, y))
        resonators = configuration_to_resonators(np.array([configuration_row(TUNING_FREQUENCY, x, y)]))
        spectra[i] = compute_erp_spectrum(resonators, frequencies=freqs)
        field = np.abs(compute_displacement(resonators, frequencies=TUNING_FREQUENCY)[..., 0])
        if fields is None:
            fields = np.empty((n, *field.shape))
        fields[i] = field
    print(f"[{sweep_axis}-sweep] computed {n} frames in {time.time() - t0:.1f}s")

    fields_um = np.array([displacement_um(f) for f in fields])
    erp_min, erp_max = spectra.min(), spectra.max()
    pad = 0.08 * (erp_max - erp_min)

    fig, ax_erp, ax_field, cax = make_two_panel_figure()
    ax_erp.set_xlim(freqs.min(), freqs.max())
    ax_erp.set_ylim(erp_min - pad, erp_max + pad)
    ax_erp.set_xlabel(FREQ_LABEL)
    ax_erp.set_ylabel(ERP_LABEL)
    for k, f in enumerate(plate_modes_in_range):
        ax_erp.axvline(f, color=MODE_COLOR, ls="--", lw=1, zorder=0,
                       label="Plate natural frequencies" if k == 0 else None)
    ax_erp.axvline(TUNING_FREQUENCY, color=TUNING_COLOR, ls="--", lw=1.4, alpha=0.8,
                   label=f"Resonator tuning frequency ({TUNING_FREQUENCY:g} Hz)")
    (erp_line,) = ax_erp.plot([], [], color=ERP_COLOR, lw=1.9)
    ax_erp.legend(loc="upper left", fontsize=9)
    erp_title = ax_erp.set_title("")

    heat = PlateHeatmap(ax_field, cax, fields_um[0])
    field_title = ax_field.set_title("")

    swept = "length" if sweep_axis == "x" else "width"
    fixed_name = "width" if sweep_axis == "x" else "length"
    fixed_axis = "y" if sweep_axis == "x" else "x"
    finish(fig, f"Moving one resonator along the plate {swept}: tuning frequency {TUNING_FREQUENCY:g} Hz, "
                f"mass {M_RES:g} kg, fixed position along the plate {fixed_name} ({fixed_value:g} m)")

    def update(i):
        x, y = positions[i]
        erp_line.set_data(freqs, spectra[i])
        erp_title.set_text(f"ERP spectrum, resonator at position ({x:.2f} m, {y:.2f} m)")
        heat.update(fields_um[i], [(x, y)])
        field_title.set_text(f"Plate displacement at {TUNING_FREQUENCY:g} Hz ({heat.largest_value_text()})")
        return erp_line,

    fps = 15
    anim = animation.FuncAnimation(fig, update, frames=n, blit=False, interval=1000 / fps)
    anim.save(out_path, writer=animation.PillowWriter(fps=fps), dpi=GIF_DPI)
    print("Saved:", out_path)


edge = 0.05
build_sweep("x", fixed_value=0.25, sweep_values=np.linspace(edge, Lx - edge, 121),
            out_path=OUT_DIR / "erp_x_sweep.gif")
build_sweep("y", fixed_value=0.70, sweep_values=np.linspace(edge, Ly - edge, 121),
            out_path=OUT_DIR / "erp_y_sweep.gif")
