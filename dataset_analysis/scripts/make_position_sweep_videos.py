import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from utils.physics import freqs, omega_n, Lx, Ly, xf, yf, X_grid, Y_grid
from utils.erp_dataset import configuration_to_resonators
from utils.solver import compute_erp_spectrum, compute_displacement

F_T_FIXED = 90.0  # resonator tuning frequency, held fixed for both sweeps
plate_modes_in_range = [f for f in (omega_n / (2*np.pi)) if freqs.min() <= f <= freqs.max()]
extent = [0.0, Lx, 0.0, Ly]


def build_sweep(sweep_axis, fixed_value, sweep_values, out_path):
    n = sweep_values.size
    spectra = np.empty((n, freqs.size), dtype=np.float64)
    fields = np.empty((n, *X_grid.shape), dtype=np.float64)

    t0 = time.time()
    for i, v in enumerate(sweep_values):
        x, y = (v, fixed_value) if sweep_axis == "x" else (fixed_value, v)
        configuration = np.array([[F_T_FIXED, x, y]], dtype=np.float64)
        resonators = configuration_to_resonators(configuration)
        spectra[i] = compute_erp_spectrum(resonators, frequencies=freqs)
        disp = compute_displacement(resonators, frequencies=F_T_FIXED)  # (ny, nx, 1)
        fields[i] = np.abs(disp[..., 0])
    print(f"[{sweep_axis}-sweep] computed {n} frames in {time.time()-t0:.1f}s")

    erp_min, erp_max = spectra.min(), spectra.max()
    erp_pad = 0.08 * (erp_max - erp_min)
    field_vmax = fields.max()

    fig, (ax_erp, ax_field) = plt.subplots(1, 2, figsize=(13, 5.2))

    # Left: ERP spectrum
    ax_erp.set_xlim(freqs.min(), freqs.max())
    ax_erp.set_ylim(erp_min - erp_pad, erp_max + erp_pad)
    ax_erp.set_xlabel("Frequency (Hz)")
    ax_erp.set_ylabel("ERP (dB)")
    ax_erp.grid(True, alpha=0.3)
    for f in plate_modes_in_range:
        ax_erp.axvline(f, color="silver", ls="--", lw=1, zorder=0)
    ax_erp.axvline(F_T_FIXED, color="crimson", ls="--", lw=1.2, alpha=0.7, label=f"resonator f_t={F_T_FIXED:g}Hz")
    erp_line, = ax_erp.plot([], [], color="#1f77b4", lw=1.8)
    ax_erp.legend(loc="upper left", fontsize=8)
    erp_title = ax_erp.set_title("")

    # Right: plate displacement magnitude field
    im = ax_field.imshow(
        fields[0], extent=extent, origin="lower", aspect="auto",
        cmap="inferno", vmin=0.0, vmax=field_vmax,
    )
    fig.colorbar(im, ax=ax_field, label="|displacement| (m)")
    force_marker, = ax_field.plot([xf], [yf], marker="*", color="cyan", ms=14,
                                    mec="black", mew=0.8, ls="", label="Force F0")
    res_marker, = ax_field.plot([], [], marker="o", color="lime", ms=10,
                                  mec="black", mew=0.8, ls="", label="Resonator")
    ax_field.set_xlabel("x (m)")
    ax_field.set_ylabel("y (m)")
    ax_field.legend(loc="upper right", fontsize=8)
    field_title = ax_field.set_title("")

    fig.suptitle(f"Resonator {sweep_axis}-sweep, fixed f_t={F_T_FIXED:g} Hz "
                 f"({'y' if sweep_axis=='x' else 'x'} fixed at {fixed_value:g} m)")
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    def init():
        erp_line.set_data([], [])
        res_marker.set_data([], [])
        return erp_line, res_marker, im, erp_title, field_title

    def update(i):
        v = sweep_values[i]
        x, y = (v, fixed_value) if sweep_axis == "x" else (fixed_value, v)
        erp_line.set_data(freqs, spectra[i])
        erp_title.set_text(f"ERP spectrum ({sweep_axis}={v:.3f} m)")
        res_marker.set_data([x], [y])
        im.set_data(fields[i])
        field_title.set_text(f"Plate |displacement| at f={F_T_FIXED:g}Hz  (resonator at x={x:.3f}, y={y:.3f})")
        return erp_line, res_marker, im, erp_title, field_title

    fps = 15
    anim = animation.FuncAnimation(
        fig, update, frames=n, init_func=init, blit=False, interval=1000 / fps
    )
    anim.save(out_path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    print("Saved:", out_path)


edge = 0.05
x_values = np.linspace(edge, Lx - edge, 121)
y_values = np.linspace(edge, Ly - edge, 121)

build_sweep(
    "x", fixed_value=0.25, sweep_values=x_values,
    out_path="dataset_analysis/videos/erp_x_sweep.gif",
)
build_sweep(
    "y", fixed_value=0.70, sweep_values=y_values,
    out_path="dataset_analysis/videos/erp_y_sweep.gif",
)
