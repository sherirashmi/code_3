import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from utils.physics import freqs, omega_n, Lx, Ly, xf, yf, X_grid, Y_grid
from utils.erp_dataset import configuration_to_resonators
from utils.solver import compute_erp_spectrum, compute_displacement

F_T_FIXED = 90.0

# Fix x at the plate's own max-displacement point (bare-plate forced response
# at f=F_T_FIXED), found via argmax|displacement| -- the true antinode line,
# not an arbitrary choice.
bare_disp = compute_displacement([], frequencies=F_T_FIXED)
mag = np.abs(bare_disp[..., 0])
iy, ix = np.unravel_index(np.argmax(mag), mag.shape)
X_FIXED = float(X_grid[iy, ix])
print(f"Fixed x = {X_FIXED:.4f} m (plate's max-displacement point at f={F_T_FIXED}Hz)")

plate_modes_in_range = [f for f in (omega_n / (2*np.pi)) if freqs.min() <= f <= freqs.max()]
extent = [0.0, Lx, 0.0, Ly]

edge = 0.05
y_values = np.linspace(edge, Ly - edge, 121)
n = y_values.size

spectra = np.empty((n, freqs.size), dtype=np.float64)
fields = np.empty((n, *X_grid.shape), dtype=np.float64)

t0 = time.time()
for i, y in enumerate(y_values):
    configuration = np.array([[F_T_FIXED, X_FIXED, y]], dtype=np.float64)
    resonators = configuration_to_resonators(configuration)
    spectra[i] = compute_erp_spectrum(resonators, frequencies=freqs)
    disp = compute_displacement(resonators, frequencies=F_T_FIXED)
    fields[i] = np.abs(disp[..., 0])
print(f"Computed {n} frames in {time.time()-t0:.1f}s")

erp_min, erp_max = spectra.min(), spectra.max()
erp_pad = 0.08 * (erp_max - erp_min)
field_vmax = fields.max()

fig, (ax_erp, ax_field) = plt.subplots(1, 2, figsize=(13, 5.2))

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

im = ax_field.imshow(
    fields[0], extent=extent, origin="lower", aspect="auto",
    cmap="inferno", vmin=0.0, vmax=field_vmax,
)
fig.colorbar(im, ax=ax_field, label="|displacement| (m)")
ax_field.axvline(X_FIXED, color="white", lw=1, ls=":", alpha=0.7, label="max-displacement line")
force_marker, = ax_field.plot([xf], [yf], marker="*", color="cyan", ms=14,
                                mec="black", mew=0.8, ls="", label="Force F0")
res_marker, = ax_field.plot([], [], marker="o", color="lime", ms=10,
                              mec="black", mew=0.8, ls="", label="Resonator")
ax_field.set_xlabel("x (m)")
ax_field.set_ylabel("y (m)")
ax_field.legend(loc="upper right", fontsize=8)
field_title = ax_field.set_title("")

fig.suptitle(f"Resonator y-sweep along the max-displacement line x={X_FIXED:.3f}m, fixed f_t={F_T_FIXED:g} Hz")
fig.tight_layout(rect=[0, 0, 1, 0.94])

def init():
    erp_line.set_data([], [])
    res_marker.set_data([], [])
    return erp_line, res_marker, im, erp_title, field_title

def update(i):
    y = y_values[i]
    erp_line.set_data(freqs, spectra[i])
    erp_title.set_text(f"ERP spectrum (y={y:.3f} m)")
    res_marker.set_data([X_FIXED], [y])
    im.set_data(fields[i])
    field_title.set_text(f"Plate |displacement| at f={F_T_FIXED:g}Hz  (resonator at x={X_FIXED:.3f}, y={y:.3f})")
    return erp_line, res_marker, im, erp_title, field_title

fps = 15
anim = animation.FuncAnimation(
    fig, update, frames=n, init_func=init, blit=False, interval=1000 / fps
)
out_path = "dataset_analysis/videos/erp_y_sweep_maxdisp.gif"
anim.save(out_path, writer=animation.PillowWriter(fps=fps))
print("Saved:", out_path)
