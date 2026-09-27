import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from utils.physics import freqs, omega_n
from utils.erp_dataset import configuration_to_resonators
from utils.solver import compute_erp_spectrum

# One resonator, fixed spatial position; only f_t varies frame to frame.
X0, Y0 = 0.70, 0.25

ft_values = np.arange(10.0, 160.0 + 1e-9, 1.0)  # 151 frames, 1 Hz steps
bare_erp = compute_erp_spectrum([], frequencies=freqs)

t0 = time.time()
spectra = np.empty((ft_values.size, freqs.size), dtype=np.float64)
for i, f_t in enumerate(ft_values):
    configuration = np.array([[f_t, X0, Y0]], dtype=np.float64)
    resonators = configuration_to_resonators(configuration)
    spectra[i] = compute_erp_spectrum(resonators, frequencies=freqs)
print(f"Computed {ft_values.size} full spectra in {time.time()-t0:.1f}s")

plate_modes_in_range = [f for f in (omega_n / (2*np.pi)) if freqs.min() <= f <= freqs.max()]

y_min = min(spectra.min(), bare_erp.min())
y_max = max(spectra.max(), bare_erp.max())
pad = 0.08 * (y_max - y_min)

fig, ax = plt.subplots(figsize=(9, 5))
ax.set_xlim(freqs.min(), freqs.max())
ax.set_ylim(y_min - pad, y_max + pad)
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("ERP (dB)")
ax.grid(True, alpha=0.3)

ax.plot(freqs, bare_erp, color="gray", lw=1.2, ls=":", label="Bare plate (no resonator)")
for i, f in enumerate(plate_modes_in_range):
    ax.axvline(f, color="silver", ls="--", lw=1, zorder=0)

line, = ax.plot([], [], color="#1f77b4", lw=1.8, label="ERP with 1 resonator")
ft_marker = ax.axvline(ft_values[0], color="crimson", lw=1.5, alpha=0.8, label="Resonator f_t")
title = ax.set_title("")
ax.legend(loc="upper left", fontsize=9)
fig.text(0.5, 0.955, f"Resonator fixed at (x={X0}, y={Y0}); f_t sweeps 10-160 Hz",
          ha="center", fontsize=9, color="dimgray")
fig.subplots_adjust(top=0.86)

def init():
    line.set_data([], [])
    return line, ft_marker, title

def update(i):
    line.set_data(freqs, spectra[i])
    ft_marker.set_xdata([ft_values[i], ft_values[i]])
    title.set_text(f"f_t = {ft_values[i]:6.1f} Hz")
    return line, ft_marker, title

fps = 15
anim = animation.FuncAnimation(
    fig, update, frames=ft_values.size, init_func=init, blit=True, interval=1000 / fps
)

out_path = "dataset_analysis/videos/erp_ft_sweep_1_resonator.gif"
anim.save(out_path, writer=animation.PillowWriter(fps=fps))
print("Saved:", out_path)
