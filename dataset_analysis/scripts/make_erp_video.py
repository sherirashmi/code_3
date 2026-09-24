import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import imageio_ffmpeg

from utils.erp_dataset import configuration_to_resonators
from utils.solver import compute_erp_spectrum

matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()

# Three resonators, fixed positions/tuning -- [f_t, x, y] within the project's
# real sampling bounds (edge_margin=0.05, Lx=1.4, Ly=0.5, f_t in [10,160]).
configuration = np.array([
    [45.0, 0.30, 0.15],   # R1
    [90.0, 0.70, 0.30],   # R2
    [130.0, 1.10, 0.40],  # R3
], dtype=np.float64)
resonators = configuration_to_resonators(configuration)

freqs_fine = np.arange(10.0, 160.0 + 1e-9, 0.25)
erp = compute_erp_spectrum(resonators, frequencies=freqs_fine)
print(f"Computed {freqs_fine.size} points, ERP range [{erp.min():.2f}, {erp.max():.2f}] dB")

fig, ax = plt.subplots(figsize=(9, 5))
ax.set_xlim(freqs_fine.min(), freqs_fine.max())
pad = 0.08 * (erp.max() - erp.min())
ax.set_ylim(erp.min() - pad, erp.max() + pad)
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("ERP (dB)")
ax.grid(True, alpha=0.3)

for f_t, x, y in configuration:
    ax.axvline(f_t, color="crimson", ls="--", lw=1, alpha=0.5)

title = ax.set_title("")
line, = ax.plot([], [], color="#1f77b4", lw=1.8)
marker, = ax.plot([], [], "o", color="darkorange", ms=7)
sweep_line = ax.axvline(freqs_fine[0], color="darkorange", lw=1, alpha=0.6)

res_label = (
    "R1: f_t=45.0Hz (0.30,0.15)   "
    "R2: f_t=90.0Hz (0.70,0.30)   "
    "R3: f_t=130.0Hz (1.10,0.40)"
)
fig.text(0.5, 0.955, res_label, ha="center", fontsize=9, color="dimgray")
fig.subplots_adjust(top=0.86)

def init():
    line.set_data([], [])
    marker.set_data([], [])
    return line, marker, sweep_line, title

def update(i):
    line.set_data(freqs_fine[: i + 1], erp[: i + 1])
    marker.set_data([freqs_fine[i]], [erp[i]])
    sweep_line.set_xdata([freqs_fine[i], freqs_fine[i]])
    title.set_text(f"ERP spectrum sweep -- f = {freqs_fine[i]:6.2f} Hz, ERP = {erp[i]:6.2f} dB")
    return line, marker, sweep_line, title

fps = 40
anim = animation.FuncAnimation(
    fig, update, frames=freqs_fine.size, init_func=init, blit=True, interval=1000 / fps
)

out_path = "dataset_analysis/videos/erp_sweep_3_resonators.mp4"
anim.save(out_path, writer=animation.FFMpegWriter(fps=fps, bitrate=2000))
print("Saved:", out_path)
