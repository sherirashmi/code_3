import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from utils.support import load_dataset
from utils.physics import omega_n, Lx, Ly, xf, yf, X_grid, Y_grid
from utils.erp_dataset import configuration_to_resonators
from utils.solver import compute_displacement

payload = load_dataset("datasets/dataset_erp_ft.pth")
freqs = np.asarray(payload["frequency_values"], dtype=np.float64)
configs = np.asarray(payload["configuration_features"], dtype=np.float64)   # (N, 3, 3) [f_t,x,y]
responses = np.asarray(payload["responses"], dtype=np.float64)[:, :, 0]     # (N, n_freq)
print(f"Loaded dataset: {configs.shape[0]} configurations, {freqs.size} frequencies")

N_SHOW = 80
rng = np.random.default_rng(42)
sample_ids = rng.choice(configs.shape[0], size=N_SHOW, replace=False)

plate_modes_in_range = [f for f in (omega_n / (2*np.pi)) if freqs.min() <= f <= freqs.max()]
extent = [0.0, Lx, 0.0, Ly]

fields = np.empty((N_SHOW, *X_grid.shape), dtype=np.float64)
peak_freqs = np.empty(N_SHOW)
t0 = time.time()
for i, cid in enumerate(sample_ids):
    configuration = configs[cid]              # (3,3) [f_t,x,y]
    spectrum = responses[cid]
    peak_idx = int(np.argmax(spectrum))
    peak_freqs[i] = freqs[peak_idx]
    resonators = configuration_to_resonators(configuration)
    disp = compute_displacement(resonators, frequencies=peak_freqs[i])
    fields[i] = np.abs(disp[..., 0])
print(f"Computed {N_SHOW} frames in {time.time()-t0:.1f}s")

erp_min, erp_max = responses[sample_ids].min(), responses[sample_ids].max()
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
erp_line, = ax_erp.plot([], [], color="#1f77b4", lw=1.8)
peak_marker, = ax_erp.plot([], [], "o", color="darkorange", ms=8, label="dataset's true peak")
ax_erp.legend(loc="upper left", fontsize=8)
erp_title = ax_erp.set_title("")

im = ax_field.imshow(
    fields[0], extent=extent, origin="lower", aspect="auto",
    cmap="inferno", vmin=0.0, vmax=field_vmax,
)
fig.colorbar(im, ax=ax_field, label="|displacement| (m)")
force_marker, = ax_field.plot([xf], [yf], marker="*", color="cyan", ms=14,
                                mec="black", mew=0.8, ls="", label="Force F0")
res_markers, = ax_field.plot([], [], marker="o", color="lime", ms=10,
                               mec="black", mew=0.8, ls="", label="Resonators")
ax_field.set_xlabel("x (m)")
ax_field.set_ylabel("y (m)")
ax_field.legend(loc="upper right", fontsize=8)
field_title = ax_field.set_title("")

fig.suptitle("Real dataset configurations (datasets/dataset_erp_ft.pth) -- 3 resonators each")
fig.tight_layout(rect=[0, 0, 1, 0.94])

def init():
    erp_line.set_data([], [])
    peak_marker.set_data([], [])
    res_markers.set_data([], [])
    return erp_line, peak_marker, res_markers, im, erp_title, field_title

def update(i):
    cid = sample_ids[i]
    configuration = configs[cid]
    spectrum = responses[cid]
    erp_line.set_data(freqs, spectrum)
    peak_marker.set_data([peak_freqs[i]], [spectrum.max()])
    erp_title.set_text(f"Configuration #{cid}  (ERP peak {spectrum.max():.1f} dB @ {peak_freqs[i]:.1f} Hz)")

    res_markers.set_data(configuration[:, 1], configuration[:, 2])
    im.set_data(fields[i])
    ft_str = ", ".join(f"{ft:.1f}" for ft in configuration[:, 0])
    field_title.set_text(f"|displacement| at peak freq={peak_freqs[i]:.1f}Hz  (resonator f_t = {ft_str} Hz)")
    return erp_line, peak_marker, res_markers, im, erp_title, field_title

fps = 6
anim = animation.FuncAnimation(
    fig, update, frames=N_SHOW, init_func=init, blit=False, interval=1000 / fps
)
out_path = "dataset_analysis/videos/erp_dataset_configs.gif"
anim.save(out_path, writer=animation.PillowWriter(fps=fps))
print("Saved:", out_path)
