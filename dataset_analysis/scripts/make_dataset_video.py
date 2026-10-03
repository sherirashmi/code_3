import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import time

import matplotlib.animation as animation
import numpy as np

from gif_common import (ERP_COLOR, ERP_LABEL, FREQ_LABEL, GIF_DPI, MODE_COLOR, PlateHeatmap,
                        displacement_um, finish, shared_vmax, make_two_panel_figure, video_dir)
from utils.erp_dataset import configuration_to_resonators
from utils.physics import omega_n
from utils.solver import compute_displacement
from utils.support import load_dataset

payload = load_dataset("datasets/dataset_erp_ft.pth")
freqs = np.asarray(payload["frequency_values"], dtype=np.float64)
configs = np.asarray(payload["configuration_features"], dtype=np.float64)  # (N, 3, 5): m, k, f_t, x, y
responses = np.asarray(payload["responses"], dtype=np.float64)[:, :, 0]
print(f"Loaded dataset: {configs.shape[0]} configurations, {freqs.size} frequencies")

N_SHOW = 80
rng = np.random.default_rng(42)
sample_ids = rng.choice(configs.shape[0], size=N_SHOW, replace=False)
plate_modes_in_range = [f for f in (omega_n / (2 * np.pi)) if freqs.min() <= f <= freqs.max()]

fields_um = []
peak_freqs = np.empty(N_SHOW)
t0 = time.time()
for i, cid in enumerate(sample_ids):
    peak_freqs[i] = freqs[int(np.argmax(responses[cid]))]
    field = np.abs(compute_displacement(configuration_to_resonators(configs[cid]), frequencies=peak_freqs[i])[..., 0])
    fields_um.append(displacement_um(field))
print(f"Computed {N_SHOW} frames in {time.time() - t0:.1f}s")

erp_min, erp_max = responses[sample_ids].min(), responses[sample_ids].max()
pad = 0.08 * (erp_max - erp_min)

fig, ax_erp, ax_field, cax = make_two_panel_figure()
ax_erp.set_xlim(freqs.min(), freqs.max())
ax_erp.set_ylim(erp_min - pad, erp_max + pad)
ax_erp.set_xlabel(FREQ_LABEL)
ax_erp.set_ylabel(ERP_LABEL)
for f in plate_modes_in_range:
    ax_erp.axvline(f, color=MODE_COLOR, ls="--", lw=1, zorder=0)
(erp_line,) = ax_erp.plot([], [], color=ERP_COLOR, lw=1.9)
(peak_marker,) = ax_erp.plot([], [], "o", color="#e08a1e", ms=8, mec="black", mew=0.6, label="Highest ERP peak")
ax_erp.legend(loc="upper left", fontsize=9)
erp_title = ax_erp.set_title("")

heat = PlateHeatmap(ax_field, cax, fields_um[0], vmax=shared_vmax(fields_um), resonator_label="Resonators")
field_title = ax_field.set_title("")
finish(fig, "Example configurations from the dataset (three resonators each)")


def update(i):
    cid = sample_ids[i]
    configuration = configs[cid]
    spectrum = responses[cid]
    erp_line.set_data(freqs, spectrum)
    peak_marker.set_data([peak_freqs[i]], [spectrum.max()])
    erp_title.set_text(f"Configuration {cid}: highest peak {spectrum.max():.1f} dB at {peak_freqs[i]:.1f} Hz")
    heat.update(fields_um[i], list(zip(configuration[:, 3], configuration[:, 4])))
    tuning = ", ".join(f"{ft:.0f}" for ft in configuration[:, 2])
    field_title.set_text(f"Plate displacement at {peak_freqs[i]:.1f} Hz ({heat.largest_value_text()})\n"
                         f"resonator tuning frequencies: {tuning} Hz")
    return erp_line,


fps = 6
anim = animation.FuncAnimation(fig, update, frames=N_SHOW, blit=False, interval=1000 / fps)
out_path = video_dir("10k") / "erp_dataset_configs.gif"
anim.save(out_path, writer=animation.PillowWriter(fps=fps), dpi=GIF_DPI)
print("Saved:", out_path)
