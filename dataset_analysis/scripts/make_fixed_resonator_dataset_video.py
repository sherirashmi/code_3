"""GIF of the dataset "100k_2res_fixed_m0.2_ft72_18modes": two identical resonators
(m = 0.2 kg, f_t = 72 Hz) whose positions alone vary.

Left: the ERP spectrum of each configuration, with the 18 plate modes, the common
tuning frequency and the 5th to 95th percentile range of the whole dataset. Right: the plate drawn to
scale with the excitation force and both resonator positions.

Any other dataset tag works too (e.g. ``100k``, three resonators with random mass, tuning
frequency and position): the per-resonator mass and tuning frequency are then shown as well.

Run from the repository root:
    python dataset_analysis/scripts/make_fixed_resonator_dataset_video.py [dataset tag]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from gif_common import ERP_COLOR, ERP_LABEL, FREQ_LABEL, GIF_DPI, MODE_COLOR, OUT_DIR, TUNING_COLOR, style_axes
import utils.physics as physics
from utils.erp_dataset import DATASETS
from utils.physics import Lx, Ly, edge_margin, xf, yf
from utils.support import load_dataset

TAG = next((a for a in sys.argv[1:] if not a.startswith("--")), "100k_2res_fixed_m0.2_ft72_18modes")
N_SHOW = 60
FPS = 2
RES_COLORS = ("#2a78d6", "#eb6834", "#2e9e5b", "#8a4fd0")
RES_MARKERS = ("o", "s", "^", "D")

spec = DATASETS[TAG]
physics.set_modal_resolution(*spec["modal_resolution"], verbose=False)
fixed = spec.get("fixed_resonator")  # (mass, tuning frequency) when every resonator is identical
if fixed is not None:
    mass, tuning = fixed
    stiffness = mass * (2.0 * np.pi * tuning) ** 2

configs, responses = [], []
for f in spec["files"]:
    payload = load_dataset(f)
    configs.append(np.asarray(payload["configuration_features"], dtype=np.float64))  # (n, 2, 5): m, k, f_t, x, y
    responses.append(np.asarray(payload["responses"], dtype=np.float64)[:, :, 0])
freqs = np.asarray(payload["frequency_values"], dtype=np.float64)
configs, responses = np.concatenate(configs), np.concatenate(responses)
print(f"Loaded {configs.shape[0]:,} configurations, {freqs.size} frequencies")
NUM_RES = configs.shape[1]

sample_ids = np.random.default_rng(42).choice(configs.shape[0], size=N_SHOW, replace=False)
low, high = np.percentile(responses, [5, 95], axis=0)
plate_modes = [f for f in physics.omega_n / (2 * np.pi) if freqs.min() <= f <= freqs.max()]

fig = plt.figure(figsize=(14.0, 5.6))
gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.05], left=0.065, right=0.98, top=0.85, bottom=0.25, wspace=0.2)
ax_erp, ax_plate = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

# ---- left: ERP spectrum ------------------------------------------------------------
style_axes(ax_erp)
ax_erp.fill_between(freqs, low, high, color="#d9d9d4", lw=0, label="Dataset range (5th to 95th percentile)")
for j, f in enumerate(plate_modes):
    ax_erp.axvline(f, color=MODE_COLOR, ls="--", lw=0.9, zorder=0, label="Bare-plate natural frequencies" if j == 0 else None)
if fixed is not None:
    ax_erp.axvline(tuning, color=TUNING_COLOR, lw=1.4, ls="-.", zorder=1,
                   label=f"Resonator tuning frequency $f_t = {tuning:.0f}$ Hz")
else:  # tuning frequencies differ per configuration: one line per resonator, moved every frame
    tuning_lines = [ax_erp.axvline(0, color=RES_COLORS[r], lw=1.4, ls="-.", zorder=1,
                                   label=f"Tuning frequency of resonator {r + 1}") for r in range(NUM_RES)]
(erp_line,) = ax_erp.plot([], [], color="black", lw=1.8, zorder=3, label="ERP of this configuration")
pad = 0.05 * (high.max() - low.min())
ax_erp.set_xlim(freqs.min(), freqs.max())
ax_erp.set_ylim(min(low.min(), responses[sample_ids].min()) - pad, max(high.max(), responses[sample_ids].max()) + pad)
ax_erp.set_xlabel(FREQ_LABEL)
ax_erp.set_ylabel(ERP_LABEL)
ax_erp.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2 if fixed is not None else 3, frameon=False, fontsize=9)
erp_title = ax_erp.set_title("")

# ---- right: plate with resonator positions ----------------------------------------
ax_plate.add_patch(Rectangle((0, 0), Lx, Ly, facecolor="#f4f4f1", ec="black", lw=1.2, zorder=0))
ax_plate.add_patch(Rectangle((edge_margin, edge_margin), Lx - 2 * edge_margin, Ly - 2 * edge_margin, fill=False,
                             ec="#9a9a94", ls="--", lw=0.9, zorder=1, label="Sampled region (5 cm edge margin)"))
ax_plate.plot([xf], [yf], marker="*", color="#35d0ff", ms=17, mec="black", mew=0.9, ls="", zorder=6,
              label="Excitation force")
res_markers = []
for r in range(NUM_RES):
    (marker,) = ax_plate.plot([], [], marker=RES_MARKERS[r], color=RES_COLORS[r], ms=11, mec="black", mew=1.0,
                              ls="", zorder=7, label=f"Resonator {r + 1}")
    res_markers.append(marker)
res_labels = [ax_plate.text(0, 0, "", fontsize=9, zorder=5,
                          bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85)) for _ in range(NUM_RES)]
ax_plate.set_xlim(-0.02, Lx + 0.02)
ax_plate.set_ylim(-0.02, Ly + 0.02)
ax_plate.set_aspect("equal")
ax_plate.set_xticks(np.arange(0, Lx + 1e-9, 0.2))
ax_plate.set_yticks(np.arange(0, Ly + 1e-9, 0.1))
ax_plate.set_xlabel("Position $x$ (m)")
ax_plate.set_ylabel("Position $y$ (m)")
ax_plate.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2 if fixed is not None else 3, frameon=False, fontsize=9)
plate_title = ax_plate.set_title("")

if fixed is not None:
    heading = (f"Dataset with two identical resonators ($m = {mass:g}$ kg, $f_t = {tuning:.0f}$ Hz, "
               f"$k = {stiffness / 1000:.1f}$ kN/m): only the positions vary")
else:
    number = {1: "one", 2: "two", 3: "three", 4: "four"}.get(NUM_RES, str(NUM_RES))
    heading = (f"Dataset with {number} resonators: mass, tuning frequency and position vary "
               f"({configs.shape[0]:,} configurations)")
fig.suptitle(heading, y=0.965, fontsize=13)


def update(i):
    cid = sample_ids[i]
    config, spectrum = configs[cid], responses[cid]
    erp_line.set_data(freqs, spectrum)
    peak = int(np.argmax(spectrum))
    erp_title.set_text(f"Configuration {cid + 1:,} of {configs.shape[0]:,}: "
                       f"highest peak {spectrum[peak]:.1f} dB at {freqs[peak]:.1f} Hz")
    for r in range(NUM_RES):
        x, y = config[r, 3], config[r, 4]
        if fixed is None:
            tuning_lines[r].set_xdata([config[r, 2], config[r, 2]])
        res_markers[r].set_data([x], [y])
        right = x > Lx - 0.45  # keep the label on the plate
        res_labels[r].set_position((x - 0.03 if right else x + 0.03, y + 0.025))
        res_labels[r].set_ha("right" if right else "left")
        res_labels[r].set_text(f"R{r + 1} ({x:.2f}, {y:.2f}) m" if fixed is not None else
                               f"R{r + 1}: {config[r, 0]:.2f} kg, tuned to {config[r, 2]:.0f} Hz")
    plate_title.set_text("Plate (1.4 m $\\times$ 0.5 m) with the resonator positions")
    extra = () if fixed is not None else tuple(tuning_lines)
    return (erp_line, *res_markers, *res_labels, *extra)


if "--preview" in sys.argv:  # one still frame for checking the layout
    update(0)
    fig.savefig(OUT_DIR / f"preview_{TAG}.png", dpi=GIF_DPI)
    sys.exit()

anim = animation.FuncAnimation(fig, update, frames=N_SHOW, blit=False, interval=1000 / FPS)
out_path = OUT_DIR / ("erp_dataset_2res_fixed_m0.2_ft72.gif" if fixed is not None else f"erp_dataset_{TAG}.gif")
anim.save(out_path, writer=animation.PillowWriter(fps=FPS), dpi=GIF_DPI)
print("Saved:", out_path)
