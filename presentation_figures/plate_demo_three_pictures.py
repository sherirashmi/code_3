"""Three pictures for the plate with three resonators (positions as in plate_three_resonators):
(1) displacement field at one frequency, (2) velocity field from that displacement, (3) the ERP sweep.

Solver: utils/solver.py (150 plate modes, 1 N point force at (0.865, 0.309) m, resonator damping 1 N s/m).
Displacement |w| in mm; velocity |v| = omega |w| in mm/s (same length unit); ERP from the velocity as in compute_erp_spectrum.
"""
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

import utils.plot_style  # noqa: F401
from utils.physics import Lx, Ly, freqs, xf, yf
from utils.solver import compute_displacement, compute_erp_spectrum, compute_velocity

F_SHOW = float(sys.argv[1]) if len(sys.argv) > 1 else 103.0  # Hz, the frequency of pictures 1 and 2
# (m_i [kg], f_t,i [Hz], x_i [m], y_i [m]); positions follow the schematic
DESIGN = [(0.5, 45.0, 0.28, 0.25), (0.8, 90.0, 0.65, 0.40), (0.6, 130.0, 1.10, 0.22)]
RES = [dict(m=m, f_t=ft, k=m * (2 * math.pi * ft) ** 2, x=x, y=y, c=1.0) for m, ft, x, y in DESIGN]
OUT = ROOT / "presentation_figures"
BLUE, RED = "#2a78d6", "#c2412c"

w = compute_displacement(RES, frequencies=[F_SHOW])  # (ny, nx, 1), complex, metres for 1 N
v = compute_velocity(w, frequencies=[F_SHOW])[..., 0]
w_mm = np.abs(w[..., 0]) * 1e3
v_mm = np.abs(v) * 1e3
erp = compute_erp_spectrum(RES)


def save(fig, name):
    for ext, kw in (("png", dict(dpi=220)), ("pdf", {}), ("svg", {})):
        fig.savefig(OUT / f"{name}.{ext}", facecolor="white", **kw)
    plt.close(fig)
    print("saved", name)


def field_figure(field, label, name, title, cmap="magma", star="#35d0ff", dot="#7CFC00"):
    fig, ax = plt.subplots(figsize=(11.5, 5.4))
    im = ax.imshow(field, extent=[0, Lx, 0, Ly], origin="lower", aspect="equal", cmap=cmap, vmin=0.0, interpolation="bicubic")
    ax.add_patch(Rectangle((0, 0), Lx, Ly, fill=False, ec="black", lw=1.2))
    ax.plot([xf], [yf], marker="*", color=star, ms=18, mec="black", mew=0.9, ls="", label="Excitation force", zorder=6)
    ax.plot([r["x"] for r in RES], [r["y"] for r in RES], marker="o", color=dot, ms=11, mec="black", mew=1.0, ls="",
            label="Resonators", zorder=7)
    for i, r in enumerate(RES, 1):
        ax.annotate(rf"$m_{i}$", (r["x"], r["y"]), xytext=(8, 7), textcoords="offset points", color="white", fontsize=14, zorder=8)
    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Ly)
    ax.set_xticks(np.arange(0, Lx + 1e-9, 0.2))
    ax.set_yticks(np.arange(0, Ly + 1e-9, 0.1))
    ax.set_xlabel("Position $x$ (m)")
    ax.set_ylabel("Position $y$ (m)")
    ax.set_title(title, fontsize=14)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label(label)
    fig.tight_layout()
    save(fig, name)


f = f"{F_SHOW:g}"
field_figure(w_mm, r"Displacement magnitude $|w|$ (mm)", "plate_demo_1_displacement",
             rf"Displacement field at ${f}$ Hz (force $1$ N, three resonators)")
field_figure(v_mm, r"Velocity magnitude $|v|=\omega\,|w|$ (mm/s)", "plate_demo_2_velocity",
             rf"Velocity field at ${f}$ Hz", cmap="viridis", star="white", dot="#ff7f2a")

fig, ax = plt.subplots(figsize=(11.5, 5.0))
ax.plot(freqs, erp, color=BLUE, lw=2.2)
for m, ft, _, _ in DESIGN:
    ax.axvline(ft, color="#9a9a94", ls=":", lw=1.2)
ax.plot([], [], color="#9a9a94", ls=":", label="Resonator tuning frequencies")
ax.axvline(F_SHOW, color=RED, ls="--", lw=1.6, label=rf"Frequency of the field plots, ${f}$ Hz")
ax.plot([F_SHOW], [np.interp(F_SHOW, freqs, erp)], "o", color=RED, ms=8)
ax.set_xlim(freqs[0], freqs[-1])
ax.set_ylim(erp.min() - 3, erp.max() + 8)
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Equivalent radiated power, ERP (dB)")
ax.set_title("ERP from the frequency sweep", fontsize=14)
ax.grid(True, color="#d9d9d4", lw=0.6)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(loc="lower right", frameon=False)
fig.tight_layout()
save(fig, "plate_demo_3_erp_sweep")
print("max |w| (mm):", w_mm.max(), " max |v| (mm/s):", v_mm.max())
