import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.pyplot as plt

from gif_common import ERP_COLOR, ERP_LABEL, FREQ_LABEL, GIF_DPI, OUT_DIR, TUNING_COLOR, position_text, style_axes
from utils.solver import compute_erp_spectrum
from utils.physics import freqs, fmin, fmax, m_min, m_max, Lx, Ly, edge_margin

# Fixed position for every resonator in every case/video, well inside plate bounds.
X0, Y0 = 0.70, 0.25
C_DAMPING = 1.0
K_PRACTICAL_LOW, K_PRACTICAL_HIGH = 400.0, 1.0e6  # practical stiffness range established earlier

N_FRAMES = 50


def spectrum(m, k, f_t):
    resonator = {"f_t": float(f_t), "x": X0, "y": Y0, "m": float(m), "c": C_DAMPING, "k": float(k)}
    return compute_erp_spectrum([resonator], frequencies=freqs)


def make_video(
    filename,
    title,
    case_a_label,
    case_b_label,
    param_series,  # list of (m, k, f_t) per frame, case A
    param_series_b,  # list of (m, k, f_t) per frame, case B
    varying_name,
):
    n = len(param_series)
    spectra_a = [spectrum(*p) for p in param_series]
    spectra_b = [spectrum(*p) for p in param_series_b]

    y_all = np.concatenate([np.concatenate(spectra_a), np.concatenate(spectra_b)])
    y_lo, y_hi = float(y_all.min()) - 3, float(y_all.max()) + 3

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle(title, fontsize=13)
    fig.text(0.5, 0.905, f"One resonator, {position_text(X0, Y0)}", ha="center", fontsize=10, color="#3d3c39")

    lines = []
    ft_lines = []
    texts = []
    for ax, label in zip((ax1, ax2), (case_a_label, case_b_label)):
        style_axes(ax)
        (line,) = ax.plot([], [], lw=2.0, color=ERP_COLOR)
        ft_line = ax.axvline(0, color=TUNING_COLOR, ls="--", lw=1.5, label="Resonator tuning frequency")
        ax.set_xlim(fmin, fmax)
        ax.set_ylim(y_lo, y_hi)
        ax.set_xlabel(FREQ_LABEL)
        ax.set_ylabel(ERP_LABEL)
        ax.set_title(label, fontsize=11)
        txt = ax.text(
            0.02, 0.97, "", transform=ax.transAxes, va="top", ha="left", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="#b5b5ae"),
        )
        ax.legend(loc="lower right", fontsize=9)
        lines.append(line)
        ft_lines.append(ft_line)
        texts.append(txt)

    def fmt(m, k, f_t):
        return (f"Mass: {m:.3f} kg\nStiffness: {k:,.0f} N/m\nTuning frequency: {f_t:.1f} Hz")

    def init():
        for line in lines:
            line.set_data([], [])
        return lines + ft_lines + texts

    def update(i):
        for line, ft_line, txt, spectra, params in (
            (lines[0], ft_lines[0], texts[0], spectra_a, param_series),
            (lines[1], ft_lines[1], texts[1], spectra_b, param_series_b),
        ):
            line.set_data(freqs, spectra[i])
            m, k, f_t = params[i]
            ft_line.set_xdata([f_t, f_t])
            txt.set_text(fmt(m, k, f_t))
        return lines + ft_lines + texts

    anim = FuncAnimation(fig, update, frames=n, init_func=init, blit=False, interval=120)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    writer = PillowWriter(fps=8)
    path = OUT_DIR / filename
    anim.save(path, writer=writer, dpi=GIF_DPI)
    plt.close(fig)
    print(f"Saved {path}")


f_t_sweep = np.linspace(fmin, fmax, N_FRAMES)
m_sweep = np.linspace(m_min, m_max, N_FRAMES)
k_sweep = np.geomspace(K_PRACTICAL_LOW, K_PRACTICAL_HIGH, N_FRAMES)

# ------------------------------------------------------------------
# Video 1: fixed mass (two cases), sweep f_t 10->160 Hz, k derived
# ------------------------------------------------------------------
m_a, m_b = m_min, m_max
series_a = [(m_a, m_a * (2 * np.pi * f) ** 2, f) for f in f_t_sweep]
series_b = [(m_b, m_b * (2 * np.pi * f) ** 2, f) for f in f_t_sweep]
make_video(
    "video1_fixed_mass_sweep_ft.gif",
    r"Sweeping the tuning frequency ($10$ to $160$ Hz) at fixed mass; stiffness follows $k = m\,(2\pi f_t)^2$",
    f"Case A: light resonator, mass {m_a:.2f} kg",
    f"Case B: heavy resonator, mass {m_b:.2f} kg",
    series_a, series_b, "f_t",
)

# ------------------------------------------------------------------
# Video 2: fixed k (two cases), sweep f_t 10->160 Hz, m derived
# ------------------------------------------------------------------
k_a, k_b = 2000.0, 400000.0
series_a = [(k_a / (2 * np.pi * f) ** 2, k_a, f) for f in f_t_sweep]
series_b = [(k_b / (2 * np.pi * f) ** 2, k_b, f) for f in f_t_sweep]
make_video(
    "video2_fixed_k_sweep_ft.gif",
    r"Sweeping the tuning frequency ($10$ to $160$ Hz) at fixed stiffness; mass follows $m = k\,/\,(2\pi f_t)^2$",
    f"Case A: stiffness {k_a:,.0f} N/m",
    f"Case B: stiffness {k_b:,.0f} N/m",
    series_a, series_b, "f_t",
)

# ------------------------------------------------------------------
# Video 3: fixed f_t (two cases), sweep m over [m_min, m_max], k derived
# ------------------------------------------------------------------
ft_a, ft_b = 30.0, 130.0
series_a = [(m, m * (2 * np.pi * ft_a) ** 2, ft_a) for m in m_sweep]
series_b = [(m, m * (2 * np.pi * ft_b) ** 2, ft_b) for m in m_sweep]
make_video(
    "video3_fixed_ft_sweep_mass.gif",
    f"Sweeping the resonator mass (${m_min:g}$ to ${m_max:g}$ kg) at fixed tuning frequency; "
    r"stiffness follows $k = m\,(2\pi f_t)^2$",
    f"Case A: tuning frequency {ft_a:.0f} Hz",
    f"Case B: tuning frequency {ft_b:.0f} Hz",
    series_a, series_b, "m",
)

# ------------------------------------------------------------------
# Video 4: fixed f_t (two cases), sweep k over practical range, m derived
# ------------------------------------------------------------------
series_a = [(k / (2 * np.pi * ft_a) ** 2, k, ft_a) for k in k_sweep]
series_b = [(k / (2 * np.pi * ft_b) ** 2, k, ft_b) for k in k_sweep]
make_video(
    "video4_fixed_ft_sweep_k.gif",
    f"Sweeping the resonator stiffness (${K_PRACTICAL_LOW:g}$ to ${K_PRACTICAL_HIGH:,.0f}$ N/m) at fixed tuning frequency; "
    r"mass follows $m = k\,/\,(2\pi f_t)^2$",
    f"Case A: tuning frequency {ft_a:.0f} Hz",
    f"Case B: tuning frequency {ft_b:.0f} Hz",
    series_a, series_b, "k",
)

print("ALL DONE")
