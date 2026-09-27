"""Render a video: fixed base 3-resonator configuration, all 15 design
values (m,k,f_t,x,y per resonator) perturbed by smoothly evolving noise
frame-by-frame, showing how the TRUE ERP spectrum (real coupled-plate
solver, not a surrogate) responds -- left panel ERP, right panel plate
layout, exactly the two-panel convention already used by utils.plotting's
plot_erp()/plot_erp_comparison().

f_t is not consumed by the solver itself (compute_coupled_matrices only
uses m, k, x, y) -- it's derived here as sqrt(k/m)/(2*pi) purely so the
displayed tuning-line/annotation always matches the actual m,k being
solved (matching the dataset's own convention), rather than drifting
independently and becoming a label that lies about the real resonance.
"""
from __future__ import annotations

import sys
import time

sys.path.insert(0, "/home/user/code_3")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import imageio.v2 as imageio

from utils.physics import Lx, Ly, edge_margin, fmax, fmin, freqs, m_max, m_min
from utils.solver import compute_erp_spectrum
from utils.plotting import _draw_plate_layout, _mark_resonator_tuning_lines

OUT_PATH = "dataset_analysis/videos/erp_config_noise_sensitivity.mp4"

rng = np.random.default_rng(42)

NUM_RES = 3
N_FRAMES = 150
FPS = 15
RAMP_FRAMES = 30  # noise amplitude ramps 0 -> full over this many frames

# Physical bounds (same convention as surrogate_inverse.py's _DESIGN_PHYSICAL_BOUNDS)
M_LO, M_HI = m_min, m_max
X_LO, X_HI = edge_margin, Lx - edge_margin
Y_LO, Y_HI = edge_margin, Ly - edge_margin
K_LO, K_HI = m_min * (2 * np.pi * fmin) ** 2, m_max * (2 * np.pi * fmax) ** 2
LOGK_LO, LOGK_HI = np.log(K_LO), np.log(K_HI)

# Fixed base configuration (three resonators tuned across the band)
base_m = np.array([0.3, 0.5, 0.7])
base_ft = np.array([30.0, 70.0, 120.0])
base_k = base_m * (2 * np.pi * base_ft) ** 2
base_x = np.array([0.30, 0.70, 1.10])
base_y = np.array([0.15, 0.35, 0.25])

# OU (mean-reverting random walk) parameters: theta = reversion rate.
# For m/x/y, sigma is a small fraction of that field's own GLOBAL bound
# range. For k, a fraction of the global bound range would be huge (k
# spans ~2500x from lo to hi), so it instead gets an absolute log-space
# sigma representing a small RELATIVE (multiplicative) wobble around each
# resonator's own base k -- e.g. sigma_log=0.015 with theta=0.10 settles
# to a steady-state relative fluctuation of roughly exp(0.015/sqrt(2*0.10
# - 0.10**2)) - 1 ~= 3.5%, not a fraction of the full dataset k range.
THETA = 0.10
SIGMA_FRAC = {"m": 0.008, "x": 0.008, "y": 0.008}
SIGMA_LOGK_ABS = 0.015


def _ou_trajectory(base, lo, hi, sigma, n_frames, ramp_frames, rng, n_res, sigma_is_frac=True):
    if sigma_is_frac:
        sigma = sigma * (hi - lo)
    traj = np.empty((n_frames, n_res))
    current = np.array(base, dtype=np.float64).copy()
    for t in range(n_frames):
        ramp = min(1.0, t / max(1, ramp_frames))
        noise = rng.normal(0.0, sigma, size=n_res) * ramp
        current = current + THETA * (np.asarray(base) - current) + noise
        current = np.clip(current, lo, hi)
        traj[t] = current
    return traj


m_traj = _ou_trajectory(base_m, M_LO, M_HI, SIGMA_FRAC["m"], N_FRAMES, RAMP_FRAMES, rng, NUM_RES)
logk_traj = _ou_trajectory(
    np.log(base_k), LOGK_LO, LOGK_HI, SIGMA_LOGK_ABS, N_FRAMES, RAMP_FRAMES, rng, NUM_RES,
    sigma_is_frac=False,
)
x_traj = _ou_trajectory(base_x, X_LO, X_HI, SIGMA_FRAC["x"], N_FRAMES, RAMP_FRAMES, rng, NUM_RES)
y_traj = _ou_trajectory(base_y, Y_LO, Y_HI, SIGMA_FRAC["y"], N_FRAMES, RAMP_FRAMES, rng, NUM_RES)
k_traj = np.exp(logk_traj)
ft_traj = np.sqrt(k_traj / m_traj) / (2 * np.pi)  # derived, display-only

print(f"Solving {N_FRAMES} configurations with the real coupled-plate solver...")
t_start = time.time()
erp_frames = np.empty((N_FRAMES, freqs.size))
config_frames = np.empty((N_FRAMES, NUM_RES, 5))  # [m,k,f_t,x,y]
for t in range(N_FRAMES):
    resonators = [
        {"m": m_traj[t, i], "k": k_traj[t, i], "x": x_traj[t, i], "y": y_traj[t, i]}
        for i in range(NUM_RES)
    ]
    erp_frames[t] = compute_erp_spectrum(resonators, frequencies=freqs)
    config_frames[t] = np.stack(
        [m_traj[t], k_traj[t], ft_traj[t], x_traj[t], y_traj[t]], axis=-1
    )
    if (t + 1) % 25 == 0 or t == N_FRAMES - 1:
        elapsed = time.time() - t_start
        print(f"  frame {t + 1}/{N_FRAMES}  ({elapsed:.1f}s elapsed)")

erp_lo, erp_hi = erp_frames.min(), erp_frames.max()
pad = 0.08 * (erp_hi - erp_lo)
YLIM = (erp_lo - pad, erp_hi + pad)
print(f"ERP range across all frames: [{erp_lo:.2f}, {erp_hi:.2f}] dB")

print("Rendering frames...")
t_render_start = time.time()
writer = imageio.get_writer(OUT_PATH, fps=FPS, codec="libx264", quality=8, macro_block_size=1)
for t in range(N_FRAMES):
    fig, (ax, ax_plate) = plt.subplots(
        1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1.6, 1]}
    )

    ax.plot(freqs, erp_frames[t], lw=2, color="#4C72B0", label="ERP")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("ERP (dB)")
    ax.set_xlim(freqs.min(), freqs.max())
    ax.set_ylim(*YLIM)
    ax.set_title(f"ERP Spectrum -- frame {t + 1}/{N_FRAMES}")
    ax.grid(True, alpha=0.4)
    _mark_resonator_tuning_lines(ax, config_frames[t])
    ax.legend(loc="upper right", fontsize=9)

    _draw_plate_layout(ax_plate, config_frames[t])

    ramp = min(1.0, t / max(1, RAMP_FRAMES))
    fig.suptitle(
        f"Real-solver ERP response to noise on all 15 design values (noise amplitude x{ramp:.2f})",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    writer.append_data(image)
    plt.close(fig)

    if (t + 1) % 25 == 0 or t == N_FRAMES - 1:
        elapsed = time.time() - t_render_start
        print(f"  rendered {t + 1}/{N_FRAMES}  ({elapsed:.1f}s elapsed)")

writer.close()
print(f"Saved video: {OUT_PATH}")
