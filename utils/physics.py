"""
==================================================
Project     : Vibro-Acoustic Metamaterials
Module      : Physics
Description : Physical parameters and modal basis
==================================================
"""

from __future__ import annotations

import numpy as np

from utils.support import grid


# ==================================================
# Plate Geometry
# ==================================================

Lx = 1.4
Ly = 0.5
h = 0.005

nx, ny = 280, 100
X_grid, Y_grid, dA = grid(Lx=Lx, Ly=Ly, nx=nx, ny=ny)


# ==================================================
# Material Properties
# ==================================================

E = 7.1e10 * (1.0 + 1j * 0.001)
rho = 2800.0
nu = 0.3
D = E * h**3 / (12.0 * (1.0 - nu**2))
M_plate = rho * h * Lx * Ly


# ==================================================
# Acoustic Properties
# ==================================================

rho_L = 1.21
c_L = 343.0
P_ref = 1e-12


# ==================================================
# External Excitation / Measurement Point
# ==================================================

F0 = 1.0

xf = 0.865
yf = 0.309

x_meas = Lx - xf
y_meas = Ly - yf


# ==================================================
# Frequency Range
# ==================================================

fmin = 10.0
fmax = 160.0
df = 0.5

freqs = np.arange(fmin, fmax + 0.5 * df, df, dtype=np.float64)
n_freqs = freqs.size


# ==================================================
# Resonator Parameters
# ==================================================

# Legacy fixed stiffness -- no longer used by the main ERP dataset generator
# (m and k are now independently sampled per resonator, see resonator_bounds
# below), kept only for standalone scripts (e.g. others/position_dependency.py)
# that still derive mass from a single fixed k.
k = 584000
num_res = 3

edge_margin = 0.05

# Practical tuned-mass-damper mass range: ~1-10% of the plate's own mass
# (M_plate = rho*h*Lx*Ly ~= 9.8 kg), the standard mass-ratio guideline for a
# lightweight vibration absorber that doesn't dominate the host structure.
m_min = 0.1
m_max = 1.0


def resonator_bounds(n_res: int = num_res) -> np.ndarray:
    """Return [x, y, f_t, m] bounds repeated for ``n_res`` resonators.

    ``f_t`` and ``m`` are sampled as the two independent primary quantities
    (LHS on both directly), and resonator stiffness ``k = m*(2*pi*f_t)**2``
    is derived afterward -- sampling ``m``/``k`` directly instead would make
    the resulting f_t distribution skewed rather than uniform, since
    f_t = (1/2pi)*sqrt(k/m) is a nonlinear function of two independently
    sampled quantities.
    """
    if n_res <= 0:
        raise ValueError("n_res must be positive.")
    single = np.array(
        [
            [edge_margin, Lx - edge_margin],
            [edge_margin, Ly - edge_margin],
            [fmin, fmax],
            [m_min, m_max],
        ],
        dtype=np.float64,
    )
    return np.tile(single, (n_res, 1))


bounds = resonator_bounds(num_res)


# ==================================================
# Modal Expansion Parameters
# ==================================================

Nx = 15
Ny = 10


def omega_mn(m: int, n: int) -> complex:
    kx = m * np.pi / Lx
    ky = n * np.pi / Ly
    return np.sqrt(D / (rho * h)) * (kx**2 + ky**2)


def phi_mn(m: int, n: int, x: np.ndarray | float, y: np.ndarray | float) -> np.ndarray:
    norm = 2.0 / np.sqrt(M_plate)
    return norm * np.sin(m * np.pi * np.asarray(x) / Lx) * np.sin(
        n * np.pi * np.asarray(y) / Ly
    )


def modal_table(nx_modes: int, ny_modes: int) -> np.ndarray:
    modes_list: list[list[float]] = [
        [float(np.real(omega_mn(m, n))), float(m), float(n)]
        for m in range(1, nx_modes + 1)
        for n in range(1, ny_modes + 1)
    ]
    return np.asarray(sorted(modes_list, key=lambda row: row[0]), dtype=np.float64)


modes = modal_table(nx_modes=Nx, ny_modes=Ny)
omega_n = modes[:, 0]
omega_sq = omega_n**2
m_idx = modes[:, 1].astype(np.int32)
n_idx = modes[:, 2].astype(np.int32)
N = len(modes)


def mode_shapes(x: np.ndarray | float, y: np.ndarray | float) -> np.ndarray:
    """Return modal basis with shape ``(N_modes, N_points)``."""
    x_arr = np.atleast_1d(np.asarray(x, dtype=np.float64))
    y_arr = np.atleast_1d(np.asarray(y, dtype=np.float64))
    if x_arr.size != y_arr.size:
        raise ValueError("x and y must contain the same number of points.")

    norm = 2.0 / np.sqrt(M_plate)
    return norm * np.sin(np.pi * np.outer(m_idx, x_arr) / Lx) * np.sin(
        np.pi * np.outer(n_idx, y_arr) / Ly
    )
