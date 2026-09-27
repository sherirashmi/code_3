"""
==================================================
Project     : Vibro-Acoustic Metamaterials
Module      : Solver
Description : Coupled plate-resonator solver
==================================================
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from scipy.signal import find_peaks

from utils.physics import (
    F0,
    N,
    P_ref,
    X_grid,
    Y_grid,
    c_L,
    dA,
    freqs,
    mode_shapes,
    omega_n,
    rho_L,
    xf,
    yf,
)


# Cached full-grid modal basis and its spatial Gram matrix.
_FULL_GRID_PHI: np.ndarray | None = None
_FULL_GRID_GRAM: np.ndarray | None = None


# ==================================================
# Coupled Matrices
# ==================================================

def compute_coupled_matrices(
    resonators: Sequence[dict[str, float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Construct modal mass, damping, stiffness and forcing arrays."""
    n_res = len(resonators)
    n_dofs = N + n_res

    M = np.zeros((n_dofs, n_dofs), dtype=np.complex128)
    C = np.zeros_like(M)
    K = np.zeros_like(M)

    M[:N, :N] = np.eye(N, dtype=np.complex128)
    K[:N, :N] = np.diag(omega_n**2)

    f = np.zeros(n_dofs, dtype=np.complex128)
    f[:N] = F0 * mode_shapes(xf, yf).reshape(N)

    if not resonators:
        return M, C, K, f

    phi_res = mode_shapes(
        np.asarray([r["x"] for r in resonators], dtype=np.float64),
        np.asarray([r["y"] for r in resonators], dtype=np.float64),
    )

    for i, res in enumerate(resonators):
        phi = phi_res[:, i]
        m_i = float(res["m"])
        c_i = float(res.get("c", 1.0))
        k_i = float(res["k"])
        dof = N + i

        M[dof, dof] = m_i

        C[:N, :N] += c_i * np.outer(phi, phi)
        C[:N, dof] = -c_i * phi
        C[dof, :N] = -c_i * phi
        C[dof, dof] = c_i

        K[:N, :N] += k_i * np.outer(phi, phi)
        K[:N, dof] = -k_i * phi
        K[dof, :N] = -k_i * phi
        K[dof, dof] = k_i

    return M, C, K, f


# ==================================================
# Modal Solver
# ==================================================

def solve_modal_response(
    resonators: Sequence[dict[str, float]],
    frequencies: np.ndarray | float = freqs,
) -> np.ndarray:
    """Solve coupled modal coordinates for one or many frequencies.

    Returns
    -------
    np.ndarray
        Plate modal coordinates with shape ``(n_frequencies, N)``.
    """
    frequencies_arr = np.atleast_1d(np.asarray(frequencies, dtype=np.float64))
    omega = 2.0 * np.pi * frequencies_arr

    M, C, K, rhs = compute_coupled_matrices(resonators)
    q_plate = np.empty((frequencies_arr.size, N), dtype=np.complex128)

    # Frequency-by-frequency solve keeps memory bounded and is typically
    # faster overall than repeatedly rebuilding the coupled matrices.
    for i, omega_i in enumerate(omega):
        dynamic_stiffness = K + 1j * omega_i * C - omega_i**2 * M
        q_plate[i] = np.linalg.solve(dynamic_stiffness, rhs)[:N]

    return q_plate


# ==================================================
# Displacement Solver
# ==================================================

def _get_full_grid_phi() -> np.ndarray:
    global _FULL_GRID_PHI
    if _FULL_GRID_PHI is None:
        _FULL_GRID_PHI = mode_shapes(X_grid.ravel(), Y_grid.ravel())
    return _FULL_GRID_PHI


def compute_displacement(
    resonators: Sequence[dict[str, float]],
    x: np.ndarray | float | None = None,
    y: np.ndarray | float | None = None,
    frequencies: np.ndarray | float = freqs,
) -> np.ndarray:
    """Compute complex plate displacement at a point, points, or full grid.

    Usage
    -----
    ``compute_displacement(resonators)``
        Returns the complete grid with shape ``(ny, nx, n_frequencies)``.

    ``compute_displacement(resonators, x=x_meas, y=y_meas)``
        Returns one-point displacement with shape ``(n_frequencies,)``.

    ``compute_displacement(resonators, x=X, y=Y)``
        Returns displacement on arbitrary same-shaped coordinate arrays with
        shape ``X.shape + (n_frequencies,)``.
    """
    frequencies_arr = np.atleast_1d(np.asarray(frequencies, dtype=np.float64))
    q_plate = solve_modal_response(resonators, frequencies=frequencies_arr)

    if x is None and y is None:
        phi_eval = _get_full_grid_phi()
        displacement = q_plate @ phi_eval
        return displacement.reshape(
            frequencies_arr.size, *X_grid.shape
        ).transpose(1, 2, 0)

    if (x is None) != (y is None):
        raise ValueError("x and y must either both be provided or both be omitted.")

    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if x_arr.shape != y_arr.shape:
        raise ValueError("x and y must have the same shape.")

    scalar_point = x_arr.ndim == 0
    original_shape = x_arr.shape
    phi_eval = mode_shapes(x_arr.ravel(), y_arr.ravel())
    displacement = q_plate @ phi_eval

    if scalar_point:
        return displacement[:, 0]

    return displacement.reshape(frequencies_arr.size, *original_shape).transpose(
        *range(1, len(original_shape) + 1), 0
    )


# ==================================================
# Velocity
# ==================================================

def compute_velocity(
    displacement: np.ndarray,
    frequencies: np.ndarray | float = freqs,
) -> np.ndarray:
    """Convert displacement to harmonic velocity for any spatial shape."""
    displacement = np.asarray(displacement)
    frequencies_arr = np.atleast_1d(np.asarray(frequencies, dtype=np.float64))
    if displacement.shape[-1] != frequencies_arr.size:
        raise ValueError(
            "The last displacement dimension must match the number of frequencies."
        )

    omega_shape = (1,) * (displacement.ndim - 1) + (frequencies_arr.size,)
    omega = (2.0 * np.pi * frequencies_arr).reshape(omega_shape)
    return 1j * omega * displacement


# ==================================================
# Equivalent Radiated Power
# ==================================================

def compute_erp(velocity_field: np.ndarray) -> np.ndarray:
    """Compute ERP from a velocity field whose last axis is frequency."""
    velocity_field = np.asarray(velocity_field)
    if velocity_field.ndim < 3:
        raise ValueError(
            "ERP requires a spatial velocity field, e.g. shape (ny, nx, n_freqs)."
        )

    spatial_axes = tuple(range(velocity_field.ndim - 1))
    velocity_energy = np.sum(np.abs(velocity_field) ** 2, axis=spatial_axes) * dA
    power = 0.5 * rho_L * c_L * velocity_energy
    return 10.0 * np.log10(np.maximum(power, np.finfo(float).tiny) / P_ref)


def _get_full_grid_gram() -> np.ndarray:
    """Return cached Phi Phi^T dA used for direct ERP evaluation."""
    global _FULL_GRID_GRAM
    if _FULL_GRID_GRAM is None:
        phi_grid = _get_full_grid_phi()
        _FULL_GRID_GRAM = (phi_grid @ phi_grid.T) * dA
    return _FULL_GRID_GRAM


def compute_erp_spectrum(
    resonators: Sequence[dict[str, float]],
    frequencies: np.ndarray | float = freqs,
) -> np.ndarray:
    """Compute ERP without constructing the full displacement/velocity cube.

    This is algebraically equivalent to computing the full grid first, but it
    uses a cached modal spatial Gram matrix and therefore uses much less memory.
    """
    frequencies_arr = np.atleast_1d(np.asarray(frequencies, dtype=np.float64))
    q_plate = solve_modal_response(resonators, frequencies=frequencies_arr)
    gram = _get_full_grid_gram()

    displacement_energy = np.einsum(
        "fi,ij,fj->f", np.conjugate(q_plate), gram, q_plate, optimize=True
    ).real
    velocity_energy = (2.0 * np.pi * frequencies_arr) ** 2 * displacement_energy
    power = 0.5 * rho_L * c_L * velocity_energy
    return 10.0 * np.log10(np.maximum(power, np.finfo(float).tiny) / P_ref)


# ==================================================
# Peak Detection
# ==================================================

def get_peak_frequencies(
    erp: np.ndarray,
    frequencies: np.ndarray = freqs,
    prominence: float = 3.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    peak_idx, _ = find_peaks(erp, prominence=prominence)
    return frequencies[peak_idx], erp[peak_idx], peak_idx


# ==================================================
# Resonator Field
# ==================================================

def gaussian_delta(
    X: np.ndarray,
    Y: np.ndarray,
    x0: float,
    y0: float,
    sigma: float = 0.01,
) -> np.ndarray:
    return np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2.0 * sigma**2)) / (
        2.0 * np.pi * sigma**2
    )


def get_resonator_field(
    resonators: Sequence[dict[str, float]],
    X: np.ndarray = X_grid,
    Y: np.ndarray = Y_grid,
) -> np.ndarray:
    field = np.zeros_like(X, dtype=np.float64)
    for res in resonators:
        field += float(res["m"]) * gaussian_delta(X, Y, res["x"], res["y"])
    return field


# ==================================================
# CNN Input Preparation
# ==================================================

def create_cnn_input(resonator_field: torch.Tensor) -> torch.Tensor:
    field_min = torch.min(resonator_field)
    field_max = torch.max(resonator_field)
    normalized = (resonator_field - field_min) / (field_max - field_min + 1e-12)
    return normalized.unsqueeze(0)
