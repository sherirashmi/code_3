"""Physics-informed loss for the displacement forward operators: a residual
on the Kirchhoff-Love plate equation, evaluated at randomly sampled
collocation points AWAY from any point-force/resonator-coupling location
(where the true PDE has a Dirac-delta forcing term this residual doesn't
include -- see below for why that's the correct thing to do, not a
shortcut).

Governing equation (utils/physics.py's D, rho, h implement exactly this;
utils/solver.py's compute_coupled_matrices is the same physics in modal
form):
    D * nabla^4(w) + rho*h*w_ddot =
        f_d(t) * delta(x-x_d) * delta(y-y_d)
        - sum_i delta(x-x_si) * delta(y-y_si) * [k_si*(w-u_si) + c_si*(w_dot-u_si_dot)]

Under harmonic time dependence (w_ddot = -omega^2 * w) and away from every
delta-supported term, this reduces to the homogeneous residual actually
evaluated here:
    D * nabla^4(w) - rho*h*omega^2*w  =  0

This is not a hand-wave: a Dirac delta has zero density everywhere except
exactly at its own point, so for randomly sampled collocation points, the
force and every resonator-coupling term vanish almost surely, leaving the
homogeneous equation as the correct residual everywhere else on the plate
-- exactly the same reasoning classical PINN collocation methods use for
point-forced problems (residual points away from singularities; the
singular behavior itself is supplied by the boundary/data terms instead).

Why the residual is split into real/imaginary parts by hand, not evaluated
with torch.complex: D is complex here (D = E*h^3/(12*(1-nu^2)), E =
E0*(1+1j*eta) -- this project's structural-damping convention, see
utils/physics.py), and torch.autograd.grad's handling of a complex-valued
output differentiated w.r.t. REAL inputs (x, y) needs Wirtinger-calculus
care that's easy to get subtly wrong. Computing every spatial derivative
on the two REAL tensors w_real, w_imag independently, and only combining
them algebraically at the very end (once every derivative is already a
plain real number), sidesteps that entirely:
    D*w = (D_r + i*D_i)*(w_r + i*w_i)
        = (D_r*w_r - D_i*w_i) + i*(D_r*w_i + D_i*w_r)
applied to nabla^4(w) exactly the same way (nabla^4 is linear, so it
commutes with taking real/imaginary parts).

nabla^4(w) = d^4w/dx^4 + 2*d^4w/dx^2dy^2 + d^4w/dy^4 (the biharmonic
operator), computed via 8 nested torch.autograd.grad(..., create_graph=True)
calls per real-valued component (w_real, w_imag) -- 16 total per
collocation point, expensive, which is why this uses far fewer collocation
points per training step than the data loss (matching the reference
paper's own physics-collocation set being much smaller than its data-
training set).

Honest caveat: an untrained network's high-order derivatives can be wild
early in training (a classical PINN pain point), and this residual's raw
physical scale is dominated by the inertial term rho*h*omega^2*w (~1e7 at
the top of this project's 160 Hz band) versus the bending term D*nabla^4(w)
(~1e-3 to 1, depending on local curvature) -- normalized below by the
inertial term's own characteristic scale so the loss stays O(1)-ish, but
this has NOT been carefully tuned (no warmup schedule, no adaptive
weighting); it's a reasonable default, verified only to run without
producing NaNs on a short smoke run, not a carefully validated weighting.
"""

from __future__ import annotations

from typing import Callable, Mapping

import numpy as np
import torch
import torch.nn as nn

from utils.erp_dataset import normalize_configuration_array, normalize_frequency_array
from utils.physics import D, Lx, Ly, edge_margin, fmax, fmin, h, resonator_bounds, rho
from utils.support import lhs_sampling

D_REAL = float(D.real)
D_IMAG = float(D.imag)
MASS_PER_AREA = float(rho * h)


def _biharmonic(w: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """w: (P, 1) real-valued network output for one component (real or
    imag). x, y: (P, 1) leaf tensors with requires_grad=True, one
    independent collocation point per row. Returns (P, 1) =
    d^4w/dx^4 + 2*d^4w/dx^2dy^2 + d^4w/dy^4.
    """
    ones = torch.ones_like(w)
    w_x = torch.autograd.grad(w, x, grad_outputs=ones, create_graph=True)[0]
    w_xx = torch.autograd.grad(w_x, x, grad_outputs=torch.ones_like(w_x), create_graph=True)[0]
    w_xxx = torch.autograd.grad(w_xx, x, grad_outputs=torch.ones_like(w_xx), create_graph=True)[0]
    w_xxxx = torch.autograd.grad(w_xxx, x, grad_outputs=torch.ones_like(w_xxx), create_graph=True)[0]

    w_y = torch.autograd.grad(w, y, grad_outputs=ones, create_graph=True)[0]
    w_yy = torch.autograd.grad(w_y, y, grad_outputs=torch.ones_like(w_y), create_graph=True)[0]
    w_yyy = torch.autograd.grad(w_yy, y, grad_outputs=torch.ones_like(w_yy), create_graph=True)[0]
    w_yyyy = torch.autograd.grad(w_yyy, y, grad_outputs=torch.ones_like(w_yyy), create_graph=True)[0]

    w_xxy = torch.autograd.grad(w_xx, y, grad_outputs=torch.ones_like(w_xx), create_graph=True)[0]
    w_xxyy = torch.autograd.grad(w_xxy, y, grad_outputs=torch.ones_like(w_xxy), create_graph=True)[0]

    return w_xxxx + 2.0 * w_xxyy + w_yyyy


def make_physics_loss_fn(
    norm_params: Mapping[str, object],
    num_configs: int = 64,
    num_res: int | None = None,
    points_per_step: int = 32,
    seed: int = 727,
) -> Callable[[nn.Module], torch.Tensor]:
    """Builds ``physics_loss_fn(model) -> scalar tensor``.

    A FIXED pool of ``num_configs`` LHS-sampled resonator configurations is
    generated once here (mirroring the reference paper's own held-out
    physics-collocation configuration set, generated once before training
    rather than resampled every step). Each call to the returned function
    re-samples ``points_per_step`` fresh (configuration, x, y, frequency)
    collocation points from that pool -- one frequency per point (not the
    full shared 301-point grid): the linear steady-state harmonic PDE has
    no cross-frequency coupling, so the residual at each frequency is an
    independent equation anyway, and using one frequency per point keeps
    every collocation point's spatial derivatives unambiguous (see
    _biharmonic's docstring) without the extra cost of evaluating a full
    frequency grid per point.
    """
    num_res = int(norm_params["num_res"]) if num_res is None else int(num_res)
    bounds = resonator_bounds(num_res)
    design_samples = lhs_sampling(num_configs, bounds=bounds, seed=seed)  # (num_configs, num_res*4), [x,y,f_t,m]

    configurations_physical = np.empty((num_configs, num_res, 5), dtype=np.float32)
    for i in range(num_configs):
        quad = design_samples[i].reshape(num_res, 4)
        x_r, y_r, f_t, m = quad[:, 0], quad[:, 1], quad[:, 2], quad[:, 3]
        k = m * (2.0 * np.pi * f_t) ** 2
        configurations_physical[i] = np.stack([m, k, f_t, x_r, y_r], axis=1)

    rng = np.random.default_rng(seed + 1)
    disp_scale = float(norm_params["displacement_std"])

    def physics_loss_fn(model: nn.Module) -> torch.Tensor:
        idx = rng.integers(0, num_configs, size=points_per_step)
        config_physical = configurations_physical[idx]
        config_norm = normalize_configuration_array(config_physical, norm_params)

        xs = rng.uniform(edge_margin, Lx - edge_margin, size=(points_per_step, 1)).astype(np.float32)
        ys = rng.uniform(edge_margin, Ly - edge_margin, size=(points_per_step, 1)).astype(np.float32)
        freqs_hz = rng.uniform(fmin, fmax, size=points_per_step).astype(np.float32)
        freq_norm = normalize_frequency_array(freqs_hz, norm_params).astype(np.float32)

        model_device = next(model.parameters()).device
        configuration_t = torch.from_numpy(config_norm).to(model_device)  # (P, num_res, 5)
        frequency_t = torch.from_numpy(freq_norm).to(model_device)[:, None, None]  # (P, 1, 1)
        x_t = torch.from_numpy(xs).to(model_device).requires_grad_(True)  # (P, 1)
        y_t = torch.from_numpy(ys).to(model_device).requires_grad_(True)  # (P, 1)

        pred = model(configuration_t, frequency_t, x_t, y_t)  # (P, 1, 2)
        w_real = pred[:, 0, 0:1]
        w_imag = pred[:, 0, 1:2]

        biharm_real = _biharmonic(w_real, x_t, y_t) * disp_scale
        biharm_imag = _biharmonic(w_imag, x_t, y_t) * disp_scale
        w_real_phys = w_real * disp_scale
        w_imag_phys = w_imag * disp_scale

        omega_t = torch.from_numpy((2.0 * np.pi * freqs_hz).astype(np.float32)).to(model_device)[:, None]
        omega_sq = omega_t**2

        residual_real = D_REAL * biharm_real - D_IMAG * biharm_imag - MASS_PER_AREA * omega_sq * w_real_phys
        residual_imag = D_REAL * biharm_imag + D_IMAG * biharm_real - MASS_PER_AREA * omega_sq * w_imag_phys

        # Normalize by the inertial term's own characteristic scale so the
        # loss stays O(1)-ish regardless of frequency -- see module
        # docstring's caveat about this not being carefully tuned.
        scale = MASS_PER_AREA * omega_sq * disp_scale + 1e-8
        residual_real = residual_real / scale
        residual_imag = residual_imag / scale

        return (residual_real**2 + residual_imag**2).mean()

    return physics_loss_fn
