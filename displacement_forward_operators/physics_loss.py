"""Physics-informed loss for the displacement forward operators: a MIXED-
FORMULATION residual on the Kirchhoff-Love plate equation, in VELOCITY
(not displacement), split into two coupled 2nd-order equations instead of
one 4th-order equation.

Why velocity, not displacement: v = i*omega*w is a constant (w.r.t. x, y)
multiple of w, so the same governing equation holds for v as for w --
substituting v = i*omega*w into D*nabla^4(w) - rho*h*omega^2*w = 0 and
multiplying through by i*omega gives exactly
    D*nabla^4(v) - rho*h*omega^2*v = 0
(nabla^4 is linear and i*omega doesn't depend on x, y, so it passes
straight through the spatial derivative). This does NOT reduce the
derivative order needed -- v and w have identical spatial dependence, so
this alone buys nothing computationally. What it does buy: velocity is
what ERP actually needs (ERP = rho*c*integral(|v|^2)), so every
architecture's primary output is now the quantity actually evaluated,
with no intermediate "predict displacement, then derive velocity" step
and no displacement-vs-velocity loss-weighting question left open.

Why the mixed formulation: nabla^4 = nabla^2(nabla^2), the Laplacian
applied twice. Introduce M = nabla^2(v) as a SECOND, independently
learned output (not derived from v by differentiating it further) and
split the one 4th-order equation into two 2nd-order ones:
    (A) M - nabla^2(v) = 0                      [definition of M]
    (B) D*nabla^2(M) - rho*h*omega^2*v = 0       [the actual physics]
Equation A only needs v's own 2nd derivatives (v_xx, v_yy); equation B
only needs M's own 2nd derivatives (M_xx, M_yy) -- M is the network's own
output, so getting M_xx does NOT require differentiating v three more
times. Per component (real or imag): 4 calls for equation A + 4 calls
for equation B = 8, versus 10 for the direct 4th-order route -- a modest
20% fewer calls, but the more important change is that no derivative
CHAIN goes deeper than 2 nested calls (v_x->v_xx) instead of 4
(w_x->w_xx->w_xxx->w_xxxx), which is what actually matters for early-
training numerical stability (a freshly-initialized network's 4th-order
output derivatives can be wild; 2nd-order ones are much better-behaved).

Collocation points are still sampled away from the point-force/resonator
locations (see the previous version's reasoning, unchanged: a Dirac delta
is measure-zero for randomly sampled points, so the homogeneous residual
is exactly correct everywhere else, not an approximation).

D is complex (structural damping), so as before every derivative is
computed on real-valued tensors (v_real, v_imag, M_real, M_imag
independently) and combined algebraically only once every derivative is
already a plain real number -- sidesteps Wirtinger-calculus subtleties
of differentiating a complex output w.r.t. real inputs.

M has no ground-truth label anywhere in this project's dataset (it's a
pure auxiliary field, standard practice in mixed-formulation PINNs) --
only the physics residual ever constrains it, never the data loss.

Honest caveat, same as before: the relative weighting between equation A,
equation B, and the data loss is a reasonable default (each residual
normalized by its own characteristic scale below), not carefully tuned.
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


def _laplacian_of_own_output(w: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """w: (P, 1) real-valued network output. x, y: (P, 1) leaf tensors with
    requires_grad=True. Returns (P, 1) = d^2w/dx^2 + d^2w/dy^2 -- 4
    autograd.grad calls, 2 levels deep at most.
    """
    w_x = torch.autograd.grad(w, x, grad_outputs=torch.ones_like(w), create_graph=True)[0]
    w_xx = torch.autograd.grad(w_x, x, grad_outputs=torch.ones_like(w_x), create_graph=True)[0]
    w_y = torch.autograd.grad(w, y, grad_outputs=torch.ones_like(w), create_graph=True)[0]
    w_yy = torch.autograd.grad(w_y, y, grad_outputs=torch.ones_like(w_y), create_graph=True)[0]
    return w_xx + w_yy


def make_physics_loss_fn(
    norm_params: Mapping[str, object],
    num_configs: int = 64,
    num_res: int | None = None,
    points_per_step: int = 32,
    seed: int = 727,
) -> Callable[[nn.Module], torch.Tensor]:
    """Builds ``physics_loss_fn(model) -> scalar tensor``.

    Same collocation strategy as before: a fixed pool of ``num_configs``
    LHS-sampled configurations built once, ``points_per_step`` fresh
    (configuration, x, y, one frequency) points resampled every call.
    ``model`` is expected to output 4 channels now:
    ``(v_real, v_imag, M_real, M_imag)``.
    """
    num_res = int(norm_params["num_res"]) if num_res is None else int(num_res)
    bounds = resonator_bounds(num_res)
    design_samples = lhs_sampling(num_configs, bounds=bounds, seed=seed)

    configurations_physical = np.empty((num_configs, num_res, 5), dtype=np.float32)
    for i in range(num_configs):
        quad = design_samples[i].reshape(num_res, 4)
        x_r, y_r, f_t, m = quad[:, 0], quad[:, 1], quad[:, 2], quad[:, 3]
        k = m * (2.0 * np.pi * f_t) ** 2
        configurations_physical[i] = np.stack([m, k, f_t, x_r, y_r], axis=1)

    rng = np.random.default_rng(seed + 1)
    velocity_scale = float(norm_params["velocity_std"])
    # M = laplacian(v) has units of [velocity]/[length]^2; with no direct
    # label to calibrate against, a characteristic scale is approximated
    # from velocity's own scale over the plate's shorter dimension --
    # a heuristic, not a derived constant (see module docstring's caveat).
    m_scale = velocity_scale / (min(Lx, Ly) ** 2)

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

        pred = model(configuration_t, frequency_t, x_t, y_t)  # (P, 1, 4)
        v_real = pred[:, 0, 0:1]
        v_imag = pred[:, 0, 1:2]
        m_real = pred[:, 0, 2:3]
        m_imag = pred[:, 0, 3:4]

        # Physical units: network outputs are normalized, so scale up by
        # the same constant factor each represents (a constant multiplier
        # commutes with differentiation, same reasoning as basis_flow/
        # padding_inn's normalized-space losses elsewhere in this project).
        v_real_phys = v_real * velocity_scale
        v_imag_phys = v_imag * velocity_scale
        m_real_phys = m_real * m_scale
        m_imag_phys = m_imag * m_scale

        lap_v_real = _laplacian_of_own_output(v_real, x_t, y_t) * velocity_scale
        lap_v_imag = _laplacian_of_own_output(v_imag, x_t, y_t) * velocity_scale
        lap_m_real = _laplacian_of_own_output(m_real, x_t, y_t) * m_scale
        lap_m_imag = _laplacian_of_own_output(m_imag, x_t, y_t) * m_scale

        # (A) M - laplacian(v) = 0 -- pure definition, no physical constants.
        residual_a_real = m_real_phys - lap_v_real
        residual_a_imag = m_imag_phys - lap_v_imag

        # (B) D*laplacian(M) - rho*h*omega^2*v = 0, D complex.
        omega_t = torch.from_numpy((2.0 * np.pi * freqs_hz).astype(np.float32)).to(model_device)[:, None]
        omega_sq = omega_t**2
        residual_b_real = D_REAL * lap_m_real - D_IMAG * lap_m_imag - MASS_PER_AREA * omega_sq * v_real_phys
        residual_b_imag = D_REAL * lap_m_imag + D_IMAG * lap_m_real - MASS_PER_AREA * omega_sq * v_imag_phys

        # Normalize each residual by its own characteristic scale.
        scale_a = m_scale + 1e-8
        scale_b = MASS_PER_AREA * omega_sq * velocity_scale + 1e-8
        loss_a = (residual_a_real / scale_a) ** 2 + (residual_a_imag / scale_a) ** 2
        loss_b = (residual_b_real / scale_b) ** 2 + (residual_b_imag / scale_b) ** 2

        return (loss_a + loss_b).mean()

    return physics_loss_fn
