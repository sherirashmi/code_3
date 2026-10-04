"""Design parameterisations for the inverse models.

Two ways to represent one design (``num_res`` resonators) as the flat
vector a model predicts / scores (sizes below are for 3 resonators):

``full15`` (legacy)
    ``[m, k, f_t, x, y]`` per resonator, each z-scored -> 15 numbers. ``k``
    is redundant (``k = m (2 pi f_t)^2`` in the data), so a model can output
    inconsistent ``(m, k, f_t)`` triples and out-of-range values; predicted
    designs are therefore projected afterwards (``common.enforce_physical_consistency``).

``bounded12``
    Only the physically independent variables ``[m, f_t, x, y]`` per
    resonator -> 12 numbers. Each variable ``v`` with generation bounds
    ``[lo, hi]`` is mapped to an unbounded coordinate

        u = (v - lo) / (hi - lo),   w = logit(u),   flat = (w - mu_w) / sd_w

    (``mu_w``/``sd_w`` fitted on the training split). The inverse map
    ``v = lo + (hi - lo) * sigmoid(sd_w * flat + mu_w)`` lands strictly
    inside the bounds for ANY model output, and ``k`` is derived exactly,
    so no projection/clipping is ever needed: the design that is scored is
    exactly the design that is evaluated, and model densities stay exact.

    A density over ``flat`` converts to a density over the physical
    variables ``(m, f_t, x, y)`` with the change of variables

        log p_phys(v) = log p_flat(flat) + sum_i log |d flat_i / d v_i|,
        d flat / d v  = 1 / (sd_w * u (1 - u) * (hi - lo)).
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import torch

from utils.physics import Lx, Ly, fmax, fmin, m_max, m_min

FULL15 = "full15"
BOUNDED12 = "bounded12"
DESIGN_PARAMS = (FULL15, BOUNDED12)

# (field name, index in [m, k, f_t, x, y], lower bound, upper bound)
BOUNDED_FIELDS = (
    ("m", 0, m_min, m_max),
    ("f_t", 2, fmin, fmax),
    # Positions are bounded by the plate itself, not the 5 cm sampling margin:
    # grid datasets put resonators exactly on the margin (x = 0.05, 1.35 m),
    # which would sit on the logit's singularity with margin bounds.
    ("x", 3, 0.0, Lx),
    ("y", 4, 0.0, Ly),
)
_EPS = 1e-6


def design_dim(design_param: str, num_res: int) -> int:
    if design_param not in DESIGN_PARAMS:
        raise ValueError(f"design_param must be one of {DESIGN_PARAMS}, got {design_param!r}.")
    return num_res * (4 if design_param == BOUNDED12 else 5)


def _unit(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip((values - lo) / (hi - lo), _EPS, 1.0 - _EPS)


def fit_bounded_stats(raw_configurations: np.ndarray) -> dict[str, float]:
    """Logit-space mean/std per field from TRAINING designs ``(N, R, 5)``;
    stored in ``norm_params`` (keys ``b12_<field>_mean/_std``) so every
    checkpoint carries them."""
    raw = np.asarray(raw_configurations, dtype=np.float64)
    stats: dict[str, float] = {}
    for name, idx, lo, hi in BOUNDED_FIELDS:
        u = _unit(raw[..., idx], lo, hi)
        w = np.log(u) - np.log1p(-u)
        stats[f"b12_{name}_mean"] = float(w.mean())
        stats[f"b12_{name}_std"] = float(max(w.std(), 1e-6))
    return stats


def _stats_tensors(norm_params: Mapping[str, object], device, dtype):
    mean = torch.tensor([float(norm_params[f"b12_{n}_mean"]) for n, *_ in BOUNDED_FIELDS], device=device, dtype=dtype)
    std = torch.tensor([float(norm_params[f"b12_{n}_std"]) for n, *_ in BOUNDED_FIELDS], device=device, dtype=dtype)
    lo = torch.tensor([b[2] for b in BOUNDED_FIELDS], device=device, dtype=dtype)
    hi = torch.tensor([b[3] for b in BOUNDED_FIELDS], device=device, dtype=dtype)
    return mean, std, lo, hi


def encode_bounded(raw_configurations: np.ndarray, norm_params: Mapping[str, object]) -> np.ndarray:
    """Physical ``(..., R, 5)`` -> normalised bounded coordinates ``(..., R, 4)``."""
    raw = np.asarray(raw_configurations, dtype=np.float64)
    out = np.empty(raw.shape[:-1] + (4,), dtype=np.float64)
    for j, (name, idx, lo, hi) in enumerate(BOUNDED_FIELDS):
        u = _unit(raw[..., idx], lo, hi)
        w = np.log(u) - np.log1p(-u)
        out[..., j] = (w - float(norm_params[f"b12_{name}_mean"])) / float(norm_params[f"b12_{name}_std"])
    return out.astype(np.float32)


def decode_bounded_torch(flat: torch.Tensor, num_res: int, norm_params: Mapping[str, object]) -> torch.Tensor:
    """Differentiable ``(..., R*4)`` -> physical ``(..., R, 5)`` [m, k, f_t, x, y]."""
    mean, std, lo, hi = _stats_tensors(norm_params, flat.device, flat.dtype)
    coords = flat.reshape(*flat.shape[:-1], num_res, 4)
    values = lo + (hi - lo) * torch.sigmoid(coords * std + mean)  # (..., R, 4): m, f_t, x, y
    m, f_t, x, y = values.unbind(dim=-1)
    k = m * (2.0 * torch.pi * f_t) ** 2
    return torch.stack((m, k, f_t, x, y), dim=-1)


def decode_bounded(flat: np.ndarray, num_res: int, norm_params: Mapping[str, object]) -> np.ndarray:
    with torch.no_grad():
        return decode_bounded_torch(torch.as_tensor(np.asarray(flat), dtype=torch.float64), num_res, norm_params).numpy().astype(np.float32)


def bounded_log_abs_det(flat: torch.Tensor, num_res: int, norm_params: Mapping[str, object]) -> torch.Tensor:
    """``sum_i log |d flat_i / d v_i|`` for each design ``(...,)``; add it to
    ``log p_flat`` to get the density over physical ``(m, f_t, x, y)``."""
    mean, std, lo, hi = _stats_tensors(norm_params, flat.device, flat.dtype)
    coords = flat.reshape(*flat.shape[:-1], num_res, 4)
    u = torch.sigmoid(coords * std + mean).clamp(_EPS, 1 - _EPS)
    log_dflat_dv = -torch.log(std) - torch.log(u) - torch.log1p(-u) - torch.log(hi - lo)
    return log_dflat_dv.sum(dim=(-1, -2))


__all__ = [
    "FULL15",
    "BOUNDED12",
    "DESIGN_PARAMS",
    "design_dim",
    "fit_bounded_stats",
    "encode_bounded",
    "decode_bounded",
    "decode_bounded_torch",
    "bounded_log_abs_det",
]
