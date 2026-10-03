"""Position-only design space: [x, y] per resonator, sorted by x.

Every resonator of the dataset has the same, fixed mass and tuning frequency
(``fixed_resonator`` in the dataset registry), so the only unknowns are the
positions -- ``2 * num_res`` numbers. Each coordinate ``v`` in ``[lo, hi]`` (the
plate: ``[0, Lx]`` for x, ``[0, Ly]`` for y) is mapped to an unbounded coordinate

    u = (v - lo) / (hi - lo),   w = logit(u),   flat = (w - mu_w) / sd_w

(``mu_w`` / ``sd_w`` fitted on the training split). The inverse map lands
strictly on the plate for ANY model output, so no clipping is ever needed and
model densities stay exact. A density over ``flat`` converts to a density over
the physical positions with ``sum_i log |d flat_i / d v_i|`` (see
:func:`position_log_abs_det`).

Ordering: the resonators are identical, so swapping two of them gives exactly
the same ERP. Sorting them by ascending x removes this labelling ambiguity
(sorting by f_t, as in ``erp_inverse_operators``, cannot -- every f_t is the
same). The logit map is monotonic, so the order also holds in ``flat`` space.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import torch

from utils.physics import Lx, Ly

# (field name, index in [m, k, f_t, x, y], lower bound, upper bound)
POSITION_FIELDS = (("x", 3, 0.0, Lx), ("y", 4, 0.0, Ly))
_EPS = 1e-6


def design_dim(num_res: int) -> int:
    return 2 * int(num_res)


def sort_by_x(configuration: np.ndarray) -> np.ndarray:
    """``(..., num_res, C)`` physical designs with resonators in ascending x
    (index 3 for [m, k, f_t, x, y] rows, index 0 for [x, y] rows)."""
    configuration = np.asarray(configuration)
    x_index = 3 if configuration.shape[-1] == 5 else 0
    order = np.argsort(configuration[..., x_index], axis=-1, kind="stable")
    return np.take_along_axis(configuration, order[..., None], axis=-2)


def _unit(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip((values - lo) / (hi - lo), _EPS, 1.0 - _EPS)


def fit_position_stats(raw_configurations: np.ndarray) -> dict[str, float]:
    """Logit-space mean / std of x and y from TRAINING designs ``(N, R, 5)``;
    stored in ``norm_params`` (keys ``pos_<field>_mean/_std``)."""
    raw = np.asarray(raw_configurations, dtype=np.float64)
    stats: dict[str, float] = {}
    for name, idx, lo, hi in POSITION_FIELDS:
        u = _unit(raw[..., idx], lo, hi)
        w = np.log(u) - np.log1p(-u)
        stats[f"pos_{name}_mean"] = float(w.mean())
        stats[f"pos_{name}_std"] = float(max(w.std(), 1e-6))
    return stats


def _stats_tensors(norm_params: Mapping[str, object], device, dtype):
    mean = torch.tensor([float(norm_params[f"pos_{n}_mean"]) for n, *_ in POSITION_FIELDS], device=device, dtype=dtype)
    std = torch.tensor([float(norm_params[f"pos_{n}_std"]) for n, *_ in POSITION_FIELDS], device=device, dtype=dtype)
    lo = torch.tensor([f[2] for f in POSITION_FIELDS], device=device, dtype=dtype)
    hi = torch.tensor([f[3] for f in POSITION_FIELDS], device=device, dtype=dtype)
    return mean, std, lo, hi


def encode_positions(raw_configurations: np.ndarray, norm_params: Mapping[str, object]) -> np.ndarray:
    """Physical ``(..., R, 5)`` -> normalised position coordinates ``(..., R, 2)``."""
    raw = np.asarray(raw_configurations, dtype=np.float64)
    out = np.empty(raw.shape[:-1] + (2,), dtype=np.float64)
    for j, (name, idx, lo, hi) in enumerate(POSITION_FIELDS):
        u = _unit(raw[..., idx], lo, hi)
        w = np.log(u) - np.log1p(-u)
        out[..., j] = (w - float(norm_params[f"pos_{name}_mean"])) / float(norm_params[f"pos_{name}_std"])
    return out.astype(np.float32)


def decode_positions_torch(flat: torch.Tensor, num_res: int, norm_params: Mapping[str, object]) -> torch.Tensor:
    """Differentiable ``(..., 2R)`` -> physical positions ``(..., R, 2)`` [x, y] in metres."""
    mean, std, lo, hi = _stats_tensors(norm_params, flat.device, flat.dtype)
    coords = flat.reshape(*flat.shape[:-1], num_res, 2)
    return lo + (hi - lo) * torch.sigmoid(coords * std + mean)


def decode_design(flat: np.ndarray, num_res: int, norm_params: Mapping[str, object]) -> np.ndarray:
    """Normalised ``(..., 2R)`` -> full physical design ``(..., R, 5)`` [m, k, f_t, x, y]
    with the dataset's fixed m, f_t and k = m (2 pi f_t)^2, resonators sorted by x."""
    with torch.no_grad():
        xy = decode_positions_torch(torch.as_tensor(np.asarray(flat), dtype=torch.float64), num_res, norm_params).numpy()
    m, f_t = float(norm_params["fixed_m"]), float(norm_params["fixed_f_t"])
    out = np.empty(xy.shape[:-1] + (5,), dtype=np.float64)
    out[..., 0] = m
    out[..., 1] = m * (2.0 * np.pi * f_t) ** 2
    out[..., 2] = f_t
    out[..., 3:5] = xy
    return sort_by_x(out).astype(np.float32)


def position_log_abs_det(flat: torch.Tensor, num_res: int, norm_params: Mapping[str, object]) -> torch.Tensor:
    """``sum_i log |d flat_i / d v_i|`` per design ``(...,)``; added to ``log p_flat``
    it gives the density over the physical positions (per m^(2R))."""
    mean, std, lo, hi = _stats_tensors(norm_params, flat.device, flat.dtype)
    coords = flat.reshape(*flat.shape[:-1], num_res, 2)
    u = torch.sigmoid(coords * std + mean).clamp(_EPS, 1 - _EPS)
    log_dflat_dv = -torch.log(std) - torch.log(u) - torch.log1p(-u) - torch.log(hi - lo)
    return log_dflat_dv.sum(dim=(-1, -2))
