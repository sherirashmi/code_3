"""iLNO: the invertible-operator scaffold (see ifno.py / common.py) with LNO's
own layers.

LNO (erp_forward_operators/lno.py) = resonator encoder (context c) -> pole
predictor (num_poles complex-conjugate pole/residue pairs per design) ->
pole features r/(s - p) + conj at s = i f -> + resonance-query features + f
-> lift -> local frequency refinement -> projection.

Here:
  - Lift P = LNO's own design path: c -> design-dependent poles/residues ->
    pole features at every f, concatenated with q(f) and f -> Linear ->
    2*width.
  - Gate L = an LNO-style layer that does not see the design: a set of
    LEARNED, design-independent poles gives fixed pole features of f, which
    are added to a pointwise transform of the latent, followed by LNO's
    local frequency refinement. (The design-dependent poles cannot be in the
    gate: the inverse direction does not know the design.)
"""

from __future__ import annotations

from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from erp_forward_operators.neural_operator_utils import MLP, FrequencyRefinement1d, ResonanceQueryEncoder
from erp_invertible_operators.common import InvertibleOperatorBase, build_design_encoder
from erp_invertible_operators.coupling import InvertibleCouplingStack
from utils.physics import freqs as _frequency_grid_hz


def pole_features(f: torch.Tensor, sigma: torch.Tensor, omega: torch.Tensor, r_re: torch.Tensor,
                  r_im: torch.Tensor) -> torch.Tensor:
    """Re/Im of r/(s - p) + conj(r)/(s - conj(p)), s = i f, p = -sigma + i omega.
    f: (..., F); pole parameters broadcastable to (..., 1, P) -> (..., F, 2P)."""
    s = torch.complex(torch.zeros_like(f), f)[..., None]
    pole = torch.complex(-sigma, omega)
    res = torch.complex(r_re, r_im)
    term = res / (s - pole) + res.conj() / (s - pole.conj())
    return torch.cat((term.real, term.imag), dim=-1)


class LNOGateLayer1d(nn.Module):
    """Fixed learned pole features of f + pointwise transform -> local refinement."""

    def __init__(self, width: int, frequency_norm: torch.Tensor, num_poles: int = 8) -> None:
        super().__init__()
        self.register_buffer("f", frequency_norm.clone())
        lo, hi = float(frequency_norm.min()), float(frequency_norm.max())
        self.omega = nn.Parameter(torch.linspace(lo, hi, num_poles))  # spread over the band
        self.log_sigma = nn.Parameter(torch.full((num_poles,), -2.0))
        self.r_re = nn.Parameter(0.1 * torch.randn(num_poles))
        self.r_im = nn.Parameter(0.1 * torch.randn(num_poles))
        self.pointwise = nn.Conv1d(width, width, kernel_size=1)
        self.pole_proj = nn.Linear(2 * num_poles, width)
        self.refine = FrequencyRefinement1d(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = pole_features(self.f, F.softplus(self.log_sigma) + 1e-3, self.omega, self.r_re, self.r_im)  # (F, 2P)
        return self.refine(F.silu(self.pointwise(x) + self.pole_proj(feats).T[None]))


class ILNO(InvertibleOperatorBase):
    def __init__(
        self,
        design_dim: int,
        n_freq: int = 301,
        width: int = 24,
        num_blocks: int = 4,
        num_poles: int = 12,
        gate_poles: int = 8,
        context_dim: int = 48,
        config_hidden: int = 64,
        pole_hidden: int = 48,
        query_dim: int = 24,
        z_dim: int = 6,
        vae_hidden: int = 64,
        tau: float = 1.0,
        activation: str = "silu",
        use_sorted_branch: bool = False,
        encoder: str = "set",
        coordinate_features: str = "zscored",
        design_param: str = "full15",
        num_res: int = 3,
        gate: str = "softplus",
        gate_scale: float = 2.0,
        readout: str = "pooled",
        readout_bins: int = 16,
    ) -> None:
        super().__init__()
        self._init_design(design_dim, design_param, num_res)
        self.n_freq, self.width, self.num_poles = int(n_freq), int(width), int(num_poles)
        self._register_frequency_grid(torch.from_numpy(_frequency_grid_hz.astype("float32")))
        # ---- Lift P: LNO's design-dependent pole/residue basis ----
        self.encoder = "set+sorted" if use_sorted_branch and encoder == "set" else encoder
        self.configuration_encoder = build_design_encoder(self.encoder, False, self.num_res, config_hidden, context_dim)
        self.pole_predictor = MLP([context_dim, pole_hidden, pole_hidden, 4 * self.num_poles], activation=nn.SiLU)
        self.resonance_query = ResonanceQueryEncoder(hidden_dim=query_dim, element_dim=query_dim, output_dim=query_dim)
        self.lift_p = nn.Linear(2 * self.num_poles + query_dim + 1, 2 * self.width)
        self.lift_p_activation = nn.SiLU()
        # ---- Coupling blocks gated by an LNO-style layer (fixed learned poles) ----
        self.blocks = InvertibleCouplingStack(
            lambda: LNOGateLayer1d(self.width, self.frequency_norm, num_poles=gate_poles), num_blocks, tau=tau,
            gate=gate, gate_scale=gate_scale)
        self._build_standard_parts(query_dim, activation, readout, readout_bins, vae_hidden, z_dim)
        self._init_features(coordinate_features)

    def _lift_forward(self, configuration: torch.Tensor, frequency: torch.Tensor) -> torch.Tensor:
        raw = self.pole_predictor(self.configuration_encoder(configuration)).view(-1, 1, self.num_poles, 4)
        poles = pole_features(frequency[..., 0], F.softplus(raw[..., 0]) + 1e-3, F.softplus(raw[..., 1]),
                              raw[..., 2], raw[..., 3])  # (B, F, 2P)
        query = self.resonance_query(configuration, frequency)
        lifted = self.lift_p(torch.cat((poles, query, frequency), dim=-1))
        return self.lift_p_activation(lifted).transpose(1, 2)

    def _lift_inverse(self, spectrum: torch.Tensor, frequency: torch.Tensor) -> torch.Tensor:
        return self._standard_lift_inverse(spectrum, frequency)


def build_model(design_dim: int, **kwargs) -> ILNO:
    return ILNO(design_dim=design_dim, **kwargs)


DEFAULT_MODEL_CONFIG: Mapping[str, object] = {
    "n_freq": 301, "width": 24, "num_blocks": 4, "num_poles": 12, "gate_poles": 8, "context_dim": 48,
    "config_hidden": 64, "pole_hidden": 48, "query_dim": 24, "z_dim": 6, "vae_hidden": 64, "tau": 1.0,
    "activation": "silu",
}
