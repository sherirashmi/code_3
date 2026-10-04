"""iDNO: the invertible-operator scaffold (see ifno.py / common.py) with DNO's
own layers.

DNO (erp_forward_operators/dno.py) = resonator encoder (context c) +
frequency-encoder MLP + resonance-query features -> lift -> repeated
FiLM-modulated residual blocks (gamma, beta from [c, q(f)]) -> local
frequency refinement -> output head.

Here:
  - Lift P = DNO's own lift: [c, emb(f), q(f)] -> Linear -> 2*width, SiLU.
  - Gate L = DNO's FiLM residual block + the local frequency refinement.
    The gate must not see the design (the inverse direction does not know
    it), so its FiLM is driven by a learned embedding of the frequency
    (known in both directions) instead of [c, q(f)]; the design enters
    only through the lift, as in every family member.
"""

from __future__ import annotations

from typing import Mapping

import torch
import torch.nn as nn

from erp_forward_operators.dno import FiLMResidualBlock
from erp_forward_operators.neural_operator_utils import MLP, FrequencyRefinement1d, ResonanceQueryEncoder
from erp_invertible_operators.common import InvertibleOperatorBase, build_design_encoder
from erp_invertible_operators.coupling import InvertibleCouplingStack
from utils.physics import freqs as _frequency_grid_hz


class DNOGateLayer1d(nn.Module):
    """FiLM residual block (FiLM from a frequency embedding) + local refinement."""

    def __init__(self, width: int, frequency_norm: torch.Tensor, cond_dim: int = 16, activation: str = "silu") -> None:
        super().__init__()
        self.register_buffer("f", frequency_norm.clone()[:, None])  # (F, 1)
        self.condition = MLP([1, cond_dim, cond_dim], activation=nn.SiLU)
        self.block = FiLMResidualBlock(width, cond_dim, activation=activation)
        self.refine = FrequencyRefinement1d(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.block(x.transpose(1, 2), self.condition(self.f))  # (B, F, width), FiLM broadcast over B
        return self.refine(h.transpose(1, 2))


class IDNO(InvertibleOperatorBase):
    def __init__(
        self,
        design_dim: int,
        n_freq: int = 301,
        width: int = 24,
        num_blocks: int = 4,
        context_dim: int = 48,
        config_hidden: int = 64,
        frequency_dim: int = 24,
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
        self.n_freq, self.width = int(n_freq), int(width)
        self._register_frequency_grid(torch.from_numpy(_frequency_grid_hz.astype("float32")))
        # ---- Lift P: DNO's context + frequency encoder + resonance query ----
        self.encoder = "set+sorted" if use_sorted_branch and encoder == "set" else encoder
        self.configuration_encoder = build_design_encoder(self.encoder, False, self.num_res, config_hidden, context_dim)
        self.frequency_encoder = MLP([1, frequency_dim, frequency_dim], activation=nn.SiLU)
        self.resonance_query = ResonanceQueryEncoder(hidden_dim=query_dim, element_dim=query_dim, output_dim=query_dim)
        self.lift_p = nn.Linear(context_dim + frequency_dim + query_dim, 2 * self.width)
        self.lift_p_activation = nn.SiLU()
        # ---- Coupling blocks gated by DNO's FiLM residual block ----
        self.blocks = InvertibleCouplingStack(
            lambda: DNOGateLayer1d(self.width, self.frequency_norm, activation=activation), num_blocks, tau=tau,
            gate=gate, gate_scale=gate_scale)
        self._build_standard_parts(frequency_dim, activation, readout, readout_bins, vae_hidden, z_dim)
        self._init_features(coordinate_features)

    def _lift_forward(self, configuration: torch.Tensor, frequency: torch.Tensor) -> torch.Tensor:
        context = self.configuration_encoder(configuration)[:, None, :].expand(-1, frequency.shape[1], -1)
        parts = (context, self.frequency_encoder(frequency), self.resonance_query(configuration, frequency))
        return self.lift_p_activation(self.lift_p(torch.cat(parts, dim=-1))).transpose(1, 2)

    def _lift_inverse(self, spectrum: torch.Tensor, frequency: torch.Tensor) -> torch.Tensor:
        return self._standard_lift_inverse(spectrum, frequency)


def build_model(design_dim: int, **kwargs) -> IDNO:
    return IDNO(design_dim=design_dim, **kwargs)


DEFAULT_MODEL_CONFIG: Mapping[str, object] = {
    "n_freq": 301, "width": 24, "num_blocks": 4, "context_dim": 48, "config_hidden": 64, "frequency_dim": 24,
    "query_dim": 24, "z_dim": 6, "vae_hidden": 64, "tau": 1.0, "activation": "silu",
}
