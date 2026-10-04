"""iWNO: the invertible-operator scaffold (see ifno.py / common.py) with WNO's
own layer.

WNO (erp_forward_operators/wno.py) = resonator encoder (context c) +
resonance-query features + f -> lift -> repeated multi-level Haar wavelet
blocks (analysis, per-band Conv1d, synthesis, + local conv) -> projection.

Here:
  - Lift P = WNO's own lift: [c, q(f), f] -> Linear -> 2*width.
  - Gate L = WNO's multi-level Haar wavelet block, unchanged (mixing along
    the frequency axis on several scales at once).
"""

from __future__ import annotations

from typing import Mapping

import torch
import torch.nn as nn

from erp_forward_operators.neural_operator_utils import ResonanceQueryEncoder
from erp_forward_operators.wno import MultiLevelHaarWaveletBlock1d
from erp_invertible_operators.common import InvertibleOperatorBase, build_design_encoder
from erp_invertible_operators.coupling import InvertibleCouplingStack
from utils.physics import freqs as _frequency_grid_hz


class IWNO(InvertibleOperatorBase):
    def __init__(
        self,
        design_dim: int,
        n_freq: int = 301,
        width: int = 24,
        num_blocks: int = 4,
        levels: int = 3,
        config_hidden: int = 64,
        query_dim: int = 24,
        z_dim: int = 6,
        vae_hidden: int = 64,
        tau: float = 1.0,
        activation: str = "gelu",
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
        # ---- Lift P: WNO's context + resonance query + f ----
        self.encoder = "set+sorted" if use_sorted_branch and encoder == "set" else encoder
        self.configuration_encoder = build_design_encoder(self.encoder, False, self.num_res, config_hidden, 2 * self.width)
        self.resonance_query = ResonanceQueryEncoder(hidden_dim=query_dim, element_dim=query_dim, output_dim=query_dim)
        self.lift_p = nn.Linear(2 * self.width + query_dim + 1, 2 * self.width)
        # ---- Coupling blocks gated by WNO's wavelet block ----
        self.blocks = InvertibleCouplingStack(
            lambda: MultiLevelHaarWaveletBlock1d(self.width, levels=levels, dropout=0.0, activation=activation),
            num_blocks, tau=tau, gate=gate, gate_scale=gate_scale)
        self._build_standard_parts(query_dim, activation, readout, readout_bins, vae_hidden, z_dim)
        self._init_features(coordinate_features)

    def _lift_forward(self, configuration: torch.Tensor, frequency: torch.Tensor) -> torch.Tensor:
        context = self.configuration_encoder(configuration)[:, None, :].expand(-1, frequency.shape[1], -1)
        query = self.resonance_query(configuration, frequency)
        return self.lift_p(torch.cat((context, query, frequency), dim=-1)).transpose(1, 2)

    def _lift_inverse(self, spectrum: torch.Tensor, frequency: torch.Tensor) -> torch.Tensor:
        return self._standard_lift_inverse(spectrum, frequency)


def build_model(design_dim: int, **kwargs) -> IWNO:
    return IWNO(design_dim=design_dim, **kwargs)


DEFAULT_MODEL_CONFIG: Mapping[str, object] = {
    "n_freq": 301, "width": 24, "num_blocks": 4, "levels": 3, "config_hidden": 64, "query_dim": 24, "z_dim": 6,
    "vae_hidden": 64, "tau": 1.0, "activation": "gelu",
}
