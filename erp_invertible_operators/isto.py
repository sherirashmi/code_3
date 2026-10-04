"""iSTO: the invertible-operator scaffold (see ifno.py / common.py) with the
set transformer operator's own layers.

STO (erp_forward_operators/set_transformer_operator.py) = per-resonator node
lift (with plate mode shapes) -> transformer encoder (self-attention between
resonators) -> every frequency queries the resonator tokens by
cross-attention with a learned detuning bias -> frequency mixer (local +
dilated Conv1d) -> output head.

Here:
  - Lift P = STO's own design path: node lift + transformer encoder +
    detuning-biased cross-attention from the frequencies to the resonators,
    -> Linear -> 2*width. The resonators are tokens of an attention layer,
    which is permutation invariant by construction, so there is no set or
    sorted encoder to choose (as for the forward STO).
  - Gate L = STO's frequency mixer (local Conv1d k=5 + dilated Conv1d k=3,
    residual, GroupNorm), unchanged. (The cross-attention cannot be in the
    gate: the inverse direction does not know the resonators.)
"""

from __future__ import annotations

from typing import Mapping

import torch
import torch.nn as nn

from erp_forward_operators.neural_operator_utils import MLP, modal_features, resolve_activation
from erp_forward_operators.set_transformer_operator import DetuningCrossAttention, FrequencyMixer
from erp_invertible_operators.common import InvertibleOperatorBase
from erp_invertible_operators.coupling import InvertibleCouplingStack
from utils.physics import freqs as _frequency_grid_hz


class ISTO(InvertibleOperatorBase):
    uses_modal_features = True  # node features: plate mode shapes in physical mode

    def __init__(
        self,
        design_dim: int,
        n_freq: int = 301,
        width: int = 24,
        num_blocks: int = 4,
        token_width: int = 32,
        heads: int = 4,
        encoder_depth: int = 2,
        ff_dim: int = 64,
        modal_harmonics: int = 4,
        frequency_dim: int = 24,
        z_dim: int = 6,
        vae_hidden: int = 64,
        tau: float = 1.0,
        activation: str = "gelu",
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
        self.n_freq, self.width, self.modal_harmonics = int(n_freq), int(width), int(modal_harmonics)
        self._register_frequency_grid(torch.from_numpy(_frequency_grid_hz.astype("float32")))
        act = resolve_activation(activation)
        # ---- Lift P: STO's resonator tokens + detuning-biased cross-attention ----
        node_dim = 5 + 2 * self.modal_harmonics + self.modal_harmonics ** 2
        self.node_lift = MLP([node_dim, token_width, token_width], activation=act)
        layer = nn.TransformerEncoderLayer(d_model=token_width, nhead=heads, dim_feedforward=ff_dim, dropout=0.0,
                                           activation=act(), batch_first=True)
        self.token_encoder = nn.TransformerEncoder(layer, num_layers=encoder_depth)
        self.frequency_query = MLP([1, token_width, token_width], activation=act)
        self.cross_attention = DetuningCrossAttention(token_width, heads, dropout=0.0, activation=act)
        self.cross_norm = nn.LayerNorm(token_width)
        self.lift_p = nn.Linear(token_width, 2 * self.width)
        self.lift_p_activation = act()
        # ---- Coupling blocks gated by STO's frequency mixer ----
        self.blocks = InvertibleCouplingStack(lambda: FrequencyMixer(self.width, activation=activation), num_blocks,
                                              tau=tau, gate=gate, gate_scale=gate_scale)
        self._build_standard_parts(frequency_dim, activation, readout, readout_bins, vae_hidden, z_dim)
        self._init_features(coordinate_features)

    def _lift_forward(self, configuration: torch.Tensor, frequency: torch.Tensor) -> torch.Tensor:
        tokens = self.token_encoder(self.node_lift(modal_features(self, configuration, self.modal_harmonics)))
        query = self.frequency_query(frequency)
        h = self.cross_norm(query + self.cross_attention(query, tokens, configuration, frequency))
        return self.lift_p_activation(self.lift_p(h)).transpose(1, 2)

    def _lift_inverse(self, spectrum: torch.Tensor, frequency: torch.Tensor) -> torch.Tensor:
        return self._standard_lift_inverse(spectrum, frequency)


def build_model(design_dim: int, **kwargs) -> ISTO:
    return ISTO(design_dim=design_dim, **kwargs)


DEFAULT_MODEL_CONFIG: Mapping[str, object] = {
    "n_freq": 301, "width": 24, "num_blocks": 4, "token_width": 32, "heads": 4, "encoder_depth": 2, "ff_dim": 64,
    "modal_harmonics": 4, "frequency_dim": 24, "z_dim": 6, "vae_hidden": 64, "tau": 1.0, "activation": "gelu",
}
