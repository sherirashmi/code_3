"""iSIREN: the invertible-operator scaffold (see ifno.py / common.py) with
SIREN's own layers.

SIREN (erp_forward_operators/siren_operator.py) = resonator encoder (context
c) + resonance-query features; [f, q(f)] -> stack of sine layers
sin(w0 (gamma * (W h + b) + beta)) with gamma, beta from c -> local
frequency refinement -> linear output.

Here:
  - Lift P = SIREN's own design path: [f, q(f)] -> modulated sine layers
    (FiLM from the context c) -> Linear -> 2*width.
  - Gate L = an unmodulated sine layer sin(w0 (W x + b)) applied pointwise
    to the latent, + the local frequency refinement. (The FiLM modulation
    from c cannot be in the gate: the inverse direction does not know the
    design.)
"""

from __future__ import annotations

import math
from typing import Mapping

import torch
import torch.nn as nn

from erp_forward_operators.neural_operator_utils import FrequencyRefinement1d, ResonanceQueryEncoder
from erp_forward_operators.siren_operator import ModulatedSineLayer
from erp_invertible_operators.common import InvertibleOperatorBase, build_design_encoder
from erp_invertible_operators.coupling import InvertibleCouplingStack
from utils.physics import freqs as _frequency_grid_hz


class SineGateLayer1d(nn.Module):
    """Pointwise sine layer (SIREN initialisation) + local refinement."""

    def __init__(self, width: int, omega_0: float = 20.0) -> None:
        super().__init__()
        self.omega_0 = float(omega_0)
        self.linear = nn.Conv1d(width, width, kernel_size=1)
        bound = math.sqrt(6.0 / width) / self.omega_0
        with torch.no_grad():
            self.linear.weight.uniform_(-bound, bound)
            self.linear.bias.uniform_(-bound, bound)
        self.refine = FrequencyRefinement1d(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.refine(torch.sin(self.omega_0 * self.linear(x)))


class ISIREN(InvertibleOperatorBase):
    def __init__(
        self,
        design_dim: int,
        n_freq: int = 301,
        width: int = 24,
        num_blocks: int = 4,
        siren_depth: int = 3,
        siren_hidden: int = 48,
        omega_0: float = 20.0,
        context_dim: int = 48,
        config_hidden: int = 64,
        query_dim: int = 16,
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
        # ---- Lift P: SIREN's modulated sine stack ----
        self.encoder = "set+sorted" if use_sorted_branch and encoder == "set" else encoder
        self.configuration_encoder = build_design_encoder(self.encoder, False, self.num_res, config_hidden, context_dim)
        self.resonance_query = ResonanceQueryEncoder(hidden_dim=query_dim, element_dim=query_dim, output_dim=query_dim)
        layers = [ModulatedSineLayer(1 + query_dim, siren_hidden, context_dim, first=True, omega_0=omega_0)]
        layers += [ModulatedSineLayer(siren_hidden, siren_hidden, context_dim, omega_0=omega_0)
                   for _ in range(siren_depth - 1)]
        self.sine_layers = nn.ModuleList(layers)
        self.lift_p = nn.Linear(siren_hidden, 2 * self.width)
        # ---- Coupling blocks gated by a sine layer ----
        self.blocks = InvertibleCouplingStack(lambda: SineGateLayer1d(self.width, omega_0), num_blocks, tau=tau,
                                              gate=gate, gate_scale=gate_scale)
        self._build_standard_parts(query_dim, activation, readout, readout_bins, vae_hidden, z_dim)
        self._init_features(coordinate_features)

    def _lift_forward(self, configuration: torch.Tensor, frequency: torch.Tensor) -> torch.Tensor:
        context = self.configuration_encoder(configuration)
        h = torch.cat((frequency, self.resonance_query(configuration, frequency)), dim=-1)
        for layer in self.sine_layers:
            h = layer(h, context)
        return self.lift_p(h).transpose(1, 2)

    def _lift_inverse(self, spectrum: torch.Tensor, frequency: torch.Tensor) -> torch.Tensor:
        return self._standard_lift_inverse(spectrum, frequency)


def build_model(design_dim: int, **kwargs) -> ISIREN:
    return ISIREN(design_dim=design_dim, **kwargs)


DEFAULT_MODEL_CONFIG: Mapping[str, object] = {
    "n_freq": 301, "width": 24, "num_blocks": 4, "siren_depth": 3, "siren_hidden": 48, "omega_0": 20.0,
    "context_dim": 48, "config_hidden": 64, "query_dim": 16, "z_dim": 6, "vae_hidden": 64, "tau": 1.0,
    "activation": "silu",
}
