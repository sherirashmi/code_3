"""Displacement-field SIREN neural operator: (configuration, frequency, x,
y) -> (v_real, v_imag, M_real, M_imag).

Adapted from forward_operators/siren_operator.py. Unchanged: the depth
FiLM-modulated sine layers (sin(omega_0 * (gamma*Linear(h)+beta))) and
FrequencyRefinement1d, still along the shared, ordered frequency axis.
Changed: context comes from FieldContextEncoder (position-aware) instead
of ResonatorSetEncoder alone; query_features from FieldResonanceQueryEncoder;
output is 4 channels, not 1.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from forward_operators.neural_operator_utils import FrequencyRefinement1d

from .displacement_operator_utils import FieldContextEncoder, FieldResonanceQueryEncoder


class ModulatedSineLayer(nn.Module):
    """SIREN layer with configuration-dependent FiLM modulation (unchanged)."""

    def __init__(self, in_features: int, out_features: int, context_dim: int, *, first: bool = False, omega_0: float = 20.0) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.omega_0 = float(omega_0)
        self.linear = nn.Linear(in_features, out_features)
        self.film = nn.Linear(context_dim, 2 * out_features)

        with torch.no_grad():
            if first:
                bound = 1.0 / max(1, in_features)
            else:
                bound = math.sqrt(6.0 / max(1, in_features)) / self.omega_0
            self.linear.weight.uniform_(-bound, bound)
            self.linear.bias.uniform_(-bound, bound)
            self.film.weight.zero_()
            self.film.bias.zero_()

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.film(context).chunk(2, dim=-1)
        gamma = 1.0 + 0.25 * torch.tanh(gamma)[:, None, :]
        beta = 0.10 * beta[:, None, :]
        preactivation = gamma * self.linear(x) + beta
        return torch.sin(self.omega_0 * preactivation)


class DisplacementSIRENOperator(nn.Module):
    def __init__(
        self,
        num_res: int,
        context_dim: int = 57,
        query_dim: int = 14,
        hidden_dim: int = 75,
        depth: int = 4,
        omega_0: float = 20.0,
        config_hidden: int = 57,
        query_hidden: int = 28,
    ) -> None:
        super().__init__()
        if depth <= 0:
            raise ValueError("depth must be positive.")
        self.num_res = int(num_res)
        self.omega_0 = float(omega_0)
        self.field_encoder = FieldContextEncoder(output_dim=context_dim, resonator_hidden=config_hidden)
        self.resonance_query = FieldResonanceQueryEncoder(
            hidden_dim=query_hidden, element_dim=query_hidden, output_dim=query_dim
        )

        input_dim = 1 + query_dim
        layers: list[nn.Module] = [ModulatedSineLayer(input_dim, hidden_dim, context_dim, first=True, omega_0=omega_0)]
        layers.extend(ModulatedSineLayer(hidden_dim, hidden_dim, context_dim, omega_0=omega_0) for _ in range(depth - 1))
        self.layers = nn.ModuleList(layers)
        self.frequency_refinement = FrequencyRefinement1d(hidden_dim)
        self.output = nn.Linear(hidden_dim, 4)

        with torch.no_grad():
            bound = math.sqrt(6.0 / hidden_dim) / self.omega_0
            self.output.weight.uniform_(-bound, bound)
            self.output.bias.zero_()

    def forward(
        self, configuration: torch.Tensor, frequency: torch.Tensor, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        context = self.field_encoder(configuration, x, y)
        query_features = self.resonance_query(configuration, frequency, x, y)
        h = torch.cat((frequency, query_features), dim=-1)
        for layer in self.layers:
            h = layer(h, context)
        h = self.frequency_refinement(h.transpose(1, 2)).transpose(1, 2)
        return self.output(h)


def build_model(num_res: int, **kwargs) -> DisplacementSIRENOperator:
    return DisplacementSIRENOperator(num_res=num_res, **kwargs)


DEFAULT_MODEL_CONFIG = {
    "context_dim": 57,
    "query_dim": 14,
    "hidden_dim": 75,
    "config_hidden": 57,
    "query_hidden": 28,
    "depth": 4,
    "omega_0": 20.0,
}
