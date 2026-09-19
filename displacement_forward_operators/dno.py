"""Displacement-field Deep Neural Operator (DNO): (configuration, frequency,
x, y) -> (displacement_real, displacement_imag).

Adapted from forward_operators/dno.py. Unchanged: the deep FiLM-
conditioned residual MLP tower (depth blocks, each gated by
context+detuning-aware query) and FrequencyRefinement1d -- still valid
since frequency stays the same shared, ordered grid. Changed: the branch
context now comes from FieldContextEncoder (resonator configuration fused
with the query position's modal-basis encoding, see
displacement_operator_utils.py) instead of ResonatorSetEncoder alone; the
detuning-aware query now also carries a spatial-distance-to-each-resonator
feature (FieldResonanceQueryEncoder); output is 2 channels, not 1.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from forward_operators.neural_operator_utils import MLP, FrequencyRefinement1d, ResidualMLPBlock, resolve_activation

from .displacement_operator_utils import FieldContextEncoder, FieldResonanceQueryEncoder


class FiLMResidualBlock(nn.Module):
    """Residual MLP block modulated by configuration/query conditioning (unchanged from forward_operators/dno.py)."""

    def __init__(
        self, width: int, condition_dim: int, dropout: float = 0.0, activation: str | type[nn.Module] = "silu"
    ) -> None:
        super().__init__()
        activation_cls = resolve_activation(activation)
        self.block = ResidualMLPBlock(width, activation=activation_cls)
        self.film = nn.Linear(condition_dim, 2 * width)
        self.activation = activation_cls()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        h = self.block(x)
        gamma, beta = self.film(condition).chunk(2, dim=-1)
        gamma = 1.0 + 0.20 * torch.tanh(gamma)
        beta = 0.10 * beta
        return self.dropout(self.activation(gamma * h + beta))


class DisplacementDNO(nn.Module):
    def __init__(
        self,
        num_res: int,
        hidden_dim: int = 54,
        context_dim: int = 57,
        frequency_dim: int = 28,
        query_dim: int = 28,
        depth: int = 4,
        dropout: float = 0.0,
        activation: str | type[nn.Module] = "silu",
    ) -> None:
        super().__init__()
        self.num_res = int(num_res)
        activation_cls = resolve_activation(activation)
        self.field_encoder = FieldContextEncoder(output_dim=context_dim, resonator_hidden=hidden_dim)
        self.frequency_encoder = MLP([1, frequency_dim, frequency_dim], activation=activation_cls)
        self.resonance_query = FieldResonanceQueryEncoder(
            hidden_dim=query_dim, element_dim=query_dim, output_dim=query_dim
        )
        self.lift = nn.Linear(context_dim + frequency_dim + query_dim, hidden_dim)
        self.lift_activation = activation_cls()
        condition_dim = context_dim + query_dim
        self.blocks = nn.ModuleList(
            [FiLMResidualBlock(hidden_dim, condition_dim, dropout=dropout, activation=activation_cls) for _ in range(depth)]
        )
        self.frequency_refinement = FrequencyRefinement1d(hidden_dim)
        self.output = MLP([hidden_dim, hidden_dim // 2, 2], activation=activation_cls)

    def forward(
        self, configuration: torch.Tensor, frequency: torch.Tensor, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        context_global = self.field_encoder(configuration, x, y)
        context = context_global[:, None, :].expand(-1, frequency.shape[1], -1)
        freq = self.frequency_encoder(frequency)
        query = self.resonance_query(configuration, frequency, x, y)

        h = self.lift_activation(self.lift(torch.cat((context, freq, query), dim=-1)))
        condition = torch.cat((context, query), dim=-1)
        for block in self.blocks:
            h = block(h, condition)

        h = self.frequency_refinement(h.transpose(1, 2)).transpose(1, 2)
        return self.output(h)


def build_model(num_res: int, **kwargs) -> DisplacementDNO:
    return DisplacementDNO(num_res=num_res, **kwargs)


DEFAULT_MODEL_CONFIG = {
    "hidden_dim": 54,
    "context_dim": 57,
    "frequency_dim": 28,
    "query_dim": 28,
    "depth": 4,
    "dropout": 0.0,
    "activation": "silu",
}
