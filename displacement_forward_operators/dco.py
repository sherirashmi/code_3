"""Displacement-field Deep Cat Operator (DCO): (configuration, frequency,
x, y) -> (v_real, v_imag, M_real, M_imag).

Adapted from erp_forward_operators/dco.py. Unchanged: concatenate
(branch, trunk, query) -> lift -> depth plain (unconditioned) residual MLP
blocks -> FrequencyRefinement1d -> output, all along the still-shared,
ordered frequency axis. Changed: branch is FieldContextEncoder (position-
aware) instead of ResonatorSetEncoder alone; query is
FieldResonanceQueryEncoder; output is 4 channels, not 1.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from erp_forward_operators.neural_operator_utils import MLP, FrequencyRefinement1d, ResidualMLPBlock, resolve_activation

from .displacement_operator_utils import FieldContextEncoder, FieldResonanceQueryEncoder


class DisplacementDCO(nn.Module):
    def __init__(
        self,
        num_res: int,
        hidden_dim: int = 67,
        branch_dim: int = 56,
        trunk_dim: int = 56,
        query_dim: int = 28,
        depth: int = 4,
        dropout: float = 0.0,
        activation: str | type[nn.Module] = "silu",
    ) -> None:
        super().__init__()
        self.num_res = int(num_res)
        activation_cls = resolve_activation(activation)
        self.branch = FieldContextEncoder(output_dim=branch_dim, resonator_hidden=hidden_dim)
        self.trunk = MLP([1, hidden_dim, trunk_dim], activation=activation_cls)
        self.resonance_query = FieldResonanceQueryEncoder(
            hidden_dim=query_dim, element_dim=query_dim, output_dim=query_dim
        )
        self.lift = nn.Linear(branch_dim + trunk_dim + query_dim, hidden_dim)
        self.lift_activation = activation_cls()
        self.blocks = nn.ModuleList([ResidualMLPBlock(hidden_dim, activation=activation_cls) for _ in range(depth)])
        self.block_dropout = nn.Dropout(dropout)
        self.frequency_refinement = FrequencyRefinement1d(hidden_dim)
        self.output = MLP([hidden_dim, hidden_dim // 2, 4], activation=activation_cls)

    def forward(
        self, configuration: torch.Tensor, frequency: torch.Tensor, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        branch = self.branch(configuration, x, y)[:, None, :]
        branch = branch.expand(-1, frequency.shape[1], -1)
        trunk = self.trunk(frequency)
        query = self.resonance_query(configuration, frequency, x, y)

        h = self.lift_activation(self.lift(torch.cat((branch, trunk, query), dim=-1)))
        for block in self.blocks:
            h = self.block_dropout(block(h))
        h = self.frequency_refinement(h.transpose(1, 2)).transpose(1, 2)
        return self.output(h)


def build_model(num_res: int, **kwargs) -> DisplacementDCO:
    return DisplacementDCO(num_res=num_res, **kwargs)


DEFAULT_MODEL_CONFIG = {
    "hidden_dim": 67,
    "branch_dim": 56,
    "trunk_dim": 56,
    "query_dim": 28,
    "depth": 4,
    "dropout": 0.0,
    "activation": "silu",
}
