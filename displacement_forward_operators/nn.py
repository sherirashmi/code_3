"""Displacement-field plain feedforward baseline: (configuration,
frequency, x, y) -> (v_real, v_imag, M_real, M_imag).

Adapted from forward_operators/nn.py, kept as the same deliberate "does
any of that sophistication even matter" ablation control: Linear -> ReLU
-> Dropout, no residual connections, no FiLM, no branch/trunk split, no
spectral/wavelet/attention/message-passing mechanism. It still gets
exactly the same *information* every other architecture in this folder
has -- FieldContextEncoder (position-aware resonator context) and
FieldResonanceQueryEncoder (frequency-detuning AND spatial-distance
features) -- so any accuracy gap reflects architecture, not access to
different inputs. Output is 4 channels, not 1.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from forward_operators.neural_operator_utils import FrequencyRefinement1d, resolve_activation

from .displacement_operator_utils import FieldContextEncoder, FieldResonanceQueryEncoder


class DisplacementNN(nn.Module):
    def __init__(
        self,
        num_res: int,
        hidden_dim: int = 85,
        depth: int = 6,
        context_dim: int = 57,
        query_dim: int = 28,
        dropout: float = 0.1,
        activation: str | type[nn.Module] = "relu",
    ) -> None:
        super().__init__()
        if depth <= 0:
            raise ValueError("depth must be positive.")
        self.num_res = int(num_res)
        activation_cls = resolve_activation(activation)
        self.field_encoder = FieldContextEncoder(output_dim=context_dim, resonator_hidden=context_dim)
        self.resonance_query = FieldResonanceQueryEncoder(
            hidden_dim=query_dim, element_dim=query_dim, output_dim=query_dim
        )

        layers: list[nn.Module] = []
        in_dim = context_dim + query_dim + 1
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(activation_cls())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        self.mlp = nn.Sequential(*layers)

        self.frequency_refinement = FrequencyRefinement1d(hidden_dim)
        self.output = nn.Linear(hidden_dim, 4)

    def forward(
        self, configuration: torch.Tensor, frequency: torch.Tensor, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        context = self.field_encoder(configuration, x, y)[:, None, :]
        context = context.expand(-1, frequency.shape[1], -1)
        query = self.resonance_query(configuration, frequency, x, y)

        h = torch.cat((context, query, frequency), dim=-1)
        h = self.mlp(h)
        h = self.frequency_refinement(h.transpose(1, 2)).transpose(1, 2)
        return self.output(h)


def build_model(num_res: int, **kwargs) -> DisplacementNN:
    return DisplacementNN(num_res=num_res, **kwargs)


DEFAULT_MODEL_CONFIG = {
    "hidden_dim": 85,
    "depth": 6,
    "context_dim": 57,
    "query_dim": 28,
    "dropout": 0.1,
    "activation": "relu",
}
