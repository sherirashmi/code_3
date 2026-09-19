"""Displacement-field Graph Neural Operator (GNO): (configuration,
frequency, x, y) -> (displacement_real, displacement_imag).

Adapted from forward_operators/gno.py. Unchanged: complete-graph message
passing between resonator nodes (GraphMessageLayer, geometric/frequency
edge features) and the learned per-query-frequency softmax attention
pooling ("kernel integral") over those nodes -- GNO's own branch
mechanism is already custom (not ResonatorSetEncoder), so
FieldContextEncoder isn't used here at all. Changed, faithfully extending
GNO's OWN idiom rather than swapping in a shared class: the query kernel's
pairwise features already include frequency detuning (query_frequency -
f_t) per resonator; a spatial-distance feature (query position vs. each
resonator's own (x,y)) is added alongside it, the exact position-side
counterpart to detuning's frequency-side role. Output is 2 channels, not 1.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from forward_operators.neural_operator_utils import MLP, FrequencyRefinement1d, physics_aware_resonator_features, resolve_activation


class GraphMessageLayer(nn.Module):
    """Complete-graph message passing with geometric/frequency edge features (unchanged)."""

    def __init__(self, width: int, dropout: float = 0.0, activation: str | type[nn.Module] = "silu") -> None:
        super().__init__()
        activation_cls = resolve_activation(activation)
        self.message = MLP([2 * width + 7, width, width], activation=activation_cls)
        self.update = MLP([2 * width, width, width], activation=activation_cls)
        self.norm = nn.LayerNorm(width)
        self.activation = activation_cls()
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        b, n, width = h.shape
        hi = h[:, :, None, :].expand(b, n, n, width)
        hj = h[:, None, :, :].expand(b, n, n, width)
        rel = features[:, None, :, :] - features[:, :, None, :]
        distance = torch.linalg.vector_norm(rel[..., 3:5], dim=-1, keepdim=True)
        ft_gap = rel[..., 2:3].abs()
        edge = torch.cat((rel, distance, ft_gap), dim=-1)
        message = self.message(torch.cat((hi, hj, edge), dim=-1))

        if n > 1:
            mask = (~torch.eye(n, dtype=torch.bool, device=h.device))[None, :, :, None]
            aggregate = (message * mask).sum(dim=2) / float(n - 1)
        else:
            aggregate = torch.zeros_like(h)

        delta = self.update(torch.cat((h, aggregate), dim=-1))
        return self.dropout(self.activation(self.norm(h + delta)))


class DisplacementGNO(nn.Module):
    def __init__(
        self,
        num_res: int,
        width: int = 60,
        depth: int = 3,
        frequency_dim: int = 28,
        modal_harmonics: int = 4,
        dropout: float = 0.1,
        activation: str | type[nn.Module] = "silu",
    ) -> None:
        super().__init__()
        self.num_res = int(num_res)
        self.modal_harmonics = int(modal_harmonics)
        activation_cls = resolve_activation(activation)
        node_input_dim = 5 + 2 * self.modal_harmonics + self.modal_harmonics**2
        self.node_lift = MLP([node_input_dim, width, width], activation=activation_cls)
        self.layers = nn.ModuleList(
            [GraphMessageLayer(width, dropout=dropout, activation=activation_cls) for _ in range(depth)]
        )
        self.frequency_encoder = MLP([1, frequency_dim, frequency_dim], activation=activation_cls)
        # node, raw resonator, frequency embedding, query f, delta, |delta|, delta^2, spatial_distance
        query_input_dim = width + 5 + frequency_dim + 5
        self.query_kernel = MLP([query_input_dim, width, width, width], activation=activation_cls)
        self.attention_score = MLP([width, width // 2, 1], activation=activation_cls)
        self.attention_dropout = nn.Dropout(dropout)
        self.frequency_refinement = FrequencyRefinement1d(width)
        self.output = MLP([width, width // 2, 2], activation=activation_cls)

    def forward(
        self, configuration: torch.Tensor, frequency: torch.Tensor, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        node_features = physics_aware_resonator_features(configuration, harmonics=self.modal_harmonics)
        h = self.node_lift(node_features)
        for layer in self.layers:
            h = layer(h, configuration)

        freq = self.frequency_encoder(frequency)
        b, f, _ = freq.shape
        n = configuration.shape[1]
        node = h[:, None, :, :].expand(b, f, n, -1)
        raw = configuration[:, None, :, :].expand(b, f, n, -1)
        query_embedding = freq[:, :, None, :].expand(b, f, n, -1)
        query_frequency = frequency[:, :, None, :].expand(b, f, n, 1)
        f_t = configuration[:, None, :, 2:3].expand(b, f, n, 1)
        detuning = query_frequency - f_t

        res_x = configuration[:, None, :, 3:4].expand(b, f, n, 1)
        res_y = configuration[:, None, :, 4:5].expand(b, f, n, 1)
        qx = x[:, None, None, :].expand(b, f, n, 1)
        qy = y[:, None, None, :].expand(b, f, n, 1)
        spatial_distance = torch.sqrt((qx - res_x) ** 2 + (qy - res_y) ** 2 + 1e-12)

        pair = torch.cat(
            (node, raw, query_embedding, query_frequency, detuning, detuning.abs(), detuning.square(), spatial_distance),
            dim=-1,
        )
        kernel_values = self.query_kernel(pair)
        weights = torch.softmax(self.attention_score(kernel_values), dim=2)
        weights = self.attention_dropout(weights)
        integral = (weights * kernel_values).sum(dim=2)
        integral = self.frequency_refinement(integral.transpose(1, 2)).transpose(1, 2)
        return self.output(integral)


def build_model(num_res: int, **kwargs) -> DisplacementGNO:
    return DisplacementGNO(num_res=num_res, **kwargs)


DEFAULT_MODEL_CONFIG = {
    "width": 60,
    "depth": 3,
    "frequency_dim": 28,
    "modal_harmonics": 4,
    "dropout": 0.1,
    "activation": "silu",
}
