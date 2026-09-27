"""Displacement-field Set Transformer Operator (STO): (configuration,
frequency, x, y) -> (v_real, v_imag, M_real, M_imag).

Adapted from erp_forward_operators/set_transformer_operator.py. Unchanged:
resonator self-attention (nn.TransformerEncoder over resonator tokens)
followed by detuning-biased frequency cross-attention and FrequencyMixer,
all along the still-shared, ordered frequency axis. Changed, faithfully
extending STO's own idiom: DetuningCrossAttention's learned per-head
attention bias already used (raw config, query frequency, detuning,
|detuning|) as its bias-net input; a spatial-distance feature (query
position vs. each resonator's own (x,y)) is added alongside it, same
position-side counterpart to detuning used in gno.py/dno.py/etc. Output
is 4 channels, not 1.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from erp_forward_operators.neural_operator_utils import MLP, physics_aware_resonator_features, resolve_activation


class DetuningCrossAttention(nn.Module):
    """Multi-head frequency-to-resonator attention with learned detuning+spatial bias."""

    def __init__(self, width: int, heads: int, dropout: float = 0.0, activation: str | type[nn.Module] = "silu") -> None:
        super().__init__()
        if width % heads != 0:
            raise ValueError("width must be divisible by heads.")
        self.width = int(width)
        self.heads = int(heads)
        self.head_dim = self.width // self.heads
        self.q_proj = nn.Linear(width, width)
        self.k_proj = nn.Linear(width, width)
        self.v_proj = nn.Linear(width, width)
        self.out_proj = nn.Linear(width, width)
        # raw [m,k,f_t,x,y], query f, delta, |delta|, spatial_distance -> one bias per head
        self.bias_net = MLP([9, width // 2, heads], activation=resolve_activation(activation))
        self.attn_dropout = nn.Dropout(dropout)

    def forward(
        self, query: torch.Tensor, tokens: torch.Tensor, configuration: torch.Tensor, frequency: torch.Tensor,
        x: torch.Tensor, y: torch.Tensor,
    ) -> torch.Tensor:
        b, f, _ = query.shape
        n = tokens.shape[1]

        q = self.q_proj(query).view(b, f, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(tokens).view(b, n, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(tokens).view(b, n, self.heads, self.head_dim).transpose(1, 2)

        score = torch.einsum("bhfd,bhnd->bhfn", q, k) / math.sqrt(float(self.head_dim))

        raw = configuration[:, None, :, :].expand(b, f, n, -1)
        query_f = frequency[:, :, None, :].expand(b, f, n, 1)
        f_t = configuration[:, None, :, 2:3].expand(b, f, n, 1)
        delta = query_f - f_t

        res_x = configuration[:, None, :, 3:4].expand(b, f, n, 1)
        res_y = configuration[:, None, :, 4:5].expand(b, f, n, 1)
        qx = x[:, None, None, :].expand(b, f, n, 1)
        qy = y[:, None, None, :].expand(b, f, n, 1)
        spatial_distance = torch.sqrt((qx - res_x) ** 2 + (qy - res_y) ** 2 + 1e-12)

        bias_features = torch.cat((raw, query_f, delta, delta.abs(), spatial_distance), dim=-1)
        bias = self.bias_net(bias_features).permute(0, 3, 1, 2)
        attention = torch.softmax(score + bias, dim=-1)
        attention = self.attn_dropout(attention)

        attended = torch.einsum("bhfn,bhnd->bhfd", attention, v)
        attended = attended.transpose(1, 2).contiguous().view(b, f, self.width)
        return self.out_proj(attended)


class FrequencyMixer(nn.Module):
    """Memory-efficient local/dilated frequency interaction after cross-attention (unchanged)."""

    def __init__(self, width: int, activation: str | type[nn.Module] = "gelu") -> None:
        super().__init__()
        self.local = nn.Conv1d(width, width, kernel_size=5, padding=2)
        self.dilated = nn.Conv1d(width, width, kernel_size=3, padding=2, dilation=2)
        self.norm = nn.GroupNorm(1, width)
        self.activation = resolve_activation(activation)()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.norm(x + self.local(x) + self.dilated(x)))


class DisplacementSetTransformerOperator(nn.Module):
    def __init__(
        self,
        num_res: int,
        width: int = 56,
        heads: int = 4,
        depth: int = 2,
        ff_dim: int = 144,
        modal_harmonics: int = 4,
        dropout: float = 0.1,
        activation: str | type[nn.Module] = "gelu",
    ) -> None:
        super().__init__()
        self.num_res = int(num_res)
        self.modal_harmonics = int(modal_harmonics)
        activation_cls = resolve_activation(activation)
        node_input_dim = 5 + 2 * self.modal_harmonics + self.modal_harmonics**2
        self.node_lift = MLP([node_input_dim, width, width], activation=activation_cls)
        layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=heads, dim_feedforward=ff_dim, dropout=dropout,
            activation=activation_cls(), batch_first=True, norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.frequency_query = MLP([1, width, width], activation=activation_cls)
        self.cross_attention = DetuningCrossAttention(width, heads, dropout=dropout, activation=activation_cls)
        self.cross_norm = nn.LayerNorm(width)
        self.frequency_mixer = FrequencyMixer(width, activation=activation_cls)
        self.output = MLP([width, width, width // 2, 4], activation=activation_cls)

    def forward(
        self, configuration: torch.Tensor, frequency: torch.Tensor, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        node_features = physics_aware_resonator_features(configuration, harmonics=self.modal_harmonics)
        tokens = self.encoder(self.node_lift(node_features))
        query = self.frequency_query(frequency)
        attended = self.cross_attention(query, tokens, configuration, frequency, x, y)
        h = self.cross_norm(query + attended)
        h = self.frequency_mixer(h.transpose(1, 2)).transpose(1, 2)
        return self.output(h)


def build_model(num_res: int, **kwargs) -> DisplacementSetTransformerOperator:
    return DisplacementSetTransformerOperator(num_res=num_res, **kwargs)


DEFAULT_MODEL_CONFIG = {
    "width": 56,
    "heads": 4,
    "depth": 2,
    "ff_dim": 144,
    "modal_harmonics": 4,
    "dropout": 0.1,
    "activation": "gelu",
}
