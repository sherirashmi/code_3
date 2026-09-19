"""Displacement-field Wavelet Neural Operator (WNO): (configuration,
frequency, x, y) -> (displacement_real, displacement_imag).

Adapted from forward_operators/wno.py. Unchanged: the learned multi-level
Haar wavelet analysis/mixing/synthesis blocks run along the frequency axis
exactly as before -- valid for the same reason FNO's spectral conv stays
valid (frequency is still a shared, ordered grid; see
displacement_operator_utils.py's module docstring). Changed: branch
context from FieldContextEncoder (position-aware); query from
FieldResonanceQueryEncoder; final projection outputs 2 channels, not 1.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from forward_operators.neural_operator_utils import resolve_activation

from .displacement_operator_utils import FieldContextEncoder, FieldResonanceQueryEncoder


class MultiLevelHaarWaveletBlock1d(nn.Module):
    """Learned multi-scale Haar analysis/mixing/synthesis residual block (unchanged)."""

    def __init__(self, width: int, levels: int = 3, dropout: float = 0.0, activation: str | type[nn.Module] = "gelu") -> None:
        super().__init__()
        if levels <= 0:
            raise ValueError("levels must be positive.")
        self.levels = int(levels)
        self.low_mix = nn.ModuleList([nn.Conv1d(width, width, kernel_size=3, padding=1) for _ in range(levels)])
        self.high_mix = nn.ModuleList([nn.Conv1d(width, width, kernel_size=3, padding=1) for _ in range(levels)])
        self.coarse_mix = nn.Conv1d(width, width, kernel_size=3, padding=1)
        self.local = nn.Conv1d(width, width, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(1, width)
        self.activation = resolve_activation(activation)()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inv_sqrt2 = 1.0 / math.sqrt(2.0)
        current = x
        details: list[torch.Tensor] = []
        original_lengths: list[int] = []

        for level in range(self.levels):
            original_n = current.shape[-1]
            original_lengths.append(original_n)
            if original_n % 2:
                padded = F.pad(current, (0, 1), mode="replicate")
            else:
                padded = current

            even = padded[..., 0::2]
            odd = padded[..., 1::2]
            low = (even + odd) * inv_sqrt2
            high = (even - odd) * inv_sqrt2

            low = self.activation(self.low_mix[level](low))
            high = self.activation(self.high_mix[level](high))
            details.append(high)
            current = low

        current = self.activation(self.coarse_mix(current))

        for level in reversed(range(self.levels)):
            high = details[level]
            even = (current + high) * inv_sqrt2
            odd = (current - high) * inv_sqrt2
            reconstructed = torch.empty(*even.shape[:-1], even.shape[-1] * 2, device=x.device, dtype=x.dtype)
            reconstructed[..., 0::2] = even
            reconstructed[..., 1::2] = odd
            current = reconstructed[..., : original_lengths[level]]

        return self.dropout(self.activation(self.norm(x + current + self.local(x))))


class DisplacementWNO(nn.Module):
    def __init__(
        self,
        num_res: int,
        width: int = 30,
        depth: int = 4,
        levels: int = 3,
        config_hidden: int = 57,
        query_dim: int = 22,
        dropout: float = 0.1,
        activation: str | type[nn.Module] = "gelu",
    ) -> None:
        super().__init__()
        self.num_res = int(num_res)
        activation_cls = resolve_activation(activation)
        self.field_encoder = FieldContextEncoder(output_dim=width, resonator_hidden=config_hidden)
        self.resonance_query = FieldResonanceQueryEncoder(
            hidden_dim=query_dim, element_dim=query_dim, output_dim=query_dim
        )
        self.lift = nn.Linear(width + query_dim + 1, width)
        self.blocks = nn.ModuleList(
            [MultiLevelHaarWaveletBlock1d(width, levels=levels, dropout=dropout, activation=activation_cls) for _ in range(depth)]
        )
        self.project = nn.Sequential(nn.Linear(width, width), activation_cls(), nn.Linear(width, 2))

    def forward(
        self, configuration: torch.Tensor, frequency: torch.Tensor, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        context = self.field_encoder(configuration, x, y)[:, None, :]
        context = context.expand(-1, frequency.shape[1], -1)
        query = self.resonance_query(configuration, frequency, x, y)
        h = self.lift(torch.cat((context, query, frequency), dim=-1)).transpose(1, 2)
        for block in self.blocks:
            h = block(h)
        return self.project(h.transpose(1, 2))


def build_model(num_res: int, **kwargs) -> DisplacementWNO:
    return DisplacementWNO(num_res=num_res, **kwargs)


DEFAULT_MODEL_CONFIG = {
    "width": 30,
    "depth": 4,
    "levels": 3,
    "config_hidden": 57,
    "query_dim": 22,
    "dropout": 0.1,
    "activation": "gelu",
}
