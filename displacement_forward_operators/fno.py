"""Displacement-field Fourier Neural Operator (FNO): (configuration,
frequency, x, y) -> (v_real, v_imag, M_real, M_imag).

Adapted from erp_forward_operators/fno.py. Unchanged: the SpectralConv1d
Fourier-mode-truncation mixing runs along the frequency axis exactly as
before -- this is the one architecture where preserving that axis as a
shared, ordered grid mattered most (see displacement_operator_utils.py's
module docstring: rfft along a scattered axis would be meaningless).
Changed: branch context comes from FieldContextEncoder (position-aware)
instead of ResonatorSetEncoder alone; the query uses
FieldResonanceQueryEncoder (adds spatial distance to each resonator);
final projection outputs 4 channels, not 1.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from erp_forward_operators.neural_operator_utils import resolve_activation

from .displacement_operator_utils import FieldContextEncoder, FieldResonanceQueryEncoder


class SpectralConv1d(nn.Module):
    """Learned convolution on retained Fourier modes (unchanged from erp_forward_operators/fno.py)."""

    def __init__(self, in_channels: int, out_channels: int, modes: int) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.modes = int(modes)
        scale = 1.0 / max(1, in_channels * out_channels)
        weight = scale * torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat)
        self.weight = nn.Parameter(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[-1]
        x_ft = torch.fft.rfft(x, dim=-1)
        n_modes = min(self.modes, x_ft.shape[-1])
        out_ft = torch.zeros(x.shape[0], self.out_channels, x_ft.shape[-1], device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :n_modes] = torch.einsum("bim,iom->bom", x_ft[:, :, :n_modes], self.weight[:, :, :n_modes])
        return torch.fft.irfft(out_ft, n=n, dim=-1)


class FNOBlock1d(nn.Module):
    """Global Fourier mixing plus local kernel-3 peak refinement (unchanged)."""

    def __init__(self, width: int, modes: int, dropout: float = 0.0, activation: str | type[nn.Module] = "gelu") -> None:
        super().__init__()
        self.spectral = SpectralConv1d(width, width, modes)
        self.local = nn.Conv1d(width, width, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(1, width)
        self.activation = resolve_activation(activation)()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x + self.spectral(x) + self.local(x)
        return self.dropout(self.activation(self.norm(y)))


class DisplacementFNO(nn.Module):
    def __init__(
        self,
        num_res: int,
        width: int = 23,
        modes: int = 35,
        depth: int = 4,
        config_hidden: int = 70,
        query_dim: int = 26,
        padding: int = 8,
        dropout: float = 0.1,
        activation: str | type[nn.Module] = "gelu",
    ) -> None:
        super().__init__()
        self.num_res = int(num_res)
        self.padding = int(padding)
        activation_cls = resolve_activation(activation)
        self.field_encoder = FieldContextEncoder(output_dim=width, resonator_hidden=config_hidden)
        self.resonance_query = FieldResonanceQueryEncoder(
            hidden_dim=query_dim, element_dim=query_dim, output_dim=query_dim
        )
        self.lift = nn.Linear(width + query_dim + 1, width)
        self.blocks = nn.ModuleList(
            [FNOBlock1d(width, modes, dropout=dropout, activation=activation_cls) for _ in range(depth)]
        )
        self.project = nn.Sequential(nn.Linear(width, width), activation_cls(), nn.Linear(width, 4))

    def forward(
        self, configuration: torch.Tensor, frequency: torch.Tensor, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        context = self.field_encoder(configuration, x, y)[:, None, :]
        context = context.expand(-1, frequency.shape[1], -1)
        query = self.resonance_query(configuration, frequency, x, y)
        h = self.lift(torch.cat((context, query, frequency), dim=-1)).transpose(1, 2)

        if self.padding > 0:
            h = F.pad(h, (self.padding, self.padding), mode="replicate")
        for block in self.blocks:
            h = block(h)
        if self.padding > 0:
            h = h[..., self.padding : -self.padding]

        return self.project(h.transpose(1, 2))


def build_model(num_res: int, **kwargs) -> DisplacementFNO:
    return DisplacementFNO(num_res=num_res, **kwargs)


DEFAULT_MODEL_CONFIG = {
    "width": 23,
    "modes": 35,
    "depth": 4,
    "config_hidden": 70,
    "query_dim": 26,
    "padding": 8,
    "dropout": 0.1,
    "activation": "gelu",
}
