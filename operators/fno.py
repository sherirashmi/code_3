"""Peak-aware 1D Fourier Neural Operator (FNO) for ERP spectra."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.neural_operator_utils import (
    ResonanceQueryEncoder,
    ResonatorSetEncoder,
    resolve_activation,
    run_operator_experiment,
)


class SpectralConv1d(nn.Module):
    """Learned convolution on retained Fourier modes."""

    def __init__(self, in_channels: int, out_channels: int, modes: int) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.modes = int(modes)
        scale = 1.0 / max(1, in_channels * out_channels)
        weight = scale * torch.randn(
            in_channels, out_channels, modes, dtype=torch.cfloat
        )
        self.weight = nn.Parameter(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[-1]
        x_ft = torch.fft.rfft(x, dim=-1)
        n_modes = min(self.modes, x_ft.shape[-1])
        out_ft = torch.zeros(
            x.shape[0],
            self.out_channels,
            x_ft.shape[-1],
            device=x.device,
            dtype=torch.cfloat,
        )
        out_ft[:, :, :n_modes] = torch.einsum(
            "bim,iom->bom",
            x_ft[:, :, :n_modes],
            self.weight[:, :, :n_modes],
        )
        return torch.fft.irfft(out_ft, n=n, dim=-1)


class FNOBlock1d(nn.Module):
    """Global Fourier mixing plus local kernel-3 peak refinement.

    ``dropout`` fights the overfitting FNO otherwise shows on this dataset:
    training loss reaches the lowest value of any architecture here while
    validation loss plateaus well above DCO/DNO and drifts back up late in
    training — the retained Fourier modes give it enough global capacity to
    fit each training spectrum's coefficients fairly exactly without that
    capacity transferring to held-out configurations.
    """

    def __init__(
        self,
        width: int,
        modes: int,
        dropout: float = 0.0,
        activation: str | type[nn.Module] = "gelu",
    ) -> None:
        super().__init__()
        self.spectral = SpectralConv1d(width, width, modes)
        self.local = nn.Conv1d(width, width, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(1, width)
        self.activation = resolve_activation(activation)()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x + self.spectral(x) + self.local(x)
        return self.dropout(self.activation(self.norm(y)))


class FNO(nn.Module):
    """FNO with more retained modes, local mixing, padding and detuning features."""

    def __init__(
        self,
        num_res: int,
        width: int = 64,
        modes: int = 64,
        depth: int = 4,
        config_hidden: int = 128,
        query_dim: int = 48,
        padding: int = 8,
        dropout: float = 0.1,
        activation: str | type[nn.Module] = "gelu",
    ) -> None:
        super().__init__()
        self.num_res = int(num_res)
        self.padding = int(padding)
        activation_cls = resolve_activation(activation)
        self.configuration_encoder = ResonatorSetEncoder(
            hidden_dim=config_hidden,
            element_dim=config_hidden,
            output_dim=width,
        )
        self.resonance_query = ResonanceQueryEncoder(
            hidden_dim=query_dim,
            element_dim=query_dim,
            output_dim=query_dim,
        )
        self.lift = nn.Linear(width + query_dim + 1, width)
        self.blocks = nn.ModuleList(
            [
                FNOBlock1d(width, modes, dropout=dropout, activation=activation_cls)
                for _ in range(depth)
            ]
        )
        self.project = nn.Sequential(
            nn.Linear(width, width),
            activation_cls(),
            nn.Linear(width, 1),
        )

    def forward(self, configuration: torch.Tensor, frequency: torch.Tensor) -> torch.Tensor:
        context = self.configuration_encoder(configuration)[:, None, :]
        context = context.expand(-1, frequency.shape[1], -1)
        query = self.resonance_query(configuration, frequency)
        x = self.lift(torch.cat((context, query, frequency), dim=-1)).transpose(1, 2)

        if self.padding > 0:
            x = F.pad(x, (self.padding, self.padding), mode="replicate")
        for block in self.blocks:
            x = block(x)
        if self.padding > 0:
            x = x[..., self.padding : -self.padding]

        return self.project(x.transpose(1, 2))


def build_model(num_res: int, **kwargs) -> FNO:
    return FNO(num_res=num_res, **kwargs)


# width tuned to the shared ~550K-parameter budget (was 64 -> ~1.20M params).
# modes stays 64: it indexes retained rfft modes along the frequency-query
# sequence axis, independent of channel width, so it isn't rescaled here.
DEFAULT_MODEL_CONFIG = {
    "width": 41,
    "modes": 64,
    "depth": 4,
    "config_hidden": 128,
    "query_dim": 48,
    "padding": 8,
    "dropout": 0.1,
    "activation": "gelu",
}

# Search space for random_search_operator(): explores FNO's own knobs at a
# fixed (parameter-matched) width.
SEARCH_SPACE = {
    "modes": [32, 48, 64, 80],
    "depth": [3, 4, 5],
    "padding": [4, 8, 12],
    "dropout": [0.0, 0.05, 0.1, 0.15],
    "activation": ["gelu", "silu"],
}


def main(
    action: str = "train",
    num_configurations: int = 500,
    batch_size: int = 16,
    epochs: int = 200,
    learning_rate: float = 5e-4,
    dataset_file: str = "datasets/dataset_erp_ft.pth",
    regenerate_dataset: bool = False,
    seed: int = 727,
    configuration=None,
    plot: bool = True,
):
    return run_operator_experiment(
        operator_name="FNO",
        build_model=build_model,
        model_config=DEFAULT_MODEL_CONFIG,
        action=action,
        num_configurations=num_configurations,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        dataset_file=dataset_file,
        regenerate_dataset=regenerate_dataset,
        seed=seed,
        configuration=configuration,
        plot=plot,
    )


if __name__ == "__main__":
    results = main(action="train", num_configurations=500)
