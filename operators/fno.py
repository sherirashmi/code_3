"""1D Fourier Neural Operator (FNO) for ERP spectra."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.neural_operator_utils import ResonatorSetEncoder, run_operator_experiment


class SpectralConv1d(nn.Module):
    """Learned convolution on the lowest Fourier modes."""

    def __init__(self, in_channels: int, out_channels: int, modes: int) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.modes = int(modes)
        scale = 1.0 / max(1, in_channels * out_channels)
        weight = scale * torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat)
        self.weight = nn.Parameter(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,F)
        n = x.shape[-1]
        x_ft = torch.fft.rfft(x, dim=-1)
        n_modes = min(self.modes, x_ft.shape[-1])
        out_ft = torch.zeros(
            x.shape[0], self.out_channels, x_ft.shape[-1],
            device=x.device, dtype=torch.cfloat,
        )
        out_ft[:, :, :n_modes] = torch.einsum(
            "bim,iom->bom",
            x_ft[:, :, :n_modes],
            self.weight[:, :, :n_modes],
        )
        return torch.fft.irfft(out_ft, n=n, dim=-1)


class FNOBlock1d(nn.Module):
    def __init__(self, width: int, modes: int) -> None:
        super().__init__()
        self.spectral = SpectralConv1d(width, width, modes)
        self.local = nn.Conv1d(width, width, kernel_size=1)
        self.norm = nn.GroupNorm(1, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.spectral(x) + self.local(x)
        return F.gelu(self.norm(y))


class FNO(nn.Module):
    """Permutation-invariant configuration encoder + 1D FNO on frequency grid."""

    def __init__(
        self,
        num_res: int,
        width: int = 64,
        modes: int = 32,
        depth: int = 4,
        config_hidden: int = 128,
    ) -> None:
        super().__init__()
        self.num_res = int(num_res)
        self.configuration_encoder = ResonatorSetEncoder(
            hidden_dim=config_hidden,
            element_dim=config_hidden,
            output_dim=width,
        )
        self.lift = nn.Linear(width + 1, width)
        self.blocks = nn.ModuleList([FNOBlock1d(width, modes) for _ in range(depth)])
        self.project = nn.Sequential(
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, 1),
        )

    def forward(self, configuration: torch.Tensor, frequency: torch.Tensor) -> torch.Tensor:
        context = self.configuration_encoder(configuration)[:, None, :]
        context = context.expand(-1, frequency.shape[1], -1)
        x = self.lift(torch.cat((context, frequency), dim=-1))  # (B,F,C)
        x = x.transpose(1, 2)                                  # (B,C,F)
        for block in self.blocks:
            x = block(x)
        x = x.transpose(1, 2)
        return self.project(x)


def build_model(num_res: int, **kwargs) -> FNO:
    return FNO(num_res=num_res, **kwargs)


DEFAULT_MODEL_CONFIG = {
    "width": 64,
    "modes": 32,
    "depth": 4,
    "config_hidden": 128,
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
