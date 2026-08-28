"""Lightweight 1D Haar Wavelet Neural Operator (WNO) for ERP spectra."""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.neural_operator_utils import ResonatorSetEncoder, run_operator_experiment


class HaarWaveletBlock1d(nn.Module):
    """One learned Haar analysis/mixing/synthesis operator block.

    The transform is fixed and dependency-free. Learned convolutions act on the
    low- and high-frequency wavelet coefficients before exact Haar synthesis.
    """

    def __init__(self, width: int) -> None:
        super().__init__()
        self.low_mix = nn.Conv1d(width, width, kernel_size=3, padding=1)
        self.high_mix = nn.Conv1d(width, width, kernel_size=3, padding=1)
        self.local = nn.Conv1d(width, width, kernel_size=1)
        self.norm = nn.GroupNorm(1, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,F). Haar requires pairs, so duplicate the last point if odd.
        original_n = x.shape[-1]
        if original_n % 2:
            x_pad = F.pad(x, (0, 1), mode="replicate")
        else:
            x_pad = x

        even = x_pad[..., 0::2]
        odd = x_pad[..., 1::2]
        inv_sqrt2 = 1.0 / math.sqrt(2.0)
        low = (even + odd) * inv_sqrt2
        high = (even - odd) * inv_sqrt2

        low = self.low_mix(low)
        high = self.high_mix(high)

        even_rec = (low + high) * inv_sqrt2
        odd_rec = (low - high) * inv_sqrt2
        reconstructed = torch.empty_like(x_pad)
        reconstructed[..., 0::2] = even_rec
        reconstructed[..., 1::2] = odd_rec
        reconstructed = reconstructed[..., :original_n]

        return F.gelu(self.norm(reconstructed + self.local(x)))


class WNO(nn.Module):
    """Permutation-invariant configuration encoder + Haar wavelet operator."""

    def __init__(
        self,
        num_res: int,
        width: int = 64,
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
        self.blocks = nn.ModuleList([HaarWaveletBlock1d(width) for _ in range(depth)])
        self.project = nn.Sequential(
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, 1),
        )

    def forward(self, configuration: torch.Tensor, frequency: torch.Tensor) -> torch.Tensor:
        context = self.configuration_encoder(configuration)[:, None, :]
        context = context.expand(-1, frequency.shape[1], -1)
        x = self.lift(torch.cat((context, frequency), dim=-1)).transpose(1, 2)
        for block in self.blocks:
            x = block(x)
        return self.project(x.transpose(1, 2))


def build_model(num_res: int, **kwargs) -> WNO:
    return WNO(num_res=num_res, **kwargs)


DEFAULT_MODEL_CONFIG = {
    "width": 64,
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
        operator_name="WNO",
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
