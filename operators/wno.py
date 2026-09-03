"""Multi-level Haar Wavelet Neural Operator (WNO) for ERP spectra."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.neural_operator_utils import (
    ResonanceQueryEncoder,
    ResonatorSetEncoder,
    run_operator_experiment,
)


class MultiLevelHaarWaveletBlock1d(nn.Module):
    """Learned multi-scale Haar analysis/mixing/synthesis residual block."""

    def __init__(self, width: int, levels: int = 3) -> None:
        super().__init__()
        if levels <= 0:
            raise ValueError("levels must be positive.")
        self.levels = int(levels)
        self.low_mix = nn.ModuleList(
            [nn.Conv1d(width, width, kernel_size=3, padding=1) for _ in range(levels)]
        )
        self.high_mix = nn.ModuleList(
            [nn.Conv1d(width, width, kernel_size=3, padding=1) for _ in range(levels)]
        )
        self.coarse_mix = nn.Conv1d(width, width, kernel_size=3, padding=1)
        self.local = nn.Conv1d(width, width, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(1, width)

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

            low = F.gelu(self.low_mix[level](low))
            high = F.gelu(self.high_mix[level](high))
            details.append(high)
            current = low

        current = F.gelu(self.coarse_mix(current))

        for level in reversed(range(self.levels)):
            high = details[level]
            even = (current + high) * inv_sqrt2
            odd = (current - high) * inv_sqrt2
            reconstructed = torch.empty(
                *even.shape[:-1],
                even.shape[-1] * 2,
                device=x.device,
                dtype=x.dtype,
            )
            reconstructed[..., 0::2] = even
            reconstructed[..., 1::2] = odd
            current = reconstructed[..., : original_lengths[level]]

        return F.gelu(self.norm(x + current + self.local(x)))


class WNO(nn.Module):
    """Physics-aware multi-level WNO with explicit resonance-query features."""

    def __init__(
        self,
        num_res: int,
        width: int = 44,
        depth: int = 4,
        levels: int = 3,
        config_hidden: int = 58,
        query_dim: int = 22,
    ) -> None:
        super().__init__()
        self.num_res = int(num_res)
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
            [MultiLevelHaarWaveletBlock1d(width, levels=levels) for _ in range(depth)]
        )
        self.project = nn.Sequential(
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, 1),
        )

    def forward(self, configuration: torch.Tensor, frequency: torch.Tensor) -> torch.Tensor:
        context = self.configuration_encoder(configuration)[:, None, :]
        context = context.expand(-1, frequency.shape[1], -1)
        query = self.resonance_query(configuration, frequency)
        x = self.lift(torch.cat((context, query, frequency), dim=-1)).transpose(1, 2)
        for block in self.blocks:
            x = block(x)
        return self.project(x.transpose(1, 2))


def build_model(num_res: int, **kwargs) -> WNO:
    return WNO(num_res=num_res, **kwargs)


DEFAULT_MODEL_CONFIG = {
    "width": 44,
    "depth": 4,
    "levels": 3,
    "config_hidden": 58,
    "query_dim": 22,
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
