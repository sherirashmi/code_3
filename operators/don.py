"""Peak-aware DeepONet (DON) for ERP spectrum prediction."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from utils.neural_operator_utils import MLP, ResonatorSetEncoder, run_operator_experiment


class FourierFrequencyEncoder(nn.Module):
    """Fixed multi-scale Fourier features for the normalized query frequency."""

    def __init__(self, num_bands: int = 6) -> None:
        super().__init__()
        if num_bands <= 0:
            raise ValueError("num_bands must be positive.")
        scales = math.pi * (2.0 ** torch.arange(num_bands, dtype=torch.float32))
        self.register_buffer("scales", scales)

    @property
    def output_dim(self) -> int:
        return 1 + 2 * int(self.scales.numel())

    def forward(self, frequency: torch.Tensor) -> torch.Tensor:
        angles = frequency * self.scales
        return torch.cat((frequency, torch.sin(angles), torch.cos(angles)), dim=-1)


class DON(nn.Module):
    """Physics-aware DeepONet with a configuration-modulated Fourier trunk.

    The classical branch/trunk inner product is retained, but the trunk receives
    multi-scale Fourier frequency features and is gently modulated by the full
    resonator configuration.  This makes moving and narrow resonances easier to
    represent than with a single fixed low-rank frequency basis.
    """

    def __init__(
        self,
        num_res: int,
        hidden_dim: int = 108,
        context_dim: int = 135,
        basis_dim: int = 216,
        fourier_bands: int = 6,
    ) -> None:
        super().__init__()
        self.num_res = int(num_res)
        self.frequency_features = FourierFrequencyEncoder(fourier_bands)
        self.configuration_encoder = ResonatorSetEncoder(
            hidden_dim=hidden_dim,
            element_dim=hidden_dim,
            output_dim=context_dim,
        )
        self.branch_head = MLP(
            [context_dim, hidden_dim, basis_dim], activation=nn.SiLU
        )
        self.trunk = MLP(
            [self.frequency_features.output_dim, hidden_dim, hidden_dim, basis_dim],
            activation=nn.SiLU,
        )
        self.trunk_modulation = MLP(
            [context_dim, hidden_dim, 2 * basis_dim], activation=nn.SiLU
        )
        self.bias = nn.Parameter(torch.zeros(1))
        self.scale = math.sqrt(float(basis_dim))

    def forward(self, configuration: torch.Tensor, frequency: torch.Tensor) -> torch.Tensor:
        context = self.configuration_encoder(configuration)
        branch = self.branch_head(context)[:, None, :]  # (B,1,P)

        trunk = self.trunk(self.frequency_features(frequency))  # (B,F,P)
        gamma, beta = self.trunk_modulation(context).chunk(2, dim=-1)
        gamma = 1.0 + 0.25 * torch.tanh(gamma[:, None, :])
        beta = 0.10 * beta[:, None, :]
        trunk = gamma * trunk + beta

        output = (branch * trunk).sum(dim=-1, keepdim=True) / self.scale
        return output + self.bias


def build_model(num_res: int, **kwargs) -> DON:
    return DON(num_res=num_res, **kwargs)


DEFAULT_MODEL_CONFIG = {
    "hidden_dim": 108,
    "context_dim": 135,
    "basis_dim": 216,
    "fourier_bands": 6,
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
        operator_name="DON",
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
