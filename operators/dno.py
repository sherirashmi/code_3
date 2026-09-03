"""Peak-aware Deep Neural Operator (DNO) for ERP spectrum prediction."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.neural_operator_utils import (
    MLP,
    ResidualMLPBlock,
    ResonanceQueryEncoder,
    ResonatorSetEncoder,
    run_operator_experiment,
)


class FiLMResidualBlock(nn.Module):
    """Residual MLP block modulated by configuration/query conditioning."""

    def __init__(self, width: int, condition_dim: int) -> None:
        super().__init__()
        self.block = ResidualMLPBlock(width)
        self.film = nn.Linear(condition_dim, 2 * width)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        h = self.block(x)
        gamma, beta = self.film(condition).chunk(2, dim=-1)
        gamma = 1.0 + 0.20 * torch.tanh(gamma)
        beta = 0.10 * beta
        return F.silu(gamma * h + beta)


class FrequencyRefinement1d(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(width, width, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(width, width, kernel_size=3, padding=1),
        )
        self.norm = nn.GroupNorm(1, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(self.norm(x + self.net(x)))


class DNO(nn.Module):
    """Deep residual operator with detuning-aware FiLM and frequency refinement."""

    def __init__(
        self,
        num_res: int,
        hidden_dim: int = 76,
        context_dim: int = 76,
        frequency_dim: int = 38,
        query_dim: int = 38,
        depth: int = 4,
    ) -> None:
        super().__init__()
        self.num_res = int(num_res)
        self.configuration_encoder = ResonatorSetEncoder(
            hidden_dim=hidden_dim,
            element_dim=hidden_dim,
            output_dim=context_dim,
        )
        self.frequency_encoder = MLP(
            [1, frequency_dim, frequency_dim], activation=nn.SiLU
        )
        self.resonance_query = ResonanceQueryEncoder(
            hidden_dim=query_dim,
            element_dim=query_dim,
            output_dim=query_dim,
        )
        self.lift = nn.Linear(context_dim + frequency_dim + query_dim, hidden_dim)
        condition_dim = context_dim + query_dim
        self.blocks = nn.ModuleList(
            [FiLMResidualBlock(hidden_dim, condition_dim) for _ in range(depth)]
        )
        self.frequency_refinement = FrequencyRefinement1d(hidden_dim)
        self.output = MLP([hidden_dim, hidden_dim // 2, 1], activation=nn.SiLU)

    def forward(self, configuration: torch.Tensor, frequency: torch.Tensor) -> torch.Tensor:
        context_global = self.configuration_encoder(configuration)
        context = context_global[:, None, :].expand(-1, frequency.shape[1], -1)
        freq = self.frequency_encoder(frequency)
        query = self.resonance_query(configuration, frequency)

        h = F.silu(self.lift(torch.cat((context, freq, query), dim=-1)))
        condition = torch.cat((context, query), dim=-1)
        for block in self.blocks:
            h = block(h, condition)

        h = self.frequency_refinement(h.transpose(1, 2)).transpose(1, 2)
        return self.output(h)


def build_model(num_res: int, **kwargs) -> DNO:
    return DNO(num_res=num_res, **kwargs)


DEFAULT_MODEL_CONFIG = {
    "hidden_dim": 76,
    "context_dim": 76,
    "frequency_dim": 38,
    "query_dim": 38,
    "depth": 4,
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
        operator_name="DNO",
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
