"""Peak-aware Deep Neural Operator (DNO) for ERP spectrum prediction."""

from __future__ import annotations

import torch
import torch.nn as nn

from utils.neural_operator_utils import (
    MLP,
    FrequencyRefinement1d,
    ResidualMLPBlock,
    ResonanceQueryEncoder,
    ResonatorSetEncoder,
    resolve_activation,
    run_operator_experiment,
)


class FiLMResidualBlock(nn.Module):
    """Residual MLP block modulated by configuration/query conditioning."""

    def __init__(
        self,
        width: int,
        condition_dim: int,
        dropout: float = 0.0,
        activation: str | type[nn.Module] = "silu",
    ) -> None:
        super().__init__()
        activation_cls = resolve_activation(activation)
        self.block = ResidualMLPBlock(width, activation=activation_cls)
        self.film = nn.Linear(condition_dim, 2 * width)
        self.activation = activation_cls()
        # Default 0.0 preserves DNO's existing (already strong) behavior;
        # non-zero only used by the hyperparameter search harness.
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        h = self.block(x)
        gamma, beta = self.film(condition).chunk(2, dim=-1)
        # This tanh is the FiLM gate's own bounding function (keeps gamma in
        # a controlled range), not the architecture's activation choice --
        # left fixed regardless of `activation`.
        gamma = 1.0 + 0.20 * torch.tanh(gamma)
        beta = 0.10 * beta
        return self.dropout(self.activation(gamma * h + beta))


class DNO(nn.Module):
    """Deep residual operator with detuning-aware FiLM and frequency refinement."""

    def __init__(
        self,
        num_res: int,
        hidden_dim: int = 128,
        context_dim: int = 128,
        frequency_dim: int = 64,
        query_dim: int = 64,
        depth: int = 4,
        dropout: float = 0.0,
        activation: str | type[nn.Module] = "silu",
    ) -> None:
        super().__init__()
        self.num_res = int(num_res)
        activation_cls = resolve_activation(activation)
        self.configuration_encoder = ResonatorSetEncoder(
            hidden_dim=hidden_dim,
            element_dim=hidden_dim,
            output_dim=context_dim,
        )
        self.frequency_encoder = MLP(
            [1, frequency_dim, frequency_dim], activation=activation_cls
        )
        self.resonance_query = ResonanceQueryEncoder(
            hidden_dim=query_dim,
            element_dim=query_dim,
            output_dim=query_dim,
        )
        self.lift = nn.Linear(context_dim + frequency_dim + query_dim, hidden_dim)
        self.lift_activation = activation_cls()
        condition_dim = context_dim + query_dim
        self.blocks = nn.ModuleList(
            [
                FiLMResidualBlock(
                    hidden_dim, condition_dim, dropout=dropout, activation=activation_cls
                )
                for _ in range(depth)
            ]
        )
        self.frequency_refinement = FrequencyRefinement1d(hidden_dim)
        self.output = MLP([hidden_dim, hidden_dim // 2, 1], activation=activation_cls)

    def forward(self, configuration: torch.Tensor, frequency: torch.Tensor) -> torch.Tensor:
        context_global = self.configuration_encoder(configuration)
        context = context_global[:, None, :].expand(-1, frequency.shape[1], -1)
        freq = self.frequency_encoder(frequency)
        query = self.resonance_query(configuration, frequency)

        h = self.lift_activation(self.lift(torch.cat((context, freq, query), dim=-1)))
        condition = torch.cat((context, query), dim=-1)
        for block in self.blocks:
            h = block(h, condition)

        h = self.frequency_refinement(h.transpose(1, 2)).transpose(1, 2)
        return self.output(h)


def build_model(num_res: int, **kwargs) -> DNO:
    return DNO(num_res=num_res, **kwargs)


# hidden_dim tuned to the shared ~550K-parameter budget (was 128 -> ~584K).
DEFAULT_MODEL_CONFIG = {
    "hidden_dim": 123,
    "context_dim": 128,
    "frequency_dim": 64,
    "query_dim": 64,
    "depth": 4,
    "dropout": 0.0,
    "activation": "silu",
}

# Search space for random_search_operator(): explores DNO's own knobs at a
# fixed (parameter-matched) hidden_dim.
SEARCH_SPACE = {
    "depth": [3, 4, 5, 6],
    "context_dim": [96, 128, 160],
    "frequency_dim": [48, 64, 96],
    "query_dim": [48, 64, 96],
    "dropout": [0.0, 0.05, 0.1],
    "activation": ["silu", "gelu", "tanh"],
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
