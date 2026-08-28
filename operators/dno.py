"""Deep Neural Operator (DNO) for ERP spectrum prediction."""

from __future__ import annotations

import torch
import torch.nn as nn

from utils.neural_operator_utils import (
    MLP,
    ResidualMLPBlock,
    ResonatorSetEncoder,
    run_operator_experiment,
)


class DNO(nn.Module):
    """Deep residual neural operator on configuration context + query frequency.

    The configuration is encoded permutation-invariantly. The latent context and
    frequency embedding are then processed jointly by a deep residual network.
    """

    def __init__(
        self,
        num_res: int,
        hidden_dim: int = 128,
        context_dim: int = 128,
        frequency_dim: int = 64,
        depth: int = 4,
    ) -> None:
        super().__init__()
        self.num_res = int(num_res)
        self.configuration_encoder = ResonatorSetEncoder(
            hidden_dim=hidden_dim,
            element_dim=hidden_dim,
            output_dim=context_dim,
        )
        self.frequency_encoder = MLP([1, frequency_dim, frequency_dim], activation=nn.SiLU)
        self.lift = nn.Linear(context_dim + frequency_dim, hidden_dim)
        self.blocks = nn.ModuleList([ResidualMLPBlock(hidden_dim) for _ in range(depth)])
        self.output = MLP([hidden_dim, hidden_dim // 2, 1], activation=nn.SiLU)

    def forward(self, configuration: torch.Tensor, frequency: torch.Tensor) -> torch.Tensor:
        context = self.configuration_encoder(configuration)[:, None, :]
        context = context.expand(-1, frequency.shape[1], -1)
        freq = self.frequency_encoder(frequency)
        h = torch.cat((context, freq), dim=-1)
        h = torch.nn.functional.silu(self.lift(h))
        for block in self.blocks:
            h = block(h)
        return self.output(h)


def build_model(num_res: int, **kwargs) -> DNO:
    return DNO(num_res=num_res, **kwargs)


DEFAULT_MODEL_CONFIG = {
    "hidden_dim": 128,
    "context_dim": 128,
    "frequency_dim": 64,
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
