"""Peak-aware Deep Cat Operator (DCO) for ERP spectrum prediction."""

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


class FrequencyRefinement1d(nn.Module):
    """Residual local frequency mixer used to sharpen neighboring ERP points."""

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


class DCO(nn.Module):
    """DeepCat operator with explicit resonance-query features and local mixing."""

    def __init__(
        self,
        num_res: int,
        hidden_dim: int = 64,
        branch_dim: int = 64,
        trunk_dim: int = 64,
        query_dim: int = 32,
        depth: int = 4,
    ) -> None:
        super().__init__()
        self.num_res = int(num_res)
        self.branch = ResonatorSetEncoder(
            hidden_dim=hidden_dim,
            element_dim=hidden_dim,
            output_dim=branch_dim,
        )
        self.trunk = MLP([1, hidden_dim, trunk_dim], activation=nn.SiLU)
        self.resonance_query = ResonanceQueryEncoder(
            hidden_dim=query_dim,
            element_dim=query_dim,
            output_dim=query_dim,
        )
        self.lift = nn.Linear(branch_dim + trunk_dim + query_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [ResidualMLPBlock(hidden_dim) for _ in range(depth)]
        )
        self.frequency_refinement = FrequencyRefinement1d(hidden_dim)
        self.output = MLP([hidden_dim, hidden_dim // 2, 1], activation=nn.SiLU)

    def forward(self, configuration: torch.Tensor, frequency: torch.Tensor) -> torch.Tensor:
        branch = self.branch(configuration)[:, None, :]
        branch = branch.expand(-1, frequency.shape[1], -1)
        trunk = self.trunk(frequency)
        query = self.resonance_query(configuration, frequency)

        h = F.silu(self.lift(torch.cat((branch, trunk, query), dim=-1)))
        for block in self.blocks:
            h = block(h)
        h = self.frequency_refinement(h.transpose(1, 2)).transpose(1, 2)
        return self.output(h)


def build_model(num_res: int, **kwargs) -> DCO:
    return DCO(num_res=num_res, **kwargs)


DEFAULT_MODEL_CONFIG = {
    "hidden_dim": 64,
    "branch_dim": 64,
    "trunk_dim": 64,
    "query_dim": 32,
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
        operator_name="DCO",
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
