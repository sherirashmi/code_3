"""Deep Cat Operator (DCO) for ERP spectrum prediction."""

from __future__ import annotations

import torch
import torch.nn as nn

from utils.neural_operator_utils import MLP, ResonatorSetEncoder, run_operator_experiment


class DCO(nn.Module):
    """DeepCat-style operator.

    Separate branch and trunk encoders are concatenated and passed through a
    nonlinear Cat network, rather than combined by the DeepONet inner product.
    """

    def __init__(
        self,
        num_res: int,
        hidden_dim: int = 128,
        branch_dim: int = 128,
        trunk_dim: int = 128,
    ) -> None:
        super().__init__()
        self.num_res = int(num_res)
        self.branch = ResonatorSetEncoder(
            hidden_dim=hidden_dim,
            element_dim=hidden_dim,
            output_dim=branch_dim,
        )
        self.trunk = MLP([1, hidden_dim, trunk_dim], activation=nn.SiLU)
        self.cat_net = MLP(
            [branch_dim + trunk_dim, hidden_dim, hidden_dim, hidden_dim // 2, 1],
            activation=nn.SiLU,
        )

    def forward(self, configuration: torch.Tensor, frequency: torch.Tensor) -> torch.Tensor:
        branch = self.branch(configuration)[:, None, :]
        branch = branch.expand(-1, frequency.shape[1], -1)
        trunk = self.trunk(frequency)
        return self.cat_net(torch.cat((branch, trunk), dim=-1))


def build_model(num_res: int, **kwargs) -> DCO:
    return DCO(num_res=num_res, **kwargs)


DEFAULT_MODEL_CONFIG = {
    "hidden_dim": 128,
    "branch_dim": 128,
    "trunk_dim": 128,
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
