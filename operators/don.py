"""DeepONet (DON) for ERP spectrum prediction."""

from __future__ import annotations

import math
import torch
import torch.nn as nn

from utils.neural_operator_utils import MLP, ResonatorSetEncoder, run_operator_experiment


class DON(nn.Module):
    """Permutation-invariant DeepONet.

    Branch: unordered resonator set [f_t, x, y] -> basis coefficients.
    Trunk : evaluation frequency -> basis functions.
    Output: DeepONet inner product + learned bias.
    """

    def __init__(
        self,
        num_res: int,
        hidden_dim: int = 128,
        basis_dim: int = 128,
    ) -> None:
        super().__init__()
        self.num_res = int(num_res)
        self.branch = ResonatorSetEncoder(
            hidden_dim=hidden_dim,
            element_dim=hidden_dim,
            output_dim=basis_dim,
        )
        self.trunk = MLP([1, hidden_dim, hidden_dim, basis_dim], activation=nn.SiLU)
        self.bias = nn.Parameter(torch.zeros(1))
        self.scale = math.sqrt(float(basis_dim))

    def forward(self, configuration: torch.Tensor, frequency: torch.Tensor) -> torch.Tensor:
        branch = self.branch(configuration)[:, None, :]          # (B,1,P)
        trunk = self.trunk(frequency)                            # (B,F,P)
        output = (branch * trunk).sum(dim=-1, keepdim=True) / self.scale
        return output + self.bias


def build_model(num_res: int, **kwargs) -> DON:
    return DON(num_res=num_res, **kwargs)


DEFAULT_MODEL_CONFIG = {
    "hidden_dim": 128,
    "basis_dim": 128,
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
