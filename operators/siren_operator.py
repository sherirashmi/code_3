"""SIREN-based neural operator for sharp ERP spectra."""

from __future__ import annotations

import math
import torch
import torch.nn as nn

from utils.neural_operator_utils import ResonatorSetEncoder, run_operator_experiment


class SineLayer(nn.Module):
    """SIREN layer with the initialization proposed for periodic activations."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        first: bool = False,
        omega_0: float = 30.0,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.omega_0 = float(omega_0)
        self.linear = nn.Linear(in_features, out_features)

        with torch.no_grad():
            if first:
                bound = 1.0 / in_features
            else:
                bound = math.sqrt(6.0 / in_features) / self.omega_0
            self.linear.weight.uniform_(-bound, bound)
            self.linear.bias.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * self.linear(x))


class SIRENOperator(nn.Module):
    """Continuous frequency-coordinate operator using sinusoidal activations."""

    def __init__(
        self,
        num_res: int,
        context_dim: int = 64,
        hidden_dim: int = 128,
        depth: int = 4,
        omega_0: float = 30.0,
    ) -> None:
        super().__init__()
        self.num_res = int(num_res)
        self.configuration_encoder = ResonatorSetEncoder(
            hidden_dim=128,
            element_dim=128,
            output_dim=context_dim,
        )
        input_dim = context_dim + 1
        layers: list[nn.Module] = [
            SineLayer(input_dim, hidden_dim, first=True, omega_0=omega_0)
        ]
        for _ in range(depth - 1):
            layers.append(SineLayer(hidden_dim, hidden_dim, omega_0=omega_0))
        self.siren = nn.Sequential(*layers)
        self.output = nn.Linear(hidden_dim, 1)

        with torch.no_grad():
            bound = math.sqrt(6.0 / hidden_dim) / omega_0
            self.output.weight.uniform_(-bound, bound)
            self.output.bias.zero_()

    def forward(self, configuration: torch.Tensor, frequency: torch.Tensor) -> torch.Tensor:
        context = self.configuration_encoder(configuration)[:, None, :]
        context = context.expand(-1, frequency.shape[1], -1)
        h = self.siren(torch.cat((context, frequency), dim=-1))
        return self.output(h)


def build_model(num_res: int, **kwargs) -> SIRENOperator:
    return SIRENOperator(num_res=num_res, **kwargs)


DEFAULT_MODEL_CONFIG = {
    "context_dim": 64,
    "hidden_dim": 128,
    "depth": 4,
    "omega_0": 30.0,
}


def main(
    action: str = "train",
    num_configurations: int = 500,
    batch_size: int = 16,
    epochs: int = 250,
    learning_rate: float = 2e-4,
    dataset_file: str = "datasets/dataset_erp_ft.pth",
    regenerate_dataset: bool = False,
    seed: int = 727,
    configuration=None,
    plot: bool = True,
):
    return run_operator_experiment(
        operator_name="SIREN_NO",
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
