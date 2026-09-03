"""Peak-aware SIREN neural operator for ERP spectra."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from utils.neural_operator_utils import (
    ResonanceQueryEncoder,
    ResonatorSetEncoder,
    run_operator_experiment,
)


class ModulatedSineLayer(nn.Module):
    """SIREN layer with configuration-dependent FiLM modulation."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        context_dim: int,
        *,
        first: bool = False,
        omega_0: float = 20.0,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.omega_0 = float(omega_0)
        self.linear = nn.Linear(in_features, out_features)
        self.film = nn.Linear(context_dim, 2 * out_features)

        with torch.no_grad():
            if first:
                bound = 1.0 / max(1, in_features)
            else:
                bound = math.sqrt(6.0 / max(1, in_features)) / self.omega_0
            self.linear.weight.uniform_(-bound, bound)
            self.linear.bias.uniform_(-bound, bound)
            # Start near an ordinary SIREN and learn modulation gradually.
            self.film.weight.zero_()
            self.film.bias.zero_()

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.film(context).chunk(2, dim=-1)
        gamma = 1.0 + 0.25 * torch.tanh(gamma)[:, None, :]
        beta = 0.10 * beta[:, None, :]
        preactivation = gamma * self.linear(x) + beta
        return torch.sin(self.omega_0 * preactivation)


class SIRENOperator(nn.Module):
    """Continuous operator with detuning features and layer-wise configuration modulation."""

    def __init__(
        self,
        num_res: int,
        context_dim: int = 104,
        query_dim: int = 26,
        hidden_dim: int = 104,
        depth: int = 4,
        omega_0: float = 20.0,
    ) -> None:
        super().__init__()
        if depth <= 0:
            raise ValueError("depth must be positive.")
        self.num_res = int(num_res)
        self.omega_0 = float(omega_0)
        self.configuration_encoder = ResonatorSetEncoder(
            hidden_dim=context_dim,
            element_dim=context_dim,
            output_dim=context_dim,
        )
        self.resonance_query = ResonanceQueryEncoder(
            hidden_dim=2 * query_dim,
            element_dim=2 * query_dim,
            output_dim=query_dim,
        )

        input_dim = 1 + query_dim
        layers: list[nn.Module] = [
            ModulatedSineLayer(
                input_dim,
                hidden_dim,
                context_dim,
                first=True,
                omega_0=omega_0,
            )
        ]
        for _ in range(depth - 1):
            layers.append(
                ModulatedSineLayer(
                    hidden_dim,
                    hidden_dim,
                    context_dim,
                    omega_0=omega_0,
                )
            )
        self.layers = nn.ModuleList(layers)
        self.output = nn.Linear(hidden_dim, 1)

        with torch.no_grad():
            bound = math.sqrt(6.0 / hidden_dim) / self.omega_0
            self.output.weight.uniform_(-bound, bound)
            self.output.bias.zero_()

    def forward(self, configuration: torch.Tensor, frequency: torch.Tensor) -> torch.Tensor:
        context = self.configuration_encoder(configuration)
        query_features = self.resonance_query(configuration, frequency)
        h = torch.cat((frequency, query_features), dim=-1)
        for layer in self.layers:
            h = layer(h, context)
        return self.output(h)


def build_model(num_res: int, **kwargs) -> SIRENOperator:
    return SIRENOperator(num_res=num_res, **kwargs)


DEFAULT_MODEL_CONFIG = {
    "context_dim": 104,
    "query_dim": 26,
    "hidden_dim": 104,
    "depth": 4,
    "omega_0": 20.0,
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
