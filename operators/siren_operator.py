"""Peak-aware SIREN neural operator for ERP spectra."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from utils.neural_operator_utils import (
    FrequencyRefinement1d,
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
        context_dim: int = 128,
        query_dim: int = 32,
        hidden_dim: int = 128,
        depth: int = 4,
        omega_0: float = 20.0,
        config_hidden: int = 128,
        query_hidden: int = 64,
    ) -> None:
        super().__init__()
        if depth <= 0:
            raise ValueError("depth must be positive.")
        self.num_res = int(num_res)
        self.omega_0 = float(omega_0)
        # config_hidden/query_hidden were hardcoded at 128/64 here regardless
        # of any other size parameter -- a real fixed floor that made SIREN
        # unable to reach a smaller parameter budget purely by shrinking
        # context_dim/query_dim/hidden_dim, unlike every other architecture
        # in this project (whose ResonatorSetEncoder/ResonanceQueryEncoder
        # calls already tie their internal width to a scalable parameter).
        # Exposing them keeps SIREN consistent with the rest.
        self.configuration_encoder = ResonatorSetEncoder(
            hidden_dim=config_hidden,
            element_dim=config_hidden,
            output_dim=context_dim,
        )
        self.resonance_query = ResonanceQueryEncoder(
            hidden_dim=query_hidden,
            element_dim=query_hidden,
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
        # Every other operator in this project mixes neighboring frequency
        # samples somewhere (a local conv "refinement" stage); a plain SIREN
        # predicts each query frequency independently. Adding the same shared
        # refinement block here (as a residual on top of the sine stack) gives
        # SIREN parity with the rest rather than leaving it the only
        # purely-pointwise architecture in the comparison.
        self.frequency_refinement = FrequencyRefinement1d(hidden_dim)
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
        h = self.frequency_refinement(h.transpose(1, 2)).transpose(1, 2)
        return self.output(h)


def build_model(num_res: int, **kwargs) -> SIRENOperator:
    return SIRENOperator(num_res=num_res, **kwargs)


# All size dimensions -- including config_hidden/query_hidden, now that
# they're real parameters instead of hardcoded 128/64 -- scaled down
# proportionally from the ~550K-matched config to the project's new ~110K
# budget (was hidden_dim=170 -> ~549K).
DEFAULT_MODEL_CONFIG = {
    "context_dim": 57,
    "query_dim": 14,
    "hidden_dim": 75,
    "config_hidden": 57,
    "query_hidden": 28,
    "depth": 4,
    "omega_0": 20.0,
}

# Search space for random_search_operator(): explores SIREN's own knobs at a
# fixed (parameter-matched) hidden_dim. omega_0 is the most sensitive SIREN
# hyperparameter (Sitzmann et al.) so it's included even though it wasn't
# swept before.
SEARCH_SPACE = {
    "omega_0": [10.0, 20.0, 30.0],
    "depth": [3, 4, 5, 6],
    "query_dim": [11, 14, 21],
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
