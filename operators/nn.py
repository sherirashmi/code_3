"""Plain feedforward neural-network baseline for ERP spectrum prediction.

Every other file in ``operators/`` implements a specific neural-operator
idea (branch/trunk product, spectral/wavelet transform, message passing,
attention, sinusoidal representation, ...). This file is the "does any of
that sophistication even matter" baseline: a bog-standard stack of
Linear -> ReLU -> Dropout layers, no residual connections, no FiLM
conditioning, no branch/trunk split, no spectral or wavelet transform, no
attention, no message passing.

It is still given exactly the same *information* every other architecture
in this project was equalized to have (see the fairness pass):

- physics-aware resonator harmonics via the shared ``ResonatorSetEncoder``
- explicit per-query detuning features via the shared ``ResonanceQueryEncoder``
- the shared local ``FrequencyRefinement1d`` cross-frequency mixing stage

so any accuracy gap between NN and DON/DNO/DCO/FNO/WNO/GNO/STO/SIREN
reflects architecture, not access to different inputs. Its capacity is
matched to the same ~110K-parameter budget the other 8 operators were
tuned to for the same reason.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from operators.neural_operator_utils import (
    FrequencyRefinement1d,
    ResonanceQueryEncoder,
    ResonatorSetEncoder,
    resolve_activation,
    run_operator_experiment,
)


class NN(nn.Module):
    """Plain feedforward MLP: Linear -> ReLU -> Dropout, stacked ``depth`` times."""

    def __init__(
        self,
        num_res: int,
        hidden_dim: int = 191,
        depth: int = 6,
        context_dim: int = 128,
        query_dim: int = 64,
        dropout: float = 0.0,
        activation: str | type[nn.Module] = "relu",
    ) -> None:
        super().__init__()
        if depth <= 0:
            raise ValueError("depth must be positive.")
        self.num_res = int(num_res)
        activation_cls = resolve_activation(activation)
        self.configuration_encoder = ResonatorSetEncoder(
            hidden_dim=context_dim,
            element_dim=context_dim,
            output_dim=context_dim,
        )
        self.resonance_query = ResonanceQueryEncoder(
            hidden_dim=query_dim,
            element_dim=query_dim,
            output_dim=query_dim,
        )

        layers: list[nn.Module] = []
        in_dim = context_dim + query_dim + 1
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(activation_cls())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        self.mlp = nn.Sequential(*layers)

        # Every other operator in this project mixes neighboring frequency
        # samples somewhere; this baseline gets the same shared block so it
        # isn't handicapped relative to the rest by a missing tool.
        self.frequency_refinement = FrequencyRefinement1d(hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, configuration: torch.Tensor, frequency: torch.Tensor) -> torch.Tensor:
        context = self.configuration_encoder(configuration)[:, None, :]
        context = context.expand(-1, frequency.shape[1], -1)
        query = self.resonance_query(configuration, frequency)

        h = torch.cat((context, query, frequency), dim=-1)
        h = self.mlp(h)
        h = self.frequency_refinement(h.transpose(1, 2)).transpose(1, 2)
        return self.output(h)


def build_model(num_res: int, **kwargs) -> NN:
    return NN(num_res=num_res, **kwargs)


# All size dimensions scaled down proportionally from the ~550K-matched
# config to the project's new ~110K budget (was hidden_dim=191 -> ~549K).
DEFAULT_MODEL_CONFIG = {
    "hidden_dim": 85,
    "depth": 6,
    "context_dim": 57,
    "query_dim": 28,
    "dropout": 0.1,
    "activation": "relu",
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
        operator_name="NN",
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
