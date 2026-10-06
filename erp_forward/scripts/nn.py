"""Plain feedforward neural-network baseline for ERP spectrum prediction.

Every other file in ``erp_forward/scripts/`` implements a specific neural-operator
idea (branch/trunk product, spectral/wavelet transform, message passing,
attention, sinusoidal representation, ...). This file is the "does any of
that sophistication even matter" baseline: a bog-standard stack of
Linear -> ReLU -> Dropout layers, no residual connections, no FiLM
conditioning, no branch/trunk split, no spectral or wavelet transform, no
attention, no message passing -- and, unlike this project's other shared-
tooling convention, NO physics-aware feature encoders either: no
``ResonatorSetEncoder`` (permutation-invariant resonator-set pooling), no
``ResonanceQueryEncoder`` (per-query detuning features), no
``FrequencyRefinement1d`` (cross-frequency conv mixing). Every one of those
is itself a piece of architectural sophistication (set-pooling, attention-
like query interaction, local convolution) -- keeping them would make this
"baseline" secretly not a plain MLP at all.

Input is the RAW ``[m,k,f_t,x,y]`` configuration (already normalized
upstream, just flattened across resonators) concatenated with the RAW
query frequency, fed straight into an ordinary MLP applied independently
at each frequency point (the same weights are reused across every query
frequency -- there's no cross-frequency interaction anywhere in this model).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from erp_forward.scripts.neural_operator_utils import resolve_activation, run_operator_experiment


class NN(nn.Module):
    """Plain feedforward MLP: Linear -> ReLU -> Dropout, stacked ``depth`` times.

    Takes the flattened raw configuration (``num_res*5`` values) and the raw
    query frequency (1 value) as input -- no learned feature encoders of any
    kind upstream of the MLP itself.
    """

    def __init__(
        self,
        num_res: int,
        hidden_dim: int = 128,
        depth: int = 6,
        dropout: float = 0.0,
        activation: str | type[nn.Module] = "relu",
    ) -> None:
        super().__init__()
        if depth <= 0:
            raise ValueError("depth must be positive.")
        self.num_res = int(num_res)
        activation_cls = resolve_activation(activation)

        layers: list[nn.Module] = []
        in_dim = self.num_res * 5 + 1  # flattened [m,k,f_t,x,y]*num_res + query frequency
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(activation_cls())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        self.mlp = nn.Sequential(*layers)
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, configuration: torch.Tensor, frequency: torch.Tensor) -> torch.Tensor:
        """``configuration``: ``(batch, num_res, 5)``. ``frequency``: ``(batch, n_freq, 1)``."""
        batch, n_freq = frequency.shape[0], frequency.shape[1]
        flat_configuration = configuration.reshape(batch, 1, self.num_res * 5).expand(-1, n_freq, -1)

        h = torch.cat((flat_configuration, frequency), dim=-1)
        h = self.mlp(h)
        return self.output(h)


def build_model(num_res: int, **kwargs) -> NN:
    return NN(num_res=num_res, **kwargs)


DEFAULT_MODEL_CONFIG = {
    "hidden_dim": 148,
    "depth": 6,
    "dropout": 0.1,
    "activation": "relu",
}


def main(
    action: str = "train",
    num_configurations: int = 500,
    batch_size: int = 16,
    epochs: int = 200,
    learning_rate: float = 5e-4,
    dataset_file: str = "datasets/erp/3res/10k/dataset_erp_ft.pth",
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
