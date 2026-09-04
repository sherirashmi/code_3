"""Peak-aware DeepONet (DON) for ERP spectrum prediction."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from utils.neural_operator_utils import (
    MLP,
    FrequencyRefinement1d,
    ResonatorSetEncoder,
    resolve_activation,
    run_operator_experiment,
)


class FourierFrequencyEncoder(nn.Module):
    """Fixed multi-scale Fourier features for the normalized query frequency."""

    def __init__(self, num_bands: int = 6) -> None:
        super().__init__()
        if num_bands <= 0:
            raise ValueError("num_bands must be positive.")
        scales = math.pi * (2.0 ** torch.arange(num_bands, dtype=torch.float32))
        self.register_buffer("scales", scales)

    @property
    def output_dim(self) -> int:
        return 1 + 2 * int(self.scales.numel())

    def forward(self, frequency: torch.Tensor) -> torch.Tensor:
        angles = frequency * self.scales
        return torch.cat((frequency, torch.sin(angles), torch.cos(angles)), dim=-1)


class DON(nn.Module):
    """Physics-aware, multi-term DeepONet with a configuration-modulated Fourier trunk.

    The classical branch/trunk inner product is retained as the operator's core
    mechanism (branch encodes the resonator configuration, trunk encodes the
    query frequency, combined by inner product) so this stays a recognizable
    DeepONet rather than converging onto DCO/DNO. Two additions give it the same
    tools every other operator in this project already had, closing an
    unintentional capacity/feature gap rather than changing what a DeepONet is:

    1. ``num_terms`` independent branch/trunk basis pairs are summed (a
       stacked/POD-style DeepONet) instead of one. A single shared basis is a
       low-rank map from configuration to spectrum; summing several raises the
       effective rank so multiple, independently-positioned narrow resonances
       no longer have to share one global basis.
    2. A small residual local-frequency refinement stage (the same
       ``FrequencyRefinement1d`` block DCO/DNO/GNO/STO use) sharpens the
       branch/trunk output after combination. Every other architecture mixes
       neighboring frequency samples somewhere; plain DeepONet never did.

    ``activation`` defaults to ``Tanh`` (not the ``SiLU`` most other
    operators here use) to match the original DeepONet paper (Lu et al.),
    which uses bounded activations in the branch/trunk nets to keep the
    basis functions well-conditioned for the inner-product combination.
    """

    def __init__(
        self,
        num_res: int,
        hidden_dim: int = 128,
        context_dim: int = 160,
        basis_dim: int = 256,
        fourier_bands: int = 6,
        num_terms: int = 4,
        refine_width: int = 64,
        activation: str | type[nn.Module] = "tanh",
    ) -> None:
        super().__init__()
        self.num_res = int(num_res)
        self.num_terms = int(num_terms)
        self.basis_dim = int(basis_dim)
        stacked_dim = self.num_terms * self.basis_dim
        activation_cls = resolve_activation(activation)

        self.frequency_features = FourierFrequencyEncoder(fourier_bands)
        self.configuration_encoder = ResonatorSetEncoder(
            hidden_dim=hidden_dim,
            element_dim=hidden_dim,
            output_dim=context_dim,
        )
        self.branch_head = MLP(
            [context_dim, hidden_dim, stacked_dim], activation=activation_cls
        )
        self.trunk = MLP(
            [self.frequency_features.output_dim, hidden_dim, hidden_dim, stacked_dim],
            activation=activation_cls,
        )
        self.trunk_modulation = MLP(
            [context_dim, hidden_dim, 2 * stacked_dim], activation=activation_cls
        )
        # Learned convex combination over terms keeps the sum on the same
        # scale as a single-term DeepONet regardless of num_terms.
        self.term_logits = nn.Parameter(torch.zeros(self.num_terms))
        self.bias = nn.Parameter(torch.zeros(1))
        self.scale = math.sqrt(float(basis_dim))

        self.refine_lift = nn.Linear(1, refine_width)
        self.frequency_refinement = FrequencyRefinement1d(refine_width)
        self.refine_project = nn.Linear(refine_width, 1)

    def forward(self, configuration: torch.Tensor, frequency: torch.Tensor) -> torch.Tensor:
        batch, n_freq, _ = frequency.shape
        context = self.configuration_encoder(configuration)
        branch = self.branch_head(context).view(
            batch, 1, self.num_terms, self.basis_dim
        )

        trunk = self.trunk(self.frequency_features(frequency))  # (B,F,T*P)
        gamma, beta = self.trunk_modulation(context).chunk(2, dim=-1)
        gamma = 1.0 + 0.25 * torch.tanh(gamma[:, None, :])
        beta = 0.10 * beta[:, None, :]
        trunk = (gamma * trunk + beta).view(
            batch, n_freq, self.num_terms, self.basis_dim
        )

        term_output = (branch * trunk).sum(dim=-1) / self.scale  # (B,F,T)
        term_weight = torch.softmax(self.term_logits, dim=0)
        base_output = (term_output * term_weight).sum(dim=-1, keepdim=True) + self.bias

        refined = self.refine_lift(base_output).transpose(1, 2)
        refined = self.frequency_refinement(refined)
        refined = self.refine_project(refined.transpose(1, 2))
        return base_output + refined


def build_model(num_res: int, **kwargs) -> DON:
    return DON(num_res=num_res, **kwargs)


# Every size-related dimension (not just hidden_dim) scaled down by the same
# ratio from the ~550K-matched config to land at the project's new ~110K
# budget, keeping DON's internal proportions similar rather than leaving one
# sub-module oversized relative to a shrunken hidden_dim. fourier_bands and
# num_terms are structural (not "widths"), so they stay unchanged.
DEFAULT_MODEL_CONFIG = {
    "hidden_dim": 45,
    "context_dim": 71,
    "basis_dim": 113,
    "fourier_bands": 6,
    "num_terms": 4,
    "refine_width": 28,
    "activation": "tanh",
}

# Search space for random_search_operator(): explores DON's own knobs at a
# fixed (parameter-matched) hidden_dim.
SEARCH_SPACE = {
    "basis_dim": [56, 84, 112, 140],
    "num_terms": [2, 3, 4, 6],
    "fourier_bands": [4, 6, 8],
    "refine_width": [14, 28, 42],
    "activation": ["tanh", "silu", "gelu"],
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
