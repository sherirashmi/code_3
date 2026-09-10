"""Laplace Neural Operator (LNO) for ERP spectrum prediction.

Every other operator in this project represents the configuration -> ERP
mapping through a basis native to its own domain: FNO through truncated
Fourier modes, WNO through a wavelet transform, DON through a bilinear
branch/trunk product. LNO represents it through an explicit, per-configuration
rational transfer function in the Laplace domain -- i.e. a learned set of
complex conjugate pole/residue pairs, evaluated at each query frequency. This
is the natural basis for *this* physical problem: a damped resonator's
frequency response is, by construction, a sum of complex-conjugate poles
(damping = real part, natural frequency = imaginary part) with complex
residues -- exactly the classical modal/Laplace expansion of a linear damped
system. FNO's Fourier basis assumes periodic/steady-state structure; LNO's
pole-residue basis is built for damped, resonant, decaying structure, which
is what every ERP peak/notch in this dataset actually is.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from forward_operators.neural_operator_utils import (
    MLP,
    FrequencyRefinement1d,
    ResonanceQueryEncoder,
    ResonatorSetEncoder,
    resolve_activation,
    run_operator_experiment,
)


class LNO(nn.Module):
    """Per-configuration pole/residue predictor, evaluated at query frequencies.

    The configuration is encoded once into a context vector, which predicts
    ``num_poles`` complex conjugate pole/residue pairs (a data-dependent
    rational transfer function). Each pole contributes one complex term
    ``r/(s-p) + conj(r)/(s-conj(p))`` evaluated at ``s = j*frequency`` for
    every query frequency; the resulting real/imaginary parts become
    per-pole feature channels, combined with the same detuning-aware query
    features every other operator uses and the same shared local-mixing
    stage before the final projection.
    """

    def __init__(
        self,
        num_res: int,
        width: int = 96,
        num_poles: int = 14,
        config_hidden: int = 72,
        pole_hidden: int = 72,
        query_dim: int = 26,
        dropout: float = 0.1,
        activation: str | type[nn.Module] = "silu",
    ) -> None:
        super().__init__()
        self.num_res = int(num_res)
        self.num_poles = int(num_poles)
        activation_cls = resolve_activation(activation)

        context_dim = width
        self.configuration_encoder = ResonatorSetEncoder(
            hidden_dim=config_hidden,
            element_dim=config_hidden,
            output_dim=context_dim,
        )
        self.resonance_query = ResonanceQueryEncoder(
            hidden_dim=query_dim,
            element_dim=query_dim,
            output_dim=query_dim,
        )

        # Per configuration: damping (sigma>0), natural frequency (omega>0),
        # and a complex residue (r_real, r_imag) for each of num_poles poles.
        self.pole_predictor = MLP(
            [context_dim, pole_hidden, pole_hidden, 4 * self.num_poles],
            activation=activation_cls,
        )

        self.lift = nn.Linear(2 * self.num_poles + query_dim + 1, width)
        self.dropout = nn.Dropout(dropout)
        self.refine = FrequencyRefinement1d(width)
        self.project = nn.Sequential(
            nn.Linear(width, width),
            activation_cls(),
            nn.Linear(width, 1),
        )

    def forward(self, configuration: torch.Tensor, frequency: torch.Tensor) -> torch.Tensor:
        context = self.configuration_encoder(configuration)  # (B, width)

        raw = self.pole_predictor(context).view(-1, self.num_poles, 4)
        sigma = nn.functional.softplus(raw[..., 0]) + 1e-3  # (B, P) damping > 0
        omega = nn.functional.softplus(raw[..., 1])  # (B, P) natural frequency >= 0
        r_real, r_imag = raw[..., 2], raw[..., 3]  # (B, P) residue

        pole = torch.complex(-sigma, omega)  # (B, P)
        residue = torch.complex(r_real, r_imag)  # (B, P)

        f = frequency[..., 0]  # (B, F)
        s = torch.complex(torch.zeros_like(f), f)  # j*frequency, (B, F)

        s_e = s[:, :, None]  # (B, F, 1)
        pole_e = pole[:, None, :]  # (B, 1, P)
        residue_e = residue[:, None, :]  # (B, 1, P)

        term = residue_e / (s_e - pole_e) + residue_e.conj() / (s_e - pole_e.conj())
        pole_features = torch.cat((term.real, term.imag), dim=-1)  # (B, F, 2P)

        query = self.resonance_query(configuration, frequency)  # (B, F, query_dim)
        combined = torch.cat((pole_features, query, frequency), dim=-1)
        x = self.dropout(self.lift(combined)).transpose(1, 2)  # (B, width, F)
        x = self.refine(x)
        return self.project(x.transpose(1, 2))


def build_model(num_res: int, **kwargs) -> LNO:
    return LNO(num_res=num_res, **kwargs)


# Sized to land in the project's shared ~110-120K parameter budget alongside
# every other operator (see operator_registry.py).
DEFAULT_MODEL_CONFIG = {
    "width": 90,
    "num_poles": 13,
    "config_hidden": 64,
    "pole_hidden": 64,
    "query_dim": 25,
    "dropout": 0.1,
    "activation": "silu",
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
        operator_name="LNO",
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
