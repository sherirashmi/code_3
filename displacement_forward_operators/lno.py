"""Displacement-field Laplace Neural Operator (LNO): (configuration,
frequency, x, y) -> (displacement_real, displacement_imag).

Adapted from forward_operators/lno.py, with the one architecture-specific
change in this whole folder that ISN'T "swap in FieldContextEncoder
everywhere": LNO's poles (damping + natural frequency) are kept
CONFIGURATION-ONLY, predicted from the plain ResonatorSetEncoder context
exactly as before -- a pole is a global modal property of the coupled
plate+resonator system (which frequencies it resonates at, how damped
they are), not something that depends on where on the plate you happen to
be asking. Its RESIDUES, however, become POSITION-DEPENDENT, predicted
from FieldContextEncoder's position-aware context instead: a residue is
essentially a mode shape's amplitude at the query point, which genuinely
varies spatially (a mode's contribution to the field is large near its
own antinodes and near-zero at its nodal lines) -- exactly the physical
picture the modal expansion this project's own solver uses is built on
(w(x,y,omega) = sum_mn phi_mn(x,y) * eta_mn(omega); poles come from
eta_mn's own dynamics, position-dependence comes from phi_mn(x,y)).

Everything else -- evaluating each pole's rational term at s=j*frequency,
FrequencyRefinement1d -- is unchanged. Output is 2 channels, not 1.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from forward_operators.neural_operator_utils import MLP, FrequencyRefinement1d, ResonatorSetEncoder, resolve_activation

from .displacement_operator_utils import FieldContextEncoder, FieldResonanceQueryEncoder


class DisplacementLNO(nn.Module):
    def __init__(
        self,
        num_res: int,
        width: int = 90,
        num_poles: int = 13,
        config_hidden: int = 64,
        pole_hidden: int = 64,
        query_dim: int = 25,
        dropout: float = 0.1,
        activation: str | type[nn.Module] = "silu",
    ) -> None:
        super().__init__()
        self.num_res = int(num_res)
        self.num_poles = int(num_poles)
        activation_cls = resolve_activation(activation)

        context_dim = width
        # Poles: configuration only (global modal property).
        self.configuration_encoder = ResonatorSetEncoder(
            hidden_dim=config_hidden, element_dim=config_hidden, output_dim=context_dim
        )
        self.pole_predictor = MLP([context_dim, pole_hidden, pole_hidden, 2 * self.num_poles], activation=activation_cls)

        # Residues: configuration + position (a mode shape's local amplitude).
        self.field_encoder = FieldContextEncoder(output_dim=context_dim, resonator_hidden=config_hidden)
        self.residue_predictor = MLP([context_dim, pole_hidden, pole_hidden, 2 * self.num_poles], activation=activation_cls)

        self.resonance_query = FieldResonanceQueryEncoder(
            hidden_dim=query_dim, element_dim=query_dim, output_dim=query_dim
        )

        self.lift = nn.Linear(2 * self.num_poles + query_dim + 1, width)
        self.dropout = nn.Dropout(dropout)
        self.refine = FrequencyRefinement1d(width)
        self.project = nn.Sequential(nn.Linear(width, width), activation_cls(), nn.Linear(width, 2))

    def forward(
        self, configuration: torch.Tensor, frequency: torch.Tensor, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        config_context = self.configuration_encoder(configuration)  # (B, width) -- poles
        field_context = self.field_encoder(configuration, x, y)  # (B, width) -- residues

        pole_raw = self.pole_predictor(config_context).view(-1, self.num_poles, 2)
        sigma = nn.functional.softplus(pole_raw[..., 0]) + 1e-3  # (B, P) damping > 0
        omega = nn.functional.softplus(pole_raw[..., 1])  # (B, P) natural frequency >= 0
        pole = torch.complex(-sigma, omega)  # (B, P)

        residue_raw = self.residue_predictor(field_context).view(-1, self.num_poles, 2)
        residue = torch.complex(residue_raw[..., 0], residue_raw[..., 1])  # (B, P)

        f = frequency[..., 0]  # (B, F)
        s = torch.complex(torch.zeros_like(f), f)  # j*frequency, (B, F)

        s_e = s[:, :, None]  # (B, F, 1)
        pole_e = pole[:, None, :]  # (B, 1, P)
        residue_e = residue[:, None, :]  # (B, 1, P)

        term = residue_e / (s_e - pole_e) + residue_e.conj() / (s_e - pole_e.conj())
        pole_features = torch.cat((term.real, term.imag), dim=-1)  # (B, F, 2P)

        query = self.resonance_query(configuration, frequency, x, y)  # (B, F, query_dim)
        combined = torch.cat((pole_features, query, frequency), dim=-1)
        h = self.dropout(self.lift(combined)).transpose(1, 2)  # (B, width, F)
        h = self.refine(h)
        return self.project(h.transpose(1, 2))


def build_model(num_res: int, **kwargs) -> DisplacementLNO:
    return DisplacementLNO(num_res=num_res, **kwargs)


DEFAULT_MODEL_CONFIG = {
    "width": 90,
    "num_poles": 13,
    "config_hidden": 64,
    "pole_hidden": 64,
    "query_dim": 25,
    "dropout": 0.1,
    "activation": "silu",
}
