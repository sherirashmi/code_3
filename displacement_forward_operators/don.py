"""Displacement-field DeepONet (DON): (configuration, frequency, x, y) ->
(v_real, v_imag, M_real, M_imag).

Adapted from erp_forward_operators/don.py. What's UNCHANGED: the classical
branch/trunk inner product over frequency (branch encodes resonator
configuration, trunk encodes query frequency, num_terms stacked basis
pairs, config-modulated FiLM gating of the trunk, FrequencyRefinement1d
polish) -- the whole frequency-axis mechanism is identical to the ERP
version, because frequency is still the same shared, ordered 301-point
grid (see displacement_operator_utils.py's module docstring for why that
matters: FNO/WNO's spectral/wavelet mixing elsewhere in this folder only
stays valid because of this).

What changed:
  1. ``configuration_encoder`` (ResonatorSetEncoder alone) -> ``field_encoder``
     (FieldContextEncoder: the same ResonatorSetEncoder fused with a
     position encoding built from the true modal sin/sin basis) -- see
     displacement_operator_utils.py. Position enters exactly where the
     resonator configuration already entered (branch conditioning), not
     as a new trunk axis.
  2. Output is 4 channels (v_real, v_imag, M_real, M_imag), not 1 (ERP).
     ``v`` is velocity (the primary, data-supervised quantity); ``M`` is
     the auxiliary laplacian(v) field used only by physics_loss.py's
     mixed-formulation residual (see displacement_operator_utils.py's
     module docstring), never data-supervised. The branch/trunk basis
     stays SHARED between all four channels (physically reasonable -- all
     are components of the same underlying field/its Laplacian), but the
     final term-weighting and bias are per-channel (``term_logits``/
     ``bias`` gain a size-4 axis).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from erp_forward_operators.neural_operator_utils import FrequencyRefinement1d, MLP, resolve_activation

from .displacement_operator_utils import FieldContextEncoder


class DisplacementDON(nn.Module):
    def __init__(
        self,
        num_res: int,
        hidden_dim: int = 45,
        context_dim: int = 71,
        basis_dim: int = 113,
        num_terms: int = 4,
        refine_width: int = 28,
        activation: str | type[nn.Module] = "tanh",
    ) -> None:
        super().__init__()
        self.num_res = int(num_res)
        self.num_terms = int(num_terms)
        self.basis_dim = int(basis_dim)
        stacked_dim = self.num_terms * self.basis_dim
        activation_cls = resolve_activation(activation)

        self.field_encoder = FieldContextEncoder(output_dim=context_dim, resonator_hidden=hidden_dim)
        self.branch_head = MLP([context_dim, hidden_dim, stacked_dim], activation=activation_cls)
        self.trunk = MLP([1, hidden_dim, hidden_dim, stacked_dim], activation=activation_cls)
        self.trunk_modulation = MLP([context_dim, hidden_dim, 2 * stacked_dim], activation=activation_cls)

        self.term_logits = nn.Parameter(torch.zeros(self.num_terms, 4))  # per output channel
        self.bias = nn.Parameter(torch.zeros(4))
        self.scale = math.sqrt(float(basis_dim))

        self.refine_lift = nn.Linear(4, refine_width)
        self.frequency_refinement = FrequencyRefinement1d(refine_width)
        self.refine_project = nn.Linear(refine_width, 4)

    def forward(
        self, configuration: torch.Tensor, frequency: torch.Tensor, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        batch, n_freq, _ = frequency.shape
        context = self.field_encoder(configuration, x, y)
        branch = self.branch_head(context).view(batch, 1, self.num_terms, self.basis_dim)

        trunk = self.trunk(frequency)  # (B, F, T*P)
        gamma, beta = self.trunk_modulation(context).chunk(2, dim=-1)
        gamma = 1.0 + 0.25 * torch.tanh(gamma[:, None, :])
        beta = 0.10 * beta[:, None, :]
        trunk = (gamma * trunk + beta).view(batch, n_freq, self.num_terms, self.basis_dim)

        term_output = (branch * trunk).sum(dim=-1) / self.scale  # (B, F, T)
        term_weight = torch.softmax(self.term_logits, dim=0)  # (T, 4)
        base_output = torch.einsum("bft,tc->bfc", term_output, term_weight) + self.bias  # (B, F, 4)

        refined = self.refine_lift(base_output).transpose(1, 2)
        refined = self.frequency_refinement(refined)
        refined = self.refine_project(refined.transpose(1, 2))
        return base_output + refined


def build_model(num_res: int, **kwargs) -> DisplacementDON:
    return DisplacementDON(num_res=num_res, **kwargs)


DEFAULT_MODEL_CONFIG = {
    "hidden_dim": 45,
    "context_dim": 71,
    "basis_dim": 113,
    "num_terms": 4,
    "refine_width": 28,
    "activation": "tanh",
}
