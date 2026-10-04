"""iDCO: an invertible operator built the same way as iFNO (see ifno.py's
module docstring for the general method), but with DCO's own characteristic
layer as the coupling blocks' gate function L instead of an FNO Fourier
layer.

DCO's own (non-invertible) forward pass, erp_forward_operators/dco.py, is:
lift (branch=ResonatorSetEncoder, trunk=frequency MLP, query=
ResonanceQueryEncoder, concatenated and linearly projected) -> repeated
ResidualMLPBlock (a PLAIN pointwise residual MLP -- no cross-frequency
mixing at all) -> one FrequencyRefinement1d pass (DCO's only
cross-frequency, local kernel-3 mixing, used once at the very end) ->
output head. So "DCO's core transform" genuinely is pointwise-MLP-plus-
one-local-refinement, unlike FNO's Fourier layer which mixes across ALL
frequencies (via the spectral conv) inside every single layer. That
architectural difference is preserved here: :class:`DCOGateLayer1d` below
is exactly DCO's own ``ResidualMLPBlock`` + ``FrequencyRefinement1d`` pair,
reused verbatim as the gate function -- nothing about it is contrived to
look FNO-like.

Everything else -- the lift/read-out shapes, the beta-VAE, the three-stage
loss terms -- is inherited unchanged from
:class:`~erp_invertible_operators.common.InvertibleOperatorBase`,
since none of that ever depended on which gate function the coupling
blocks use. The forward-direction lift mirrors DCO's own branch/trunk/
query lift structure directly (rather than iFNO's slightly leaner
context+query+raw-frequency concat), since DCO's real "trunk" (a frequency
embedding MLP) is a genuine, named part of its architecture worth keeping
recognizable here.
"""

from __future__ import annotations

from typing import Mapping

import torch
import torch.nn as nn

from erp_forward_operators.neural_operator_utils import (
    MLP,
    FrequencyRefinement1d,
    ResidualMLPBlock,
    ResonanceQueryEncoder,
    ResonatorSetEncoder,
    build_resonator_encoder,
    resolve_activation,
)
from erp_invertible_operators.common import DesignVAE, InvertibleOperatorBase, build_design_encoder
from erp_invertible_operators.coupling import InvertibleCouplingStack
from utils.physics import freqs as _frequency_grid_hz


# ==================================================
# DCO-specific gate function
# ==================================================


class DCOGateLayer1d(nn.Module):
    """DCO's own core transform, repurposed as the coupling blocks' gate
    function L: one pointwise residual MLP block (DCO's repeated interior
    computation, applied independently at each frequency point -- no
    cross-frequency mixing) followed by :class:`FrequencyRefinement1d`,
    the same local kernel-3 cross-frequency mixer DCO itself uses as its
    own final refinement stage before its output head.
    """

    def __init__(self, width: int, activation: str | type[nn.Module] = "silu") -> None:
        super().__init__()
        activation_cls = resolve_activation(activation)
        self.residual_block = ResidualMLPBlock(width, activation=activation_cls)
        self.refine = FrequencyRefinement1d(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, width, F). ResidualMLPBlock operates pointwise over the
        # last dim, so transpose to (B, F, width) for it, then back.
        h = self.residual_block(x.transpose(1, 2)).transpose(1, 2)
        return self.refine(h)


# ==================================================
# Full model
# ==================================================


class IDCO(InvertibleOperatorBase):
    def __init__(
        self,
        design_dim: int,
        n_freq: int = 301,
        width: int = 24,
        num_blocks: int = 4,
        branch_dim: int = 56,
        trunk_dim: int = 28,
        query_dim: int = 32,
        z_dim: int = 6,
        vae_hidden: int = 64,
        tau: float = 1.0,
        activation: str | type[nn.Module] = "silu",
        use_sorted_branch: bool = False,
        encoder: str = "set",
        coordinate_features: str = "zscored",
        design_param: str = "full15",
        num_res: int = 3,
        gate: str = "softplus",
        gate_scale: float = 2.0,
        readout: str = "pooled",
        readout_bins: int = 16,
    ) -> None:
        super().__init__()
        # full15 ([m,k,f_t,x,y]) or bounded12 ([m,f_t,x,y], k derived) designs
        self._init_design(design_dim, design_param, num_res)
        self.n_freq = int(n_freq)
        self.width = int(width)

        self._register_frequency_grid(torch.from_numpy(_frequency_grid_hz.astype("float32")))
        activation_cls = resolve_activation(activation)

        # ---- Forward-direction lift (P): DCO's own branch/trunk/query ----
        # use_sorted_branch adds the f_t-sorted resonator branch next to the pooled
        # set branch (DCO_sorted design, see SetAndSortedResonatorEncoder).
        self.use_sorted_branch = bool(use_sorted_branch)
        # encoder: "set" (pooled), "sorted" (f_t-ordered, lossless) or "set+sorted"
        self.encoder = "set+sorted" if self.use_sorted_branch and encoder == "set" else encoder
        self.branch = build_design_encoder(self.encoder, False, self.num_res, branch_dim, branch_dim)
        self.trunk = MLP([1, trunk_dim, trunk_dim], activation=activation_cls)
        self.resonance_query = ResonanceQueryEncoder(hidden_dim=query_dim, element_dim=query_dim, output_dim=query_dim)
        self.lift_p = nn.Linear(branch_dim + trunk_dim + query_dim, 2 * self.width)
        self.lift_p_activation = activation_cls()

        # ---- Inverse-direction lift (P'): no configuration known yet ----
        self.trunk_inverse = MLP([1, trunk_dim, trunk_dim], activation=activation_cls)
        self.lift_pp = nn.Linear(1 + trunk_dim, 2 * self.width)
        self.lift_pp_activation = activation_cls()

        # ---- Shared invertible coupling blocks, gated by DCO's own layer ----
        self.blocks = InvertibleCouplingStack(
            lambda: DCOGateLayer1d(self.width, activation=activation), num_blocks, tau=tau, gate=gate, gate_scale=gate_scale
        )

        # ---- Output projections ----
        self.project_q = nn.Sequential(
            nn.Linear(2 * self.width, 2 * self.width), activation_cls(), nn.Linear(2 * self.width, 1)
        )  # Q: per-point (v1,v2) -> ERP scalar
        self.project_qp = MLP(
            [self._init_readout(readout, readout_bins), vae_hidden, self.design_dim], activation=nn.SiLU
        )  # Q': pooled (v1,v2) over frequency -> flat design

        # ---- beta-VAE over the design space (Sec 3.2) ----
        self.vae = DesignVAE(self.design_dim, z_dim=z_dim, hidden=vae_hidden)
        self._init_features(coordinate_features)

    def _lift_forward(self, configuration: torch.Tensor, frequency: torch.Tensor) -> torch.Tensor:
        branch = self.branch(configuration)[:, None, :].expand(-1, frequency.shape[1], -1)
        trunk = self.trunk(frequency)
        query = self.resonance_query(configuration, frequency)
        lifted = self.lift_p_activation(self.lift_p(torch.cat((branch, trunk, query), dim=-1)))  # (B, F, 2*width)
        return lifted.transpose(1, 2)  # (B, 2*width, F)

    def _lift_inverse(self, spectrum: torch.Tensor, frequency: torch.Tensor) -> torch.Tensor:
        trunk = self.trunk_inverse(frequency)  # (B, F, trunk_dim)
        lifted = self.lift_pp_activation(self.lift_pp(torch.cat((spectrum[..., None], trunk), dim=-1)))
        return lifted.transpose(1, 2)  # (B, 2*width, F)


def build_model(design_dim: int, **kwargs) -> IDCO:
    return IDCO(design_dim=design_dim, **kwargs)


DEFAULT_MODEL_CONFIG: Mapping[str, object] = {
    "n_freq": 301,
    "width": 24,
    "num_blocks": 4,
    "branch_dim": 56,
    "trunk_dim": 28,
    "query_dim": 32,
    "z_dim": 6,
    "vae_hidden": 64,
    "tau": 1.0,
    "activation": "silu",
}
