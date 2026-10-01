"""iGNO: an invertible operator built the same way as iFNO (see ifno.py's
module docstring for the general method), with GNO's own graph message-
passing and attention query mechanism doing the lift instead of an FNO
Fourier layer or DCO's residual blocks.

GNO doesn't fit the iFNO/iDCO template as directly, and it's worth being
explicit about why, rather than quietly forcing it: GNO's genuinely
distinctive computation (erp_forward_operators/gno.py's ``GraphMessageLayer``)
passes messages BETWEEN THE num_res RESONATOR NODES -- it never mixes
across frequency at all. The coupling blocks in this family (coupling.py)
need a gate function operating on a per-FREQUENCY-point latent, since
that's the one shared, ordered grid this problem has to hang invertibility
on (see ifno.py). Graph message-passing among 3 resonator nodes has no
natural per-frequency-point form -- there's no "frequency slot" for a
graph layer to update.

The resolution used here: GNO's own two real stages both survive, just
relocated to where they naturally fit --

  - Its graph message-passing (:class:`GraphMessageLayer`, reused
    unchanged from erp_forward_operators/gno.py) runs ONCE per
    configuration, unconditioned on frequency, refining a per-resonator
    node embedding -- exactly as GNO's own ``forward`` does before it ever
    touches frequency.
  - Its per-frequency attention query (``query_kernel`` + ``attention_score``
    + softmax-over-resonators, also reused unchanged) then reads that node
    embedding off at every frequency point, producing the ``(B, F, width)``
    signal that becomes this model's lift -- literally GNO's own
    ``integral`` variable, one step before its own ``frequency_refinement``.
  - What replaces GNO's own ``frequency_refinement`` + output head is the
    invertible coupling blocks, gated by :class:`GNOGateLayer1d`: a plain
    (LayerNorm/residual-free, matching GNO's own MLPs -- GNO never uses
    DCO's ``ResidualMLPBlock``) pointwise MLP followed by
    ``FrequencyRefinement1d``, the same cross-frequency mixer GNO itself
    already uses as its final stage.

So the graph structure is real and does real work (conditioning the lift),
it just isn't literally what makes the coupling blocks invertible -- that
job still falls to a per-frequency-point gate, same as ifno.py/idco.py,
because invertibility in this family is fundamentally a per-frequency-axis
property.
"""

from __future__ import annotations

from typing import Mapping

import torch
import torch.nn as nn

from erp_forward_operators.neural_operator_utils import (
    MLP,
    FrequencyRefinement1d,
    physics_aware_resonator_features,
    resolve_activation,
)
from erp_invertible_operators.common import DesignVAE, InvertibleOperatorBase
from erp_invertible_operators.coupling import InvertibleCouplingStack
from utils.physics import freqs as _frequency_grid_hz


# ==================================================
# GNO-specific building blocks (reused verbatim from erp_forward_operators/gno.py,
# redefined locally per this repo's convention of keeping each operator
# module self-contained)
# ==================================================


class GraphMessageLayer(nn.Module):
    """Complete-graph message passing with geometric/frequency-tuning edge
    features -- identical construction to GNO's own layer.
    """

    def __init__(self, width: int, dropout: float = 0.0, activation: str | type[nn.Module] = "silu") -> None:
        super().__init__()
        activation_cls = resolve_activation(activation)
        self.message = MLP([2 * width + 7, width, width], activation=activation_cls)
        self.update = MLP([2 * width, width, width], activation=activation_cls)
        self.norm = nn.LayerNorm(width)
        self.activation = activation_cls()
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        b, n, width = h.shape
        hi = h[:, :, None, :].expand(b, n, n, width)
        hj = h[:, None, :, :].expand(b, n, n, width)
        rel = features[:, None, :, :] - features[:, :, None, :]
        distance = torch.linalg.vector_norm(rel[..., 3:5], dim=-1, keepdim=True)
        ft_gap = rel[..., 2:3].abs()
        edge = torch.cat((rel, distance, ft_gap), dim=-1)
        message = self.message(torch.cat((hi, hj, edge), dim=-1))

        if n > 1:
            mask = (~torch.eye(n, dtype=torch.bool, device=h.device))[None, :, :, None]
            aggregate = (message * mask).sum(dim=2) / float(n - 1)
        else:
            aggregate = torch.zeros_like(h)

        delta = self.update(torch.cat((h, aggregate), dim=-1))
        return self.dropout(self.activation(self.norm(h + delta)))


class GNOGateLayer1d(nn.Module):
    """GNO's own tools, repurposed as the coupling blocks' gate function L:
    a plain pointwise MLP (GNO never uses DCO's LayerNorm/residual
    ``ResidualMLPBlock`` anywhere -- its message/update/query_kernel MLPs
    are all bare) followed by :class:`FrequencyRefinement1d`, the same
    cross-frequency local mixer GNO itself uses as its own final stage.
    """

    def __init__(self, width: int, activation: str | type[nn.Module] = "silu") -> None:
        super().__init__()
        activation_cls = resolve_activation(activation)
        self.pointwise = MLP([width, width, width], activation=activation_cls)
        self.refine = FrequencyRefinement1d(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pointwise(x.transpose(1, 2)).transpose(1, 2)
        return self.refine(h)


# ==================================================
# Full model
# ==================================================


class IGNO(InvertibleOperatorBase):
    def __init__(
        self,
        design_dim: int,
        n_freq: int = 301,
        width: int = 24,
        num_blocks: int = 4,
        depth: int = 3,
        frequency_dim: int = 28,
        modal_harmonics: int = 4,
        dropout: float = 0.1,
        z_dim: int = 6,
        vae_hidden: int = 64,
        tau: float = 1.0,
        activation: str | type[nn.Module] = "silu",
    ) -> None:
        super().__init__()
        if design_dim % 5 != 0:
            raise ValueError("design_dim must be num_res*5 ([m,k,f_t,x,y] per resonator).")
        self.design_dim = int(design_dim)
        self.num_res = self.design_dim // 5
        self.n_freq = int(n_freq)
        self.width = int(width)
        self.modal_harmonics = int(modal_harmonics)

        self._register_frequency_grid(torch.from_numpy(_frequency_grid_hz.astype("float32")))
        activation_cls = resolve_activation(activation)

        # ---- Graph encoder (GNO's own node_lift + message-passing stack) ----
        node_input_dim = 5 + 2 * self.modal_harmonics + self.modal_harmonics**2
        self.node_lift = MLP([node_input_dim, width, width], activation=activation_cls)
        self.graph_layers = nn.ModuleList(
            [GraphMessageLayer(width, dropout=dropout, activation=activation_cls) for _ in range(depth)]
        )

        # ---- Per-frequency attention query (GNO's own query_kernel/attention_score) ----
        self.frequency_encoder = MLP([1, frequency_dim, frequency_dim], activation=activation_cls)
        query_input_dim = width + 5 + frequency_dim + 4  # node, raw resonator, freq embedding, f/delta/|delta|/delta^2
        self.query_kernel = MLP([query_input_dim, width, width, width], activation=activation_cls)
        self.attention_score = MLP([width, width // 2, 1], activation=activation_cls)
        self.attention_dropout = nn.Dropout(dropout)

        # ---- Forward-direction lift (P): GNO's own attention-pooled integral, projected to 2*width ----
        self.lift_p = nn.Linear(width, 2 * self.width)
        self.lift_p_activation = activation_cls()

        # ---- Inverse-direction lift (P'): no configuration known yet, reuses the same frequency_encoder ----
        self.lift_pp = nn.Linear(1 + frequency_dim, 2 * self.width)
        self.lift_pp_activation = activation_cls()

        # ---- Shared invertible coupling blocks, gated by GNO's own layer ----
        self.blocks = InvertibleCouplingStack(
            lambda: GNOGateLayer1d(self.width, activation=activation), num_blocks, tau=tau
        )

        # ---- Output projections ----
        self.project_q = nn.Sequential(
            nn.Linear(2 * self.width, 2 * self.width), activation_cls(), nn.Linear(2 * self.width, 1)
        )  # Q: per-point (v1,v2) -> ERP scalar
        self.project_qp = MLP(
            [4 * self.width, vae_hidden, self.design_dim], activation=nn.SiLU
        )  # Q': pooled (v1,v2) over frequency -> flat design

        # ---- beta-VAE over the design space (Sec 3.2) ----
        self.vae = DesignVAE(self.design_dim, z_dim=z_dim, hidden=vae_hidden)

    def _lift_forward(self, configuration: torch.Tensor, frequency: torch.Tensor) -> torch.Tensor:
        node_features = physics_aware_resonator_features(configuration, harmonics=self.modal_harmonics)
        h = self.node_lift(node_features)
        for layer in self.graph_layers:
            h = layer(h, configuration)

        freq = self.frequency_encoder(frequency)
        b, f, _ = freq.shape
        n = configuration.shape[1]
        node = h[:, None, :, :].expand(b, f, n, -1)
        raw = configuration[:, None, :, :].expand(b, f, n, -1)
        query_embedding = freq[:, :, None, :].expand(b, f, n, -1)
        query_frequency = frequency[:, :, None, :].expand(b, f, n, 1)
        f_t = configuration[:, None, :, 2:3].expand(b, f, n, 1)
        detuning = query_frequency - f_t

        pair = torch.cat(
            (node, raw, query_embedding, query_frequency, detuning, detuning.abs(), detuning.square()), dim=-1
        )
        kernel_values = self.query_kernel(pair)
        weights = torch.softmax(self.attention_score(kernel_values), dim=2)
        weights = self.attention_dropout(weights)
        integral = (weights * kernel_values).sum(dim=2)  # (B, F, width) -- GNO's own pre-refinement signal

        lifted = self.lift_p_activation(self.lift_p(integral))  # (B, F, 2*width)
        return lifted.transpose(1, 2)  # (B, 2*width, F)

    def _lift_inverse(self, spectrum: torch.Tensor, frequency: torch.Tensor) -> torch.Tensor:
        freq_feat = self.frequency_encoder(frequency)  # (B, F, frequency_dim)
        lifted = self.lift_pp_activation(self.lift_pp(torch.cat((spectrum[..., None], freq_feat), dim=-1)))
        return lifted.transpose(1, 2)


def build_model(design_dim: int, **kwargs) -> IGNO:
    return IGNO(design_dim=design_dim, **kwargs)


DEFAULT_MODEL_CONFIG: Mapping[str, object] = {
    "n_freq": 301,
    "width": 24,
    "num_blocks": 4,
    "depth": 3,
    "frequency_dim": 28,
    "modal_harmonics": 4,
    "dropout": 0.1,
    "z_dim": 6,
    "vae_hidden": 64,
    "tau": 1.0,
    "activation": "silu",
}
