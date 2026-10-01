"""Generic invertible coupling scaffold (Long, Xu, Yuan, Yang & Zhe,
"Invertible Fourier Neural Operators for Tackling Both Forward and Inverse
Problems", arXiv:2402.11722, eq 2/3), factored out of the original iFNO
port so any architecture's own per-point layer can be dropped in as the
gate function and get an exactly invertible latent core for free (the
lifts/readouts around it are not invertible, see common.py).

The paper's own instantiation always uses an FNO Fourier layer as the gate
function L. Nothing about the coupling update itself requires that -- eq
2/3 only need L to be SOME function that maps a ``(B, width, F)`` latent to
another of the same shape; the softplus gate ``S`` is what makes the update
invertible, not L. That is the "method" this module extracts: swap the
paper's Fourier-layer L for a different architecture's own characteristic
per-point layer (DCO's residual-MLP-plus-local-refinement, GNO's own
frequency-refinement stage, ...) and the SAME block/stack mechanics below
still apply unchanged. See ifno.py, idco.py and igno.py for three concrete
choices of L.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


class InvertibleCouplingBlock(nn.Module):
    """One invertible coupling block (paper eq 2/3): a pair of d-channel
    latents (v1, v2), each ``(B, d, F)``, updated by

        v1_next = v1 * S(L(v2))
        v2_next = v2 * S(L(v1_next))

    where ``L`` (``gate_net``) is any per-point network sharing weights
    across both update steps (the paper uses one Fourier layer symbol per
    block, applied to each half in turn) and ``S`` is the elementwise
    softplus transform ``S(x) = tau^-1 * log(1+exp(tau*x))`` --
    ``F.softplus`` with ``beta=tau`` is exactly this. Since ``S(x) > 0``
    everywhere, both multiplicative updates are exactly invertible by
    elementwise division, eq 3:

        v2 = v2_next / S(L(v1_next))       (v1_next is already known)
        v1 = v1_next / S(L(v2))            (using the just-recovered v2)

    A small epsilon floor on the gate guards against division blowing up
    when a freshly-initialized L output is very negative (S(x) -> 0); the
    paper does not need this in float64 tabletop experiments but it matters
    for stable float32 training here.
    """

    def __init__(self, gate_net: nn.Module, tau: float = 1.0, gate: str = "softplus", gate_scale: float = 2.0) -> None:
        super().__init__()
        if gate not in ("softplus", "bounded"):
            raise ValueError("gate must be 'softplus' or 'bounded'.")
        self.gate_net = gate_net
        self.tau = float(tau)
        self.eps = 1e-6
        self.gate = gate
        if gate == "bounded":
            # S(x) = exp(c * tanh(a * L(x))) with a learnable gain a = 0 at
            # init: every block starts as the exact identity (S = 1) and its
            # per-step scale stays in [exp(-c), exp(c)].
            self.gate_scale = float(gate_scale)
            self.gate_gain = nn.Parameter(torch.zeros(1))

    def _gate(self, x: torch.Tensor) -> torch.Tensor:
        if self.gate == "bounded":
            return torch.exp(self.gate_scale * torch.tanh(self.gate_gain * self.gate_net(x)))
        # Paper gate: positive but unbounded above, and softplus(0) = 0.69,
        # so a freshly initialised block shrinks its input rather than
        # starting at the identity.
        return F.softplus(self.gate_net(x), beta=self.tau) + self.eps

    def forward(self, v1: torch.Tensor, v2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        v1_next = v1 * self._gate(v2)
        v2_next = v2 * self._gate(v1_next)
        return v1_next, v2_next

    def inverse(self, v1_next: torch.Tensor, v2_next: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        v2 = v2_next / self._gate(v1_next)
        v1 = v1_next / self._gate(v2)
        return v1, v2


class InvertibleCouplingStack(nn.Module):
    """K stacked InvertibleCouplingBlocks -- forward runs blocks 1..K in
    order, inverse runs the SAME blocks (same weights) K..1 using each
    block's own closed-form inverse. The STACK is an exact bijection between
    its latent tensors; the surrounding lifts and readouts of the full
    models are learned and lossy, so a full model is bidirectional with an
    invertible core, not an exact design<->spectrum bijection (see
    erp_invertible_operators/common.py).
    """

    def __init__(
        self,
        gate_net_factory: Callable[[], nn.Module],
        num_blocks: int,
        tau: float = 1.0,
        gate: str = "softplus",
        gate_scale: float = 2.0,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                InvertibleCouplingBlock(gate_net_factory(), tau=tau, gate=gate, gate_scale=gate_scale)
                for _ in range(num_blocks)
            ]
        )

    def forward(self, v1: torch.Tensor, v2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        for block in self.blocks:
            v1, v2 = block(v1, v2)
        return v1, v2

    def inverse(self, v1: torch.Tensor, v2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        for block in reversed(self.blocks):
            v1, v2 = block.inverse(v1, v2)
        return v1, v2
