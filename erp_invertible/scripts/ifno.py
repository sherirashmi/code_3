"""iFNO: Invertible Fourier Neural Operator for this project's config<->ERP
problem, adapted from Long, Xu, Yuan, Yang & Zhe, "Invertible Fourier Neural
Operators for Tackling Both Forward and Inverse Problems" (arXiv:2402.11722).

This is the original member of the invertible-operator family in this
package (ifno.py/idco.py/igno.py): the gate function inside every
coupling block (see coupling.py) is an FNO Fourier layer -- a spectral
convolution over the retained Fourier modes plus a local kernel-3 conv,
exactly matching erp_forward/scripts/fno.py's own SpectralConv1d. See
idco.py/igno.py's module docstrings for how the SAME coupling scaffold is
reused with DCO's and GNO's own characteristic layers as the gate function
instead.

What the paper does, in its own setting: both the input function ``f`` and
output function ``u`` are sampled on the SAME spatial grid (e.g. a 64x64
Darcy-flow permeability map and its pressure field). It lifts samples of
each to a ``2d``-channel latent via a shared per-point MLP, stacks K
invertible coupling blocks, and reads the forward prediction off one
projection MLP (Q) or the inverse prediction off another (Q'), running the
SAME blocks/weights backward for the inverse direction. A beta-VAE is
layered on top of the projected-back input representation to give the
inverse direction a posterior to sample from (uncertainty, not just a
point estimate).

Why this can't be a literal transplant here, and what changes: this
project's "input function" (a resonator configuration: a small, unordered
set of up to num_res picks of [m,k,f_t,x,y]) and "output function" (the ERP
spectrum, a genuine function of frequency, 301 points) do NOT live on the
same grid -- so there is no single shared discretization to lift both onto,
which the paper's construction assumes. The adaptation:

  - The ONE shared, ordered grid in this problem is the 301-point frequency
    axis (fixed, deterministic -- utils.physics.freqs), so that is what
    plays the paper's "spatial grid" role, and the coupling blocks' gate
    function (here, the FNO Fourier layer) mixes along frequency, exactly
    as FNO's Fourier layer requires an ordered axis to be meaningful.
  - Forward direction (P, lift): the configuration is NOT itself a function
    of frequency, so its lift is built by broadcasting a permutation-
    invariant ResonatorSetEncoder summary of the whole resonator set to
    every one of the 301 points, concatenated with a genuine per-point
    resonance-detuning feature (ResonanceQueryEncoder) and the frequency
    value itself -- i.e. exactly the conditioning signal
    erp_forward/scripts/fno.py already uses for its own (non-invertible) FNO.
  - Inverse direction (P', lift): given a target spectrum, each of the 301
    (ERP value, frequency) pairs is lifted independently (no configuration
    known yet -- that is what we are solving for).
  - Design read-out (Q'): the design is NOT indexed by frequency -- it is
    one fixed configuration, not a curve. So instead of projecting each of
    the 301 recovered per-point latents to a design value independently,
    they are first mean+max pooled over frequency into one vector, and Q'
    projects THAT to the flat design_dim design vector.
  - beta-VAE (Sec 3.2): identical role and construction to the paper,
    specialized to this project's design space.

Three-step training (Sec 3.3) is implemented in train.py, not here -- this
module only defines the architecture and the loss TERMS each stage needs
(inherited from InvertibleOperatorBase), since which parameters are
optimized in which stage is a training-schedule decision, not a model one.
"""

from __future__ import annotations

from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from erp_forward.scripts.neural_operator_utils import (
    MLP,
    ResonanceQueryEncoder,
    ResonatorSetEncoder,
    build_resonator_encoder,
    resolve_activation,
)
from erp_invertible.scripts.common import DesignVAE, InvertibleOperatorBase, build_design_encoder
from erp_invertible.scripts.coupling import InvertibleCouplingStack
from utils.physics import freqs as _frequency_grid_hz


# ==================================================
# FNO-specific gate function (paper eq 1)
# ==================================================


class SpectralConv1d(nn.Module):
    """Learned convolution on retained Fourier modes -- same construction as
    erp_forward/scripts/fno.py's own SpectralConv1d.
    """

    def __init__(self, in_channels: int, out_channels: int, modes: int, padding: int = 0) -> None:
        super().__init__()
        # Zero-padding the (non-periodic) frequency axis before the FFT
        # stops the FFT's implicit periodicity from coupling the 10 Hz and
        # 160 Hz ends of the band (same idea as erp_forward/scripts/fno.py's
        # padding). 0 = original behaviour.
        self.padding = int(padding)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.modes = int(modes)
        scale = 1.0 / max(1, in_channels * out_channels)
        weight = scale * torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat)
        self.weight = nn.Parameter(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n_orig = x.shape[-1]
        if self.padding > 0:
            x = F.pad(x, (0, self.padding))
        n = x.shape[-1]
        x_ft = torch.fft.rfft(x, dim=-1)
        n_modes = min(self.modes, x_ft.shape[-1])
        out_ft = torch.zeros(x.shape[0], self.out_channels, x_ft.shape[-1], device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :n_modes] = torch.einsum("bim,iom->bom", x_ft[:, :, :n_modes], self.weight[:, :, :n_modes])
        return torch.fft.irfft(out_ft, n=n, dim=-1)[..., :n_orig]


class FourierLayer1d(nn.Module):
    """The paper's "standard FNO Fourier layer" L: a location-wise linear
    transform W (a kernel-3 conv) plus the spectral integral, then a
    nonlinear activation -- eq 1 of the paper. Deliberately has NO
    residual/skip connection of its own (unlike this repo's other
    FNOBlock1d): L itself is not required to be invertible, only the
    coupling structure built from it (coupling.py) is.
    """

    def __init__(self, width: int, modes: int, activation: str | type[nn.Module] = "gelu", padding: int = 0) -> None:
        super().__init__()
        self.spectral = SpectralConv1d(width, width, modes, padding=padding)
        self.local = nn.Conv1d(width, width, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(1, width)
        self.activation = resolve_activation(activation)()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.norm(self.local(x) + self.spectral(x)))


# ==================================================
# Full model
# ==================================================


class IFNO(InvertibleOperatorBase):
    def __init__(
        self,
        design_dim: int,
        n_freq: int = 301,
        width: int = 24,
        num_blocks: int = 4,
        modes: int = 48,
        config_hidden: int = 80,
        query_dim: int = 32,
        z_dim: int = 6,
        vae_hidden: int = 64,
        tau: float = 1.0,
        activation: str | type[nn.Module] = "gelu",
        use_sorted_branch: bool = False,
        encoder: str = "set",
        coordinate_features: str = "zscored",
        design_param: str = "full15",
        num_res: int = 3,
        gate: str = "softplus",
        gate_scale: float = 2.0,
        readout: str = "pooled",
        readout_bins: int = 16,
        spectral_padding: int = 0,
    ) -> None:
        super().__init__()
        # full15 ([m,k,f_t,x,y]) or bounded12 ([m,f_t,x,y], k derived) designs
        self._init_design(design_dim, design_param, num_res)
        self.n_freq = int(n_freq)
        self.width = int(width)

        self._register_frequency_grid(torch.from_numpy(_frequency_grid_hz.astype("float32")))

        # ---- Forward-direction lift (P) ----
        # use_sorted_branch adds the f_t-sorted resonator branch next to the pooled
        # set branch (DCO_sorted design, see SetAndSortedResonatorEncoder).
        self.use_sorted_branch = bool(use_sorted_branch)
        # encoder: "set" (pooled), "sorted" (f_t-ordered, lossless) or "set+sorted"
        self.encoder = "set+sorted" if self.use_sorted_branch and encoder == "set" else encoder
        self.configuration_encoder = build_design_encoder(self.encoder, False, self.num_res, config_hidden, 2 * self.width)
        self.resonance_query = ResonanceQueryEncoder(hidden_dim=query_dim, element_dim=query_dim, output_dim=query_dim)
        self.lift_p = nn.Linear(2 * self.width + query_dim + 1, 2 * self.width)

        # ---- Inverse-direction lift (P'): no configuration known yet ----
        self.freq_embed = MLP([1, query_dim, query_dim], activation=nn.SiLU)
        self.lift_pp = nn.Linear(1 + query_dim, 2 * self.width)

        # ---- Shared invertible coupling blocks, gated by an FNO Fourier layer ----
        self.blocks = InvertibleCouplingStack(
            lambda: FourierLayer1d(self.width, modes, activation=activation, padding=spectral_padding), num_blocks, tau=tau, gate=gate, gate_scale=gate_scale
        )

        # ---- Output projections ----
        activation_cls = resolve_activation(activation)
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
        context = self.configuration_encoder(configuration)[:, None, :].expand(-1, frequency.shape[1], -1)
        query = self.resonance_query(configuration, frequency)
        lifted = self.lift_p(torch.cat((context, query, frequency), dim=-1))  # (B, F, 2*width)
        return lifted.transpose(1, 2)  # (B, 2*width, F)

    def _lift_inverse(self, spectrum: torch.Tensor, frequency: torch.Tensor) -> torch.Tensor:
        freq_feat = self.freq_embed(frequency)  # (B, F, query_dim)
        lifted = self.lift_pp(torch.cat((spectrum[..., None], freq_feat), dim=-1))  # (B, F, 2*width)
        return lifted.transpose(1, 2)


def build_model(design_dim: int, **kwargs) -> IFNO:
    return IFNO(design_dim=design_dim, **kwargs)


DEFAULT_MODEL_CONFIG: Mapping[str, object] = {
    "n_freq": 301,
    "width": 24,
    "num_blocks": 4,
    "modes": 48,
    "config_hidden": 80,
    "query_dim": 32,
    "z_dim": 6,
    "vae_hidden": 64,
    "tau": 1.0,
    "activation": "gelu",
}
