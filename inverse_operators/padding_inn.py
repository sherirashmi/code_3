"""PadINN: Ardizzone et al., "Analyzing Inverse Problems with Invertible
Neural Networks" (arXiv:1808.04730, ICLR 2019).

Core idea: build ONE bijective network between parameters (here: design)
and measurements (here: ERP spectrum), instead of two separate networks
for the well-posed forward direction and the ambiguous inverse direction.
Since a bijection needs matched input/output dimension and this project's
design (15 numbers) and spectrum (301 numbers) don't match, the smaller
side is padded up to the larger side's size -- but WHICH side the padding
goes on, and whether it's zero-padding or a random latent, matters and is
easy to get backwards (an earlier draft of this file did).

This project's forward map is design -> spectrum (well-posed, already
modeled by 10 forward operators) and the inverse is spectrum -> design
(non-unique: several designs can produce near-identical spectra, mostly
through the mass/stiffness degeneracy f_t = sqrt(k/m)/(2*pi), see
common.py). That matches the paper's params -> measurement / measurement
-> params roles directly.

Where this project's shape differs from the paper's usual examples: they
usually have MORE parameters than measurements (an underdetermined
inverse problem by dimension count alone), so the *measurement* side gets
padded with a latent z, and the *parameter* side already has enough
dimensions and needs no padding. Here it's the reverse -- 301 spectrum
numbers vs. 15 design numbers -- so naively "pad whichever side is
smaller" would put the latent on the design side. That naive version
does NOT work: if z pads the design side, then inverting a fixed target
spectrum y produces a SINGLE deterministic (design, z) pair (a bijection
has exactly one preimage), with no freedom left to inject a fresh z at
inference. That collapses this model into a point-estimate inverse, no
posterior diversity at all -- the opposite of what every other model in
this package (MDN/cVAE/Flow/Diffusion/BasisFlow) provides.

The correct generalization keeps the latent on the *measurement/target*
(spectrum) side regardless of which raw dimension is bigger, because
that's the side you get to freely choose a fresh z for at inference time
(concatenate z ~ N(0, I) with the target spectrum, invert, get a
genuinely different design sample per draw). Concretely here: the design
is zero-padded (fixed, not learned) up to spectrum_dim + z_dim, and the
spectrum is paired with a real latent z of a size chosen independently of
the raw 301-vs-15 gap (16, in the same ballpark as this package's other
latent/coefficient sizes) -- not the 286 = 301-15 gap a naive reading of
the paper's formula would suggest.

Training losses (paper's Eq. 3-5, simplified since this project has full
paired (design, spectrum) data instead of the paper's typical partially
unpaired setting):
  - L_y: supervised MSE between the forward pass's predicted spectrum and
    the true one.
  - L_z: MMD (inverse-multiquadratic kernel, same kernel family the
    paper uses) between the forward pass's recovered z and a fresh
    N(0, I) sample -- this is what makes the z's the model was actually
    trained on match the z's it will be queried with at inference.
  - L_x: supervised MSE between the backward pass's recovered design (a
    true target spectrum + a fresh z) and the true design -- a direct
    simplification of the paper's own backward-reconstruction term, made
    possible by this project's full pairing.

No tractable log p(design|spectrum): this model is trained by MSE + MMD
distribution-matching, not maximum likelihood, so ``sample()`` returns a
plain tensor (no log-probability), matching ConditionalDiffusion's
convention rather than MDN/Flow/BasisFlow's.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from forward_operators.neural_operator_utils import MLP

from .common import flatten_configuration

LOG_SCALE_CLAMP = 2.0
_MMD_BANDWIDTHS = (0.2, 0.5, 0.9, 1.3, 2.0)


class AffineCoupling(nn.Module):
    """RealNVP coupling layer (no log-det tracking -- unlike basis_flow.py's,
    this model isn't trained by exact likelihood, so the Jacobian determinant
    is never used and would be dead code here).
    """

    def __init__(self, dim: int, hidden: int, mask: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("mask", mask)
        self.net = MLP([dim, hidden, hidden, 2 * dim], activation=nn.SiLU)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_masked = x * self.mask
        log_s, t = self.net(x_masked).chunk(2, dim=-1)
        log_s = torch.tanh(log_s) * LOG_SCALE_CLAMP
        inv_mask = 1.0 - self.mask
        return x_masked + inv_mask * (x * torch.exp(log_s) + t)

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        y_masked = y * self.mask
        log_s, t = self.net(y_masked).chunk(2, dim=-1)
        log_s = torch.tanh(log_s) * LOG_SCALE_CLAMP
        inv_mask = 1.0 - self.mask
        return y_masked + inv_mask * ((y - t) * torch.exp(-log_s))


class CouplingStack(nn.Module):
    """The single bijection between padded-design-space and padded-spectrum-space."""

    def __init__(self, dim: int, num_layers: int = 8, hidden: int = 96) -> None:
        super().__init__()
        self.dim = int(dim)
        layers = []
        for i in range(num_layers):
            mask = torch.zeros(dim)
            mask[i % 2 :: 2] = 1.0
            layers.append(AffineCoupling(dim, hidden, mask))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x
        for layer in self.layers:
            z = layer(z)
        return z

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        x = y
        for layer in reversed(self.layers):
            x = layer.inverse(x)
        return x


def _mmd_multiscale(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Inverse-multiquadratic-kernel MMD^2 between two equal-size samples.

    Same kernel family as the paper's reference implementation -- heavier
    tails than an RBF kernel, which behaves better for matching
    high-dimensional Gaussians than an RBF's fast-decaying tails do.
    """
    xx, yy, xy = x @ x.t(), y @ y.t(), x @ y.t()
    rx = xx.diag().unsqueeze(0).expand_as(xx)
    ry = yy.diag().unsqueeze(0).expand_as(yy)
    dxx = (rx.t() + rx - 2 * xx).clamp(min=0.0)
    dyy = (ry.t() + ry - 2 * yy).clamp(min=0.0)
    dxy = (rx.t() + ry - 2 * xy).clamp(min=0.0)
    total = 0.0
    for c in _MMD_BANDWIDTHS:
        total = total + c / (c + dxx) + c / (c + dyy) - 2 * c / (c + dxy)
    return total.mean()


class PadINN(nn.Module):
    """design <-> spectrum via one bijection, latent z padding the spectrum side."""

    def __init__(
        self,
        design_dim: int,
        n_freq: int = 301,
        z_dim: int = 16,
        num_layers: int = 8,
        hidden: int = 96,
    ) -> None:
        super().__init__()
        self.design_dim = int(design_dim)
        self.n_freq = int(n_freq)
        self.z_dim = int(z_dim)
        self.total_dim = self.n_freq + self.z_dim
        self.flow = CouplingStack(self.total_dim, num_layers=num_layers, hidden=hidden)

    def _pad_design(self, flat_design: torch.Tensor) -> torch.Tensor:
        zeros = flat_design.new_zeros(flat_design.shape[0], self.total_dim - self.design_dim)
        return torch.cat([flat_design, zeros], dim=-1)

    def predict_spectrum(self, design: torch.Tensor) -> torch.Tensor:
        """Forward-operator use: design -> predicted spectrum. Same weights as inversion."""
        flat = flatten_configuration(design)
        y_pad = self.flow(self._pad_design(flat))
        return y_pad[:, : self.n_freq]

    def training_loss(
        self,
        spectrum: torch.Tensor,
        design: torch.Tensor,
        x_weight: float = 1.0,
        z_mmd_weight: float = 50.0,
    ) -> torch.Tensor:
        flat = flatten_configuration(design)
        b = flat.shape[0]

        # Forward: real (zero-padded) design -> predicted [spectrum; z].
        y_pad_pred = self.flow(self._pad_design(flat))
        y_pred, z_pred = y_pad_pred[:, : self.n_freq], y_pad_pred[:, self.n_freq :]
        y_loss = F.mse_loss(y_pred, spectrum)

        # Shape z_pred's marginal toward N(0, I): the reason sample()'s
        # fresh z ~ N(0, I) lands somewhere the network actually saw
        # during training instead of an out-of-distribution corner.
        z_loss = _mmd_multiscale(z_pred, torch.randn_like(z_pred))

        # Backward: true spectrum + a FRESH z (not z_pred -- matching how
        # sample() actually queries the model) -> recovered design.
        z_sample = torch.randn(b, self.z_dim, device=flat.device)
        x_pad_rec = self.flow.inverse(torch.cat([spectrum, z_sample], dim=-1))
        x_loss = F.mse_loss(x_pad_rec[:, : self.design_dim], flat)

        return y_loss + x_weight * x_loss + z_mmd_weight * z_loss

    @torch.no_grad()
    def sample(self, spectrum: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """Returns ``(B, num_samples, design_dim)``, no tractable density
        (see module docstring) -- same convention as
        ConditionalDiffusion.sample(): a plain tensor, not a
        ``(samples, log_prob)`` tuple.
        """
        b = spectrum.shape[0]
        spectrum_e = spectrum[:, None, :].expand(-1, num_samples, -1).reshape(b * num_samples, -1)
        z = torch.randn(b * num_samples, self.z_dim, device=spectrum.device)
        x_pad_rec = self.flow.inverse(torch.cat([spectrum_e, z], dim=-1))
        x_rec = x_pad_rec[:, : self.design_dim]
        return x_rec.view(b, num_samples, self.design_dim)
