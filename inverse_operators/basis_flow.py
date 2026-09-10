"""BasisFlow: one invertible map, learned basis coefficients, both directions.

Synthesizes the three papers reviewed for this project into one architecture
actually buildable for this problem's shape:

- **Thorpe et al., "Learning Generalizable Neural Operators for Inverse
  Problems" (B2B^-1, arXiv:2512.18120)**: decouple *representation* from
  *inversion* by learning a compact coefficient representation of the
  function space (here, the 301-point ERP spectrum) and doing the actual
  inverse mapping in that coefficient space instead of on the raw function.
  ``SpectrumBasisAutoencoder`` below is that learned basis: it compresses a
  spectrum to a ``design_dim``-sized coefficient vector and back.

- **Long et al., "Invertible Fourier Neural Operators..." (iFNO,
  arXiv:2402.11722)** and **Kaltenbach, Perdikaris & Koutsourelakis,
  "Semi-supervised invertible neural operators..." (Comp. Mech. 2023)**:
  one invertible coupling stack, run forward for the forward problem and
  inverted for the inverse problem, instead of two unrelated networks.
  ``CouplingFlow`` below is exactly that: a RealNVP-style bijection between
  design space and the learned coefficient space, with the SAME weights
  used in both directions.

Why this shape and not a literal transplant of either paper: both iFNO and
the invertible DeepONet need their invertible block's input and output to
be the same dimension (a bijection requires it). This project's design
vector is 15 numbers (3 resonators x [m,k,f_t,x,y]); the ERP spectrum is
301 numbers. Neither paper's own domain has that mismatch (their examples
match the input/output field's discretization). B2B^-1's basis-coefficient
idea is the bridge: compress the spectrum to 15 coefficients first, and
*then* the coupling stack has two matched-dimension spaces to be a genuine
bijection between.

What "both directions, same weights" buys here, concretely:
  - Forward use: design -> CouplingFlow -> coefficients -> decode -> a full
    predicted ERP spectrum. A real forward-operator capability, from the
    same weights trained for the inverse problem -- none of this package's
    other 4 models (MDN/cVAE/Flow/Diffusion) can do this at all.
  - Inverse use: target spectrum -> encode -> coefficients (sampled from a
    learned, spectrum-conditioned Gaussian, not a fixed prior) ->
    CouplingFlow.inverse -> a design, with an exact log p(design|spectrum)
    from the change-of-variables formula (same idea the existing `Flow`
    model already uses, but conditioned through the learned basis instead
    of directly on a raw spectrum embedding).

Design representation, canonicalization, and log-probability/confidence
reporting conventions match the rest of this package exactly (see
inverse_operators/common.py and inverse_operators/flow.py).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from forward_operators.neural_operator_utils import MLP

from .common import SpectrumEncoder, flatten_configuration

LOG_SCALE_CLAMP = 2.0


class SpectrumBasisAutoencoder(nn.Module):
    """Learned basis (Thorpe et al.): spectrum <-> coeff_dim coefficients.

    The encoder reuses the same shared Conv1d ``SpectrumEncoder`` every
    model in this package conditions on, then projects to a small Gaussian
    over ``coeff_dim`` coefficients (mean + log-variance) instead of a
    single point -- this Gaussian is what CouplingFlow's inverse direction
    samples from, and what its forward direction is scored against (a
    learned, spectrum-conditioned prior in coefficient space, rather than
    the fixed standard-Gaussian prior a typical conditional flow uses).
    """

    def __init__(self, coeff_dim: int, n_freq: int, embed_dim: int = 96, hidden: int = 128) -> None:
        super().__init__()
        self.coeff_dim = int(coeff_dim)
        self.spectrum_encoder = SpectrumEncoder(embed_dim=embed_dim)
        self.coeff_head = MLP([embed_dim, hidden, 2 * coeff_dim], activation=nn.SiLU)
        self.decoder = MLP([coeff_dim, hidden, hidden, n_freq], activation=nn.SiLU)

    def encode(self, spectrum: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """spectrum (B, n_freq) -> (mu, log_var), each (B, coeff_dim)."""
        embedding = self.spectrum_encoder(spectrum)
        mu, log_var = self.coeff_head(embedding).chunk(2, dim=-1)
        log_var = log_var.clamp(min=-8.0, max=4.0)
        return mu, log_var

    def decode(self, coefficients: torch.Tensor) -> torch.Tensor:
        """(B, coeff_dim) -> reconstructed/predicted spectrum (B, n_freq)."""
        return self.decoder(coefficients)


class AffineCoupling(nn.Module):
    """RealNVP coupling layer with BOTH directions returning their log-det.

    Unlike inverse_operators/flow.py's coupling layer (which only needs a
    forward log-det, since that model always samples via the inverse
    direction blind to any density there), this one needs the inverse
    direction's log-det too -- CouplingFlow.inverse() is used at sampling
    time and its log-det is part of the reported log p(design|spectrum)
    (see the module docstring's change-of-variables derivation).
    """

    def __init__(self, dim: int, hidden: int, mask: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("mask", mask)
        self.net = MLP([dim, hidden, hidden, 2 * dim], activation=nn.SiLU)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_masked = x * self.mask
        log_s, t = self.net(x_masked).chunk(2, dim=-1)
        log_s = torch.tanh(log_s) * LOG_SCALE_CLAMP
        inv_mask = 1.0 - self.mask
        y = x_masked + inv_mask * (x * torch.exp(log_s) + t)
        log_det = (inv_mask * log_s).sum(dim=-1)
        return y, log_det

    def inverse(self, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        y_masked = y * self.mask
        log_s, t = self.net(y_masked).chunk(2, dim=-1)
        log_s = torch.tanh(log_s) * LOG_SCALE_CLAMP
        inv_mask = 1.0 - self.mask
        x = y_masked + inv_mask * ((y - t) * torch.exp(-log_s))
        log_det = -(inv_mask * log_s).sum(dim=-1)
        return x, log_det


class CouplingFlow(nn.Module):
    """The invertible block itself: design <-> coefficients, one set of weights.

    No conditioning input inside the coupling layers (unlike a typical
    conditional flow) -- conditioning on the spectrum happens entirely at
    the SpectrumBasisAutoencoder stage. This is deliberately a clean,
    unconditional bijection between the two matched-dimension spaces, which
    is what makes "run it forward for the forward problem, invert it for
    the inverse problem" a meaningful, literal description rather than a
    loose analogy.
    """

    def __init__(self, dim: int, num_layers: int = 8, hidden: int = 96) -> None:
        super().__init__()
        self.dim = int(dim)
        layers = []
        for i in range(num_layers):
            mask = torch.zeros(dim)
            mask[i % 2 :: 2] = 1.0
            layers.append(AffineCoupling(dim, hidden, mask))
        self.layers = nn.ModuleList(layers)

    def forward(self, design: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """design -> coefficients, log|det dT/d(design)| (forward-problem direction)."""
        z = design
        log_det_total = torch.zeros(design.shape[0], device=design.device)
        for layer in self.layers:
            z, log_det = layer(z)
            log_det_total = log_det_total + log_det
        return z, log_det_total

    def inverse(self, coefficients: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """coefficients -> design, log|det dT^-1/d(coeff)| (inverse-problem direction)."""
        x = coefficients
        log_det_total = torch.zeros(coefficients.shape[0], device=coefficients.device)
        for layer in reversed(self.layers):
            x, log_det = layer.inverse(x)
            log_det_total = log_det_total + log_det
        return x, log_det_total


def _gaussian_log_prob(x: torch.Tensor, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """Diagonal Gaussian log-density, summed over the last dim."""
    return (-0.5 * ((x - mu) ** 2 / torch.exp(log_var) + log_var + math.log(2 * math.pi))).sum(dim=-1)


class BasisFlow(nn.Module):
    """design <-> spectrum, both directions, via a learned basis + one invertible flow."""

    def __init__(
        self,
        design_dim: int,
        n_freq: int = 301,
        embed_dim: int = 96,
        ae_hidden: int = 128,
        num_layers: int = 8,
        flow_hidden: int = 96,
    ) -> None:
        super().__init__()
        self.design_dim = int(design_dim)
        self.basis = SpectrumBasisAutoencoder(design_dim, n_freq, embed_dim=embed_dim, hidden=ae_hidden)
        self.flow = CouplingFlow(design_dim, num_layers=num_layers, hidden=flow_hidden)

    def predict_spectrum(self, design: torch.Tensor) -> torch.Tensor:
        """Forward-operator use: design -> predicted spectrum. Same weights as inversion."""
        flat = flatten_configuration(design)
        coefficients, _ = self.flow(flat)
        return self.basis.decode(coefficients)

    def training_loss(
        self,
        spectrum: torch.Tensor,
        design: torch.Tensor,
        ae_weight: float = 0.1,
        forward_weight: float = 0.1,
    ) -> torch.Tensor:
        """-log p(design|spectrum) [primary] + basis-reconstruction + forward-consistency.

        The primary term maps the TRUE design through the flow's forward
        direction to a predicted coefficient, and scores it under the
        encoder's spectrum-conditioned Gaussian plus the forward log-det --
        the standard way a conditional normalizing flow is trained (map
        data to the latent/coefficient space, evaluate under the
        conditional prior there, add the log-det), just with a *learned*
        conditional prior instead of a fixed standard Gaussian.
        """
        flat_design = flatten_configuration(design)
        mu, log_var = self.basis.encode(spectrum)

        coefficients, log_det_fwd = self.flow(flat_design)
        log_prob = _gaussian_log_prob(coefficients, mu, log_var) + log_det_fwd
        inverse_nll = -log_prob.mean()

        recon_spectrum = self.basis.decode(mu)
        ae_loss = nn.functional.mse_loss(recon_spectrum, spectrum)

        predicted_spectrum = self.basis.decode(coefficients)
        forward_loss = nn.functional.mse_loss(predicted_spectrum, spectrum)

        return inverse_nll + ae_weight * ae_loss + forward_weight * forward_loss

    @torch.no_grad()
    def sample(self, spectrum: torch.Tensor, num_samples: int = 1):
        """Returns ``(flat_designs, log_prob)``, both ``(B, num_samples, ...)``.

        Samples coefficients from the encoder's spectrum-conditioned
        Gaussian, inverts the flow to get a design, and reports the exact
        ``log p(design|spectrum)`` of each returned design (same
        change-of-variables identity as ``training_loss``, evaluated in the
        inverse direction: log q(coeff) + log|det dT^-1/dcoeff|).
        """
        b = spectrum.shape[0]
        mu, log_var = self.basis.encode(spectrum)
        std = torch.exp(0.5 * log_var)

        mu_e = mu[:, None, :].expand(-1, num_samples, -1).reshape(b * num_samples, -1)
        std_e = std[:, None, :].expand(-1, num_samples, -1).reshape(b * num_samples, -1)
        log_var_e = log_var[:, None, :].expand(-1, num_samples, -1).reshape(b * num_samples, -1)
        coefficients = mu_e + std_e * torch.randn_like(mu_e)

        flat_design, log_det_inv = self.flow.inverse(coefficients)
        log_q_coeff = _gaussian_log_prob(coefficients, mu_e, log_var_e)
        log_prob = log_q_coeff + log_det_inv

        return flat_design.view(b, num_samples, self.design_dim), log_prob.view(b, num_samples)
