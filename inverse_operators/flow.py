"""Conditional Normalizing Flow (cINN): exact-density p(design | spectrum).

A stack of spectrum-conditioned affine coupling layers (RealNVP-style)
forms an exactly invertible map between the design space and a standard
Gaussian latent space. Unlike the VAE or MDN, this gives an *exact*
log-likelihood (no variational bound, no fixed mixture-count cap on
multi-modality) -- the standard choice in the inverse-problems literature
(Ardizzone et al., "Analyzing Inverse Problems with Invertible Neural
Networks") for exactly this kind of ambiguous, many-to-one inverse mapping.
Sampling: draw z ~ N(0, I), run the inverse coupling stack conditioned on
the target spectrum embedding.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from operators.neural_operator_utils import MLP

from .common import SpectrumEncoder, flatten_configuration

LOG_SCALE_CLAMP = 2.0


class AffineCoupling(nn.Module):
    def __init__(self, dim: int, cond_dim: int, hidden: int, mask: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("mask", mask)
        self.net = MLP([dim + cond_dim, hidden, hidden, 2 * dim], activation=nn.SiLU)

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        x_masked = x * self.mask
        log_s, t = self.net(torch.cat((x_masked, cond), dim=-1)).chunk(2, dim=-1)
        log_s = torch.tanh(log_s) * LOG_SCALE_CLAMP
        inv_mask = 1.0 - self.mask
        y = x_masked + inv_mask * (x * torch.exp(log_s) + t)
        log_det = (inv_mask * log_s).sum(dim=-1)
        return y, log_det

    def inverse(self, y: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        y_masked = y * self.mask
        log_s, t = self.net(torch.cat((y_masked, cond), dim=-1)).chunk(2, dim=-1)
        log_s = torch.tanh(log_s) * LOG_SCALE_CLAMP
        inv_mask = 1.0 - self.mask
        return y_masked + inv_mask * ((y - t) * torch.exp(-log_s))


class ConditionalFlow(nn.Module):
    def __init__(
        self,
        design_dim: int,
        num_layers: int = 8,
        hidden: int = 96,
        embed_dim: int = 96,
    ) -> None:
        super().__init__()
        self.design_dim = int(design_dim)
        self.encoder = SpectrumEncoder(embed_dim=embed_dim)

        layers = []
        for i in range(num_layers):
            mask = torch.zeros(design_dim)
            mask[i % 2 :: 2] = 1.0  # alternate even/odd masks across layers
            layers.append(AffineCoupling(design_dim, embed_dim, hidden, mask))
        self.layers = nn.ModuleList(layers)

    def _transform_flat(self, flat: torch.Tensor, cond: torch.Tensor):
        """Already-flat design -> latent z, plus total log-det-Jacobian."""
        z = flat
        log_det_total = torch.zeros(flat.shape[0], device=flat.device)
        for layer in self.layers:
            z, log_det = layer(z, cond)
            log_det_total = log_det_total + log_det
        return z, log_det_total

    def _log_prob_flat(self, flat: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        z, log_det = self._transform_flat(flat, cond)
        log_prior = -0.5 * (z**2).sum(dim=-1) - 0.5 * self.design_dim * math.log(2 * math.pi)
        return log_prior + log_det

    def forward(self, design: torch.Tensor, spectrum: torch.Tensor):
        """design -> latent z, plus total log-det-Jacobian (for training)."""
        flat = flatten_configuration(design)
        cond = self.encoder(spectrum)
        return self._transform_flat(flat, cond)

    def log_prob(self, spectrum: torch.Tensor, design: torch.Tensor) -> torch.Tensor:
        """Exact log p(design | spectrum) -- the change-of-variables density."""
        flat = flatten_configuration(design)
        cond = self.encoder(spectrum)
        return self._log_prob_flat(flat, cond)

    def training_loss(self, spectrum: torch.Tensor, design: torch.Tensor) -> torch.Tensor:
        return -self.log_prob(spectrum, design).mean()

    @torch.no_grad()
    def sample(self, spectrum: torch.Tensor, num_samples: int = 1):
        """Returns ``(flat_designs, log_prob)``, both ``(B, num_samples, ...)``.

        ``log_prob`` is the exact ``log p(design | spectrum)`` of each
        returned design under this model -- usable directly as a per-sample
        probability/confidence score.
        """
        b = spectrum.shape[0]
        cond = self.encoder(spectrum)
        cond = cond[:, None, :].expand(-1, num_samples, -1).reshape(b * num_samples, -1)
        z = torch.randn(b * num_samples, self.design_dim, device=spectrum.device)
        x = z
        for layer in reversed(self.layers):
            x = layer.inverse(x, cond)
        log_prob = self._log_prob_flat(x, cond)
        return x.view(b, num_samples, self.design_dim), log_prob.view(b, num_samples)
