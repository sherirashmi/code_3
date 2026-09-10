"""Conditional VAE: p(design | spectrum) via a spectrum-conditioned latent variable.

Encoder q(z | design, spectrum) and decoder p(design | z, spectrum) are both
conditioned on the spectrum embedding. At inference time the encoder is
discarded: sample z ~ N(0, I), decode conditioned on the target spectrum,
repeat for as many candidate designs as wanted.

The decoder outputs a full Gaussian (mean *and* log-variance) over the flat
design vector, not just a point estimate -- this gives a genuine, if
model-relative, ``p(design | z, spectrum)`` density for every sample rather
than only a reconstruction target, which lets callers report a real
probability alongside each generated design (see ``sample``'s returned
``log_prob``). The reconstruction term below is therefore a proper Gaussian
negative log-likelihood, not the plain MSE used before.

Beta (the KL weight) is deliberately NOT fixed here -- it's passed in by the
caller on every call to ``training_loss`` so it can be *annealed* (ramped
from ~0 up to its target over the first N epochs) rather than fixed from
step 1. A fixed, non-trivial beta from the very start is the classic cause
of posterior collapse: the KL term crushes q(z|x) to the prior before the
decoder has learned to use z at all, so the decoder falls back to predicting
the spectrum-conditional mean design and ignores z entirely -- which shows
up exactly as a suspiciously flat loss curve and near-zero sample-to-sample
diversity, both of which this model showed in its first training run.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from forward_operators.neural_operator_utils import MLP

from .common import SpectrumEncoder, flatten_configuration

MIN_LOG_VAR = -6.0
MAX_LOG_VAR = 2.0


class ConditionalVAE(nn.Module):
    def __init__(
        self,
        design_dim: int,
        latent_dim: int = 8,
        embed_dim: int = 96,
        hidden: int = 128,
    ) -> None:
        super().__init__()
        self.design_dim = int(design_dim)
        self.latent_dim = int(latent_dim)

        self.encoder_spectrum = SpectrumEncoder(embed_dim=embed_dim)
        self.decoder_spectrum = SpectrumEncoder(embed_dim=embed_dim)

        self.posterior_net = MLP(
            [design_dim + embed_dim, hidden, hidden, 2 * latent_dim], activation=nn.SiLU
        )
        self.decoder_net = MLP(
            [latent_dim + embed_dim, hidden, hidden, 2 * design_dim], activation=nn.SiLU
        )

    def encode(self, design: torch.Tensor, spectrum: torch.Tensor):
        flat = flatten_configuration(design)
        embedding = self.encoder_spectrum(spectrum)
        params = self.posterior_net(torch.cat((flat, embedding), dim=-1))
        mu, log_var = params.chunk(2, dim=-1)
        return mu, log_var, embedding

    def decode(self, z: torch.Tensor, embedding: torch.Tensor):
        """Returns ``(design_mean, design_log_var)``, both ``(..., design_dim)``."""
        params = self.decoder_net(torch.cat((z, embedding), dim=-1))
        mean, log_var = params.chunk(2, dim=-1)
        log_var = log_var.clamp(MIN_LOG_VAR, MAX_LOG_VAR)
        return mean, log_var

    def forward(self, design: torch.Tensor, spectrum: torch.Tensor):
        mu, log_var, embedding = self.encode(design, spectrum)
        std = torch.exp(0.5 * log_var)
        z = mu + std * torch.randn_like(std)
        design_mean, design_log_var = self.decode(z, embedding)
        return design_mean, design_log_var, mu, log_var

    def training_loss(
        self, spectrum: torch.Tensor, design: torch.Tensor, beta: float = 0.1
    ) -> torch.Tensor:
        flat = flatten_configuration(design)
        design_mean, design_log_var, mu, log_var = self.forward(design, spectrum)
        recon_nll = 0.5 * (
            ((flat - design_mean) ** 2) / design_log_var.exp()
            + design_log_var
            + math.log(2 * math.pi)
        ).sum(dim=-1).mean()
        kl = -0.5 * torch.mean(
            torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1)
        )
        return recon_nll + beta * kl

    @torch.no_grad()
    def sample(self, spectrum: torch.Tensor, num_samples: int = 1):
        """Returns ``(flat_designs, log_prob)``, both ``(B, num_samples, ...)``.

        ``flat_designs`` are drawn from the decoder's Gaussian (not just its
        mean), and ``log_prob`` is that Gaussian's log-density evaluated at
        the drawn sample -- an honest ``log p(design | z, spectrum)`` for
        the specific design returned, usable directly as a per-sample
        confidence/probability score.
        """
        b = spectrum.shape[0]
        embedding = self.decoder_spectrum(spectrum)
        embedding = embedding[:, None, :].expand(-1, num_samples, -1).reshape(b * num_samples, -1)
        z = torch.randn(b * num_samples, self.latent_dim, device=spectrum.device)
        design_mean, design_log_var = self.decode(z, embedding)
        std = torch.exp(0.5 * design_log_var)
        flat = design_mean + std * torch.randn_like(std)
        log_prob = -0.5 * (
            ((flat - design_mean) ** 2) / design_log_var.exp()
            + design_log_var
            + math.log(2 * math.pi)
        ).sum(dim=-1)
        return flat.view(b, num_samples, self.design_dim), log_prob.view(b, num_samples)
