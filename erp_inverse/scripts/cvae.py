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

from erp_forward.scripts.neural_operator_utils import MLP

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
        spectrum_encoder: str = "pooled",
    ) -> None:
        super().__init__()
        self.design_dim = int(design_dim)
        self.latent_dim = int(latent_dim)

        self.encoder_spectrum = SpectrumEncoder(embed_dim=embed_dim, mode=spectrum_encoder)
        self.decoder_spectrum = SpectrumEncoder(embed_dim=embed_dim, mode=spectrum_encoder)

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
        mu, log_var, _ = self.encode(design, spectrum)
        std = torch.exp(0.5 * log_var)
        z = mu + std * torch.randn_like(std)
        # decode() must condition on decoder_spectrum's embedding here too
        # (not encoder_spectrum's, which only exists to help the posterior
        # network q(z|design,spectrum) -- design is unavailable at sample()
        # time) -- otherwise decoder_net is trained against one embedding
        # distribution and queried at sampling time with a DIFFERENT,
        # never-trained network's embedding (decoder_spectrum previously
        # received zero gradient the entire time, since nothing in training
        # ever called it).
        decoder_embedding = self.decoder_spectrum(spectrum)
        design_mean, design_log_var = self.decode(z, decoder_embedding)
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
    def marginal_log_prob(self, spectrum: torch.Tensor, flat: torch.Tensor, num_importance: int = 64) -> torch.Tensor:
        """Importance-sampled estimate of the MARGINAL ``log p(design | spectrum)``.

        ``p(x|c) = E_{z~q(z|x,c)}[ p(x|z,c) p(z) / q(z|x,c) ]`` -- the
        posterior network q is the proposal (Burda et al., IWAE estimator):

            log p(x|c) ~= logsumexp_k [log p(x|z_k,c) + log N(z_k;0,I) - log q(z_k|x,c)] - log K

        ``spectrum``: (B, n_freq); ``flat``: (B, S, D) or (B, D). Returns
        (B, S) or (B,). This is the number comparable with the exact
        densities of MDN/Flow/BasisFlow (a lower-variance estimate with
        larger ``num_importance``).
        """
        squeeze = flat.dim() == 2
        if squeeze:
            flat = flat[:, None, :]
        b, s, d = flat.shape
        k = int(num_importance)
        spec = spectrum[:, None, :].expand(-1, s, -1).reshape(b * s, -1)
        x = flat.reshape(b * s, d)
        params = self.posterior_net(torch.cat((x, self.encoder_spectrum(spec)), dim=-1))
        mu, log_var = params.chunk(2, dim=-1)
        dec_embedding = self.decoder_spectrum(spec)
        std = torch.exp(0.5 * log_var)
        z = mu[None] + std[None] * torch.randn(k, *mu.shape, device=mu.device)  # (K, BS, L)
        log_q = (-0.5 * (((z - mu[None]) / std[None]) ** 2 + log_var[None] + math.log(2 * math.pi))).sum(-1)
        log_prior = (-0.5 * (z**2 + math.log(2 * math.pi))).sum(-1)
        mean, dec_log_var = self.decode(z, dec_embedding[None].expand(k, -1, -1))
        log_lik = (-0.5 * (((x[None] - mean) ** 2) / dec_log_var.exp() + dec_log_var + math.log(2 * math.pi))).sum(-1)
        log_p = torch.logsumexp(log_lik + log_prior - log_q, dim=0) - math.log(k)
        log_p = log_p.view(b, s)
        return log_p[:, 0] if squeeze else log_p

    @torch.no_grad()
    def sample(self, spectrum: torch.Tensor, num_samples: int = 1, num_importance: int = 64):
        """Returns ``(flat_designs, log_prob)``, both ``(B, num_samples, ...)``.

        ``flat_designs`` are drawn from the generative model (z ~ N(0, I),
        then the decoder Gaussian). ``log_prob`` is the importance-sampled
        MARGINAL ``log p(design | spectrum)`` (see :meth:`marginal_log_prob`).

        (Earlier versions returned the decoder density ``log p(design | z,
        spectrum)`` at the particular sampled z instead. That is NOT the
        posterior density -- it omits the latent prior and the integral over
        z, and since the design was just drawn from that same Gaussian it
        mostly reflects the decoder's predicted variance -- so it must not be
        compared with or ranked like the MDN/Flow densities.)
        """
        b = spectrum.shape[0]
        embedding = self.decoder_spectrum(spectrum)
        embedding = embedding[:, None, :].expand(-1, num_samples, -1).reshape(b * num_samples, -1)
        z = torch.randn(b * num_samples, self.latent_dim, device=spectrum.device)
        design_mean, design_log_var = self.decode(z, embedding)
        std = torch.exp(0.5 * design_log_var)
        flat = (design_mean + std * torch.randn_like(std)).view(b, num_samples, self.design_dim)
        log_prob = self.marginal_log_prob(spectrum, flat, num_importance=num_importance)
        return flat, log_prob

    @torch.no_grad()
    def log_prob(self, spectrum: torch.Tensor, flat: torch.Tensor) -> torch.Tensor:
        """Same interface as MDN/Flow: (approximate) marginal log p(design | spectrum)."""
        return self.marginal_log_prob(spectrum, flat)
