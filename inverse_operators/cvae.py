"""Conditional VAE: p(design | spectrum) via a spectrum-conditioned latent variable.

Encoder q(z | design, spectrum) and decoder p(design | z, spectrum) are both
conditioned on the spectrum embedding. At inference time the encoder is
discarded: sample z ~ N(0, I), decode conditioned on the target spectrum,
repeat for as many candidate designs as wanted. Simpler and faster to train
than the flow or diffusion models, but tends to mode-average across
genuinely distinct solutions more than either -- a useful, fast probabilistic
baseline rather than the sharpest one.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from utils.neural_operator_utils import MLP

from .common import SpectrumEncoder, flatten_configuration


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
            [latent_dim + embed_dim, hidden, hidden, design_dim], activation=nn.SiLU
        )

    def encode(self, design: torch.Tensor, spectrum: torch.Tensor):
        flat = flatten_configuration(design)
        embedding = self.encoder_spectrum(spectrum)
        params = self.posterior_net(torch.cat((flat, embedding), dim=-1))
        mu, log_var = params.chunk(2, dim=-1)
        return mu, log_var, embedding

    def decode(self, z: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        return self.decoder_net(torch.cat((z, embedding), dim=-1))

    def forward(self, design: torch.Tensor, spectrum: torch.Tensor):
        mu, log_var, embedding = self.encode(design, spectrum)
        std = torch.exp(0.5 * log_var)
        z = mu + std * torch.randn_like(std)
        recon = self.decode(z, embedding)
        return recon, mu, log_var

    def training_loss(
        self, spectrum: torch.Tensor, design: torch.Tensor, beta: float = 0.1
    ) -> torch.Tensor:
        flat = flatten_configuration(design)
        recon, mu, log_var = self.forward(design, spectrum)
        recon_loss = nn.functional.mse_loss(recon, flat, reduction="mean")
        kl = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        return recon_loss + beta * kl

    @torch.no_grad()
    def sample(self, spectrum: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """Returns ``(B, num_samples, design_dim)`` flat normalized design samples."""
        b = spectrum.shape[0]
        embedding = self.decoder_spectrum(spectrum)
        embedding = embedding[:, None, :].expand(-1, num_samples, -1).reshape(b * num_samples, -1)
        z = torch.randn(b * num_samples, self.latent_dim, device=spectrum.device)
        flat = self.decode(z, embedding)
        return flat.view(b, num_samples, self.design_dim)
