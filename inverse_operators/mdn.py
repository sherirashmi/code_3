"""Mixture Density Network (MDN): amortized posterior p(design | spectrum).

The simplest probabilistic inverse baseline -- a single forward pass through
the spectrum encoder predicts the parameters of a Gaussian mixture over the
flat design vector (mixture weights, per-component means, per-component
diagonal std-devs). Sampling is closed-form and instantaneous (pick a
component, sample its Gaussian) -- no iterative inference needed at all,
unlike the flow or diffusion models. This is the classic amortized
simulation-based-inference (neural posterior estimation) baseline: cheap,
exact-form density, and already able to represent genuine multi-modality
(one target spectrum -> several distinct plausible designs) as long as the
number of mixture components is enough to cover them.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from utils.neural_operator_utils import MLP

from .common import SpectrumEncoder, flatten_configuration


class MDN(nn.Module):
    def __init__(
        self,
        design_dim: int,
        num_components: int = 8,
        embed_dim: int = 96,
        hidden: int = 128,
        min_std: float = 1e-3,
    ) -> None:
        super().__init__()
        self.design_dim = int(design_dim)
        self.num_components = int(num_components)
        self.min_std = float(min_std)

        self.encoder = SpectrumEncoder(embed_dim=embed_dim)
        out_dim = num_components * (1 + 2 * design_dim)
        self.head = MLP([embed_dim, hidden, hidden, out_dim], activation=nn.SiLU)

    def _params(self, spectrum: torch.Tensor):
        embedding = self.encoder(spectrum)
        raw = self.head(embedding)
        k, d = self.num_components, self.design_dim
        logits = raw[:, :k]
        mu = raw[:, k : k + k * d].view(-1, k, d)
        log_std = raw[:, k + k * d :].view(-1, k, d)
        std = nn.functional.softplus(log_std) + self.min_std
        return logits, mu, std

    def log_prob(self, spectrum: torch.Tensor, design: torch.Tensor) -> torch.Tensor:
        flat = flatten_configuration(design)
        logits, mu, std = self._params(spectrum)
        log_weights = torch.log_softmax(logits, dim=-1)  # (B, K)
        x = flat[:, None, :]  # (B, 1, D)
        component_log_prob = (
            -0.5 * (((x - mu) / std) ** 2 + 2 * torch.log(std) + math.log(2 * math.pi))
        ).sum(dim=-1)  # (B, K)
        return torch.logsumexp(log_weights + component_log_prob, dim=-1)  # (B,)

    def training_loss(self, spectrum: torch.Tensor, design: torch.Tensor) -> torch.Tensor:
        return -self.log_prob(spectrum, design).mean()

    @torch.no_grad()
    def sample(self, spectrum: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """Returns ``(B, num_samples, design_dim)`` flat normalized design samples."""
        logits, mu, std = self._params(spectrum)
        b, k, d = mu.shape
        weights = torch.softmax(logits, dim=-1)
        component = torch.multinomial(weights, num_samples, replacement=True)  # (B, S)
        mu_s = torch.gather(mu, 1, component[:, :, None].expand(-1, -1, d))
        std_s = torch.gather(std, 1, component[:, :, None].expand(-1, -1, d))
        eps = torch.randn_like(mu_s)
        return mu_s + std_s * eps
