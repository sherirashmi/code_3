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

Mixture components are prone to collapsing onto each other under plain
random init (see ``initialize_from_data``'s docstring); ``train_all.py``
warm-starts them via k-means++ on a batch of real training designs right
after construction, before any gradient step.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from erp_forward_operators.neural_operator_utils import MLP

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

    @staticmethod
    def _mixture_log_prob(flat: torch.Tensor, logits: torch.Tensor, mu: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """``flat``: (B, D) or (B, S, D); ``logits/mu/std``: (B, K) / (B, K, D)."""
        log_weights = torch.log_softmax(logits, dim=-1)  # (B, K)
        if flat.dim() == 3:
            x = flat[:, :, None, :]  # (B, S, 1, D)
            mu_e = mu[:, None, :, :]  # (B, 1, K, D)
            std_e = std[:, None, :, :]
            weights_e = log_weights[:, None, :]  # (B, 1, K)
        else:
            x = flat[:, None, :]  # (B, 1, D)
            mu_e, std_e, weights_e = mu, std, log_weights
        component_log_prob = (
            -0.5 * (((x - mu_e) / std_e) ** 2 + 2 * torch.log(std_e) + math.log(2 * math.pi))
        ).sum(dim=-1)
        return torch.logsumexp(weights_e + component_log_prob, dim=-1)

    def log_prob(self, spectrum: torch.Tensor, design: torch.Tensor) -> torch.Tensor:
        flat = flatten_configuration(design)
        logits, mu, std = self._params(spectrum)
        return self._mixture_log_prob(flat, logits, mu, std)  # (B,)

    def training_loss(self, spectrum: torch.Tensor, design: torch.Tensor) -> torch.Tensor:
        return -self.log_prob(spectrum, design).mean()

    @torch.no_grad()
    def sample(self, spectrum: torch.Tensor, num_samples: int = 1):
        """Returns ``(flat_designs, log_prob)``, both ``(B, num_samples, ...)``.

        ``log_prob`` is the exact mixture log-density (not just the sampled
        component's) evaluated at each returned design.
        """
        logits, mu, std = self._params(spectrum)
        b, k, d = mu.shape
        weights = torch.softmax(logits, dim=-1)
        component = torch.multinomial(weights, num_samples, replacement=True)  # (B, S)
        mu_s = torch.gather(mu, 1, component[:, :, None].expand(-1, -1, d))
        std_s = torch.gather(std, 1, component[:, :, None].expand(-1, -1, d))
        eps = torch.randn_like(mu_s)
        flat = mu_s + std_s * eps  # (B, S, D)
        log_prob = self._mixture_log_prob(flat, logits, mu, std)  # (B, S)
        return flat, log_prob

    @torch.no_grad()
    def initialize_from_data(self, designs: torch.Tensor, iters: int = 25, seed: int = 0) -> None:
        """Warm-start the K mixture-component means via k-means++ clustering
        on a batch of real (flat or (num_res,5)) training designs, instead
        of leaving them at random init.

        Why: MDN mixtures are a well-documented pain to train well -- with
        components starting scattered randomly, gradient descent often lets
        several of them collapse onto the same region or never specialize
        at all ("component collapse"), simply because of a bad starting
        point, not any fundamental limit of the mixture-density idea. Since
        the training loss the first several epochs shows exactly this
        project's MDN never getting as sharp/confident as Flow or BasisFlow
        (its NLL never goes negative -- see the training-curves discussion),
        giving every component a sensible starting region of real design
        space to own, before gradient descent ever begins, costs nothing at
        training time and directly targets that failure mode.

        Sets the shared head's final layer so the FIRST forward pass's
        predicted mu for every component is close to its cluster center
        regardless of what the (still randomly initialized) spectrum
        encoder's embedding looks like; training then refines both freely
        from there. Call this once, right after constructing the model and
        before training starts, with a batch of real training designs.
        """
        flat = flatten_configuration(designs) if designs.dim() == 3 else designs
        flat = flat.to(torch.float32)
        n, d = flat.shape
        k = self.num_components
        if n < k:
            raise ValueError(f"Need at least num_components={k} designs to warm-start from, got {n}.")
        if d != self.design_dim:
            raise ValueError(f"designs' flat dim {d} does not match design_dim={self.design_dim}.")

        # k-means++ init (spread the starting centers out, rather than
        # picking them uniformly at random, which can start two centers
        # right next to each other and waste a component immediately).
        generator = torch.Generator().manual_seed(seed)
        first_idx = int(torch.randint(0, n, (1,), generator=generator).item())
        centers = [flat[first_idx]]
        for _ in range(k - 1):
            stacked = torch.stack(centers, dim=0)
            dist_sq = ((flat[:, None, :] - stacked[None, :, :]) ** 2).sum(dim=-1).min(dim=-1).values
            probs = dist_sq / dist_sq.sum().clamp(min=1e-12)
            next_idx = int(torch.multinomial(probs, 1, generator=generator).item())
            centers.append(flat[next_idx])
        centers = torch.stack(centers, dim=0)  # (K, D)

        # Plain Lloyd's-algorithm refinement.
        for _ in range(iters):
            dists = ((flat[:, None, :] - centers[None, :, :]) ** 2).sum(dim=-1)  # (N, K)
            assignment = dists.argmin(dim=-1)
            new_centers = centers.clone()
            for component in range(k):
                mask = assignment == component
                if mask.any():
                    new_centers[component] = flat[mask].mean(dim=0)
            centers = new_centers

        final_linear = self.head.net[-1]
        if not isinstance(final_linear, nn.Linear):
            raise TypeError("Expected self.head's final layer to be an nn.Linear.")
        mu_slice = slice(k, k + k * d)
        final_linear.bias[mu_slice] = centers.reshape(-1)
        final_linear.weight[mu_slice].mul_(0.01)
