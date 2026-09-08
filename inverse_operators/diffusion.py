"""Conditional diffusion model: p(design | spectrum) via iterative denoising.

The current strongest general-purpose approach for probabilistic conditional
generation. For a design vector this small (15 numbers -- 3 resonators x
[m,k,f_t,x,y]) diffusion is cheap to train and sample despite its reputation
for being expensive; that reputation comes from images, not small structured
vectors. A small MLP denoiser predicts the noise added to a design vector at
a random timestep, conditioned on both the timestep and the target spectrum
embedding (standard DDPM, Ho et al. 2020). Sampling runs the reverse
diffusion chain from pure noise down to a design sample, conditioned on the
target spectrum throughout -- naturally probabilistic (every run of the
chain from a different noise seed gives a different, independently valid
candidate design) and typically the sharpest/most diverse of these models at
representing genuinely distinct alternative solutions.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from utils.neural_operator_utils import MLP

from .common import SpectrumEncoder, flatten_configuration


class SinusoidalTimestepEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t.float()[:, None] * freqs[None, :]
        return torch.cat((torch.sin(args), torch.cos(args)), dim=-1)


class Denoiser(nn.Module):
    def __init__(self, design_dim: int, embed_dim: int, time_dim: int, hidden: int) -> None:
        super().__init__()
        self.time_embed = SinusoidalTimestepEmbedding(time_dim)
        self.net = MLP(
            [design_dim + embed_dim + time_dim, hidden, hidden, hidden, design_dim],
            activation=nn.SiLU,
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        t_embed = self.time_embed(t)
        return self.net(torch.cat((x_t, cond, t_embed), dim=-1))


class ConditionalDiffusion(nn.Module):
    def __init__(
        self,
        design_dim: int,
        num_steps: int = 100,
        embed_dim: int = 96,
        time_dim: int = 32,
        hidden: int = 128,
        cosine_s: float = 0.008,
    ) -> None:
        super().__init__()
        self.design_dim = int(design_dim)
        self.num_steps = int(num_steps)

        self.encoder = SpectrumEncoder(embed_dim=embed_dim)
        self.denoiser = Denoiser(design_dim, embed_dim, time_dim, hidden)

        # Cosine schedule (Nichol & Dhariwal, "Improved DDPM") instead of the
        # standard linear one. A linear schedule spends most of its steps in
        # the high-noise regime, where predicting the injected noise from an
        # almost-pure-noise x_t is close to ill-posed by construction -- for
        # a small design vector like this one (15 dims), that dominates the
        # averaged training loss and can make real progress at the more
        # useful low-noise steps look like a flat curve. Cosine spends more
        # of the schedule in the low/medium-noise regime instead.
        steps = num_steps + 1
        t = torch.linspace(0, num_steps, steps) / num_steps
        f_t = torch.cos((t + cosine_s) / (1 + cosine_s) * math.pi * 0.5) ** 2
        alpha_bars_full = f_t / f_t[0]
        betas = (1.0 - alpha_bars_full[1:] / alpha_bars_full[:-1]).clamp(max=0.999)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)

    def training_loss(self, spectrum: torch.Tensor, design: torch.Tensor) -> torch.Tensor:
        flat = flatten_configuration(design)
        b = flat.shape[0]
        cond = self.encoder(spectrum)
        t = torch.randint(0, self.num_steps, (b,), device=flat.device)
        alpha_bar_t = self.alpha_bars[t][:, None]
        noise = torch.randn_like(flat)
        x_t = torch.sqrt(alpha_bar_t) * flat + torch.sqrt(1 - alpha_bar_t) * noise
        noise_pred = self.denoiser(x_t, t, cond)
        return nn.functional.mse_loss(noise_pred, noise)

    @torch.no_grad()
    def sample(self, spectrum: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """Returns ``(B, num_samples, design_dim)`` flat normalized design samples."""
        b = spectrum.shape[0]
        cond = self.encoder(spectrum)
        cond = cond[:, None, :].expand(-1, num_samples, -1).reshape(b * num_samples, -1)
        n = b * num_samples

        x = torch.randn(n, self.design_dim, device=spectrum.device)
        for step in reversed(range(self.num_steps)):
            t = torch.full((n,), step, device=spectrum.device, dtype=torch.long)
            noise_pred = self.denoiser(x, t, cond)
            alpha_t = self.alphas[step]
            alpha_bar_t = self.alpha_bars[step]
            beta_t = self.betas[step]
            mean = (1.0 / torch.sqrt(alpha_t)) * (
                x - (beta_t / torch.sqrt(1 - alpha_bar_t)) * noise_pred
            )
            if step > 0:
                x = mean + torch.sqrt(beta_t) * torch.randn_like(x)
            else:
                x = mean
        return x.view(b, num_samples, self.design_dim)
