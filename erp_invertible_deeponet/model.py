"""Invertible DeepONet: branch = RealNVP bijection, trunk = basis functions of frequency.

Forward (Kaltenbach et al.):

    ERP(f; a) = sum_q b_q(a) psi_q(f) + psi_0(f),      b = T(a)  (T invertible)

* ``a``: the design, D = 4 * num_res numbers ([m, f_t, x, y] per resonator in
  the bounded logit coordinates of erp_inverse_operators/design_space.py, so
  every value maps back inside its physical range; k = m (2 pi f_t)^2).
* ``T``: a RealNVP (affine couplings) on R^Q. A bijection needs equal input and
  output size, so the paper has Q = D. With ``pad > 0`` the design is padded
  with ``pad`` latent numbers z, T acts on [a, z] and Q = D + pad basis
  functions are available (z = 0 in the forward direction). ``pad = 0`` is the
  strict Q = D model.
* ``psi``: the trunk, an MLP of the (normalised) frequency with Fourier
  features, evaluated on the fixed frequency grid -> an (F, Q) basis matrix.

Inverse: the ERP is LINEAR in b, so for a target spectrum y the coefficients
have the closed-form regularised least-squares estimate

    b* = (Psi^T Psi + lam I)^-1 Psi^T (y - psi_0),  Cov = s2 (Psi^T Psi + lam I)^-1

(Gaussian posterior with a flat prior on b and noise variance s2 = the
training residual). Samples b ~ N(b*, Cov) are pushed back through T^-1, and
the design part of [a, z] is kept: several designs per target, no iterative
optimisation, milliseconds. F (frequency points) can exceed Q; only Q = dim
of the bijection is tied to D.

Training: forward MSE on the normalised ERP plus an inverse-consistency term,
||T^-1(b*(y))_a - a||^2 + w_z ||T^-1(b*(y))_z||^2, so the closed-form inverse
of a training spectrum lands on its own design (and on z ~ 0).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

LOG_SCALE_CLAMP = 2.0


class AffineCoupling(nn.Module):
    """y_b = x_b * exp(s(x_a)) + t(x_a) on the masked-out half; exactly invertible."""

    def __init__(self, dim: int, mask: torch.Tensor, hidden: int, depth: int = 2) -> None:
        super().__init__()
        self.register_buffer("mask", mask)
        layers, width = [], dim
        for _ in range(depth):
            layers += [nn.Linear(width, hidden), nn.SiLU()]
            width = hidden
        layers.append(nn.Linear(width, 2 * dim))
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)  # start as the identity map
        nn.init.zeros_(self.net[-1].bias)

    def _st(self, x_masked):
        log_s, t = self.net(x_masked).chunk(2, dim=-1)
        return torch.tanh(log_s) * LOG_SCALE_CLAMP, t

    def forward(self, x):
        xm = x * self.mask
        log_s, t = self._st(xm)
        inv = 1.0 - self.mask
        return xm + inv * (x * torch.exp(log_s) + t)

    def inverse(self, y):
        ym = y * self.mask
        log_s, t = self._st(ym)
        inv = 1.0 - self.mask
        return ym + inv * ((y - t) * torch.exp(-log_s))


class RealNVP(nn.Module):
    """Bijection R^Q -> R^Q from affine couplings with alternating / random masks."""

    def __init__(self, dim: int, num_layers: int = 10, hidden: int = 256, seed: int = 0) -> None:
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        layers = []
        for i in range(num_layers):
            if dim <= 16:
                mask = torch.zeros(dim)
                mask[i % 2::2] = 1.0
            else:  # random halves mix all dimensions faster than even/odd
                mask = torch.zeros(dim)
                mask[torch.randperm(dim, generator=gen)[: dim // 2]] = 1.0
            layers.append(AffineCoupling(dim, mask, hidden))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def inverse(self, y):
        for layer in reversed(self.layers):
            y = layer.inverse(y)
        return y


class Trunk(nn.Module):
    """Basis functions psi_0..psi_Q of the normalised frequency (Fourier features + MLP)."""

    def __init__(self, num_basis: int, hidden: int = 256, depth: int = 4, num_fourier: int = 32) -> None:
        super().__init__()
        self.register_buffer("freqs", torch.arange(1, num_fourier + 1, dtype=torch.float32) * math.pi)
        layers, width = [], 1 + 2 * num_fourier
        for _ in range(depth):
            layers += [nn.Linear(width, hidden), nn.SiLU()]
            width = hidden
        layers.append(nn.Linear(width, num_basis + 1))
        self.net = nn.Sequential(*layers)

    def forward(self, f_norm: torch.Tensor) -> torch.Tensor:
        """f_norm: (F,) in [-1, 1] -> (F, Q + 1); column 0 is the bias function psi_0."""
        arg = f_norm[:, None] * self.freqs[None, :]
        return self.net(torch.cat((f_norm[:, None], torch.sin(arg), torch.cos(arg)), dim=-1))


class InvertibleDeepONet(nn.Module):
    def __init__(self, design_dim: int, n_freq: int, pad: int = 0, num_layers: int = 10, hidden: int = 256,
                 trunk_hidden: int = 256, ridge: float = 1e-4) -> None:
        super().__init__()
        self.design_dim, self.pad, self.n_freq = int(design_dim), int(pad), int(n_freq)
        self.num_basis = self.design_dim + self.pad  # Q
        self.ridge = float(ridge)
        self.branch = RealNVP(self.num_basis, num_layers=num_layers, hidden=hidden)
        self.trunk = Trunk(self.num_basis, hidden=trunk_hidden)
        self.register_buffer("f_norm", torch.linspace(-1.0, 1.0, self.n_freq))
        self.register_buffer("noise_var", torch.tensor(1.0))  # set after training (training residual)

    # ---- pieces -----------------------------------------------------------------
    def basis(self):
        """(Psi (F, Q), psi_0 (F,))."""
        out = self.trunk(self.f_norm)
        return out[:, 1:], out[:, 0]

    def _pad(self, a):
        if not self.pad:
            return a
        return torch.cat((a, a.new_zeros(*a.shape[:-1], self.pad)), dim=-1)

    def coefficients(self, a):
        return self.branch(self._pad(a))

    # ---- forward ------------------------------------------------------------------
    def forward(self, a):
        """Normalised design (B, D) -> normalised ERP (B, F)."""
        psi, psi0 = self.basis()
        return self.coefficients(a) @ psi.T + psi0

    # ---- inverse ------------------------------------------------------------------
    def _gram(self, psi):
        return psi.T @ psi + self.ridge * torch.eye(self.num_basis, device=psi.device, dtype=psi.dtype)

    def least_squares(self, spectrum, psi=None, psi0=None):
        """Closed-form b* (B, Q) for spectra (B, F): the ERP is linear in b."""
        if psi is None:
            psi, psi0 = self.basis()
        return torch.linalg.solve(self._gram(psi), ((spectrum - psi0) @ psi).T).T

    def coefficient_posterior(self, spectrum):
        """b* (B, Q) and the Cholesky factor (Q, Q) of Cov = s2 (Psi^T Psi + lam I)^-1."""
        psi, psi0 = self.basis()
        b_star = self.least_squares(spectrum, psi, psi0)
        chol = torch.linalg.cholesky(self.noise_var * torch.linalg.inv(self._gram(psi)))
        return b_star, chol

    def invert_coefficients(self, b):
        """b (..., Q) -> (design (..., D), latent z (..., pad))."""
        x = self.branch.inverse(b)
        return x[..., : self.design_dim], x[..., self.design_dim:]

    def point_estimate(self, spectrum):
        return self.invert_coefficients(self.least_squares(spectrum))[0]

    @torch.no_grad()
    def sample(self, spectrum, num_samples: int = 16):
        """(B, S, D) normalised designs; sample 0 is the posterior-mean (point) estimate."""
        b_star, chol = self.coefficient_posterior(spectrum)
        eps = torch.randn(b_star.shape[0], num_samples, self.num_basis, device=b_star.device)
        eps[:, 0] = 0.0
        b = b_star[:, None, :] + eps @ chol.T
        return self.invert_coefficients(b)[0]

    # ---- training -------------------------------------------------------------------
    def training_loss(self, spectrum, design, inverse_weight: float = 1.0, latent_weight: float = 0.1):
        a = design.reshape(design.shape[0], -1)
        psi, psi0 = self.basis()
        pred = self.coefficients(a) @ psi.T + psi0
        forward_loss = ((pred - spectrum) ** 2).mean()
        a_hat, z_hat = self.invert_coefficients(self.least_squares(spectrum, psi, psi0))
        inverse_loss = ((a_hat - a) ** 2).mean()
        latent_loss = (z_hat ** 2).mean() if self.pad else torch.zeros((), device=a.device)
        total = forward_loss + inverse_weight * inverse_loss + latent_weight * latent_loss
        return {"total": total, "forward": forward_loss, "inverse": inverse_loss, "latent": latent_loss}
