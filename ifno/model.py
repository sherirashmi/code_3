"""iFNO: Invertible Fourier Neural Operator for this project's config<->ERP
problem, adapted from Long, Xu, Yuan, Yang & Zhe, "Invertible Fourier Neural
Operators for Tackling Both Forward and Inverse Problems" (arXiv:2402.11722).

What the paper does, in its own setting: both the input function ``f`` and
output function ``u`` are sampled on the SAME spatial grid (e.g. a 64x64
Darcy-flow permeability map and its pressure field). It lifts samples of
each to a ``2d``-channel latent via a shared per-point MLP, stacks K
"invertible Fourier blocks" (a RealNVP-style multiplicative coupling built
from the standard FNO Fourier layer, softplus-gated so every block is an
exact, closed-form bijection -- eq 2/3 of the paper), and reads the forward
prediction off one projection MLP (Q) or the inverse prediction off another
(Q'), running the SAME blocks/weights backward for the inverse direction. A
beta-VAE is layered on top of the projected-back input representation to
give the inverse direction a posterior to sample from (uncertainty, not just
a point estimate).

Why this can't be a literal transplant here, and what changes: this
project's "input function" (a resonator configuration: a small, unordered
set of up to num_res picks of [m,k,f_t,x,y]) and "output function" (the ERP
spectrum, a genuine function of frequency, 301 points) do NOT live on the
same grid -- so there is no single shared discretization to lift both onto,
which the paper's construction assumes. The adaptation:

  - The ONE shared, ordered grid in this problem is the 301-point frequency
    axis (fixed, deterministic -- utils.physics.freqs), so that is what
    plays the paper's "spatial grid" role, and the invertible Fourier
    blocks' spectral convolution (SpectralConv1d, same construction as
    erp_forward_operators/fno.py and displacement_forward_operators/fno.py)
    mixes along frequency, exactly as FNO's Fourier layer requires an
    ordered axis to be meaningful (see those modules' own docstrings for
    why a scattered axis would break this).
  - Forward direction (P, lift): the configuration is NOT itself a function
    of frequency, so its lift is built by broadcasting a permutation-
    invariant ResonatorSetEncoder summary of the whole resonator set to
    every one of the 301 points, concatenated with a genuine per-point
    resonance-detuning feature (ResonanceQueryEncoder) and the frequency
    value itself -- i.e. exactly the conditioning signal
    erp_forward_operators/fno.py already uses for its own (non-invertible) FNO,
    just lifted through P into the 2d-channel latent this architecture's
    coupling blocks require instead of FNO's plain width-d channels.
  - Inverse direction (P', lift): given a target spectrum, each of the 301
    (ERP value, frequency) pairs is lifted independently (no configuration
    known yet -- that is what we are solving for), producing per-point
    latents which the SAME blocks (run in reverse via their closed-form
    inverse, eq 3) turn back into 301 per-point estimates of what the
    forward-direction lift would have produced.
  - Design read-out (Q'): unlike the paper's setting, where input and
    output live on the same grid so the inverse prediction is naturally
    itself a 301-point field, this problem's design is NOT indexed by
    frequency -- it is one fixed configuration, not a curve. So instead of
    projecting each of the 301 recovered per-point latents to a design
    value independently, they are first mean+max pooled over frequency
    (the same permutation/aggregation-invariant idea ResonatorSetEncoder
    already uses elsewhere in this project) into one vector, and Q'
    projects THAT to the flat design_dim design vector. This is the one
    place this port deviates from a literal equation-for-equation copy of
    the paper, and it is a necessary consequence of the domain mismatch,
    not a simplification of the coupling-block mechanism itself (which is
    implemented exactly as eq 2/3 describe, softplus gates and all).
  - beta-VAE (Sec 3.2): identical role and construction to the paper,
    specialized to this project's design space -- encodes the inverse
    pipeline's point-estimate design into a small Gaussian latent z and
    decodes back, giving `sample()` a genuine posterior to draw from
    (matching every other model in inverse_operators/'s
    ``sample(spectrum, num_samples) -> (B, num_samples, design_dim)``
    convention) instead of always returning the same point estimate.

Three-step training (Sec 3.3) is implemented in ifno/train.py, not here --
this module only defines the architecture and the loss TERMS each stage
needs (stage1_loss/stage2_loss/stage3_loss), since which parameters are
optimized in which stage is a training-schedule decision, not a model one.
"""

from __future__ import annotations

import math
from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from erp_forward_operators.neural_operator_utils import (
    MLP,
    ResonanceQueryEncoder,
    ResonatorSetEncoder,
    resolve_activation,
)
from utils.physics import freqs as _frequency_grid_hz


# ==================================================
# Invertible Fourier blocks (paper eq 1-3)
# ==================================================


class SpectralConv1d(nn.Module):
    """Learned convolution on retained Fourier modes -- same construction as
    erp_forward_operators/fno.py and displacement_forward_operators/fno.py's own
    SpectralConv1d, redefined locally here per this repo's existing
    convention of keeping each operator module self-contained.
    """

    def __init__(self, in_channels: int, out_channels: int, modes: int) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.modes = int(modes)
        scale = 1.0 / max(1, in_channels * out_channels)
        weight = scale * torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat)
        self.weight = nn.Parameter(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[-1]
        x_ft = torch.fft.rfft(x, dim=-1)
        n_modes = min(self.modes, x_ft.shape[-1])
        out_ft = torch.zeros(x.shape[0], self.out_channels, x_ft.shape[-1], device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :n_modes] = torch.einsum("bim,iom->bom", x_ft[:, :, :n_modes], self.weight[:, :, :n_modes])
        return torch.fft.irfft(out_ft, n=n, dim=-1)


class FourierLayer1d(nn.Module):
    """The paper's "standard FNO Fourier layer" L: a location-wise linear
    transform W (a kernel-3 conv, playing the same role as erp_forward_operators'
    own ``local`` conv) plus the spectral integral, then a nonlinear
    activation -- eq 1 of the paper, ``sigma(W*v + integral(kappa*v'))``.
    Deliberately has NO residual/skip connection of its own (unlike this
    repo's other FNOBlock1d, which adds ``x +``): L itself is not required
    to be invertible here, only the coupling structure built from it in
    InvertibleFourierBlock below is.
    """

    def __init__(self, width: int, modes: int, activation: str | type[nn.Module] = "gelu") -> None:
        super().__init__()
        self.spectral = SpectralConv1d(width, width, modes)
        self.local = nn.Conv1d(width, width, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(1, width)
        self.activation = resolve_activation(activation)()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.norm(self.local(x) + self.spectral(x)))


class InvertibleFourierBlock(nn.Module):
    """One invertible Fourier block (paper eq 2/3): a pair of d-channel
    latents (v1, v2), each (B, d, F), updated by

        v1_next = v1 * S(L(v2))
        v2_next = v2 * S(L(v1_next))

    where L is one shared FourierLayer1d (the paper uses one Fourier layer
    symbol per block, applied to each half in turn) and S is the elementwise
    softplus transform S(x) = tau^-1 * log(1+exp(tau*x)) -- ``F.softplus``
    with ``beta=tau`` is exactly this. Since S(x) > 0 everywhere, both
    multiplicative updates are exactly invertible by elementwise division,
    eq 3:

        v2 = v2_next / S(L(v1_next))       (v1_next is already known)
        v1 = v1_next / S(L(v2))            (using the just-recovered v2)

    A small epsilon floor on the gate guards against division blowing up
    when a freshly-initialized L output is very negative (S(x) -> 0); the
    paper does not need this in float64 tabletop experiments but it matters
    for stable float32 training here.
    """

    def __init__(self, width: int, modes: int, tau: float = 1.0, activation: str | type[nn.Module] = "gelu") -> None:
        super().__init__()
        self.gate_net = FourierLayer1d(width, modes, activation=activation)
        self.tau = float(tau)
        self.eps = 1e-6

    def _gate(self, x: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.gate_net(x), beta=self.tau) + self.eps

    def forward(self, v1: torch.Tensor, v2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        v1_next = v1 * self._gate(v2)
        v2_next = v2 * self._gate(v1_next)
        return v1_next, v2_next

    def inverse(self, v1_next: torch.Tensor, v2_next: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        v2 = v2_next / self._gate(v1_next)
        v1 = v1_next / self._gate(v2)
        return v1, v2


class InvertibleFourierStack(nn.Module):
    """K stacked InvertibleFourierBlocks -- forward runs blocks 1..K in
    order, inverse runs the SAME blocks (same weights) K..1 using each
    block's own closed-form inverse. This is what makes "one architecture,
    both directions" literal rather than a loose description.
    """

    def __init__(self, width: int, num_blocks: int, modes: int, tau: float = 1.0, activation: str | type[nn.Module] = "gelu") -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [InvertibleFourierBlock(width, modes, tau=tau, activation=activation) for _ in range(num_blocks)]
        )

    def forward(self, v1: torch.Tensor, v2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        for block in self.blocks:
            v1, v2 = block(v1, v2)
        return v1, v2

    def inverse(self, v1: torch.Tensor, v2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        for block in reversed(self.blocks):
            v1, v2 = block.inverse(v1, v2)
        return v1, v2


# ==================================================
# beta-VAE (paper Sec 3.2): posterior over designs
# ==================================================


class DesignVAE(nn.Module):
    """Encodes a design_dim point (either a TRUE design during stage-2
    pretraining, or the inverse pipeline's point-estimate design during
    stage-3/inference -- same weights, different input source, matching the
    paper's own eq 10 vs eq 6/11) to a small Gaussian latent z and back.
    """

    def __init__(self, design_dim: int, z_dim: int = 6, hidden: int = 64) -> None:
        super().__init__()
        self.design_dim = int(design_dim)
        self.z_dim = int(z_dim)
        self.encoder = MLP([design_dim, hidden, hidden], activation=nn.SiLU)
        self.mu_head = nn.Linear(hidden, z_dim)
        self.log_var_head = nn.Linear(hidden, z_dim)
        self.decoder = MLP([z_dim, hidden, hidden, design_dim], activation=nn.SiLU)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        mu = self.mu_head(h)
        log_var = self.log_var_head(h).clamp(min=-8.0, max=4.0)
        return mu, log_var

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    @staticmethod
    def reparameterize(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        return mu + std * torch.randn_like(std)


def _kl_to_standard_normal(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    return (-0.5 * (1.0 + log_var - mu.pow(2) - log_var.exp())).sum(dim=-1)


# ==================================================
# Full model
# ==================================================


class IFNO(nn.Module):
    def __init__(
        self,
        design_dim: int,
        n_freq: int = 301,
        width: int = 24,
        num_blocks: int = 4,
        modes: int = 48,
        config_hidden: int = 80,
        query_dim: int = 32,
        z_dim: int = 6,
        vae_hidden: int = 64,
        tau: float = 1.0,
        activation: str | type[nn.Module] = "gelu",
    ) -> None:
        super().__init__()
        if design_dim % 5 != 0:
            raise ValueError("design_dim must be num_res*5 ([m,k,f_t,x,y] per resonator).")
        self.design_dim = int(design_dim)
        self.num_res = self.design_dim // 5
        self.n_freq = int(n_freq)
        self.width = int(width)  # d: per-half channel width (each of v1, v2)
        activation_cls = resolve_activation(activation)

        # The one shared, ordered grid (see module docstring): fixed,
        # deterministic, independent of any specific generated dataset, so
        # it can be registered here at construction time without needing a
        # live dataset/norm_params -- keeping IFNO's constructor a plain
        # zero-arg-friendly call like every other inverse_operators model.
        frequency_hz = torch.from_numpy(_frequency_grid_hz.astype("float32"))
        if frequency_hz.numel() != self.n_freq:
            raise ValueError(f"utils.physics.freqs has {frequency_hz.numel()} points, expected n_freq={self.n_freq}.")
        freq_mean = float(frequency_hz.mean())
        freq_std = max(float(frequency_hz.std()), 1e-8)
        self.register_buffer("frequency_hz", frequency_hz)
        self.register_buffer("frequency_norm", (frequency_hz - freq_mean) / freq_std)

        # ---- Forward-direction lift (P) ----
        self.configuration_encoder = ResonatorSetEncoder(
            hidden_dim=config_hidden, element_dim=config_hidden, output_dim=2 * self.width
        )
        self.resonance_query = ResonanceQueryEncoder(hidden_dim=query_dim, element_dim=query_dim, output_dim=query_dim)
        self.lift_p = nn.Linear(2 * self.width + query_dim + 1, 2 * self.width)

        # ---- Inverse-direction lift (P'): no configuration known yet ----
        self.freq_embed = MLP([1, query_dim, query_dim], activation=nn.SiLU)
        self.lift_pp = nn.Linear(1 + query_dim, 2 * self.width)

        # ---- Shared invertible Fourier blocks ----
        self.blocks = InvertibleFourierStack(self.width, num_blocks, modes, tau=tau, activation=activation_cls)

        # ---- Output projections ----
        self.project_q = nn.Sequential(
            nn.Linear(2 * self.width, 2 * self.width), activation_cls(), nn.Linear(2 * self.width, 1)
        )  # Q: per-point (v1,v2) -> ERP scalar
        self.project_qp = MLP(
            [4 * self.width, vae_hidden, self.design_dim], activation=nn.SiLU
        )  # Q': pooled (v1,v2) over frequency -> flat design

        # ---- beta-VAE over the design space (Sec 3.2) ----
        self.vae = DesignVAE(self.design_dim, z_dim=z_dim, hidden=vae_hidden)

    # ---------------------------------------------------------
    # Frequency helper
    # ---------------------------------------------------------

    def _frequency_batch(self, batch_size: int) -> torch.Tensor:
        """(B, n_freq, 1) normalized frequency, shared/broadcast across the batch."""
        return self.frequency_norm[None, :, None].expand(batch_size, -1, -1)

    # ---------------------------------------------------------
    # Forward direction: configuration -> ERP spectrum
    # ---------------------------------------------------------

    def _lift_forward(self, configuration: torch.Tensor, frequency: torch.Tensor) -> torch.Tensor:
        """configuration (B,num_res,5), frequency (B,F,1) -> (B, 2*width, F)."""
        context = self.configuration_encoder(configuration)[:, None, :].expand(-1, frequency.shape[1], -1)
        query = self.resonance_query(configuration, frequency)
        lifted = self.lift_p(torch.cat((context, query, frequency), dim=-1))  # (B, F, 2*width)
        return lifted.transpose(1, 2)  # (B, 2*width, F)

    def _through_blocks_to_spectrum(self, v0: torch.Tensor) -> torch.Tensor:
        """Already-lifted (B, 2*width, F) -> blocks.forward -> Q -> ERP (B, F, 1)."""
        v1_0, v2_0 = v0.chunk(2, dim=1)
        v1_k, v2_k = self.blocks(v1_0, v2_0)
        combined = torch.cat((v1_k, v2_k), dim=1).transpose(1, 2)  # (B, F, 2*width)
        return self.project_q(combined)

    def predict_spectrum(self, configuration: torch.Tensor, frequency: torch.Tensor | None = None) -> torch.Tensor:
        """Forward-operator use: configuration -> predicted ERP (B, F, 1),
        directly comparable to erp_forward_operators' own model outputs.
        """
        if frequency is None:
            frequency = self._frequency_batch(configuration.shape[0])
        v0 = self._lift_forward(configuration, frequency)
        return self._through_blocks_to_spectrum(v0)

    def direct_design_reconstruction(self, configuration: torch.Tensor, frequency: torch.Tensor | None = None) -> torch.Tensor:
        """P then immediately Q' (skipping the invertible blocks entirely) --
        the round-trip consistency term J_{P,Q'} from the paper.
        """
        if frequency is None:
            frequency = self._frequency_batch(configuration.shape[0])
        v0 = self._lift_forward(configuration, frequency)  # (B, 2*width, F)
        return self.project_qp(self._pool(v0))

    # ---------------------------------------------------------
    # Inverse direction: ERP spectrum -> configuration
    # ---------------------------------------------------------

    def _lift_inverse(self, spectrum: torch.Tensor, frequency: torch.Tensor) -> torch.Tensor:
        """spectrum (B,F) normalized ERP, frequency (B,F,1) -> (B, 2*width, F)."""
        freq_feat = self.freq_embed(frequency)  # (B, F, query_dim)
        lifted = self.lift_pp(torch.cat((spectrum[..., None], freq_feat), dim=-1))  # (B, F, 2*width)
        return lifted.transpose(1, 2)

    @staticmethod
    def _pool(latent: torch.Tensor) -> torch.Tensor:
        """(B, 2*width, F) -> (B, 4*width): mean+max pool over frequency.

        Design is not itself indexed by frequency (see module docstring),
        so the per-point latents must be aggregated before Q' can predict
        one design -- the same mean+max, permutation/aggregation-invariant
        idea ResonatorSetEncoder already uses across this project.
        """
        return torch.cat((latent.mean(dim=-1), latent.amax(dim=-1)), dim=-1)

    def direct_spectrum_reconstruction(self, spectrum: torch.Tensor, frequency: torch.Tensor | None = None) -> torch.Tensor:
        """P' then immediately Q (skipping the invertible blocks) -- the
        round-trip consistency term J_{P',Q} from the paper.
        """
        if frequency is None:
            frequency = self._frequency_batch(spectrum.shape[0])
        u0 = self._lift_inverse(spectrum, frequency)  # (B, 2*width, F)
        return self.project_q(u0.transpose(1, 2))

    def _through_blocks_to_design(self, u0: torch.Tensor) -> torch.Tensor:
        """Already-lifted (B, 2*width, F) -> blocks.inverse -> pool -> Q' -> flat design."""
        v1_k, v2_k = u0.chunk(2, dim=1)
        v1_0, v2_0 = self.blocks.inverse(v1_k, v2_k)
        pooled = self._pool(torch.cat((v1_0, v2_0), dim=1))
        return self.project_qp(pooled)

    def infer_point_estimate(self, spectrum: torch.Tensor, frequency: torch.Tensor | None = None) -> torch.Tensor:
        """Full inverse pipeline WITHOUT the VAE: P' -> blocks.inverse -> pool -> Q'.

        This is psi_iFNO^-1(u) from eq 5 of the paper (pre-VAE inverse
        prediction) -- used directly as J_INV's target in stage 1, and as
        the VAE encoder's input in stage 3 / at inference time.
        """
        if frequency is None:
            frequency = self._frequency_batch(spectrum.shape[0])
        u0 = self._lift_inverse(spectrum, frequency)
        return self._through_blocks_to_design(u0)

    @torch.no_grad()
    def sample(self, spectrum: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """Posterior design samples for a target spectrum: point_estimate ->
        VAE encode -> sample z ~ q(z|point_estimate) -> decode.

        Returns a plain ``(B, num_samples, design_dim)`` tensor (no
        tractable log p(design|spectrum), same convention as
        inverse_operators/diffusion.py and padding_inn.py).
        """
        b = spectrum.shape[0]
        point_estimate = self.infer_point_estimate(spectrum)
        mu, log_var = self.vae.encode(point_estimate)
        std = torch.exp(0.5 * log_var)
        mu_e = mu[:, None, :].expand(-1, num_samples, -1).reshape(b * num_samples, -1)
        std_e = std[:, None, :].expand(-1, num_samples, -1).reshape(b * num_samples, -1)
        z = mu_e + std_e * torch.randn_like(std_e)
        designs = self.vae.decode(z)
        return designs.view(b, num_samples, self.design_dim)

    # ---------------------------------------------------------
    # Stage losses (see ifno/train.py for the 3-step schedule that uses them)
    # ---------------------------------------------------------

    def stage1_loss(self, spectrum: torch.Tensor, flat_design: torch.Tensor) -> dict[str, torch.Tensor]:
        """J_IFB = J_FWD + J_INV + J_{P,Q'} + J_{P',Q} (paper eq 7), MSE in
        place of the paper's relative-L2 (matching this package's other
        models' convention, e.g. BasisFlow/PadINN's plain MSE terms).

        Each of P's and P''s lift is computed ONCE and reused for both its
        through-the-blocks path and its direct (blocks-skipped) path --
        ``predict_spectrum``/``direct_design_reconstruction`` and
        ``infer_point_estimate``/``direct_spectrum_reconstruction`` would
        otherwise each recompute the same ResonatorSetEncoder/
        ResonanceQueryEncoder lift from scratch, doubling this loss's cost
        for no reason (this project trains on CPU; that redundancy was the
        difference between a feasible and infeasible epoch time on the
        100k-configuration dataset).
        """
        configuration = flat_design.view(-1, self.num_res, 5)
        frequency = self._frequency_batch(spectrum.shape[0])

        v0 = self._lift_forward(configuration, frequency)
        predicted_spectrum = self._through_blocks_to_spectrum(v0).squeeze(-1)
        j_fwd = F.mse_loss(predicted_spectrum, spectrum)
        direct_design = self.project_qp(self._pool(v0))
        j_pq_prime = F.mse_loss(direct_design, flat_design)

        u0 = self._lift_inverse(spectrum, frequency)
        point_estimate = self._through_blocks_to_design(u0)
        j_inv = F.mse_loss(point_estimate, flat_design)
        direct_spectrum = self.project_q(u0.transpose(1, 2)).squeeze(-1)
        j_pprime_q = F.mse_loss(direct_spectrum, spectrum)

        total = j_fwd + j_inv + j_pq_prime + j_pprime_q
        return {"total": total, "j_fwd": j_fwd, "j_inv": j_inv, "j_pq_prime": j_pq_prime, "j_pprime_q": j_pprime_q}

    def stage2_loss(self, flat_design: torch.Tensor, beta: float = 0.1) -> dict[str, torch.Tensor]:
        """J_beta-VAE (paper eq 10), pretraining the VAE alone on TRUE
        designs -- no spectrum, no invertible blocks involved.
        """
        mu, log_var = self.vae.encode(flat_design)
        z = self.vae.reparameterize(mu, log_var)
        recon = self.vae.decode(z)
        recon_loss = F.mse_loss(recon, flat_design)
        kl = _kl_to_standard_normal(mu, log_var).mean()
        total = recon_loss + beta * kl
        return {"total": total, "recon": recon_loss, "kl": kl}

    def stage3_loss(self, spectrum: torch.Tensor, flat_design: torch.Tensor, beta: float = 0.1) -> dict[str, torch.Tensor]:
        """J = J_FWD + J_{P,Q'} + J_{P',Q} + J_beta-VAE (paper eq 11), with
        the VAE now encoding the INVERSE PIPELINE's point estimate (not the
        true design directly, unlike stage 2) and decoding back toward the
        true design -- this is what actually ties inverse recovery to the
        invertible blocks' output instead of to an oracle.
        """
        configuration = flat_design.view(-1, self.num_res, 5)
        frequency = self._frequency_batch(spectrum.shape[0])

        # Same one-lift-reused-twice structure as stage1_loss (see its
        # docstring) -- halves the redundant ResonatorSetEncoder/
        # ResonanceQueryEncoder/P' work per batch.
        v0 = self._lift_forward(configuration, frequency)
        predicted_spectrum = self._through_blocks_to_spectrum(v0).squeeze(-1)
        j_fwd = F.mse_loss(predicted_spectrum, spectrum)
        direct_design = self.project_qp(self._pool(v0))
        j_pq_prime = F.mse_loss(direct_design, flat_design)

        u0 = self._lift_inverse(spectrum, frequency)
        point_estimate = self._through_blocks_to_design(u0)
        direct_spectrum = self.project_q(u0.transpose(1, 2)).squeeze(-1)
        j_pprime_q = F.mse_loss(direct_spectrum, spectrum)

        mu, log_var = self.vae.encode(point_estimate)
        z = self.vae.reparameterize(mu, log_var)
        recon = self.vae.decode(z)
        recon_loss = F.mse_loss(recon, flat_design)
        kl = _kl_to_standard_normal(mu, log_var).mean()
        j_beta_vae = recon_loss + beta * kl

        total = j_fwd + j_pq_prime + j_pprime_q + j_beta_vae
        return {
            "total": total, "j_fwd": j_fwd, "j_pq_prime": j_pq_prime, "j_pprime_q": j_pprime_q,
            "j_beta_vae": j_beta_vae, "recon": recon_loss, "kl": kl,
        }


def build_model(design_dim: int, **kwargs) -> IFNO:
    return IFNO(design_dim=design_dim, **kwargs)


DEFAULT_MODEL_CONFIG: Mapping[str, object] = {
    "n_freq": 301,
    "width": 24,
    "num_blocks": 4,
    "modes": 48,
    "config_hidden": 80,
    "query_dim": 32,
    "z_dim": 6,
    "vae_hidden": 64,
    "tau": 1.0,
    "activation": "gelu",
}
