"""Shared, architecture-agnostic pieces of the invertible-operator family
(iFNO/iDCO/iGNO): the beta-VAE posterior over designs (paper Sec 3.2,
identical across all three) and ``InvertibleOperatorBase``, which factors
out every part of the original iFNO port that never actually depended on
FNO -- the forward/inverse pipeline plumbing, pooling, sampling, and the
three-stage loss terms (eq 7/10/11). Subclasses (ifno.py/idco.py/igno.py)
only need to build their own ``self.blocks`` (an
:class:`~erp_invertible_operators.coupling.InvertibleCouplingStack`
with an architecture-specific gate net) and implement ``_lift_forward``/
``_lift_inverse`` -- how a configuration or a raw spectrum gets embedded
into the shared ``(B, 2*width, F)`` latent the coupling blocks operate on.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from erp_forward_operators.neural_operator_utils import MLP


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


def kl_to_standard_normal(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    return (-0.5 * (1.0 + log_var - mu.pow(2) - log_var.exp())).sum(dim=-1)


class InvertibleOperatorBase(nn.Module):
    """Everything about the iFNO-family pipeline that is NOT specific to
    which per-point layer plays the role of the coupling blocks' gate
    function L. A subclass's ``__init__`` must set, after calling
    ``super().__init__()``:

      - ``self.design_dim``, ``self.num_res`` (``design_dim // 5``),
        ``self.n_freq``, and register ``frequency_hz``/``frequency_norm``
        buffers (see :meth:`_register_frequency_grid`).
      - ``self.width``: per-half channel width (each of v1, v2 in the
        coupling blocks operates on ``width`` channels; the shared lifted
        latent has ``2*width`` channels).
      - ``self.blocks``: an
        :class:`~erp_invertible_operators.coupling.InvertibleCouplingStack`.
      - ``self.project_q``: ``(B, F, 2*width) -> (B, F, 1)``, per-point ERP
        readout (paper's Q).
      - ``self.project_qp``: ``(B, 4*width) -> (B, design_dim)``, pooled
        design readout (paper's Q').
      - ``self.vae``: a :class:`DesignVAE`.

    and must implement :meth:`_lift_forward` and :meth:`_lift_inverse`.
    """

    def _register_frequency_grid(self, frequency_hz: torch.Tensor) -> None:
        if frequency_hz.numel() != self.n_freq:
            raise ValueError(f"frequency grid has {frequency_hz.numel()} points, expected n_freq={self.n_freq}.")
        freq_mean = float(frequency_hz.mean())
        freq_std = max(float(frequency_hz.std()), 1e-8)
        self.register_buffer("frequency_hz", frequency_hz)
        self.register_buffer("frequency_norm", (frequency_hz - freq_mean) / freq_std)

    def _frequency_batch(self, batch_size: int) -> torch.Tensor:
        """(B, n_freq, 1) normalized frequency, shared/broadcast across the batch."""
        return self.frequency_norm[None, :, None].expand(batch_size, -1, -1)

    # ---------------------------------------------------------
    # Subclass-specific: configuration/spectrum <-> shared latent
    # ---------------------------------------------------------

    def _lift_forward(self, configuration: torch.Tensor, frequency: torch.Tensor) -> torch.Tensor:
        """configuration (B,num_res,5), frequency (B,F,1) -> (B, 2*width, F)."""
        raise NotImplementedError

    def _lift_inverse(self, spectrum: torch.Tensor, frequency: torch.Tensor) -> torch.Tensor:
        """spectrum (B,F) normalized ERP, frequency (B,F,1) -> (B, 2*width, F)."""
        raise NotImplementedError

    # ---------------------------------------------------------
    # Generic pipeline (architecture-agnostic once lifted)
    # ---------------------------------------------------------

    @staticmethod
    def _pool(latent: torch.Tensor) -> torch.Tensor:
        """(B, 2*width, F) -> (B, 4*width): mean+max pool over frequency.

        Design is not itself indexed by frequency, so the per-point
        latents must be aggregated before Q' can predict one design -- the
        same mean+max, permutation/aggregation-invariant idea
        ResonatorSetEncoder uses elsewhere in this project.
        """
        return torch.cat((latent.mean(dim=-1), latent.amax(dim=-1)), dim=-1)

    def _through_blocks_to_spectrum(self, v0: torch.Tensor) -> torch.Tensor:
        v1_0, v2_0 = v0.chunk(2, dim=1)
        v1_k, v2_k = self.blocks(v1_0, v2_0)
        combined = torch.cat((v1_k, v2_k), dim=1).transpose(1, 2)  # (B, F, 2*width)
        return self.project_q(combined)

    def _through_blocks_to_design(self, u0: torch.Tensor) -> torch.Tensor:
        v1_k, v2_k = u0.chunk(2, dim=1)
        v1_0, v2_0 = self.blocks.inverse(v1_k, v2_k)
        pooled = self._pool(torch.cat((v1_0, v2_0), dim=1))
        return self.project_qp(pooled)

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
        v0 = self._lift_forward(configuration, frequency)
        return self.project_qp(self._pool(v0))

    def direct_spectrum_reconstruction(self, spectrum: torch.Tensor, frequency: torch.Tensor | None = None) -> torch.Tensor:
        """P' then immediately Q (skipping the invertible blocks) -- the
        round-trip consistency term J_{P',Q} from the paper.
        """
        if frequency is None:
            frequency = self._frequency_batch(spectrum.shape[0])
        u0 = self._lift_inverse(spectrum, frequency)
        return self.project_q(u0.transpose(1, 2))

    def infer_point_estimate(self, spectrum: torch.Tensor, frequency: torch.Tensor | None = None) -> torch.Tensor:
        """Full inverse pipeline WITHOUT the VAE: P' -> blocks.inverse -> pool -> Q'.

        This is psi^-1(u) from eq 5 of the paper (pre-VAE inverse
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
        erp_inverse_operators/diffusion.py and padding_inn.py).
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
    # Stage losses (see train.py for the 3-step schedule that uses them)
    # ---------------------------------------------------------

    def stage1_loss(self, spectrum: torch.Tensor, flat_design: torch.Tensor) -> dict[str, torch.Tensor]:
        """J_IFB = J_FWD + J_INV + J_{P,Q'} + J_{P',Q} (paper eq 7), MSE in
        place of the paper's relative-L2 (matching this package's other
        models' convention).

        Each of P's and P''s lift is computed ONCE and reused for both its
        through-the-blocks path and its direct (blocks-skipped) path,
        halving otherwise-redundant lift work per batch.
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
        kl = kl_to_standard_normal(mu, log_var).mean()
        total = recon_loss + beta * kl
        return {"total": total, "recon": recon_loss, "kl": kl}

    def stage3_loss(
        self,
        spectrum: torch.Tensor,
        flat_design: torch.Tensor,
        beta: float = 0.1,
        inverse_weight: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        """J = J_FWD + J_{P,Q'} + J_{P',Q} + J_beta-VAE (paper eq 11), with
        the VAE now encoding the INVERSE PIPELINE's point estimate (not the
        true design directly, unlike stage 2) and decoding back toward the
        true design -- this is what actually ties inverse recovery to the
        invertible blocks' output instead of to an oracle.

        ``inverse_weight`` (default 1.0; 0.0 reproduces the paper's eq 11
        exactly) keeps the direct point-estimate term J_INV from stage 1 in
        the objective. Without it, stage 3 only supervises the inverse
        pipeline THROUGH the VAE, so the point estimate the VAE encodes is
        free to drift away from the true design while the forward terms
        dominate the gradient -- one main reason the inverse direction
        degraded during joint fine-tuning.
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
        direct_spectrum = self.project_q(u0.transpose(1, 2)).squeeze(-1)
        j_pprime_q = F.mse_loss(direct_spectrum, spectrum)

        mu, log_var = self.vae.encode(point_estimate)
        z = self.vae.reparameterize(mu, log_var)
        recon = self.vae.decode(z)
        recon_loss = F.mse_loss(recon, flat_design)
        kl = kl_to_standard_normal(mu, log_var).mean()
        j_beta_vae = recon_loss + beta * kl

        j_inv = F.mse_loss(point_estimate, flat_design)

        total = j_fwd + j_pq_prime + j_pprime_q + j_beta_vae + float(inverse_weight) * j_inv
        return {
            "total": total, "j_fwd": j_fwd, "j_inv": j_inv, "j_pq_prime": j_pq_prime, "j_pprime_q": j_pprime_q,
            "j_beta_vae": j_beta_vae, "recon": recon_loss, "kl": kl,
        }
