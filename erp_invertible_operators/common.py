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

What is and is not invertible
-----------------------------
Only the coupling stack is an exact bijection, between its own latent
tensors. The lifts (P: design -> latent, P': spectrum -> latent) and
readouts (Q: latent -> one ERP value per frequency, Q': frequency-pooled
latent -> design) are ordinary learned, lossy maps, and P' is not
constrained to reproduce the latent the forward path produced. These are
therefore bidirectional models with an invertible core -- NOT exact
design<->spectrum bijections, and they have no exact change-of-variables
density over designs. The optional cycle/alignment terms (see
``stage1_loss``) encourage the two directions to use compatible latents.

Options (all default to the original behaviour, so old checkpoints load):

* ``design_param``: ``"full15"`` (z-scored [m,k,f_t,x,y]) or
  ``"bounded12"`` ([m,f_t,x,y], logit-bounded, k derived; see
  erp_inverse_operators/design_space.py). The forward lift always receives
  the z-scored 5-field configuration, so in bounded12 mode the design is
  decoded (differentiably) and k derived before the lift.
* ``readout``: ``"pooled"`` (mean+max over frequency) or ``"binned"`` (plus
  ``readout_bins`` ordered frequency bins, so the design readout knows WHERE
  along the axis a feature occurred).
* the coupling gate (``gate`` in coupling.py) is chosen by each subclass.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from erp_forward_operators.neural_operator_utils import MLP
from erp_inverse_operators.design_space import BOUNDED12, BOUNDED_FIELDS, FULL15, decode_bounded_torch, design_dim as _design_dim

_CONFIG_FIELDS = ("m", "k", "f_t", "x", "y")


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
    function L. A subclass's ``__init__`` must, after ``super().__init__()``:

      - call :meth:`_init_design` (sets ``design_dim``, ``num_res``,
        ``design_param``) and set ``self.n_freq`` and register the frequency
        grid (:meth:`_register_frequency_grid`).
      - set ``self.width`` (per-half channel width; latent has 2*width).
      - build ``self.blocks`` (an InvertibleCouplingStack), ``self.project_q``
        (``(B,F,2*width) -> (B,F,1)``), ``self.project_qp``
        (``(B, self.readout_dim) -> (B, design_dim)``; call
        :meth:`_init_readout` first) and ``self.vae`` (a :class:`DesignVAE`).

    and must implement :meth:`_lift_forward` and :meth:`_lift_inverse`.
    """

    # ---------------------------------------------------------
    # Construction helpers
    # ---------------------------------------------------------

    def _init_design(self, design_dim: int, design_param: str = FULL15, num_res: int = 3) -> None:
        self.design_param = str(design_param)
        self.num_res = int(num_res)
        expected = _design_dim(self.design_param, self.num_res)
        if int(design_dim) != expected:
            raise ValueError(
                f"design_dim={design_dim} does not match design_param={design_param!r} with num_res={num_res} "
                f"(expected {expected})."
            )
        self.design_dim = expected
        if self.design_param == BOUNDED12:
            # Normalisation needed to turn a bounded12 design into the
            # z-scored [m,k,f_t,x,y] configuration the forward lift expects.
            # Filled by set_design_normalization(); saved in the state dict.
            self.register_buffer("config_mean", torch.zeros(5))
            self.register_buffer("config_std", torch.ones(5))
            self.register_buffer("b12_mean", torch.zeros(len(BOUNDED_FIELDS)))
            self.register_buffer("b12_std", torch.ones(len(BOUNDED_FIELDS)))

    def set_design_normalization(self, norm_params) -> None:
        """Store the dataset normalisation used by the bounded12 design path
        (no-op for full15)."""
        if self.design_param != BOUNDED12:
            return
        dev = self.config_mean.device
        self.config_mean.copy_(torch.tensor([float(norm_params[f"{f}_mean"]) for f in _CONFIG_FIELDS], device=dev))
        self.config_std.copy_(torch.tensor([float(norm_params[f"{f}_std"]) for f in _CONFIG_FIELDS], device=dev))
        self.b12_mean.copy_(torch.tensor([float(norm_params[f"b12_{n}_mean"]) for n, *_ in BOUNDED_FIELDS], device=dev))
        self.b12_std.copy_(torch.tensor([float(norm_params[f"b12_{n}_std"]) for n, *_ in BOUNDED_FIELDS], device=dev))

    def _b12_norm(self) -> dict[str, torch.Tensor]:
        stats = {}
        for j, (name, *_rest) in enumerate(BOUNDED_FIELDS):
            stats[f"b12_{name}_mean"] = self.b12_mean[j]
            stats[f"b12_{name}_std"] = self.b12_std[j]
        return stats

    def _init_readout(self, readout: str = "pooled", readout_bins: int = 16) -> int:
        """Set up Q''s input pooling; returns its input dimension."""
        if readout not in ("pooled", "binned"):
            raise ValueError("readout must be 'pooled' or 'binned'.")
        self.readout = readout
        dim = 4 * self.width
        if readout == "binned":
            self.readout_bins = int(readout_bins)
            self.readout_bin_pool = nn.AdaptiveAvgPool1d(self.readout_bins)
            self.readout_bin_proj = nn.Conv1d(2 * self.width, 4, kernel_size=1)
            dim += 4 * self.readout_bins
        self.readout_dim = dim
        return dim

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
    # Design representation
    # ---------------------------------------------------------

    def design_to_configuration(self, flat_design: torch.Tensor) -> torch.Tensor:
        """Flat design (this model's design space) -> z-scored (B, R, 5)
        [m,k,f_t,x,y] configuration for the forward lift. bounded12: exact
        differentiable decode with k = m (2 pi f_t)^2, then z-scoring."""
        if self.design_param == BOUNDED12:
            physical = decode_bounded_torch(flat_design, self.num_res, self._b12_norm())
            return (physical - self.config_mean) / self.config_std
        return flat_design.reshape(flat_design.shape[0], self.num_res, 5)

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

    def _pool(self, latent: torch.Tensor) -> torch.Tensor:
        """(B, 2*width, F) -> (B, readout_dim).

        "pooled": mean+max over frequency, (B, 4*width) -- order-free, so it
        cannot tell WHERE along the frequency axis a feature occurred.
        "binned": additionally ``readout_bins`` ordered frequency bins
        (average-pooled, 1x1-projected to 4 channels, flattened in order).
        """
        pooled = torch.cat((latent.mean(dim=-1), latent.amax(dim=-1)), dim=-1)
        if getattr(self, "readout", "pooled") == "binned":
            binned = self.readout_bin_proj(self.readout_bin_pool(latent)).flatten(1)
            pooled = torch.cat((pooled, binned), dim=-1)
        return pooled

    def _forward_latent(self, v0: torch.Tensor) -> torch.Tensor:
        v1_k, v2_k = self.blocks(*v0.chunk(2, dim=1))
        return torch.cat((v1_k, v2_k), dim=1)

    def _through_blocks_to_spectrum(self, v0: torch.Tensor) -> torch.Tensor:
        combined = self._forward_latent(v0).transpose(1, 2)  # (B, F, 2*width)
        return self.project_q(combined)

    def _through_blocks_to_design(self, u0: torch.Tensor) -> torch.Tensor:
        v1_k, v2_k = u0.chunk(2, dim=1)
        v1_0, v2_0 = self.blocks.inverse(v1_k, v2_k)
        pooled = self._pool(torch.cat((v1_0, v2_0), dim=1))
        return self.project_qp(pooled)

    def predict_spectrum(self, configuration: torch.Tensor, frequency: torch.Tensor | None = None) -> torch.Tensor:
        """Forward-operator use: z-scored configuration (B,R,5) -> predicted
        ERP (B, F, 1), directly comparable to erp_forward_operators' outputs.
        (For a flat design in this model's own design space use
        :meth:`predict_spectrum_from_design`.)
        """
        if frequency is None:
            frequency = self._frequency_batch(configuration.shape[0])
        v0 = self._lift_forward(configuration, frequency)
        return self._through_blocks_to_spectrum(v0)

    def predict_spectrum_from_design(self, flat_design: torch.Tensor) -> torch.Tensor:
        return self.predict_spectrum(self.design_to_configuration(flat_design))

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
    # Diagnostics
    # ---------------------------------------------------------

    @torch.no_grad()
    def diagnostics(self, spectrum: torch.Tensor, flat_design: torch.Tensor) -> dict[str, float]:
        """Numerical health of the invertible core on one batch:
        relative round-trip error of the coupling stack, latent-norm growth
        across the stack, gate range, and the forward/inverse latent
        alignment ||P'(y) - F(P(x))|| / ||F(P(x))||."""
        frequency = self._frequency_batch(spectrum.shape[0])
        v0 = self._lift_forward(self.design_to_configuration(flat_design), frequency)
        v1, v2 = v0.chunk(2, dim=1)
        gate_min, gate_max = float("inf"), 0.0
        for block in self.blocks.blocks:
            g1 = block._gate(v2)
            v1 = v1 * g1
            g2 = block._gate(v1)
            v2 = v2 * g2
            gate_min = min(gate_min, float(g1.min()), float(g2.min()))
            gate_max = max(gate_max, float(g1.max()), float(g2.max()))
        vk = torch.cat((v1, v2), dim=1)
        r1, r2 = self.blocks.inverse(v1, v2)
        round_trip = float((torch.cat((r1, r2), dim=1) - v0).norm() / v0.norm().clamp_min(1e-12))
        u0 = self._lift_inverse(spectrum, frequency)
        alignment = float((u0 - vk).norm() / vk.norm().clamp_min(1e-12))
        return {
            "round_trip_rel_err": round_trip,
            "latent_norm_ratio": float(vk.norm(dim=1).mean() / v0.norm(dim=1).mean().clamp_min(1e-12)),
            "gate_min": gate_min,
            "gate_max": gate_max,
            "latent_alignment": alignment,
        }

    # ---------------------------------------------------------
    # Stage losses (see train.py for the 3-step schedule that uses them)
    # ---------------------------------------------------------

    def _cycle_terms(self, predicted_spectrum, u0, vk, flat_design, cycle_weight, align_weight):
        """Optional terms tying the two directions together.

        * design cycle (``cycle_weight``): design -> predicted spectrum ->
          inverse pipeline -> design, MSE against the true design;
        * latent alignment (``align_weight``): the inverse lift of the TRUE
          spectrum should land near the forward latent F(P(x)), measured as a
          scale-free relative error (so neither latent can shrink to cheat).
        Both are soft, modestly weighted: P' only sees one ERP value per
        frequency, so exact latent agreement may be unattainable.
        """
        terms = {}
        total = predicted_spectrum.new_zeros(())
        if cycle_weight > 0:
            cycled = self.infer_point_estimate(predicted_spectrum)
            terms["j_cycle"] = F.mse_loss(cycled, flat_design)
            total = total + float(cycle_weight) * terms["j_cycle"]
        if align_weight > 0:
            terms["j_align"] = ((u0 - vk) ** 2).mean() / (vk.detach() ** 2).mean().clamp_min(1e-8)
            total = total + float(align_weight) * terms["j_align"]
        return total, terms

    def stage1_loss(
        self,
        spectrum: torch.Tensor,
        flat_design: torch.Tensor,
        cycle_weight: float = 0.0,
        align_weight: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        """J_IFB = J_FWD + J_INV + J_{P,Q'} + J_{P',Q} (paper eq 7), MSE in
        place of the paper's relative-L2, plus the optional cycle/alignment
        terms (default off).

        Each of P's and P''s lift is computed ONCE and reused for both its
        through-the-blocks path and its direct (blocks-skipped) path.
        """
        configuration = self.design_to_configuration(flat_design)
        frequency = self._frequency_batch(spectrum.shape[0])

        v0 = self._lift_forward(configuration, frequency)
        vk = self._forward_latent(v0)
        predicted_spectrum = self.project_q(vk.transpose(1, 2)).squeeze(-1)
        j_fwd = F.mse_loss(predicted_spectrum, spectrum)
        direct_design = self.project_qp(self._pool(v0))
        j_pq_prime = F.mse_loss(direct_design, flat_design)

        u0 = self._lift_inverse(spectrum, frequency)
        point_estimate = self._through_blocks_to_design(u0)
        j_inv = F.mse_loss(point_estimate, flat_design)
        direct_spectrum = self.project_q(u0.transpose(1, 2)).squeeze(-1)
        j_pprime_q = F.mse_loss(direct_spectrum, spectrum)

        extra, terms = self._cycle_terms(predicted_spectrum, u0, vk, flat_design, cycle_weight, align_weight)
        total = j_fwd + j_inv + j_pq_prime + j_pprime_q + extra
        return {"total": total, "j_fwd": j_fwd, "j_inv": j_inv, "j_pq_prime": j_pq_prime,
                "j_pprime_q": j_pprime_q, **terms}

    def stage2_loss(
        self, flat_design: torch.Tensor, beta: float = 0.1, encoder_input: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """J_beta-VAE (paper eq 10), pretraining the VAE alone.

        ``encoder_input=None`` (paper): encode the TRUE design and reconstruct
        it. Passing the (detached) stage-1 inverse point estimates instead
        trains the VAE on the same kind of input it receives in stage 3 and
        at inference, still decoding toward the true design.
        """
        source = flat_design if encoder_input is None else encoder_input
        mu, log_var = self.vae.encode(source)
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
        cycle_weight: float = 0.0,
        align_weight: float = 0.0,
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
        dominate the gradient. ``cycle_weight``/``align_weight``: optional
        terms, see :meth:`_cycle_terms`.
        """
        configuration = self.design_to_configuration(flat_design)
        frequency = self._frequency_batch(spectrum.shape[0])

        v0 = self._lift_forward(configuration, frequency)
        vk = self._forward_latent(v0)
        predicted_spectrum = self.project_q(vk.transpose(1, 2)).squeeze(-1)
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
        extra, terms = self._cycle_terms(predicted_spectrum, u0, vk, flat_design, cycle_weight, align_weight)

        total = j_fwd + j_pq_prime + j_pprime_q + j_beta_vae + float(inverse_weight) * j_inv + extra
        return {
            "total": total, "j_fwd": j_fwd, "j_inv": j_inv, "j_pq_prime": j_pq_prime, "j_pprime_q": j_pprime_q,
            "j_beta_vae": j_beta_vae, "recon": recon_loss, "kl": kl, **terms,
        }
