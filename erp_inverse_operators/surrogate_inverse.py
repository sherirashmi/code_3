"""SurrogateInverse: an inverse-design model trained with a FROZEN, already-
trained forward-operator checkpoint acting as a differentiable simulator,
instead of ground-truth designs being the only training signal.

Every other model in this package (MDN/cVAE/Flow/Diffusion/BasisFlow/PadINN)
is trained purely to match designs (or, for BasisFlow/PadINN, ALSO a
spectrum predicted by a small forward component *trained jointly* alongside
the inverse direction, from scratch). This model instead reuses one of the
already-trained, already-verified erp_forward_operators/ checkpoints (DCO by
default, the best-performing of the 10 on the 100k dataset, RMSE=2.52 dB) as
a frozen "simulator-in-the-loop": sample a candidate design from the
predicted posterior, run it through the frozen DCO surrogate, and score how
well the resulting predicted ERP spectrum matches the true target spectrum.
This is standard amortized-variational-inference-with-a-simulator practice
(the surrogate plays the role a real physics simulator would in "simulation-
based inference," except here it is itself differentiable, so gradients flow
through it into the posterior network without needing a score-function
estimator).

Design:
  1. Encode the target spectrum (shared ``SpectrumEncoder``, same tool every
     model in this package conditions on).
  2. Predict a diagonal Gaussian posterior q(design | spectrum): (mu, log_var)
     over the flat design_dim-sized design vector, in THIS model's own
     normalized design space (z-scored the same way every other inverse
     model's training target is, via common.py's conventions).
  3. Two loss terms:
       - design_nll: the TRUE design's Gaussian NLL under q(.|spectrum) --
         direct ground-truth supervision, exactly analogous to cVAE's own
         reconstruction NLL, giving gradient to both mu and log_var.
       - surrogate_mse: reparameterize-sample a design from q(.|spectrum),
         run it through the frozen DCO surrogate, and MSE the resulting
         predicted ERP spectrum against the TRUE target spectrum -- an
         ERP-space consistency signal ground-truth design MSE alone can't
         give (two different designs producing the same spectrum are
         equally right under this term, unlike raw design MSE).
  4. Both terms cross between this model's own normalization and DCO's own
     (the two are fit from different dataset subsets/sizes -- see
     ``prepare_inverse_data``'s 10k default vs. DCO's 100k training run --
     so they are NOT guaranteed numerically identical). All such crossings
     go through PHYSICAL units (m/k/f_t/x/y in real values, ERP in real dB)
     as the common intermediate, never assuming the two normalizations
     happen to match.

The surrogate's own parameters are frozen (``requires_grad_(False)``) and
never touched by the optimizer -- only used as a fixed differentiable
function. It stays in ``eval()`` mode permanently (a ``train()`` call on
this whole model does not affect the frozen submodule, see ``train()``
override below).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from erp_forward_operators.dco import build_model as build_dco_surrogate
from erp_forward_operators.neural_operator_utils import MLP

from .common import SpectrumEncoder, flatten_configuration

_CONFIG_FIELDS = ("m", "k", "f_t", "x", "y")


def _gaussian_log_prob(x: torch.Tensor, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """Diagonal Gaussian log-density, summed over the last dim."""
    return (-0.5 * ((x - mu) ** 2 / torch.exp(log_var) + log_var + math.log(2 * math.pi))).sum(dim=-1)


class SurrogateInverse(nn.Module):
    def __init__(
        self,
        design_dim: int,
        surrogate_checkpoint: str = "erp_forward_operators/models/dco_erp.pth",
        embed_dim: int = 96,
        hidden: int = 128,
    ) -> None:
        super().__init__()
        self.design_dim = int(design_dim)
        if self.design_dim % 5 != 0:
            raise ValueError("design_dim must be num_res*5 ([m,k,f_t,x,y] per resonator).")
        self.num_res = self.design_dim // 5

        self.spectrum_encoder = SpectrumEncoder(embed_dim=embed_dim)
        self.head = MLP([embed_dim, hidden, hidden, 2 * self.design_dim], activation=nn.SiLU)

        checkpoint = torch.load(Path(surrogate_checkpoint), map_location="cpu", weights_only=False)
        surrogate_norm = checkpoint["preprocessing_state"]["norm_params"]
        if int(surrogate_norm["num_res"]) != self.num_res:
            raise ValueError(
                f"Surrogate checkpoint was trained with num_res="
                f"{surrogate_norm['num_res']}, but design_dim={design_dim} implies num_res={self.num_res}."
            )
        self.surrogate = build_dco_surrogate(num_res=self.num_res, **checkpoint["model_config"])
        self.surrogate.load_state_dict(checkpoint["model_state_dict"])
        self.surrogate.eval()
        for parameter in self.surrogate.parameters():
            parameter.requires_grad_(False)

        # Surrogate's own normalization stats, as buffers -- (1, num_res, 5)
        # for the config fields, scalars for frequency/ERP -- registered so
        # model.to(device) moves them with everything else.
        surrogate_mean = torch.tensor([float(surrogate_norm[f"{f}_mean"]) for f in _CONFIG_FIELDS])
        surrogate_std = torch.tensor([float(surrogate_norm[f"{f}_std"]) for f in _CONFIG_FIELDS])
        self.register_buffer("surrogate_config_mean", surrogate_mean.view(1, 1, 5))
        self.register_buffer("surrogate_config_std", surrogate_std.view(1, 1, 5))
        self.register_buffer("surrogate_erp_mean", torch.tensor(float(surrogate_norm["erp_mean"])))
        self.register_buffer("surrogate_erp_std", torch.tensor(float(surrogate_norm["erp_std"])))

        frequency_hz = torch.from_numpy(_load_frequency_grid())
        freq_norm = (frequency_hz - float(surrogate_norm["freq_mean"])) / float(surrogate_norm["freq_std"])
        self.register_buffer("surrogate_frequency_norm", freq_norm.float())

    def train(self, mode: bool = True):
        """Keep the frozen surrogate in eval() regardless of this model's own mode."""
        super().train(mode)
        self.surrogate.eval()
        return self

    # ---------------------------------------------------------
    # Posterior q(design | spectrum)
    # ---------------------------------------------------------

    def encode(self, spectrum: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.spectrum_encoder(spectrum)
        mu, log_var = self.head(h).chunk(2, dim=-1)
        log_var = log_var.clamp(min=-8.0, max=4.0)
        return mu, log_var

    # ---------------------------------------------------------
    # Frozen-surrogate forward pass, with the physical-units bridge
    # ---------------------------------------------------------

    def _surrogate_predicted_erp(
        self, design_norm: torch.Tensor, own_norm_params: Mapping[str, object]
    ) -> torch.Tensor:
        """Flat design (THIS model's own normalized space) -> predicted ERP
        in THIS model's own normalized ERP space (so callers can MSE it
        directly against a ``spectrum`` batch from this model's own loaders).

        Bridges through physical units: this model's own normalization ->
        physical [m,k,f_t,x,y]/Hz/dB -> the surrogate's own normalization ->
        DCO forward pass -> physical dB -> this model's own ERP normalization.
        """
        batch = design_norm.shape[0]
        configuration_own = design_norm.view(batch, self.num_res, 5)

        own_mean = torch.tensor(
            [float(own_norm_params[f"{f}_mean"]) for f in _CONFIG_FIELDS], device=design_norm.device
        ).view(1, 1, 5)
        own_std = torch.tensor(
            [float(own_norm_params[f"{f}_std"]) for f in _CONFIG_FIELDS], device=design_norm.device
        ).view(1, 1, 5)
        configuration_physical = configuration_own * own_std + own_mean

        configuration_surrogate = (
            configuration_physical - self.surrogate_config_mean
        ) / self.surrogate_config_std

        frequency = self.surrogate_frequency_norm[None, :, None].expand(batch, -1, -1)
        erp_surrogate_norm = self.surrogate(configuration_surrogate, frequency).squeeze(-1)  # (B, n_freq)
        erp_physical = erp_surrogate_norm * self.surrogate_erp_std + self.surrogate_erp_mean

        own_erp_mean = float(own_norm_params["erp_mean"])
        own_erp_std = float(own_norm_params["erp_std"])
        return (erp_physical - own_erp_mean) / own_erp_std

    # ---------------------------------------------------------
    # Training / sampling
    # ---------------------------------------------------------

    def training_loss(
        self,
        spectrum: torch.Tensor,
        design: torch.Tensor,
        own_norm_params: Mapping[str, object],
        surrogate_weight: float = 1.0,
    ) -> torch.Tensor:
        flat_design = flatten_configuration(design)
        mu, log_var = self.encode(spectrum)

        design_nll = -_gaussian_log_prob(flat_design, mu, log_var).mean()

        std = torch.exp(0.5 * log_var)
        design_sample = mu + std * torch.randn_like(std)
        predicted_spectrum = self._surrogate_predicted_erp(design_sample, own_norm_params)
        surrogate_mse = F.mse_loss(predicted_spectrum, spectrum)

        return design_nll + float(surrogate_weight) * surrogate_mse

    @torch.no_grad()
    def sample(self, spectrum: torch.Tensor, num_samples: int = 1):
        """Returns ``(flat_designs, log_prob)``, both ``(B, num_samples, ...)``,
        matching MDN/cVAE/Flow's convention -- the posterior here is an
        explicit diagonal Gaussian, so its log-density is exact and cheap.
        """
        b = spectrum.shape[0]
        mu, log_var = self.encode(spectrum)
        std = torch.exp(0.5 * log_var)

        mu_e = mu[:, None, :].expand(-1, num_samples, -1).reshape(b * num_samples, -1)
        std_e = std[:, None, :].expand(-1, num_samples, -1).reshape(b * num_samples, -1)
        log_var_e = log_var[:, None, :].expand(-1, num_samples, -1).reshape(b * num_samples, -1)
        z = mu_e + std_e * torch.randn_like(std_e)
        log_prob = _gaussian_log_prob(z, mu_e, log_var_e)
        return z.view(b, num_samples, self.design_dim), log_prob.view(b, num_samples)


def _load_frequency_grid():
    from utils.physics import freqs

    return freqs.astype("float32")
