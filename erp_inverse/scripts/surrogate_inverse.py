"""SurrogateInverse: an inverse-design model trained with FROZEN, already-
trained forward-operator checkpoints acting as differentiable simulators,
instead of ground-truth designs being the only training signal.

Every other model in this package (MDN/cVAE/Flow/Diffusion/BasisFlow/PadINN)
predicts a DISTRIBUTION over designs. This model instead predicts a single
DETERMINISTIC point estimate -- no posterior, no sampling, no log-density --
and reuses already-trained, already-verified erp_forward/scripts/
checkpoints as a frozen "simulator-in-the-loop" to give it an ERP-space
training signal on top of plain design-matching MSE: run the predicted
design through the frozen surrogate(s), and score how well the resulting
predicted ERP spectra match the true target spectrum. Gradients flow
through the frozen surrogates into the design-prediction network, but
never update the surrogates themselves.

(A posterior -- first a single diagonal Gaussian, later an MDN-style
Gaussian mixture -- was implemented and used earlier in this project, but
was removed by explicit request: this model is deterministic point-
estimate only now. Its ``sample()`` therefore returns a plain tensor, not
a ``(samples, log_prob)`` tuple -- same convention already used by
``ConditionalDiffusion``/``PadINN`` for "no tractable density.")

Design:
  1. Encode the target spectrum (shared ``SpectrumEncoder``, same tool every
     model in this package conditions on).
  2. Predict a single flat design_dim-sized point estimate, in THIS
     model's own normalized design space (z-scored the same way every
     other inverse model's training target is, via common.py's
     conventions).
  3. Two loss terms:
       - design_mse: plain MSE between the predicted point and the TRUE
         design -- direct ground-truth supervision.
       - surrogate_spectrum_loss: run the predicted design through EVERY
         frozen surrogate in the ensemble, and score the resulting
         predicted ERP spectra against the TRUE target spectrum using the
         SAME ``erp_spectrum_loss`` every forward operator in this repo
         trains against (MSE + a first-difference/slope penalty + a
         multi-peak value penalty summed over every one of the true
         spectrum's resonance peaks, not plain MSE) -- an ERP-space
         consistency signal ground-truth design MSE alone can't give (two
         different designs producing the same spectrum are equally right
         under this term, unlike raw design MSE). Averaged over the
         surrogate ensemble (DCO + GNO by default) so the training signal
         isn't tied to any one forward operator's own idiosyncrasies.
  4. Both terms cross between this model's own normalization and EACH
     surrogate's own (each may be fit from a different dataset
     subset/size -- see ``prepare_inverse_data``'s 10k default vs. the
     forward operators' 100k training runs -- so they are NOT guaranteed
     numerically identical). All such crossings go through PHYSICAL units
     (m/k/f_t/x/y in real values, ERP in real dB) as the common
     intermediate, never assuming normalizations happen to match.

Every surrogate's own parameters are frozen (``requires_grad_(False)``) and
never touched by the optimizer -- only used as fixed differentiable
functions. They stay in ``eval()`` mode permanently (a ``train()`` call on
this whole model does not affect the frozen submodules, see ``train()``
override below).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from erp_forward.scripts.dco import build_model as _build_dco
from erp_forward.scripts.gno import build_model as _build_gno
from erp_forward.scripts.neural_operator_utils import MLP, build_operator_model, erp_spectrum_loss
import utils.physics as _physics
from utils.erp_dataset import recorded_modal_resolution
from utils.physics import Lx, Ly, edge_margin, fmax, fmin, m_max, m_min

from .common import SpectrumEncoder, flatten_configuration
from .design_space import decode_bounded_torch

_CONFIG_FIELDS = ("m", "k", "f_t", "x", "y")

# The actual generation bounds every training design was Latin-Hypercube-
# sampled within (see utils.erp_dataset.generate's resonator_bounds() call)
# -- i.e. literally "the range of the training dataset", not a guess. k has
# no independent bound of its own (it's derived as m*(2*pi*f_t)**2), so its
# range here is the loosest box implied by m's and f_t's own corners; this
# doesn't require every (m, k, f_t) to be mutually consistent, only that
# each field individually stays where the surrogates actually saw data.
_DESIGN_PHYSICAL_BOUNDS = {
    "m": (m_min, m_max),
    "k": (m_min * (2 * math.pi * fmin) ** 2, m_max * (2 * math.pi * fmax) ** 2),
    "f_t": (fmin, fmax),
    "x": (edge_margin, Lx - edge_margin),
    "y": (edge_margin, Ly - edge_margin),
}

# Extend this if a surrogate checkpoint from a different forward-operator
# architecture is ever used -- keyed by the "operator_name" every checkpoint
# already carries (see erp_forward.scripts.neural_operator_utils.
# save_operator_checkpoint), so a new architecture only needs one new entry.
# "dco_staged" is the staged-curriculum DCO variant (peak term introduced
# only in a late fine-tuning phase) -- same architecture/build function as
# plain DCO, just a different checkpoint's worth of trained weights.
_SURROGATE_BUILDERS = {"dco": _build_dco, "dco_staged": _build_dco, "gno": _build_gno}

from utils.paths import FORWARD_ROOT as _FORWARD_ROOT, forward_model_path as _forward_model_path

# Trained on the 100k (150-mode) dataset.
DEFAULT_SURROGATE_CHECKPOINTS = (
    str(_FORWARD_ROOT / "models" / "experiments" / "dco_variants" / "dco_staged.pth"),
    str(_FORWARD_ROOT / "models" / "100k" / "gno.pth"),
)


def surrogate_checkpoints_for(dataset_tag: str) -> tuple[str, ...]:
    """Frozen DCO+GNO surrogates trained on the SAME dataset (hence the same
    plate-mode physics) as the inverse model, when they exist; otherwise the
    100k defaults, with a warning if that means mixing modal resolutions."""
    candidates = tuple(str(_forward_model_path(name, dataset_tag)) for name in ("dco", "gno"))
    if all(Path(c).exists() for c in candidates):
        return candidates
    if dataset_tag != "100k":
        print(
            f"WARNING: no forward DCO/GNO checkpoints for dataset '{dataset_tag}' "
            f"({', '.join(candidates)}); SurrogateInverse falls back to the 100k-trained "
            "surrogates. Train DCO and GNO on this dataset first (ERP -> Forward) for a "
            "physically consistent surrogate."
        )
    return DEFAULT_SURROGATE_CHECKPOINTS


_SOFT_CLAMP_SHARPNESS = 3.0


def _soft_clamp(x: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor, sharpness: float = _SOFT_CLAMP_SHARPNESS) -> torch.Tensor:
    """Smooth alternative to ``torch.clamp`` whose live-gradient region roughly
    matches the dataset's own range, instead of extending several multiples
    beyond it.

    ``mid + half_range * tanh(sharpness * (x - mid) / half_range)`` maps all
    of R into the open interval ``(lo, hi)``. At ``x = mid`` this is exactly
    the identity scaled by ``sharpness`` (``tanh`` is linear near 0), so
    values near the range's center pass through with that same slope;
    moving away from ``mid`` it saturates smoothly toward ``lo``/``hi``
    instead of hard-clipping.

    ``sharpness`` controls how tightly the saturation hugs the true
    boundary. With ``sharpness=1`` (plain tanh), a value sitting exactly at
    the true boundary (``x = hi``) only reaches ~89% of the way there
    (``tanh(1) ~= 0.762``), and the gradient stays meaningfully nonzero for
    inputs several range-widths *beyond* the boundary too (empirically, up
    to ~8x) -- i.e. the "still gets corrective signal" zone is much wider
    than the dataset's own range, which is more permissive than intended.
    The default ``sharpness=3`` instead: (a) maps the true boundary itself
    to ~99.8% of the way to ``hi``/``lo`` (barely any compression of
    legitimate boundary-adjacent training values), and (b) concentrates the
    nonzero-gradient recovery zone to roughly the dataset's own width past
    the edge (empirically negligible by ~1.5x, versus ~8x before) -- an
    input a little outside the true range still gets pulled back, but the
    "still meaningfully responsive" region no longer reaches implausibly
    far into physically-impossible territory. Unlike ``torch.clamp``,
    whose gradient is exactly 0 the instant ``x`` leaves ``[lo, hi]``, this
    keeps a real (if small) gradient right at and just past the boundary.
    In float32, ``tanh`` itself saturates to exactly 1.0 for large enough
    arguments, so truly extreme outliers do still round to a zero
    gradient -- the same standard precision limit any bounded squashing
    function hits, not a flaw specific to this use.
    """
    mid = (lo + hi) / 2
    half_range = (hi - lo) / 2
    return mid + half_range * torch.tanh(sharpness * (x - mid) / half_range)


class SurrogateInverse(nn.Module):
    def __init__(
        self,
        design_dim: int,
        surrogate_checkpoints: Sequence[str] = DEFAULT_SURROGATE_CHECKPOINTS,
        embed_dim: int = 96,
        hidden: int = 128,
        spectrum_encoder: str = "pooled",
        num_res: int = 3,
    ) -> None:
        super().__init__()
        self.design_dim = int(design_dim)
        self.num_res = int(num_res)
        if self.design_dim not in (5 * self.num_res, 4 * self.num_res):
            raise ValueError(
                "design_dim must be num_res*5 (full15: [m,k,f_t,x,y]) or num_res*4 "
                "(bounded12: [m,f_t,x,y], see design_space.py)."
            )
        self.bounded = self.design_dim == 4 * self.num_res

        self.spectrum_encoder = SpectrumEncoder(embed_dim=embed_dim, mode=spectrum_encoder)
        self.head = MLP([embed_dim, hidden, hidden, self.design_dim], activation=nn.SiLU)

        if len(surrogate_checkpoints) == 0:
            raise ValueError("surrogate_checkpoints must not be empty.")

        surrogates: list[nn.Module] = []
        config_means, config_stds, erp_means, erp_stds, freq_norms = [], [], [], [], []
        raw_frequency_hz = torch.from_numpy(_load_frequency_grid())
        for checkpoint_path in surrogate_checkpoints:
            checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
            operator_name = str(checkpoint["operator_name"]).lower()
            if operator_name not in _SURROGATE_BUILDERS:
                raise ValueError(
                    f"Unsupported surrogate architecture '{operator_name}' in {checkpoint_path}; "
                    f"supported: {sorted(_SURROGATE_BUILDERS)}."
                )
            surrogate_norm = checkpoint["preprocessing_state"]["norm_params"]
            surrogate_modes = recorded_modal_resolution(checkpoint)
            if surrogate_modes is not None and surrogate_modes != _physics.get_modal_resolution():
                print(
                    f"WARNING: frozen surrogate {checkpoint_path} was trained with Nx x Ny = "
                    f"{surrogate_modes[0]} x {surrogate_modes[1]} plate modes, but the current dataset uses "
                    f"{_physics.Nx} x {_physics.Ny}; its spectra follow different physics."
                )
            if int(surrogate_norm["num_res"]) != self.num_res:
                raise ValueError(
                    f"Surrogate checkpoint {checkpoint_path} was trained with num_res="
                    f"{surrogate_norm['num_res']}, but design_dim={design_dim} implies num_res={self.num_res}."
                )
            surrogate = build_operator_model(
                _SURROGATE_BUILDERS[operator_name], self.num_res, checkpoint["model_config"]
            )
            surrogate.load_state_dict(checkpoint["model_state_dict"])
            surrogate.eval()
            for parameter in surrogate.parameters():
                parameter.requires_grad_(False)
            surrogates.append(surrogate)

            config_means.append([float(surrogate_norm[f"{f}_mean"]) for f in _CONFIG_FIELDS])
            config_stds.append([float(surrogate_norm[f"{f}_std"]) for f in _CONFIG_FIELDS])
            erp_means.append(float(surrogate_norm["erp_mean"]))
            erp_stds.append(float(surrogate_norm["erp_std"]))
            freq_norms.append(
                (raw_frequency_hz - float(surrogate_norm["freq_mean"])) / float(surrogate_norm["freq_std"])
            )

        self.surrogates = nn.ModuleList(surrogates)
        self.num_surrogates = len(surrogates)

        # Stacked per-surrogate normalization stats, registered as buffers so
        # model.to(device) moves them with everything else -- (S,1,1,5) for
        # the config fields, (S,) for ERP, (S, n_freq) for frequency.
        self.register_buffer(
            "surrogate_config_mean", torch.tensor(config_means).view(self.num_surrogates, 1, 1, 5)
        )
        self.register_buffer(
            "surrogate_config_std", torch.tensor(config_stds).view(self.num_surrogates, 1, 1, 5)
        )
        self.register_buffer("surrogate_erp_mean", torch.tensor(erp_means))
        self.register_buffer("surrogate_erp_std", torch.tensor(erp_stds))
        self.register_buffer("surrogate_frequency_norm", torch.stack(freq_norms).float())

        # Clamp bounds for the physical configuration fed to the frozen
        # surrogates -- see _surrogate_predicted_erp's docstring for why.
        self.register_buffer(
            "design_physical_min",
            torch.tensor([_DESIGN_PHYSICAL_BOUNDS[f][0] for f in _CONFIG_FIELDS]).view(1, 1, 5),
        )
        self.register_buffer(
            "design_physical_max",
            torch.tensor([_DESIGN_PHYSICAL_BOUNDS[f][1] for f in _CONFIG_FIELDS]).view(1, 1, 5),
        )

    def train(self, mode: bool = True):
        """Keep every frozen surrogate in eval() regardless of this model's own mode."""
        super().train(mode)
        for surrogate in self.surrogates:
            surrogate.eval()
        return self

    # ---------------------------------------------------------
    # Deterministic point-estimate prediction (no posterior)
    # ---------------------------------------------------------

    def _predict_design(self, spectrum: torch.Tensor) -> torch.Tensor:
        """Spectrum -> single flat design point estimate, ``(B, design_dim)``,
        in this model's own normalized design space.
        """
        embedding = self.spectrum_encoder(spectrum)
        return self.head(embedding)

    # ---------------------------------------------------------
    # Frozen-surrogate-ensemble forward pass, with the physical-units bridge
    # ---------------------------------------------------------

    def _surrogate_predicted_erp(
        self, design_norm: torch.Tensor, own_norm_params: Mapping[str, object]
    ) -> torch.Tensor:
        """Flat design (THIS model's own normalized space) -> predicted ERP,
        one row per frozen surrogate in the ensemble, each in THIS model's
        own normalized ERP space (so callers can score it directly against a
        ``spectrum`` batch from this model's own loaders). Returns
        ``(num_surrogates, batch, n_freq)``.

        Bridges through physical units, independently per surrogate: this
        model's own normalization -> physical [m,k,f_t,x,y]/Hz/dB -> that
        surrogate's own normalization -> its forward pass -> physical dB ->
        this model's own ERP normalization.

        The physical configuration is soft-bounded to ``_DESIGN_PHYSICAL_BOUNDS``
        (the real training-dataset generation range) before being handed to
        the surrogates, via ``_soft_clamp`` rather than a hard ``torch.clamp``.
        Early in training especially, the predicted design can land far
        outside a field's physical range, querying the frozen surrogates on
        inputs wildly outside what they were ever trained on. Their output
        there is meaningless (not "wrong physics", just extrapolation
        noise), which otherwise pollutes the consistency-loss gradient with
        signal that has nothing to do with the actual prediction quality.

        A hard ``torch.clamp`` would fix the input range but at the cost of
        an exactly-zero gradient outside the bounds -- once a prediction is
        clamped, the surrogate-consistency term gives the network no signal
        at all to pull it back, throwing away exactly the information ("how
        far out of range, and in which direction") that would help it
        recover fastest. ``_soft_clamp`` uses a tanh squash instead: it's
        identity (slope exactly ``sharpness``) at the range's center, so
        in-range values pass through essentially unchanged, and it
        saturates smoothly toward the bounds for outliers. See
        ``_soft_clamp``'s own docstring for the gradient-shape tradeoff
        versus a hard ``torch.clamp``.
        """
        batch = design_norm.shape[0]
        if self.bounded:
            # bounded12: exact, differentiable decode -- always inside the
            # generation bounds with k = m (2 pi f_t)^2, nothing to clamp.
            configuration_physical = decode_bounded_torch(design_norm, self.num_res, own_norm_params)
        else:
            configuration_own = design_norm.view(batch, self.num_res, 5)
            own_mean = torch.tensor(
                [float(own_norm_params[f"{f}_mean"]) for f in _CONFIG_FIELDS], device=design_norm.device
            ).view(1, 1, 5)
            own_std = torch.tensor(
                [float(own_norm_params[f"{f}_std"]) for f in _CONFIG_FIELDS], device=design_norm.device
            ).view(1, 1, 5)
            configuration_physical = _soft_clamp(
                configuration_own * own_std + own_mean, self.design_physical_min, self.design_physical_max
            )
            # The solver only uses m and k; derive k from the (soft-bounded)
            # m and f_t so the surrogates are only ever queried with
            # physically consistent resonators, k = m (2 pi f_t)^2.
            m = configuration_physical[..., 0]
            f_t = configuration_physical[..., 2]
            configuration_physical = torch.stack(
                (m, m * (2.0 * math.pi * f_t) ** 2, f_t, configuration_physical[..., 3], configuration_physical[..., 4]),
                dim=-1,
            )

        own_erp_mean = float(own_norm_params["erp_mean"])
        own_erp_std = float(own_norm_params["erp_std"])

        predictions = []
        for s, surrogate in enumerate(self.surrogates):
            configuration_s = (configuration_physical - self.surrogate_config_mean[s]) / self.surrogate_config_std[s]
            frequency_s = self.surrogate_frequency_norm[s][None, :, None].expand(batch, -1, -1)
            erp_s_norm = surrogate(configuration_s, frequency_s).squeeze(-1)  # (B, n_freq)
            erp_physical = erp_s_norm * self.surrogate_erp_std[s] + self.surrogate_erp_mean[s]
            predictions.append((erp_physical - own_erp_mean) / own_erp_std)
        return torch.stack(predictions, dim=0)  # (S, B, n_freq)

    # ---------------------------------------------------------
    # Training / prediction
    # ---------------------------------------------------------

    def training_loss(
        self,
        spectrum: torch.Tensor,
        design: torch.Tensor,
        own_norm_params: Mapping[str, object],
        surrogate_weight: float = 1.0,
        slope_weight: float = 0.5,
        peak_weight: float = 0.05,
        peak_window: int = 7,
    ) -> torch.Tensor:
        """``slope_weight``/``peak_weight``/``peak_window`` are passed straight
        through to ``erp_spectrum_loss`` -- the EXACT SAME loss function, same
        default weights, that every forward operator (including this model's
        own frozen DCO/GNO ensemble) trains against: MSE + a first-difference
        slope penalty + a peak-value penalty summed over EVERY one of the
        true spectrum's resonance peaks (not just the tallest one -- see
        ``erp_spectrum_loss``'s own docstring for the "ALL peaks" detail).
        """
        flat_design = flatten_configuration(design)
        predicted_design = self._predict_design(spectrum)  # (B, D)

        design_loss = F.mse_loss(predicted_design, flat_design)

        predicted = self._surrogate_predicted_erp(predicted_design, own_norm_params)  # (S, B, n_freq)
        target = spectrum[..., None]  # (B, n_freq, 1) -- erp_spectrum_loss expects a trailing dim
        surrogate_losses = [
            erp_spectrum_loss(
                predicted[s][..., None], target,
                slope_weight=slope_weight, peak_weight=peak_weight, peak_window=peak_window,
            )
            for s in range(self.num_surrogates)
        ]
        surrogate_loss = torch.stack(surrogate_losses).mean()

        return design_loss + float(surrogate_weight) * surrogate_loss

    @torch.no_grad()
    def sample(self, spectrum: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """Returns ``(B, num_samples, design_dim)`` -- a PLAIN tensor, not a
        ``(samples, log_prob)`` tuple (same "no tractable density"
        convention as ``ConditionalDiffusion``/``PadINN``'s ``sample()``).

        This model is deterministic: there is no posterior to draw from, so
        every one of the ``num_samples`` rows is the identical point
        estimate, simply repeated for interface compatibility with the
        other inverse models' ``sample(spectrum, num_samples)`` signature.
        """
        predicted = self._predict_design(spectrum)  # (B, D)
        return predicted[:, None, :].expand(-1, num_samples, -1)


def _load_frequency_grid():
    from utils.physics import freqs

    return freqs.astype("float32")
