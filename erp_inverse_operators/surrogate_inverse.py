"""SurrogateInverse: an inverse-design model trained with FROZEN, already-
trained forward-operator checkpoints acting as differentiable simulators,
instead of ground-truth designs being the only training signal.

Every other model in this package (MDN/cVAE/Flow/Diffusion/BasisFlow/PadINN)
is trained purely to match designs (or, for BasisFlow/PadINN, ALSO a
spectrum predicted by a small forward component *trained jointly* alongside
the inverse direction, from scratch). This model instead reuses
already-trained, already-verified erp_forward_operators/ checkpoints as a
frozen "simulator-in-the-loop": sample candidate designs from the predicted
posterior, run them through the frozen surrogate(s), and score how well the
resulting predicted ERP spectra match the true target spectrum. This is
standard amortized-variational-inference-with-a-simulator practice (the
surrogate plays the role a real physics simulator would in "simulation-
based inference," except here it is itself differentiable, so gradients
flow through it into the posterior network without needing a score-function
estimator).

Three design points, each fixing a real limitation found by inspection this
session:

1. **Mixture-of-Gaussians posterior, not a single Gaussian.** A single
   diagonal Gaussian can only ever represent ONE region of design space --
   but this project's inverse problem is genuinely multi-modal (the m/k
   degeneracy f_t=sqrt(k/m)/2pi alone means several distinct (m,k) pairs
   can produce the same spectrum). A single-Gaussian posterior is
   structurally forced to average distinct valid solutions into one wide,
   low-quality blob sitting between them. Swapping in an MDN-style mixture
   head (same convention as mdn.py: K components, softmax-mixture NLL,
   multinomial + gather sampling) keeps everything else about this model
   unchanged while removing that structural ceiling.

2. **Multiple reparameterized samples per training step for the surrogate
   consistency term, not one.** The original version drew exactly one
   design sample per training example to check against the surrogate --
   a single noisy draw standing in for "how good is the whole posterior,"
   which is a high-variance gradient estimate. Averaging the surrogate
   loss over ``num_surrogate_samples`` draws (batched, not looped) gives a
   materially less noisy training signal for negligible extra cost (the
   surrogate forward pass is frozen and cheap).

3. **An ensemble of surrogates, not one.** Trusting a single forward
   operator's predicted spectrum as if it were ground truth silently
   teaches this model to satisfy THAT operator's own idiosyncrasies (e.g.
   any near-degenerate-peak blind spots) rather than the real physics.
   Averaging the consistency loss across multiple, architecturally
   different frozen forward operators (DCO + GNO by default) makes the
   training signal robust to any one architecture's particular quirks.

Design:
  1. Encode the target spectrum (shared ``SpectrumEncoder``, same tool every
     model in this package conditions on).
  2. Predict a Gaussian-mixture posterior q(design | spectrum): (logits,
     mu, std) over K components, each a diagonal Gaussian over the flat
     design_dim-sized design vector, in THIS model's own normalized design
     space (z-scored the same way every other inverse model's training
     target is, via common.py's conventions).
  3. Two loss terms:
       - design_nll: the TRUE design's exact mixture log-density under
         q(.|spectrum) -- direct ground-truth supervision, analogous to
         MDN's own NLL, giving gradient to logits/mu/std alike.
       - surrogate_spectrum_loss: reparameterize-sample ``num_surrogate_
         samples`` designs from q(.|spectrum) (each from an independently
         drawn mixture component), run each through EVERY frozen surrogate
         in the ensemble, and score the resulting predicted ERP spectra
         against the TRUE target spectrum using the SAME
         ``erp_spectrum_loss`` every forward operator in this repo trains
         against (MSE + a first-difference/slope penalty + a multi-peak
         value penalty summed over every one of the true spectrum's
         resonance peaks, not plain MSE) -- an ERP-space consistency
         signal ground-truth design MSE alone can't give (two different
         designs producing the same spectrum are equally right under this
         term, unlike raw design MSE). Averaged over both the sample draws
         and the surrogate ensemble.
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

from erp_forward_operators.dco import build_model as _build_dco
from erp_forward_operators.gno import build_model as _build_gno
from erp_forward_operators.neural_operator_utils import MLP, erp_spectrum_loss
from utils.physics import Lx, Ly, edge_margin, fmax, fmin, m_max, m_min

from .common import SpectrumEncoder, flatten_configuration

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
# already carries (see erp_forward_operators.neural_operator_utils.
# save_operator_checkpoint), so a new architecture only needs one new entry.
_SURROGATE_BUILDERS = {"dco": _build_dco, "gno": _build_gno}

DEFAULT_SURROGATE_CHECKPOINTS = (
    "erp_forward_operators/models/dco_erp.pth",
    "erp_forward_operators/models/gno_erp.pth",
)


def _soft_clamp(x: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    """Smooth alternative to ``torch.clamp`` with a much wider live-gradient region.

    ``mid + half_range * tanh((x - mid) / half_range)`` maps all of R into
    the open interval ``(lo, hi)``. At ``x = mid`` this is exactly the
    identity with slope 1 (``tanh`` is linear near 0), so values already
    well inside the range pass through essentially unchanged; moving away
    from ``mid`` it saturates smoothly toward ``lo``/``hi`` instead of
    hard-clipping. Unlike ``torch.clamp``, whose gradient is exactly 0 the
    instant ``x`` leaves ``[lo, hi]``, this one's gradient only shrinks --
    an input several half-ranges past the boundary (empirically, up to
    ~8x) still gets a real, if smaller, corrective push back toward the
    trusted region. In float32, ``tanh`` itself saturates to exactly 1.0
    for large enough arguments, so truly extreme outliers (~10x+ the
    half-range out) do round to a zero gradient too -- the same standard
    precision limit any bounded squashing function hits, not a flaw
    specific to this use -- but that dead zone sits many multiples of the
    range's width farther out than a hard clamp's immediate one.
    """
    mid = (lo + hi) / 2
    half_range = (hi - lo) / 2
    return mid + half_range * torch.tanh((x - mid) / half_range)


def _mixture_log_prob(flat: torch.Tensor, logits: torch.Tensor, mu: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Same convention/formula as mdn.py's ``MDN._mixture_log_prob``.

    ``flat``: (B, D) or (B, S, D); ``logits``: (B, K); ``mu``/``std``: (B, K, D).
    """
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


class SurrogateInverse(nn.Module):
    def __init__(
        self,
        design_dim: int,
        surrogate_checkpoints: Sequence[str] = DEFAULT_SURROGATE_CHECKPOINTS,
        num_components: int = 4,
        embed_dim: int = 96,
        hidden: int = 128,
        min_std: float = 1e-3,
    ) -> None:
        super().__init__()
        self.design_dim = int(design_dim)
        if self.design_dim % 5 != 0:
            raise ValueError("design_dim must be num_res*5 ([m,k,f_t,x,y] per resonator).")
        self.num_res = self.design_dim // 5
        self.num_components = int(num_components)
        self.min_std = float(min_std)

        self.spectrum_encoder = SpectrumEncoder(embed_dim=embed_dim)
        out_dim = self.num_components * (1 + 2 * self.design_dim)
        self.head = MLP([embed_dim, hidden, hidden, out_dim], activation=nn.SiLU)

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
            if int(surrogate_norm["num_res"]) != self.num_res:
                raise ValueError(
                    f"Surrogate checkpoint {checkpoint_path} was trained with num_res="
                    f"{surrogate_norm['num_res']}, but design_dim={design_dim} implies num_res={self.num_res}."
                )
            surrogate = _SURROGATE_BUILDERS[operator_name](num_res=self.num_res, **checkpoint["model_config"])
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
    # Posterior q(design | spectrum): mixture of Gaussians (mdn.py convention)
    # ---------------------------------------------------------

    def _params(self, spectrum: torch.Tensor):
        embedding = self.spectrum_encoder(spectrum)
        raw = self.head(embedding)
        k, d = self.num_components, self.design_dim
        logits = raw[:, :k]
        mu = raw[:, k : k + k * d].view(-1, k, d)
        log_std = raw[:, k + k * d :].view(-1, k, d)
        std = nn.functional.softplus(log_std) + self.min_std
        return logits, mu, std

    @staticmethod
    def _reparameterized_samples(
        logits: torch.Tensor, mu: torch.Tensor, std: torch.Tensor, num_draws: int
    ) -> torch.Tensor:
        """Draw ``num_draws`` reparameterized, mixture-weighted samples per
        batch element. Which COMPONENT is used per draw is a non-differentiable
        categorical choice (same as MDN's own ``sample()``) -- the mixture
        weights still get gradient through ``design_nll``'s exact log-density,
        not through this sampling path; only mu/std of the chosen component
        need to be (and are) differentiable here, via the reparameterization
        trick. Returns flat ``(B*num_draws, design_dim)``.
        """
        b, k, d = mu.shape
        weights = torch.softmax(logits, dim=-1)
        component = torch.multinomial(weights, num_draws, replacement=True)  # (B, num_draws)
        mu_s = torch.gather(mu, 1, component[:, :, None].expand(-1, -1, d))
        std_s = torch.gather(std, 1, component[:, :, None].expand(-1, -1, d))
        samples = mu_s + std_s * torch.randn_like(std_s)  # (B, num_draws, D)
        return samples.reshape(b * num_draws, d)

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
        ``design_norm`` comes from a reparameterized Gaussian sample
        (``_reparameterized_samples``), which has unbounded support -- early
        in training especially, a sample can land far outside every design
        field's physical range, querying the frozen surrogates on inputs
        wildly outside what they were ever trained on. Their output there is
        meaningless (not "wrong physics", just extrapolation noise), which
        otherwise pollutes the consistency-loss gradient with signal that has
        nothing to do with the actual posterior quality.

        A hard ``torch.clamp`` would fix the input range but at the cost of
        an exactly-zero gradient outside the bounds -- once a sample is
        clamped, the surrogate-consistency term gives the posterior no
        signal at all to pull it back, throwing away exactly the information
        ("how far out of range, and in which direction") that would help it
        recover fastest. ``_soft_clamp`` uses a tanh squash instead: it's
        identity (slope exactly 1) at the range's center, so in-range values
        pass through essentially unchanged, and it saturates smoothly toward
        the bounds for outliers. See ``_soft_clamp``'s own docstring for the
        gradient-shape tradeoff versus a hard ``torch.clamp``.
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
        configuration_physical = _soft_clamp(
            configuration_physical, self.design_physical_min, self.design_physical_max
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
    # Training / sampling
    # ---------------------------------------------------------

    def training_loss(
        self,
        spectrum: torch.Tensor,
        design: torch.Tensor,
        own_norm_params: Mapping[str, object],
        surrogate_weight: float = 1.0,
        slope_weight: float = 0.5,
        peak_weight: float = 0.05,
        num_surrogate_samples: int = 4,
    ) -> torch.Tensor:
        flat_design = flatten_configuration(design)
        logits, mu, std = self._params(spectrum)

        design_nll = -_mixture_log_prob(flat_design, logits, mu, std).mean()

        b = spectrum.shape[0]
        design_samples = self._reparameterized_samples(logits, mu, std, num_surrogate_samples)  # (B*M, D)
        predicted = self._surrogate_predicted_erp(design_samples, own_norm_params)  # (S, B*M, n_freq)

        spectrum_expanded = (
            spectrum[:, None, :].expand(-1, num_surrogate_samples, -1).reshape(b * num_surrogate_samples, -1)
        )
        target = spectrum_expanded[..., None]  # (B*M, n_freq, 1) -- erp_spectrum_loss expects a trailing dim
        surrogate_losses = [
            erp_spectrum_loss(predicted[s][..., None], target, slope_weight=slope_weight, peak_weight=peak_weight)
            for s in range(self.num_surrogates)
        ]
        surrogate_loss = torch.stack(surrogate_losses).mean()

        return design_nll + float(surrogate_weight) * surrogate_loss

    @torch.no_grad()
    def sample(self, spectrum: torch.Tensor, num_samples: int = 1):
        """Returns ``(flat_designs, log_prob)``, both ``(B, num_samples, ...)``,
        matching MDN/cVAE/Flow's convention -- the posterior here is a
        mixture of Gaussians (same family as MDN), so its exact mixture
        log-density is cheap and reported for every returned sample, not
        just the sampled component's.
        """
        logits, mu, std = self._params(spectrum)
        b, k, d = mu.shape
        weights = torch.softmax(logits, dim=-1)
        component = torch.multinomial(weights, num_samples, replacement=True)  # (B, S)
        mu_s = torch.gather(mu, 1, component[:, :, None].expand(-1, -1, d))
        std_s = torch.gather(std, 1, component[:, :, None].expand(-1, -1, d))
        flat = mu_s + std_s * torch.randn_like(std_s)  # (B, S, D)
        log_prob = _mixture_log_prob(flat, logits, mu, std)  # (B, S)
        return flat, log_prob


def _load_frequency_grid():
    from utils.physics import freqs

    return freqs.astype("float32")
