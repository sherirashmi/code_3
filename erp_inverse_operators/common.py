"""Shared building blocks for probabilistic inverse-design models.

Every inverse model here answers the same question: given a target ERP
spectrum, what resonator configuration ([m,k,f_t,x,y] per resonator) would
produce it? The forward problem (config -> spectrum) is what erp_forward_operators/
already solves; every model in this package solves the reverse direction,
and does it *probabilistically* -- returning a distribution over plausible
designs rather than one point estimate, since the forward mapping is
generally many-to-one (different configurations can produce very similar
spectra, e.g. the mass/stiffness coupling-strength degeneracy explored
earlier in this project), which makes the inverse direction ill-posed for
any single-point regression.

Design representation
----------------------
Every configuration is ``(num_res, 5)`` in ``[m,k,f_t,x,y]`` order, exactly
like the forward operators. Resonators within a configuration have no
canonical order in the raw dataset (they were sampled independently per
slot), which introduces a spurious combinatorial ambiguity on top of the
genuine physical one -- swapping resonator labels doesn't change the
spectrum at all. To keep the generative models from having to learn that
irrelevant symmetry, every configuration is canonicalized by sorting its
resonators by ascending tuning frequency (f_t) before use as a training
target. Models predict/sample a flat ``num_res*5``-dim vector in this
canonical, normalized (z-scored) order; :func:`unflatten_configuration`
converts back to ``(num_res, 5)`` and ``denormalize_configuration_array``
(from utils.erp_dataset) converts back to physical units.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from utils.erp_dataset import (
    denormalize_configuration_array,
    normalize_configuration_array,
    normalize_erp_array,
)
from erp_forward_operators.neural_operator_utils import (
    MLP,
    _configuration_features,
    _split_ids,
    prepare_operator_data,
)
from utils.support import device

from erp_inverse_operators.design_space import (
    BOUNDED12,
    FULL15,
    decode_bounded,
    encode_bounded,
    fit_bounded_stats,
)


def canonicalize_by_ft(configuration: np.ndarray) -> np.ndarray:
    """Sort resonators within each configuration by ascending raw f_t (index 2).

    Removes the spurious resonator-label permutation ambiguity (swapping two
    resonators doesn't change the spectrum) so the generative models only
    have to represent the genuine physical non-uniqueness, not an arbitrary
    labeling artifact of how the dataset was sampled.
    """
    configuration = np.asarray(configuration)
    order = np.argsort(configuration[..., 2], axis=-1)
    return np.take_along_axis(configuration, order[..., None], axis=-2)


def flatten_configuration(configuration: torch.Tensor) -> torch.Tensor:
    """``(..., num_res, 5) -> (..., num_res*5)``."""
    return configuration.reshape(*configuration.shape[:-2], -1)


def unflatten_configuration(flat: torch.Tensor, num_res: int) -> torch.Tensor:
    """``(..., num_res*5) -> (..., num_res, 5)``."""
    return flat.reshape(*flat.shape[:-1], num_res, 5)


POOLED = "pooled"
POSITIONAL = "positional"
SPECTRUM_ENCODERS = (POOLED, POSITIONAL)


class SpectrumEncoder(nn.Module):
    """Encodes a full normalized ERP spectrum into a fixed-size embedding.

    Every inverse model conditions on this same embedding.

    ``mode="pooled"`` (default, what all existing checkpoints use): a small
    Conv1d stack over frequency followed by global mean+max pooling. The
    receptive field is only ~13 bins (~6 Hz) and the global pooling keeps
    HOW STRONG each feature is but not WHERE along the frequency axis it
    occurred -- yet peak/notch locations are exactly what identify the
    resonators' f_t.

    ``mode="positional"``: keeps frequency position. The normalised
    frequency coordinate is fed as a second input channel (an explicit
    positional encoding), the conv features are average-pooled into
    ``num_bins`` ordered frequency bins and FLATTENED (bin order retained),
    and concatenated with the global mean/max summary before the head.
    """

    def __init__(
        self,
        width: int = 48,
        depth: int = 3,
        embed_dim: int = 96,
        mode: str = POOLED,
        num_bins: int = 16,
    ) -> None:
        super().__init__()
        if mode not in SPECTRUM_ENCODERS:
            raise ValueError(f"mode must be one of {SPECTRUM_ENCODERS}, got {mode!r}.")
        self.mode = mode
        in_channels = 2 if mode == POSITIONAL else 1
        layers: list[nn.Module] = [nn.Conv1d(in_channels, width, kernel_size=5, padding=2), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Conv1d(width, width, kernel_size=5, padding=2), nn.SiLU()]
        self.conv = nn.Sequential(*layers)
        head_in = 2 * width
        if mode == POSITIONAL:
            self.num_bins = int(num_bins)
            self.bin_pool = nn.AdaptiveAvgPool1d(self.num_bins)
            # Small per-bin projection keeps the flattened vector compact.
            self.bin_proj = nn.Conv1d(width, 8, kernel_size=1)
            head_in += 8 * self.num_bins
        self.head = MLP([head_in, embed_dim, embed_dim], activation=nn.SiLU)
        self.embed_dim = embed_dim

    def forward(self, spectrum: torch.Tensor) -> torch.Tensor:
        # spectrum: (B, n_freq) normalized ERP values.
        x = spectrum[:, None, :]
        if self.mode == POSITIONAL:
            position = torch.linspace(-1.0, 1.0, spectrum.shape[-1], device=spectrum.device, dtype=spectrum.dtype)
            x = torch.cat((x, position[None, None, :].expand(spectrum.shape[0], 1, -1)), dim=1)
        x = self.conv(x)  # (B, width, n_freq)
        pooled = torch.cat((x.mean(dim=-1), x.amax(dim=-1)), dim=-1)
        if self.mode == POSITIONAL:
            binned = self.bin_proj(self.bin_pool(x)).flatten(1)  # (B, 8*num_bins), ordered by frequency
            pooled = torch.cat((pooled, binned), dim=-1)
        return self.head(pooled)


class InverseDesignDataset(Dataset):
    """One item = (normalized ERP spectrum, normalized canonical design).

    The design is ``(num_res, 5)`` z-scored [m,k,f_t,x,y] for ``full15`` or
    ``(num_res, 4)`` bounded-logit [m,f_t,x,y] for ``bounded12`` (see
    design_space.py); both sorted by ascending f_t.
    """

    def __init__(self, dataset, configuration_ids: np.ndarray, design_param: str = FULL15) -> None:
        ids = np.asarray(configuration_ids, dtype=np.int64)
        norm = dataset.norm_params
        raw = canonicalize_by_ft(_configuration_features(dataset)[ids])
        configuration = normalize_design_physical(raw, norm, design_param)
        response = normalize_erp_array(
            np.asarray(dataset.responses, dtype=np.float32)[ids, :, 0], norm
        )
        self.configuration = torch.from_numpy(configuration.astype(np.float32))
        self.spectrum = torch.from_numpy(response.astype(np.float32))

    def __len__(self) -> int:
        return self.configuration.shape[0]

    def __getitem__(self, index: int):
        return self.spectrum[index], self.configuration[index]


def prepare_inverse_data(
    num_configurations: int = 10000,
    batch_size: int = 32,
    dataset_file: str | list[str] | tuple[str, ...] = "datasets/dataset_erp_ft.pth",
    seed: int = 727,
    design_param: str = FULL15,
):
    """Build spectrum->design train/val/test loaders sharing the forward split.

    ``design_param`` selects the design representation (design_space.py).
    The bounded-logit statistics are always fitted on the training split and
    stored in ``dataset.norm_params`` (so every checkpoint carries them).
    """
    dataset, _ = prepare_operator_data(
        num_configurations=num_configurations,
        batch_size=16,
        dataset_file=dataset_file,
        regenerate_dataset=False,
        seed=seed,
    )
    splits = _split_ids(dataset)
    train_raw = canonicalize_by_ft(_configuration_features(dataset)[splits["train"]])
    dataset.norm_params.update(fit_bounded_stats(train_raw))
    dataset.design_param = design_param
    pin_memory = device.type == "cuda"
    loaders = {
        name: DataLoader(
            InverseDesignDataset(dataset, ids, design_param),
            batch_size=batch_size,
            shuffle=(name == "train"),
            pin_memory=pin_memory,
        )
        for name, ids in splits.items()
    }
    return dataset, loaders


def enforce_physical_consistency(configuration: np.ndarray) -> np.ndarray:
    """Project predicted physical ``(..., num_res, 5)`` designs onto the
    feasible, self-consistent design space the dataset was generated in.

    Every model predicts ``m``, ``k`` and ``f_t`` as three independent
    outputs, but the solver only uses ``m`` and ``k`` -- so an unprojected
    prediction resonates at ``sqrt(k/m)/(2*pi)``, generally NOT at the
    predicted ``f_t``, and its solver-validated spectrum misses the target's
    peaks even when ``f_t`` itself was predicted well. ``f_t`` is the
    best-identified quantity (it sets the peak/notch frequency directly), so
    it is kept and the stiffness is re-derived exactly as at generation time,
    ``k = m*(2*pi*f_t)**2``. ``m``, ``f_t``, ``x``, ``y`` are clipped to the
    generation bounds (the solver is meaningless for negative masses or
    resonators off the plate).
    """
    from utils.physics import Lx, Ly, edge_margin, fmax, fmin, m_max, m_min

    out = np.array(configuration, dtype=np.float64, copy=True)
    out[..., 0] = np.clip(out[..., 0], m_min, m_max)
    out[..., 2] = np.clip(out[..., 2], fmin, fmax)
    out[..., 3] = np.clip(out[..., 3], edge_margin, Lx - edge_margin)
    out[..., 4] = np.clip(out[..., 4], edge_margin, Ly - edge_margin)
    out[..., 1] = out[..., 0] * (2.0 * np.pi * out[..., 2]) ** 2
    return out.astype(np.float32)


def design_param_of(flat_design: np.ndarray | torch.Tensor, num_res: int) -> str:
    """Representation of FLAT design vector(s): 12 numbers -> bounded12, else full15."""
    return BOUNDED12 if int(flat_design.shape[-1]) == 4 * num_res else FULL15


def denormalize_design(
    flat_design: np.ndarray,
    num_res: int,
    norm_params: Mapping[str, object],
    *,
    consistent: bool = True,
) -> np.ndarray:
    """Normalized FLAT design(s) ``(..., D)`` -> physical ``(..., num_res, 5)`` [m,k,f_t,x,y].

    ``full15`` (D = 15): z-score inverse; ``consistent=True`` (default) then
    applies :func:`enforce_physical_consistency` (k re-derived, bounds
    clipped) so the solver can evaluate it. ``bounded12`` (D = 12): exact
    sigmoid decode -- always in bounds with k derived, nothing to project.
    """
    flat_design = np.asarray(flat_design)
    if design_param_of(flat_design, num_res) == BOUNDED12:
        return decode_bounded(flat_design, num_res, norm_params)
    configuration = flat_design.reshape(*flat_design.shape[:-1], num_res, 5)
    physical = denormalize_configuration_array(configuration, norm_params)
    return enforce_physical_consistency(physical) if consistent else physical


def normalize_design_physical(
    physical: np.ndarray, norm_params: Mapping[str, object], design_param: str = FULL15
) -> np.ndarray:
    """Physical ``(..., num_res, 5)`` -> normalized design ``(..., num_res, 5|4)``."""
    if design_param == BOUNDED12:
        return encode_bounded(physical, norm_params)
    return normalize_configuration_array(physical, norm_params)


def sort_resonators_by_ft(physical: np.ndarray) -> np.ndarray:
    """Physical ``(..., num_res, 5)`` designs with resonators in ascending f_t
    order -- needed before comparing individual resonator parameters (the
    spectrum, and so the solver score, does not depend on the order)."""
    return canonicalize_by_ft(physical)


def save_checkpoint(
    model: nn.Module,
    norm_params: Mapping[str, object],
    filename: str,
    extra: Mapping[str, object] | None = None,
) -> None:
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model_state_dict": model.state_dict(), "norm_params": dict(norm_params)}
    if extra:
        payload.update(dict(extra))
    torch.save(payload, path)
    print(f"Checkpoint saved: {path}")
