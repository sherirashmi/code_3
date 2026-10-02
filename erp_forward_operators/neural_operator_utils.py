"""
==================================================
Project     : Vibro-Acoustic Metamaterials
Module      : Neural Operator Utilities
Description : Shared ERP neural-operator training/evaluation workflow
==================================================

All operator models use the same convention:

    configuration : (batch, num_res, 5) normalized [m, k, f_t, x, y]
    frequency     : (batch, n_freq, 1) normalized evaluation frequencies
    output        : (batch, n_freq, 1) normalized ERP

Dataset preprocessing is owned by ``erp_dataset.py``. Plot construction and
file output are owned by ``plotting.py``. This module owns reusable network
blocks, full-spectrum loaders, training/evaluation and checkpoint workflows.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import utils.physics as _physics
from utils.erp_dataset import (
    DEFAULT_DATASET_FILE,
    apply_model_modal_resolution,
    configuration_to_resonators,
    dataset_tag_for,
    denormalize_configuration_array,
    denormalize_erp_array,
    normalize_configuration_array,
    normalize_erp_array,
    normalize_frequency_array,
    prepare_erp_dataset,
)
from utils.physics import freqs, num_res as default_num_res
from utils.plotting import (
    operator_plot_dir,
    plot_erp_comparison,
    plot_loss_curves,
    plot_prediction_scatter,
)
from utils.paths import forward_model_path
from utils.solver import compute_erp_spectrum
from utils.support import device, seed_everything


# ==================================================
# Small reusable network blocks
# ==================================================


# Registry of activation choices every operator's own backbone can select
# from (the shared ResonatorSetEncoder/ResonanceQueryEncoder/
# FrequencyRefinement1d blocks keep their own fixed activations -- those are
# common "same information" tooling every architecture is equalized to have,
# not part of what makes one architecture different from another).
ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
    "mish": nn.Mish,
}


def resolve_activation(activation: str | type[nn.Module]) -> type[nn.Module]:
    """Resolve an activation choice to its ``nn.Module`` subclass.

    Accepts either a name registered in ``ACTIVATIONS`` (the form stored in
    ``DEFAULT_MODEL_CONFIG`` so checkpoints stay plain-data serializable) or
    an ``nn.Module`` subclass passed straight through, so existing call sites
    that already pass a class (e.g. ``nn.SiLU``) keep working unchanged.
    """
    if isinstance(activation, str):
        key = activation.lower().strip()
        if key not in ACTIVATIONS:
            raise ValueError(
                f"Unknown activation '{activation}'. Choose one of: {sorted(ACTIVATIONS)}"
            )
        return ACTIVATIONS[key]
    if isinstance(activation, type) and issubclass(activation, nn.Module):
        return activation
    raise TypeError("activation must be a registered name or an nn.Module subclass.")


class MLP(nn.Module):
    """Compact fully connected network."""

    def __init__(
        self,
        widths: Sequence[int],
        activation: type[nn.Module] = nn.Tanh,
        final_activation: nn.Module | None = None,
    ) -> None:
        super().__init__()
        widths = [int(v) for v in widths]
        if len(widths) < 2 or any(v <= 0 for v in widths):
            raise ValueError("widths must contain at least two positive integers.")

        layers: list[nn.Module] = []
        for i, (din, dout) in enumerate(zip(widths[:-1], widths[1:], strict=True)):
            layers.append(nn.Linear(din, dout))
            if i < len(widths) - 2:
                layers.append(activation())
        if final_activation is not None:
            layers.append(final_activation)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualMLPBlock(nn.Module):
    """Two-layer residual MLP block."""

    def __init__(self, width: int, activation: type[nn.Module] = nn.Tanh) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(width, width),
            activation(),
            nn.Linear(width, width),
        )
        self.norm = nn.LayerNorm(width)
        self.activation = activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.norm(x + self.net(x)))


class FrequencyRefinement1d(nn.Module):
    """Shared residual local frequency mixer used to sharpen ERP peaks.

    Every operator uses this exact module for its final cross-frequency
    mixing stage, so no architecture gets a stronger or weaker "peak
    sharpening" tool than any other purely as an implementation accident.
    Operates on ``(batch, width, n_freq)`` feature maps.
    """

    def __init__(self, width: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(width, width, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(width, width, kernel_size=3, padding=1),
        )
        self.norm = nn.GroupNorm(1, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(self.norm(x + self.net(x)))


def physics_aware_resonator_features(
    configuration: torch.Tensor,
    harmonics: int = 10,
    coordinate_affine: torch.Tensor | None = None,
) -> torch.Tensor:
    """Augment normalized ``[m, k, f_t, x, y]`` with plate-inspired sine features.

    ``coordinate_affine=None`` (the default, ``coordinate_features="zscored"``)
    applies ``sin(i*pi*x)`` / ``sin(j*pi*y)`` directly to the z-scored
    coordinates. Those are centred on the plate middle, so every such
    feature is odd about the centre and none of them can represent the
    physical modes with odd index (``sin(m*pi*x/Lx)``, m = 1, 3, 5, ... are
    even about the centre). It is kept as the default because every
    existing checkpoint was trained with it.

    ``coordinate_affine`` (shape ``(2, 2)``, rows ``[scale, shift]`` for x
    and y; see :func:`enable_physical_features`) first maps the z-scored
    coordinates to ``x/Lx`` and ``y/Ly`` in ``[0, 1]``, so the harmonics are
    exactly the plate's own simply-supported mode shapes
    ``sin(m*pi*x/Lx)``, ``sin(n*pi*y/Ly)``. Low-order tensor-product
    terms are included so the network does not have to rediscover the
    dominant modal spatial interactions from raw coordinates. ``m`` and ``k``
    are passed through unaugmented (mass ratio / coupling-strength
    information, not spatial), alongside ``f_t`` which every architecture
    also uses directly for resonance-detuning features.
    """
    if configuration.ndim < 2 or configuration.shape[-1] != 5:
        raise ValueError("configuration must end with normalized [m, k, f_t, x, y].")
    if harmonics < 0:
        raise ValueError("harmonics cannot be negative.")

    if harmonics == 0:
        return configuration

    m = configuration[..., 0:1]
    k = configuration[..., 1:2]
    f_t = configuration[..., 2:3]
    x = configuration[..., 3:4]
    y = configuration[..., 4:5]

    u, v = x, y
    if coordinate_affine is not None:
        u = x * coordinate_affine[0, 0] + coordinate_affine[0, 1]  # x / Lx
        v = y * coordinate_affine[1, 0] + coordinate_affine[1, 1]  # y / Ly
    x_modes = [torch.sin(math.pi * float(i) * u) for i in range(1, harmonics + 1)]
    y_modes = [torch.sin(math.pi * float(j) * v) for j in range(1, harmonics + 1)]
    products = [xm * yn for xm in x_modes for yn in y_modes]
    return torch.cat([m, k, f_t, x, y, *x_modes, *y_modes, *products], dim=-1)


# ==================================================
# Optional physical feature scaling (coordinate_features="physical")
# ==================================================
#
# Modules that build modal features set ``uses_modal_features = True``;
# modules that build query-resonator detuning set ``uses_detuning = True``.
# By default they behave exactly as before (and have no extra buffers, so
# existing checkpoints load unchanged). ``enable_physical_features`` gives
# them buffers that map the z-scored inputs to physical scales:
#
# * ``coord_affine``     z-scored x, y  ->  x/Lx, y/Ly
# * ``detuning_affine``  z-scored f_t   ->  f_t on the query-frequency scale,
#   so ``query - f_t`` is exactly ``(f - f_t[Hz]) / freq_std``: one scale for
#   every resonator, instead of the difference of two separately z-scored
#   quantities.
#
# The buffer values come from the training data's norm_params
# (``set_physical_feature_normalization``) and are saved in the state dict.

COORDINATE_FEATURE_MODES = ("zscored", "physical")


def modal_features(module: nn.Module, configuration: torch.Tensor, harmonics: int) -> torch.Tensor:
    """``physics_aware_resonator_features`` using ``module``'s coordinate mode."""
    return physics_aware_resonator_features(
        configuration,
        harmonics=harmonics,
        coordinate_affine=getattr(module, "coord_affine", None),
    )


def resonance_detuning(module: nn.Module, query: torch.Tensor, f_t: torch.Tensor) -> torch.Tensor:
    """Normalized detuning ``query - f_t`` using ``module``'s detuning scale."""
    affine = getattr(module, "detuning_affine", None)
    if affine is None:
        return query - f_t
    return query - (f_t * affine[0] + affine[1])


def enable_physical_features(model: nn.Module) -> nn.Module:
    """Switch every feature-building submodule of ``model`` to physical scales.

    Registers identity-initialised buffers; their real values are set by
    :func:`set_physical_feature_normalization` (training) or come from the
    checkpoint's state dict (loading).
    """
    for module in model.modules():
        if getattr(module, "uses_modal_features", False) and not hasattr(module, "coord_affine"):
            module.register_buffer("coord_affine", torch.tensor([[1.0, 0.0], [1.0, 0.0]]))
        if getattr(module, "uses_detuning", False) and not hasattr(module, "detuning_affine"):
            module.register_buffer("detuning_affine", torch.tensor([1.0, 0.0]))
    return model


def set_physical_feature_normalization(model: nn.Module, norm_params: Mapping[str, object]) -> None:
    """Fill the buffers of :func:`enable_physical_features` from ``norm_params``."""
    lx, ly = float(_physics.Lx), float(_physics.Ly)
    coord = torch.tensor(
        [
            [float(norm_params["x_std"]) / lx, float(norm_params["x_mean"]) / lx],
            [float(norm_params["y_std"]) / ly, float(norm_params["y_mean"]) / ly],
        ]
    )
    freq_std = float(norm_params["freq_std"])
    detuning = torch.tensor(
        [
            float(norm_params["f_t_std"]) / freq_std,
            (float(norm_params["f_t_mean"]) - float(norm_params["freq_mean"])) / freq_std,
        ]
    )
    for module in model.modules():
        if hasattr(module, "coord_affine"):
            module.coord_affine.copy_(coord.to(module.coord_affine))
        if hasattr(module, "detuning_affine"):
            module.detuning_affine.copy_(detuning.to(module.detuning_affine))


# model_config keys that are not constructor arguments of any architecture.
BUILD_OPTION_KEYS = ("coordinate_features", "permutation_augment")


def build_operator_model(
    build_model: Callable[..., nn.Module],
    num_res: int,
    model_config: Mapping[str, object],
    norm_params: Mapping[str, object] | None = None,
) -> nn.Module:
    """Build a forward operator from a (checkpoint) ``model_config``.

    Strips the build options in ``BUILD_OPTION_KEYS`` before calling the
    architecture's constructor and applies ``coordinate_features``. Pass
    ``norm_params`` when building a fresh model for training; when loading a
    checkpoint the physical buffers come from its state dict instead.
    """
    config = dict(model_config)
    options = {key: config.pop(key) for key in BUILD_OPTION_KEYS if key in config}
    model = build_model(num_res=num_res, **config)
    mode = str(options.get("coordinate_features", "zscored"))
    if mode not in COORDINATE_FEATURE_MODES:
        raise ValueError(f"coordinate_features must be one of {COORDINATE_FEATURE_MODES}, got {mode!r}.")
    if mode == "physical":
        enable_physical_features(model)
        if norm_params is not None:
            set_physical_feature_normalization(model, norm_params)
    model.permutation_augment = bool(options.get("permutation_augment", False))
    return model


class ResonatorSetEncoder(nn.Module):
    """Permutation-invariant, physics-aware encoder for resonator sets.

    Every resonator is embedded with shared weights, using normalized raw
    ``[m,k,f_t,x,y]`` plus low-order plate-inspired spatial sine features.
    Mean and max pooling preserve permutation invariance while retaining both
    distributed and dominant-resonator information.
    """

    uses_modal_features = True

    def __init__(
        self,
        feature_dim: int = 5,
        hidden_dim: int = 128,
        element_dim: int = 128,
        output_dim: int = 128,
        modal_harmonics: int = 10,
    ) -> None:
        super().__init__()
        if int(feature_dim) != 5:
            raise ValueError("ResonatorSetEncoder expects [m,k,f_t,x,y] feature_dim=5.")
        self.modal_harmonics = int(modal_harmonics)
        augmented_dim = 5 + 2 * self.modal_harmonics + self.modal_harmonics**2
        self.element_net = MLP(
            [augmented_dim, hidden_dim, hidden_dim, element_dim],
            activation=nn.Tanh,
        )
        self.fusion_net = MLP(
            [2 * element_dim, hidden_dim, output_dim],
            activation=nn.Tanh,
        )

    def forward(self, configuration: torch.Tensor) -> torch.Tensor:
        if configuration.ndim != 3 or configuration.shape[-1] != 5:
            raise ValueError("configuration must have shape (batch, num_res, 5).")
        features = modal_features(self, configuration, self.modal_harmonics)
        h = self.element_net(features)
        pooled = torch.cat((h.mean(dim=1), h.max(dim=1).values), dim=-1)
        return self.fusion_net(pooled)


class SortedResonatorEncoder(nn.Module):
    """Lossless, order-canonicalized counterpart to ``ResonatorSetEncoder``.

    ``ResonatorSetEncoder`` gets permutation invariance from mean+max
    pooling: invariant and smooth everywhere, but not injective -- distinct
    resonator sets can collide onto the same pooled vector (mean is a
    many-to-one average over the 3 resonators; per-channel max keeps each
    channel's winning value but not which resonator won which channel, so
    two different triples can share an identical max vector).

    This encoder gets permutation invariance the opposite way: canonicalize
    resonator order by sorting on ascending ``f_t`` (the same convention
    ``erp_inverse_operators.common.canonicalize_by_ft`` uses for its
    targets), then keep every resonator's full augmented feature vector,
    concatenated rather than pooled. Nothing is discarded, so it can't
    collide the way pooling can -- at the cost of a new failure mode pooling
    doesn't have: right where two resonators' ``f_t`` values cross, which
    one lands in which slot flips, so the flattened feature vector jumps
    discontinuously even though the true spectrum varies smoothly through
    that crossing. That's exactly the region (near-degenerate resonators)
    where this dataset is already hardest, so this encoder is meant to
    augment ``ResonatorSetEncoder``, not replace it: concatenate both
    branches so the smooth pooled summary is always available as a
    fallback, with this branch supplying the individual-resonator detail
    pooling throws away.
    """

    uses_modal_features = True

    def __init__(
        self,
        num_res: int,
        feature_dim: int = 5,
        hidden_dim: int = 128,
        output_dim: int = 128,
        modal_harmonics: int = 10,
    ) -> None:
        super().__init__()
        if int(feature_dim) != 5:
            raise ValueError("SortedResonatorEncoder expects [m,k,f_t,x,y] feature_dim=5.")
        self.num_res = int(num_res)
        self.modal_harmonics = int(modal_harmonics)
        augmented_dim = 5 + 2 * self.modal_harmonics + self.modal_harmonics**2
        self.net = MLP(
            [self.num_res * augmented_dim, hidden_dim, hidden_dim, output_dim],
            activation=nn.Tanh,
        )

    def forward(self, configuration: torch.Tensor) -> torch.Tensor:
        if configuration.ndim != 3 or configuration.shape[-1] != 5:
            raise ValueError("configuration must have shape (batch, num_res, 5).")
        order = torch.argsort(configuration[..., 2], dim=-1)
        sorted_configuration = torch.gather(
            configuration, dim=1, index=order[..., None].expand(-1, -1, 5)
        )
        features = modal_features(self, sorted_configuration, self.modal_harmonics)
        flat = features.reshape(features.shape[0], -1)
        return self.net(flat)


class SetAndSortedResonatorEncoder(nn.Module):
    """Pooled set branch + f_t-sorted branch, the DCO_sorted design as a
    reusable block (see ``SortedResonatorEncoder``'s docstring).

    Both branches see the same normalised ``[m, k, f_t, x, y]`` resonators:

    * ``set_encoder``: :class:`ResonatorSetEncoder`, shared per-resonator
      MLP + mean/max pooling -- permutation invariant and smooth, but lossy.
    * ``sorted_encoder``: :class:`SortedResonatorEncoder`, resonators sorted
      by ascending ``f_t`` and concatenated -- permutation invariant through
      the canonical order, lossless, but discontinuous where two ``f_t``
      cross.

    Their outputs are concatenated and linearly fused back to ``output_dim``,
    so this is a drop-in replacement for ``ResonatorSetEncoder`` with the
    same input/output shapes, ``(B, num_res, 5) -> (B, output_dim)``.
    """

    def __init__(
        self,
        num_res: int,
        feature_dim: int = 5,
        hidden_dim: int = 128,
        element_dim: int = 128,
        output_dim: int = 128,
        modal_harmonics: int = 10,
    ) -> None:
        super().__init__()
        self.set_encoder = ResonatorSetEncoder(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            element_dim=element_dim,
            output_dim=output_dim,
            modal_harmonics=modal_harmonics,
        )
        self.sorted_encoder = SortedResonatorEncoder(
            num_res=num_res,
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            modal_harmonics=modal_harmonics,
        )
        self.fusion = nn.Linear(2 * output_dim, output_dim)

    def forward(self, configuration: torch.Tensor) -> torch.Tensor:
        pooled = self.set_encoder(configuration)
        ordered = self.sorted_encoder(configuration)
        return self.fusion(torch.cat((pooled, ordered), dim=-1))


def build_resonator_encoder(
    *,
    use_sorted_branch: bool,
    num_res: int,
    hidden_dim: int = 128,
    element_dim: int = 128,
    output_dim: int = 128,
    modal_harmonics: int = 10,
) -> nn.Module:
    """Configuration encoder used by the set-encoder architectures.

    ``use_sorted_branch=False`` returns a plain :class:`ResonatorSetEncoder`
    (identical parameters/state-dict keys as before, so existing checkpoints
    load unchanged); ``True`` returns :class:`SetAndSortedResonatorEncoder`.
    """
    if use_sorted_branch:
        return SetAndSortedResonatorEncoder(
            num_res=num_res,
            hidden_dim=hidden_dim,
            element_dim=element_dim,
            output_dim=output_dim,
            modal_harmonics=modal_harmonics,
        )
    return ResonatorSetEncoder(
        hidden_dim=hidden_dim,
        element_dim=element_dim,
        output_dim=output_dim,
        modal_harmonics=modal_harmonics,
    )


class ResonanceQueryEncoder(nn.Module):
    """Permutation-invariant resonator/query interaction encoder.

    For every requested frequency, each resonator is combined with the query
    frequency and a normalized detuning proxy ``frequency - f_t``.  A shared
    element network followed by mean/max pooling keeps the result invariant to
    resonator ordering while exposing near-resonance information explicitly.

    By default frequency and resonator tuning frequency use the project's
    existing (separate) normalization rules, so the detuning is the
    difference of two z-scores; with ``coordinate_features="physical"`` it
    is ``(f - f_t) / freq_std`` exactly (see ``resonance_detuning``).
    """

    uses_modal_features = True
    uses_detuning = True

    def __init__(
        self,
        hidden_dim: int = 64,
        element_dim: int = 64,
        output_dim: int = 64,
        modal_harmonics: int = 10,
    ) -> None:
        super().__init__()
        self.modal_harmonics = int(modal_harmonics)
        resonator_dim = 5 + 2 * self.modal_harmonics + self.modal_harmonics**2
        interaction_dim = resonator_dim + 4  # f, delta, |delta|, delta^2
        self.element_net = MLP(
            [interaction_dim, hidden_dim, hidden_dim, element_dim],
            activation=nn.Tanh,
        )
        self.fusion_net = MLP(
            [2 * element_dim, hidden_dim, output_dim],
            activation=nn.Tanh,
        )

    def forward(
        self,
        configuration: torch.Tensor,
        frequency: torch.Tensor,
    ) -> torch.Tensor:
        if configuration.ndim != 3 or configuration.shape[-1] != 5:
            raise ValueError("configuration must have shape (batch, num_res, 5).")
        if frequency.ndim != 3 or frequency.shape[-1] != 1:
            raise ValueError("frequency must have shape (batch, n_freq, 1).")
        if configuration.shape[0] != frequency.shape[0]:
            raise ValueError("configuration and frequency batch sizes must match.")

        b, n, _ = configuration.shape
        f = frequency.shape[1]
        resonator = modal_features(self, configuration, self.modal_harmonics)
        resonator = resonator[:, None, :, :].expand(b, f, n, -1)

        query = frequency[:, :, None, :].expand(b, f, n, 1)
        f_t = configuration[:, None, :, 2:3].expand(b, f, n, 1)
        delta = resonance_detuning(self, query, f_t)
        pair = torch.cat((resonator, query, delta, delta.abs(), delta.square()), dim=-1)
        h = self.element_net(pair)
        pooled = torch.cat((h.mean(dim=2), h.max(dim=2).values), dim=-1)
        return self.fusion_net(pooled)


# ==================================================
# Dataset compatibility / full-spectrum loaders
# ==================================================


def _configuration_features(dataset) -> np.ndarray:
    """Read current configuration features, with legacy name support."""
    if hasattr(dataset, "configuration_features"):
        value = dataset.configuration_features
    elif hasattr(dataset, "design_features"):
        value = dataset.design_features
    else:
        raise AttributeError("ERP dataset has no configuration/design feature array.")
    return np.asarray(value, dtype=np.float32)


def _split_ids(dataset) -> dict[str, np.ndarray]:
    """Read current configuration split IDs, with legacy name support."""
    if hasattr(dataset, "split_configuration_ids"):
        value = dataset.split_configuration_ids
    elif hasattr(dataset, "split_design_ids"):
        value = dataset.split_design_ids
    else:
        value = None
    if value is None:
        raise RuntimeError("ERP dataset has not been split.")
    return {name: np.asarray(ids, dtype=np.int64) for name, ids in value.items()}


class ERPSpectrumDataset(Dataset):
    """One item = one complete resonator configuration and ERP spectrum."""

    def __init__(self, dataset, configuration_ids: Sequence[int]) -> None:
        if dataset.norm_params is None:
            raise RuntimeError("Dataset normalization must be available first.")

        ids = np.asarray(configuration_ids, dtype=np.int64)
        norm = dataset.norm_params

        configuration = normalize_configuration_array(
            _configuration_features(dataset)[ids], norm
        )
        frequency = normalize_frequency_array(dataset.frequency_values, norm)
        response = normalize_erp_array(
            np.asarray(dataset.responses, dtype=np.float32)[ids, :, 0], norm
        )

        self.configuration = torch.from_numpy(configuration.astype(np.float32))
        self.frequency = torch.from_numpy(frequency[:, None].astype(np.float32))
        self.response = torch.from_numpy(response[..., None].astype(np.float32))

    def __len__(self) -> int:
        return self.configuration.shape[0]

    def __getitem__(self, index: int):
        return self.configuration[index], self.frequency, self.response[index]


def build_spectrum_loaders(
    dataset,
    batch_size: int = 16,
    num_workers: int = 0,
    pin_memory: bool | None = None,
    seed: int = 727,
) -> dict[str, DataLoader]:
    """Create configuration-level loaders for full-spectrum operator training."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    splits = _split_ids(dataset)
    if pin_memory is None:
        pin_memory = device.type == "cuda"

    generator = torch.Generator()
    generator.manual_seed(seed)

    loaders: dict[str, DataLoader] = {}
    for name in ("train", "val", "test"):
        subset = ERPSpectrumDataset(dataset, splits[name])
        loaders[name] = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=(name == "train"),
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(num_workers > 0),
            generator=generator if name == "train" else None,
        )
    return loaders


def prepare_operator_data(
    num_configurations: int = 500,
    *,
    batch_size: int = 16,
    dataset_file: str | Sequence[str] = DEFAULT_DATASET_FILE,
    regenerate_dataset: bool = False,
    num_generate: int | None = None,
    num_res: int = default_num_res,
    seed: int = 727,
    preprocessing_state: Mapping[str, object] | None = None,
    num_workers: int = 0,
    verbose: bool = True,
):
    """Prepare the common ERP data and configuration-level spectrum loaders."""
    dataset, _ = prepare_erp_dataset(
        num_samples=num_configurations,
        batch_size=max(64, batch_size),
        num_res=num_res,
        dataset_file=dataset_file,
        regenerate_dataset=regenerate_dataset,
        num_generate=num_generate,
        seed=seed,
        preprocessing_state=preprocessing_state,
        num_workers=0,
        verbose=verbose,
    )
    loaders = build_spectrum_loaders(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
    )
    return dataset, loaders


# ==================================================
# Loss, training and evaluation
# ==================================================


def spectrum_peak_mask(spectrum: torch.Tensor, window: int = 7) -> torch.Tensor:
    """Boolean mask, ``(batch, n_freq, 1)``: True at every local-maximum
    frequency of ``spectrum`` -- ALL of that spectrum's resonance peaks
    (multiple resonators plus the bare-plate's own modes can each produce
    one), not just its single global maximum.

    Two stages, both vectorized (no per-spectrum Python loop, since
    ``target`` has no gradient here -- only indexing into ``prediction`` at
    the resulting positions needs one):

    1. A point is a *candidate* if it's strictly greater than both
       immediate neighbors (edges compared to their one real neighbor, via
       ``-inf`` padding rather than replicate-padding). Strict (``>``, not
       ``>=``) matters: a flat run of exactly-tied values -- e.g. a
       background region where several far-off resonance tails have
       underflowed to the same float32 value -- correctly yields NO
       candidate there, where a ``>=``-against-a-window test would mark
       the entire tied plateau as "peaks".
    2. Non-maximum suppression keeps only the locally dominant candidate
       within ``window`` bins, so numerical jitter that produces two
       candidates immediately next to one real resonance doesn't get
       double-counted, without needing exact-tie handling (two genuine
       candidate peak heights coinciding exactly is not a real concern for
       continuous physics values).

    The spectrum's global maximum is always included even if both stages
    above miss it (a perfectly monotonic spectrum still needs one peak
    term).
    """
    if spectrum.dim() != 3 or spectrum.shape[-1] != 1:
        raise ValueError("spectrum must have shape (batch, n_freq, 1).")
    if window < 1 or window % 2 == 0:
        raise ValueError("window must be a positive odd integer.")
    x = spectrum.detach().transpose(1, 2)  # (B, 1, n_freq); no grad needed for peak *locations*

    neg_inf = float("-inf")
    left = F.pad(x, (1, 0), mode="constant", value=neg_inf)[..., :-1]
    right = F.pad(x, (0, 1), mode="constant", value=neg_inf)[..., 1:]
    candidate = (x > left) & (x > right)

    candidate_values = torch.where(candidate, x, torch.full_like(x, neg_inf))
    pooled_candidates = F.max_pool1d(candidate_values, kernel_size=window, stride=1, padding=window // 2)
    dominant = candidate & (x >= pooled_candidates)

    mask = dominant.transpose(1, 2)  # (B, n_freq, 1)
    global_idx = spectrum.argmax(dim=1, keepdim=True)
    mask = mask.scatter(1, global_idx, True)
    return mask


def _peak_squared_error_sum(
    prediction: torch.Tensor, target: torch.Tensor, window: int = 7
) -> torch.Tensor:
    """Per-spectrum SUM (not mean) of squared error at every one of the
    TRUE spectrum's local-maximum frequencies -- see ``spectrum_peak_mask``.
    Returns ``(batch,)``; the number of frequencies summed over varies per
    spectrum with how many resonance peaks it actually has, so a spectrum
    with more distinguishable peaks contributes more to this term.
    """
    peak_mask = spectrum_peak_mask(target, window=window).to(prediction.dtype)
    squared_error = (prediction - target) ** 2
    return (squared_error * peak_mask).sum(dim=1).squeeze(-1)


def erp_spectrum_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    slope_weight: float = 0.5,
    peak_weight: float = 0.05,
    peak_window: int = 7,
) -> torch.Tensor:
    """Normalized ERP MSE plus a first-difference penalty and a multi-peak penalty.

    ``prediction``/``target``: ``(batch, n_freq, 1)``.

    The peak term reads off BOTH curves at the TRUE spectrum's own peak
    frequencies (every local maximum, via ``spectrum_peak_mask`` -- not
    just the single tallest one) and sums the squared error at each --
    NOT the model's own predicted peak locations, which would need a
    differentiable stand-in for argmax (e.g. soft-argmax) since
    ``torch.argmax``/local-max tests have zero gradient almost everywhere.
    Using the true peak locations is fully differentiable w.r.t. the
    prediction (a plain mask-and-sum, not an index-selection operation on
    the prediction itself) and directly targets the failure mode plain MSE
    allows: a model can reach a low average error by slightly smoothing
    over a sharp resonance peak, since a pointwise loss spreads that error
    thinly across many frequency bins instead of concentrating it where
    the peaks actually are. Summing (not averaging) over each spectrum's
    own peaks means a configuration with more resolvable resonances -- more
    terms in this sum -- contributes proportionally more peak-error, rather
    than every spectrum being treated as if it had exactly one peak to get
    right.
    """
    mse = F.mse_loss(prediction, target)
    loss = mse
    if slope_weight > 0.0 and prediction.shape[1] >= 2:
        dp = prediction[:, 1:] - prediction[:, :-1]
        dt = target[:, 1:] - target[:, :-1]
        loss = loss + float(slope_weight) * F.mse_loss(dp, dt)
    if peak_weight > 0.0:
        peak_loss = _peak_squared_error_sum(prediction, target, window=peak_window).mean()
        loss = loss + float(peak_weight) * peak_loss
    return loss


class UncertaintyWeightedERPLoss(nn.Module):
    """Homoscedastic-uncertainty weighting (Kendall, Gal & Cipolla, 2018)
    for ``erp_spectrum_loss``'s slope and peak terms.

    The base MSE term stays a fixed anchor (so the total loss can't be
    trivially driven toward zero by inflating every uncertainty at once);
    the slope/peak terms are each weighted by ``exp(-log_var)`` and pay a
    ``+log_var`` penalty for doing so, so down-weighting a term costs
    something and the model can't get it for free. ``log_var_slope`` and
    ``log_var_peak`` are ``nn.Parameter``s -- include this module's own
    parameters in the optimizer (e.g. ``itertools.chain(model.parameters(),
    weighting.parameters())``) alongside the network's.
    """

    def __init__(
        self,
        init_slope_weight: float = 0.5,
        init_peak_weight: float = 0.05,
        peak_window: int = 7,
    ) -> None:
        super().__init__()
        self.log_var_slope = nn.Parameter(torch.tensor(-math.log(init_slope_weight), dtype=torch.float32))
        self.log_var_peak = nn.Parameter(torch.tensor(-math.log(init_peak_weight), dtype=torch.float32))
        self.peak_window = int(peak_window)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = F.mse_loss(prediction, target)
        if prediction.shape[1] >= 2:
            dp = prediction[:, 1:] - prediction[:, :-1]
            dt = target[:, 1:] - target[:, :-1]
            slope_loss = F.mse_loss(dp, dt)
            loss = loss + torch.exp(-self.log_var_slope) * slope_loss + self.log_var_slope
        peak_loss = _peak_squared_error_sum(prediction, target, window=self.peak_window).mean()
        loss = loss + torch.exp(-self.log_var_peak) * peak_loss + self.log_var_peak
        return loss

    def current_weights(self) -> dict[str, float]:
        return {
            "slope_weight": torch.exp(-self.log_var_slope).item(),
            "peak_weight": torch.exp(-self.log_var_peak).item(),
        }


class DWAWeightedERPLoss:
    """Dynamic Weight Averaging (Liu, Johns & Davison, 2019) for
    ``erp_spectrum_loss``'s slope and peak terms.

    Applied as a per-epoch multiplicative *adjustment factor* on top of
    fixed base weights, rather than replacing them outright -- DWA's
    original formulation assumes the weighted terms start on comparable
    scales, which slope/peak MSE don't (different units), so the base
    weight keeps that scale-appropriate starting point and DWA only
    adjusts the relative *pace* of the two terms epoch to epoch (a term
    whose raw loss is dropping slower than the other gets upweighted).

    Not an ``nn.Module`` -- carries no learnable parameters, only a little
    per-epoch loss-history state. Call it like a loss function inside the
    training step (returns ``(loss, raw_term_losses)``); call
    :meth:`end_epoch` once per epoch with that epoch's mean raw
    (unweighted) slope/peak losses to update next epoch's factors.
    """

    def __init__(
        self,
        temperature: float = 2.0,
        base_slope_weight: float = 0.5,
        base_peak_weight: float = 0.05,
        peak_window: int = 7,
    ) -> None:
        self.temperature = temperature
        self.base_weight = {"slope": base_slope_weight, "peak": base_peak_weight}
        self.factor = {"slope": 1.0, "peak": 1.0}
        self.peak_window = int(peak_window)
        self._history: list[dict[str, float]] = []

    def __call__(self, prediction: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        loss = F.mse_loss(prediction, target)
        raw: dict[str, float] = {}
        if prediction.shape[1] >= 2:
            dp = prediction[:, 1:] - prediction[:, :-1]
            dt = target[:, 1:] - target[:, :-1]
            slope_loss = F.mse_loss(dp, dt)
            loss = loss + self.base_weight["slope"] * self.factor["slope"] * slope_loss
            raw["slope"] = slope_loss.detach().item()
        peak_loss = _peak_squared_error_sum(prediction, target, window=self.peak_window).mean()
        loss = loss + self.base_weight["peak"] * self.factor["peak"] * peak_loss
        raw["peak"] = peak_loss.detach().item()
        return loss, raw

    def end_epoch(self, epoch_avg_raw: Mapping[str, float]) -> None:
        self._history.append(dict(epoch_avg_raw))
        if len(self._history) < 2:
            return
        prev, curr = self._history[-2], self._history[-1]
        ratio = {k: curr[k] / max(prev[k], 1e-8) for k in ("slope", "peak")}
        exp_val = {k: math.exp(ratio[k] / self.temperature) for k in ("slope", "peak")}
        denom = sum(exp_val.values())
        num_terms = len(exp_val)
        for k in ("slope", "peak"):
            self.factor[k] = num_terms * exp_val[k] / denom

    def current_weights(self) -> dict[str, float]:
        return {
            "slope_weight": self.base_weight["slope"] * self.factor["slope"],
            "peak_weight": self.base_weight["peak"] * self.factor["peak"],
        }


def permute_resonators(configuration: torch.Tensor) -> torch.Tensor:
    """Random resonator order per sample, ``(B, num_res, 5) -> (B, num_res, 5)``."""
    b, n, d = configuration.shape
    order = torch.argsort(torch.rand(b, n, device=configuration.device), dim=1)
    return torch.gather(configuration, 1, order[..., None].expand(b, n, d))


def train_operator(
    model: nn.Module,
    loaders: Mapping[str, DataLoader],
    *,
    epochs: int = 200,
    lr: float = 5e-4,
    weight_decay: float = 1e-4,
    slope_weight: float = 0.5,
    peak_weight: float = 0.05,
    lbfgs_epochs: int = 0,
    lbfgs_max_iter: int = 20,
    lbfgs_history_size: int = 10,
    plot: bool = True,
    save_plots: bool = True,
    operator_name: str = "operator",
    plots_dir: str | Path | None = None,
    resume_path: str | Path | None = None,
    save_every: int = 5,
) -> tuple[nn.Module, dict[str, list[float]]]:
    """Train a common-API neural operator and retain best-validation weights.

    ``resume_path`` (optional) makes long runs survive interruptions: every
    ``save_every`` epochs the complete training state (weights, optimizer,
    LR schedule, loss history, best-validation weights) is written there,
    together with ``<resume_path stem>_progress.csv`` and ``_progress.png``
    (loss so far). If the file already exists when training starts, training
    continues from the saved epoch instead of epoch 1. The file is removed
    by the caller once the final checkpoint has been saved.

    ``lbfgs_epochs`` (default 0, disabled) appends a full-batch L-BFGS
    fine-tuning phase after the ordinary AdamW+cosine-schedule loop -- the
    standard "Adam then L-BFGS polish" recipe from physics-informed/
    scientific-ML training. Each of the ``lbfgs_epochs`` outer steps calls
    ``optimizer.step(closure)`` once (internally up to ``lbfgs_max_iter``
    strong-Wolfe line-search iterations against the *entire* training set,
    not a mini-batch), and both phases share the same best-validation-loss
    checkpointing, so the final restored weights are whichever phase
    actually reached the lower validation loss.
    """
    if epochs <= 0:
        raise ValueError("epochs must be positive.")
    if lbfgs_epochs < 0:
        raise ValueError("lbfgs_epochs cannot be negative.")

    model = model.to(device)
    # Set by build_operator_model from model_config["permutation_augment"]:
    # shuffle the resonator order of every training batch, so a model that is
    # not permutation invariant by construction (the plain NN) learns that
    # the order carries no information.
    permutation_augment = bool(getattr(model, "permutation_augment", False))
    if permutation_augment:
        print("Resonator-permutation augmentation: ON (random order every training batch)")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    history = {"train": [], "val": []}
    best_val = math.inf
    best_state = None
    start_epoch = 0

    resume_file = Path(resume_path) if resume_path is not None else None
    if resume_file is not None and resume_file.exists():
        state = torch.load(resume_file, map_location=device, weights_only=False)
        if int(state.get("epochs", -1)) == int(epochs):
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            history = {k: list(v) for k, v in state["history"].items()}
            best_val = float(state["best_val"])
            best_state = state["best_state"]
            start_epoch = int(state["epoch"])
            print(f"Resuming {operator_name} from {resume_file} at epoch {start_epoch + 1}/{epochs}")
        else:
            print(f"Ignoring {resume_file}: it was saved for a run with a different number of epochs.")

    def save_progress(epoch_done: int) -> None:
        resume_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = resume_file.with_name(resume_file.name + ".tmp")
        torch.save(
            {
                "epoch": epoch_done,
                "epochs": int(epochs),
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "history": history,
                "best_val": best_val,
                "best_state": best_state,
            },
            tmp,
        )
        tmp.replace(resume_file)  # atomic: an interruption never leaves a half-written file
        stem = resume_file.name.split(".")[0]
        rows = "\n".join(
            f"{i + 1},{t:.8e},{v:.8e}" for i, (t, v) in enumerate(zip(history["train"], history["val"]))
        )
        (resume_file.parent / f"{stem}_progress.csv").write_text("epoch,train_loss,val_loss\n" + rows + "\n")
        try:
            plot_loss_curves(
                history["train"],
                history["val"],
                title=f"{operator_name} training loss (epoch {epoch_done}/{epochs})",
                ylabel="Normalized loss",
                log_y=True,
                save_path=resume_file.parent / f"{stem}_progress.png",
                show=False,
            )
        except Exception as exc:  # a plotting problem must never stop training
            print(f"(progress plot skipped: {exc})")

    for epoch in range(start_epoch, epochs):
        model.train()
        train_sum = 0.0
        train_count = 0

        for configuration, frequency, target in loaders["train"]:
            configuration = configuration.to(device, dtype=torch.float32, non_blocking=True)
            frequency = frequency.to(device, dtype=torch.float32, non_blocking=True)
            target = target.to(device, dtype=torch.float32, non_blocking=True)

            if permutation_augment:
                configuration = permute_resonators(configuration)

            optimizer.zero_grad(set_to_none=True)
            prediction = model(configuration, frequency)
            loss = erp_spectrum_loss(prediction, target, slope_weight=slope_weight, peak_weight=peak_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            batch = configuration.shape[0]
            train_sum += loss.item() * batch
            train_count += batch

        train_loss = train_sum / max(train_count, 1)
        history["train"].append(train_loss)

        model.eval()
        val_sum = 0.0
        val_count = 0
        with torch.inference_mode():
            for configuration, frequency, target in loaders["val"]:
                configuration = configuration.to(device, dtype=torch.float32, non_blocking=True)
                frequency = frequency.to(device, dtype=torch.float32, non_blocking=True)
                target = target.to(device, dtype=torch.float32, non_blocking=True)
                prediction = model(configuration, frequency)
                loss = erp_spectrum_loss(prediction, target, slope_weight=slope_weight, peak_weight=peak_weight)
                batch = configuration.shape[0]
                val_sum += loss.item() * batch
                val_count += batch

        val_loss = val_sum / max(val_count, 1)
        history["val"].append(val_loss)
        scheduler.step()

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())

        print(
            f"epoch {epoch + 1:4d}/{epochs} | "
            f"train={train_loss:.6e} | val={val_loss:.6e}"
        )
        if resume_file is not None and ((epoch + 1) % max(int(save_every), 1) == 0 or epoch + 1 == epochs):
            save_progress(epoch + 1)

    if lbfgs_epochs > 0:
        lbfgs_optimizer = torch.optim.LBFGS(
            model.parameters(),
            lr=1.0,
            max_iter=lbfgs_max_iter,
            history_size=lbfgs_history_size,
            line_search_fn="strong_wolfe",
        )

        # Each optimizer.step() call below can invoke this closure many times
        # internally (strong-Wolfe line search re-evaluates the *entire*
        # training set repeatedly) before returning. The closure walks the
        # training set in the same mini-batches as the AdamW loop above and
        # accumulates gradients across them: summing per-sample gradients
        # this way is mathematically identical to one gradient computed from
        # a single full-batch forward/backward pass, but keeps only one
        # mini-batch of activations in memory at a time instead of jumping
        # straight from batch_size=16 to the entire dataset in one shot,
        # which was large enough to crash the process outright on a CPU
        # machine before it ever reached the save step. With no output
        # between internal evaluations, one step over a large dataset can
        # still take a while, so progress prints as each evaluation completes.
        closure_calls = 0

        def closure() -> torch.Tensor:
            nonlocal closure_calls
            closure_calls += 1
            lbfgs_optimizer.zero_grad(set_to_none=True)
            loss_sum = 0.0
            sample_count = 0
            for configuration, frequency, target in loaders["train"]:
                configuration = configuration.to(device, dtype=torch.float32)
                frequency = frequency.to(device, dtype=torch.float32)
                target = target.to(device, dtype=torch.float32)
                batch = configuration.shape[0]
                prediction = model(configuration, frequency)
                batch_loss = erp_spectrum_loss(prediction, target, slope_weight=slope_weight, peak_weight=peak_weight)
                (batch_loss * batch).backward()
                loss_sum += batch_loss.item() * batch
                sample_count += batch
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad /= sample_count
            mean_loss = loss_sum / sample_count
            print(
                f"  L-BFGS step {step + 1:4d}/{lbfgs_epochs} "
                f"(evaluating full training set, call {closure_calls:3d}) | "
                f"loss={mean_loss:.6e}",
                end="\r",
                flush=True,
            )
            return torch.tensor(mean_loss, device=device)

        n_train = len(loaders["train"].dataset)
        print(
            f"Starting L-BFGS phase: {lbfgs_epochs} step(s), full training set of "
            f"{n_train} configurations per evaluation (processed in mini-batches "
            f"to limit memory use), up to {lbfgs_max_iter} internal line-search "
            "iterations per step. This can take a while on large datasets -- "
            "progress prints below as each evaluation completes."
        )
        model.train()
        for step in range(lbfgs_epochs):
            closure_calls = 0
            train_loss = float(lbfgs_optimizer.step(closure).detach())
            print()  # end the in-place progress line before the summary below
            history["train"].append(train_loss)

            model.eval()
            val_sum = 0.0
            val_count = 0
            with torch.inference_mode():
                for configuration, frequency, target in loaders["val"]:
                    configuration = configuration.to(device, dtype=torch.float32)
                    frequency = frequency.to(device, dtype=torch.float32)
                    target = target.to(device, dtype=torch.float32)
                    prediction = model(configuration, frequency)
                    batch_loss = erp_spectrum_loss(prediction, target, slope_weight=slope_weight, peak_weight=peak_weight)
                    batch = configuration.shape[0]
                    val_sum += batch_loss.item() * batch
                    val_count += batch
            val_loss = val_sum / max(val_count, 1)
            model.train()
            history["val"].append(val_loss)

            if val_loss < best_val:
                best_val = val_loss
                best_state = copy.deepcopy(model.state_dict())

            print(
                f"L-BFGS step {step + 1:4d}/{lbfgs_epochs} | "
                f"train={train_loss:.6e} | val={val_loss:.6e}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    if save_plots or plot:
        plot_dir = operator_plot_dir(operator_name, plots_dir) if save_plots else None
        plot_loss_curves(
            history["train"],
            history["val"],
            title=f"{operator_name} ERP training loss",
            ylabel="Normalized loss",
            log_y=True,
            save_path=(plot_dir / "loss_curve.png") if plot_dir else None,
            show=plot,
        )

    return model, history


def spectrum_error_diagnostics(
    pred: np.ndarray,
    true: np.ndarray,
    configurations: np.ndarray,
    frequency_values: np.ndarray,
    *,
    band_fraction: float = 0.05,
    stratum_fraction: float = 0.10,
    peak_window: int = 7,
) -> dict[str, float]:
    """Where the test error sits, in dB (all values are RMSEs unless noted).

    * frequency bands: the lowest / highest ``band_fraction`` of the
      frequency axis vs the interior -- shows edge effects such as the FNO's
      periodic FFT wrap;
    * resonance peaks: every local maximum of the TRUE spectrum (the same
      ``spectrum_peak_mask`` the training loss uses) vs everything else;
    * hard-case strata (``stratum_fraction`` of the test configurations
      each): the closest pair of tuning frequencies ``min |f_t,i - f_t,j|``
      and the resonator closest to a plate edge, each vs the rest.

    ``pred``/``true`` are ``(N, n_freq)`` in dB, ``configurations``
    ``(N, num_res, 5)`` in physical units. The strata are defined by
    percentiles of the test set, so every model scored on the same split
    uses the same configurations.
    """
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    config = np.asarray(configurations, dtype=np.float64)
    se = (pred - true) ** 2
    n_freq = se.shape[1]
    band = max(1, int(round(band_fraction * n_freq)))

    def rmse(values: np.ndarray) -> float:
        return float(np.sqrt(values.mean())) if values.size else float("nan")

    peak_mask = (
        spectrum_peak_mask(torch.from_numpy(true[..., None]).float(), window=peak_window)
        .squeeze(-1).numpy().astype(bool)
    )

    f_t = np.sort(config[..., 2], axis=1)
    min_ft_gap = np.diff(f_t, axis=1).min(axis=1) if f_t.shape[1] > 1 else np.full(len(f_t), np.inf)
    x, y = config[..., 3], config[..., 4]
    edge_distance = np.minimum.reduce(
        [x, float(_physics.Lx) - x, y, float(_physics.Ly) - y]
    ).min(axis=1)
    close_ft = min_ft_gap <= np.quantile(min_ft_gap, stratum_fraction)
    near_edge = edge_distance <= np.quantile(edge_distance, stratum_fraction)

    freq = np.asarray(frequency_values, dtype=np.float64)
    return {
        "rmse_low_band_db": rmse(se[:, :band]),
        "rmse_interior_db": rmse(se[:, band:-band]),
        "rmse_high_band_db": rmse(se[:, -band:]),
        "low_band_hz": float(freq[band - 1]),
        "high_band_hz": float(freq[-band]),
        "rmse_at_peaks_db": rmse(se[peak_mask]),
        "rmse_off_peaks_db": rmse(se[~peak_mask]),
        "peaks_per_spectrum_mean": float(peak_mask.sum(axis=1).mean()),
        "rmse_close_ft_db": rmse(se[close_ft]),
        "rmse_not_close_ft_db": rmse(se[~close_ft]),
        "close_ft_threshold_hz": float(np.quantile(min_ft_gap, stratum_fraction)),
        "rmse_near_edge_db": rmse(se[near_edge]),
        "rmse_not_near_edge_db": rmse(se[~near_edge]),
        "near_edge_threshold_m": float(np.quantile(edge_distance, stratum_fraction)),
    }


def evaluate_operator(
    model: nn.Module,
    loaders: Mapping[str, DataLoader],
    norm_params: Mapping[str, object],
    frequency_values: np.ndarray,
    *,
    plot: bool = True,
    num_plot: int = 5,
    save_plots: bool = True,
    operator_name: str = "operator",
    plots_dir: str | Path | None = None,
) -> dict[str, object]:
    """Evaluate complete test spectra in physical ERP units.

    Saves a full-test parity plot and up to ``num_plot`` complete ERP spectrum
    comparisons when ``save_plots=True``. ``plot`` controls display only.
    """
    model = model.to(device)
    model.eval()
    pred_batches: list[np.ndarray] = []
    true_batches: list[np.ndarray] = []
    configuration_batches: list[np.ndarray] = []

    with torch.inference_mode():
        for configuration, frequency, target in loaders["test"]:
            configuration_batches.append(configuration.numpy())
            configuration = configuration.to(device, dtype=torch.float32, non_blocking=True)
            frequency = frequency.to(device, dtype=torch.float32, non_blocking=True)
            prediction = model(configuration, frequency)
            pred_batches.append(prediction.cpu().numpy())
            true_batches.append(target.numpy())

    pred_norm = np.concatenate(pred_batches, axis=0)[..., 0]
    true_norm = np.concatenate(true_batches, axis=0)[..., 0]
    pred = denormalize_erp_array(pred_norm, norm_params)
    true = denormalize_erp_array(true_norm, norm_params)
    configurations = denormalize_configuration_array(
        np.concatenate(configuration_batches, axis=0), norm_params
    )

    error = pred - true
    mse = float(np.mean(error**2))
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(mse))
    mean_error = float(np.mean(error))
    error_std = float(np.std(error))

    # Global agreement statistics across every test-spectrum frequency point.
    true_flat = true.reshape(-1).astype(np.float64, copy=False)
    pred_flat = pred.reshape(-1).astype(np.float64, copy=False)
    true_centered = true_flat - np.mean(true_flat)
    pred_centered = pred_flat - np.mean(pred_flat)
    correlation_denominator = float(
        np.sqrt(np.sum(true_centered**2) * np.sum(pred_centered**2))
    )
    pearson_correlation = (
        float(np.sum(true_centered * pred_centered) / correlation_denominator)
        if correlation_denominator > 0.0
        else float("nan")
    )

    r2_denominator = float(np.sum(true_centered**2))
    r2_score = (
        float(1.0 - np.sum((pred_flat - true_flat) ** 2) / r2_denominator)
        if r2_denominator > 0.0
        else float("nan")
    )

    # Also summarize Pearson correlation spectrum-by-spectrum. This captures
    # whether each individual predicted ERP curve follows the ground-truth
    # spectral shape, rather than only measuring agreement after flattening.
    per_spectrum_pearson: list[float] = []
    for true_spectrum, pred_spectrum in zip(true, pred, strict=True):
        true_spectrum = np.asarray(true_spectrum, dtype=np.float64)
        pred_spectrum = np.asarray(pred_spectrum, dtype=np.float64)
        true_spectrum_centered = true_spectrum - np.mean(true_spectrum)
        pred_spectrum_centered = pred_spectrum - np.mean(pred_spectrum)
        denominator = float(
            np.sqrt(
                np.sum(true_spectrum_centered**2)
                * np.sum(pred_spectrum_centered**2)
            )
        )
        if denominator > 0.0:
            per_spectrum_pearson.append(
                float(
                    np.sum(true_spectrum_centered * pred_spectrum_centered)
                    / denominator
                )
            )

    mean_spectrum_pearson = (
        float(np.mean(per_spectrum_pearson))
        if per_spectrum_pearson
        else float("nan")
    )

    freq = np.asarray(frequency_values, dtype=np.float64)
    true_peak_idx = np.argmax(true, axis=1)
    pred_peak_idx = np.argmax(pred, axis=1)
    peak_frequency_mae = float(
        np.mean(np.abs(freq[pred_peak_idx] - freq[true_peak_idx]))
    )
    true_peak_amp = true[np.arange(true.shape[0]), true_peak_idx]
    pred_at_true_peak = pred[np.arange(pred.shape[0]), true_peak_idx]
    peak_amplitude_mae = float(np.mean(np.abs(pred_at_true_peak - true_peak_amp)))

    print("=" * 68)
    print(f"ERP test RMSE               : {rmse:.6f} dB")
    print(f"ERP test MAE                : {mae:.6f} dB")
    print(f"ERP mean error (bias)       : {mean_error:.6f} dB")
    print(f"ERP error standard deviation: {error_std:.6f} dB")
    print(f"Pearson correlation         : {pearson_correlation:.6f}")
    print(f"Mean spectrum Pearson corr. : {mean_spectrum_pearson:.6f}")
    print(f"R^2 score                   : {r2_score:.6f}")
    print(f"Dominant peak frequency MAE : {peak_frequency_mae:.4f} Hz")
    print(f"ERP error at true peak MAE  : {peak_amplitude_mae:.6f} dB")
    diagnostics = spectrum_error_diagnostics(pred, true, configurations, freq)
    d = diagnostics
    print("-" * 68)
    print(f"RMSE by frequency band      : low {d['rmse_low_band_db']:.3f} | interior "
          f"{d['rmse_interior_db']:.3f} | high {d['rmse_high_band_db']:.3f} dB")
    print(f"RMSE at / off true peaks    : {d['rmse_at_peaks_db']:.3f} / {d['rmse_off_peaks_db']:.3f} dB "
          f"({d['peaks_per_spectrum_mean']:.1f} peaks per spectrum)")
    print(f"RMSE closest-f_t 10% / rest : {d['rmse_close_ft_db']:.3f} / {d['rmse_not_close_ft_db']:.3f} dB "
          f"(min |df_t| <= {d['close_ft_threshold_hz']:.2f} Hz)")
    print(f"RMSE near-edge 10% / rest   : {d['rmse_near_edge_db']:.3f} / {d['rmse_not_near_edge_db']:.3f} dB "
          f"(edge distance <= {d['near_edge_threshold_m']:.3f} m)")
    print("=" * 68)

    plot_dir = operator_plot_dir(operator_name, plots_dir) if save_plots else None
    n_to_plot = min(max(int(num_plot), 0), true.shape[0])

    if save_plots or plot:
        for i in range(n_to_plot):
            plot_erp_comparison(
                freq,
                true[i],
                pred[i],
                title=f"{operator_name} - test configuration {i + 1}",
                configuration=configurations[i],
                save_path=(
                    plot_dir / f"erp_spectrum_test_config_{i + 1:02d}.png"
                    if plot_dir
                    else None
                ),
                show=plot,
            )

        plot_prediction_scatter(
            true,
            pred,
            xlabel="Ground Truth ERP (dB)",
            ylabel="Predicted ERP (dB)",
            title=f"{operator_name} - test ERP prediction vs ground truth",
            save_path=(plot_dir / "prediction_vs_ground_truth.png") if plot_dir else None,
            show=plot,
        )

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "mean_error_db": mean_error,
        "error_std_db": error_std,
        "pearson_correlation": pearson_correlation,
        "mean_spectrum_pearson_correlation": mean_spectrum_pearson,
        "r2_score": r2_score,
        "peak_frequency_mae_hz": peak_frequency_mae,
        "peak_amplitude_mae_db": peak_amplitude_mae,
        "diagnostics": diagnostics,
        "predictions": pred,
        "targets": true,
        "configurations": configurations,
        "plot_directory": str(plot_dir) if plot_dir is not None else None,
    }


# ==================================================
# Prediction and checkpoint helpers
# ==================================================


def predict_erp_spectrum(
    model: nn.Module,
    configuration: np.ndarray,
    norm_params: Mapping[str, object],
    *,
    frequency_values: np.ndarray = freqs,
    compare_solver: bool = True,
    plot: bool = True,
    save_plots: bool = True,
    operator_name: str = "operator",
    plots_dir: str | Path | None = None,
) -> dict[str, object]:
    """Predict ERP for one raw [m,k,f_t,x,y] resonator configuration."""
    configuration = np.asarray(configuration, dtype=np.float32)
    normalized_configuration = normalize_configuration_array(configuration, norm_params)

    frequency_values = np.asarray(frequency_values, dtype=np.float32)
    normalized_frequency = normalize_frequency_array(frequency_values, norm_params)

    config_tensor = torch.from_numpy(normalized_configuration).unsqueeze(0).to(device)
    freq_tensor = torch.from_numpy(normalized_frequency[:, None]).unsqueeze(0).to(device)

    model = model.to(device)
    model.eval()
    with torch.inference_mode():
        pred_norm = model(config_tensor, freq_tensor).cpu().numpy()[0, :, 0]
    prediction = denormalize_erp_array(pred_norm, norm_params)

    result: dict[str, object] = {
        "configuration": configuration,
        "frequencies": frequency_values,
        "prediction": prediction,
    }

    ground_truth = None
    if compare_solver:
        ground_truth = compute_erp_spectrum(
            configuration_to_resonators(configuration),
            frequencies=frequency_values,
        )
        result["ground_truth"] = ground_truth

    if save_plots or plot:
        plot_dir = operator_plot_dir(operator_name, plots_dir) if save_plots else None
        if ground_truth is not None:
            plot_erp_comparison(
                frequency_values,
                ground_truth,
                prediction,
                title=f"{operator_name} - ERP spectrum prediction",
                configuration=configuration,
                save_path=(plot_dir / "prediction_spectrum.png") if plot_dir else None,
                show=plot,
            )
        else:
            # Use the same comparison plot API; when no solver curve exists,
            # no plot is created because a meaningful comparison is unavailable.
            print("Prediction computed; no ground truth requested for comparison plot.")
        result["plot_directory"] = str(plot_dir) if plot_dir is not None else None

    return result


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_operator_checkpoint(
    model: nn.Module,
    *,
    operator_name: str,
    model_config: Mapping[str, object],
    preprocessing_state: Mapping[str, object],
    filename: str | Path,
    training_config: Mapping[str, object] | None = None,
    history: Mapping[str, list[float]] | None = None,
) -> None:
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "operator_name": operator_name,
            "model_config": dict(model_config),
            "model_state_dict": model.state_dict(),
            "preprocessing_state": preprocessing_state,
            # Plate-mode basis (Nx, Ny) of the training data; every later
            # solver calculation made with this model re-applies it.
            "modal_resolution": list(
                preprocessing_state.get("modal_resolution") or _physics.get_modal_resolution()
            ),
            # Hyperparameters + loss history, so every checkpoint documents
            # exactly how it was trained (epochs, lr, loss weights, ...).
            "training_config": dict(training_config or {}),
            "history": {k: [float(v) for v in vals] for k, vals in (history or {}).items()},
        },
        path,
    )
    print(f"Checkpoint saved: {path}")


def load_operator_checkpoint(filename: str) -> dict[str, object]:
    path = Path(filename)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Train the operator first.")
    return torch.load(path, map_location=device, weights_only=False)


# ==================================================
# Common experiment workflow
# ==================================================


def run_operator_experiment(
    *,
    operator_name: str,
    build_model: Callable[..., nn.Module],
    model_config: Mapping[str, object],
    action: str = "train",
    num_configurations: int = 500,
    batch_size: int = 16,
    epochs: int = 200,
    learning_rate: float = 5e-4,
    weight_decay: float = 1e-4,
    slope_weight: float = 0.5,
    peak_weight: float = 0.05,
    lbfgs_epochs: int = 0,
    lbfgs_max_iter: int = 20,
    lbfgs_history_size: int = 10,
    dataset_file: str = DEFAULT_DATASET_FILE,
    regenerate_dataset: bool = False,
    num_generate: int | None = None,
    seed: int = 727,
    num_workers: int = 0,
    checkpoint_file: str | Path | None = None,
    configuration: np.ndarray | None = None,
    plot: bool = True,
    save_plots: bool = True,
    plots_dir: str | Path | None = None,
    num_evaluation_plots: int = 5,
    evaluate_after_training: bool = False,
) -> dict[str, object]:
    """Train, evaluate, or predict with one operator architecture."""
    action = str(action).lower().strip()
    if action not in {"train", "evaluate", "predict"}:
        raise ValueError("action must be 'train', 'evaluate', or 'predict'.")

    seed_everything(seed)
    if checkpoint_file is None:
        # erp_forward_operators/models/GENERAL/<dataset>/<operator>.pth
        checkpoint_file = forward_model_path(operator_name, dataset_tag_for(dataset_file))

    if action == "train":
        # <checkpoint>.resume.pt: mid-training state, saved every 5 epochs
        # (see train_operator); an interrupted run continues from it.
        resume_path = Path(checkpoint_file).with_suffix(".resume.pt")
        dataset, loaders = prepare_operator_data(
            num_configurations=num_configurations,
            batch_size=batch_size,
            dataset_file=dataset_file,
            regenerate_dataset=regenerate_dataset,
            num_generate=num_generate,
            seed=seed,
            num_workers=num_workers,
            verbose=True,
        )
        model = build_operator_model(
            build_model, dataset.num_res, model_config, norm_params=dataset.norm_params
        ).to(device)
        print(f"{operator_name} trainable parameters: {parameter_count(model):,}")
        model, history = train_operator(
            model,
            loaders,
            epochs=epochs,
            lr=learning_rate,
            weight_decay=weight_decay,
            slope_weight=slope_weight,
            peak_weight=peak_weight,
            lbfgs_epochs=lbfgs_epochs,
            lbfgs_max_iter=lbfgs_max_iter,
            lbfgs_history_size=lbfgs_history_size,
            plot=plot,
            save_plots=save_plots,
            operator_name=operator_name,
            plots_dir=plots_dir,
            resume_path=resume_path,
        )
        save_operator_checkpoint(
            model,
            operator_name=operator_name,
            model_config=model_config,
            preprocessing_state=dataset.preprocessing_state(),
            filename=checkpoint_file,
            training_config={
                "optimizer": "AdamW",
                "learning_rate": float(learning_rate),
                "weight_decay": float(weight_decay),
                "lr_schedule": "CosineAnnealingLR(T_max=epochs, eta_min=1e-6)",
                "epochs": int(epochs),
                "batch_size": int(batch_size),
                "grad_clip_norm": 5.0,
                "loss": "MSE + slope_weight*MSE(first difference) + peak_weight*sum(peak sq. error)",
                "slope_weight": float(slope_weight),
                "peak_weight": float(peak_weight),
                "lbfgs_epochs": int(lbfgs_epochs),
                "num_configurations": int(num_configurations),
                "split": "80/10/10 train/val/test by configuration",
                "seed": int(seed),
                "dataset_tag": dataset_tag_for(dataset_file),
            },
            history=history,
        )
        resume_path.unlink(missing_ok=True)  # final checkpoint written; progress state no longer needed

        result: dict[str, object] = {
            "action": action,
            "operator_name": operator_name,
            "model": model,
            "dataset": dataset,
            "loaders": loaders,
            "history": history,
            "checkpoint_file": str(checkpoint_file),
        }
        if evaluate_after_training:
            result["metrics"] = evaluate_operator(
                model,
                loaders,
                dataset.norm_params,
                dataset.frequency_values,
                plot=plot,
                num_plot=num_evaluation_plots,
                save_plots=save_plots,
                operator_name=operator_name,
                plots_dir=plots_dir,
            )
        return result

    checkpoint = load_operator_checkpoint(checkpoint_file)
    saved_config = dict(checkpoint["model_config"])
    preprocessing_state = checkpoint["preprocessing_state"]
    selected_ids = np.asarray(preprocessing_state["selected_source_ids"], dtype=np.int64)

    # Prefer the raw file the checkpoint actually recorded at training time
    # (see ERPDataset.preprocessing_state()) over whatever dataset_file the
    # caller passed -- selected_source_ids only makes sense against the same
    # raw file it was selected from. Checkpoints saved before this field
    # existed have no "dataset_file" key, so .get() falls back to the
    # caller-supplied dataset_file, same as before.
    recorded_dataset_file = preprocessing_state.get("dataset_file")
    effective_dataset_file = (
        recorded_dataset_file if recorded_dataset_file is not None else dataset_file
    )
    if recorded_dataset_file is not None and recorded_dataset_file != dataset_file:
        print(
            f"Note: using this checkpoint's recorded training dataset "
            f"({recorded_dataset_file}) instead of the requested {dataset_file!r}."
        )

    dataset, loaders = prepare_operator_data(
        num_configurations=int(selected_ids.size),
        batch_size=batch_size,
        dataset_file=effective_dataset_file,
        regenerate_dataset=False,
        seed=seed,
        preprocessing_state=preprocessing_state,
        num_workers=num_workers,
        verbose=True,
    )
    try:
        model = build_operator_model(build_model, dataset.num_res, saved_config).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
    except (TypeError, RuntimeError) as exc:
        raise RuntimeError(
            f"Checkpoint {checkpoint_file} does not match the current {operator_name} architecture "
            f"(the model code changed after it was trained). Retrain {operator_name} on this dataset. "
            f"Details: {type(exc).__name__}: {str(exc)[:200]}"
        ) from exc
    model.eval()
    # Solver references for this model use the Nx x Ny it was trained with.
    apply_model_modal_resolution(checkpoint, effective_dataset_file, model_name=operator_name)

    if action == "evaluate":
        metrics = evaluate_operator(
            model,
            loaders,
            dataset.norm_params,
            dataset.frequency_values,
            plot=plot,
            num_plot=num_evaluation_plots,
            save_plots=save_plots,
            operator_name=operator_name,
            plots_dir=plots_dir,
        )
        return {
            "action": action,
            "operator_name": operator_name,
            "model": model,
            "metrics": metrics,
            "frequency_values": np.asarray(dataset.frequency_values).copy(),
            "history": checkpoint.get("history"),
            "dataset": dataset,
        }

    if configuration is None:
        test_id = int(_split_ids(dataset)["test"][0])
        configuration = _configuration_features(dataset)[test_id].copy()
        print("No configuration supplied; using the first test configuration:")
        print(configuration)

    prediction = predict_erp_spectrum(
        model,
        configuration,
        dataset.norm_params,
        frequency_values=dataset.frequency_values,
        compare_solver=True,
        plot=plot,
        save_plots=save_plots,
        operator_name=operator_name,
        plots_dir=plots_dir,
    )
    return {
        "action": action,
        "operator_name": operator_name,
        "model": model,
        "prediction": prediction,
    }


def make_operator_runner(
    *,
    operator_name: str,
    build_model: Callable[..., nn.Module],
    model_config: Mapping[str, object],
    default_epochs: int = 200,
    default_learning_rate: float = 5e-4,
) -> Callable[..., dict[str, object]]:
    """Create the repeated per-architecture ``main`` experiment wrapper once."""

    def runner(
        action: str = "train",
        num_configurations: int = 500,
        batch_size: int = 16,
        epochs: int = default_epochs,
        learning_rate: float = default_learning_rate,
        lbfgs_epochs: int = 0,
        dataset_file: str | Sequence[str] = DEFAULT_DATASET_FILE,
        regenerate_dataset: bool = False,
        seed: int = 727,
        configuration=None,
        plot: bool = True,
        save_plots: bool = True,
        plots_dir: str | Path | None = None,
        num_evaluation_plots: int = 5,
        evaluate_after_training: bool = False,
        checkpoint_file: str | None = None,
        model_config_overrides: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        # e.g. {"use_sorted_branch": True}; saved in the checkpoint's
        # model_config, so evaluate/predict rebuild the same variant.
        config = {**dict(model_config), **dict(model_config_overrides or {})}
        return run_operator_experiment(
            operator_name=operator_name,
            build_model=build_model,
            model_config=config,
            action=action,
            num_configurations=num_configurations,
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate,
            lbfgs_epochs=lbfgs_epochs,
            dataset_file=dataset_file,
            regenerate_dataset=regenerate_dataset,
            seed=seed,
            configuration=configuration,
            checkpoint_file=checkpoint_file,
            plot=plot,
            save_plots=save_plots,
            plots_dir=plots_dir,
            num_evaluation_plots=num_evaluation_plots,
            evaluate_after_training=evaluate_after_training,
        )

    runner.__name__ = "main"
    runner.__doc__ = f"Run train/evaluate/predict workflow for {operator_name}."
    return runner


__all__ = [
    "ACTIVATIONS",
    "resolve_activation",
    "MLP",
    "ResidualMLPBlock",
    "FrequencyRefinement1d",
    "physics_aware_resonator_features",
    "COORDINATE_FEATURE_MODES",
    "modal_features",
    "resonance_detuning",
    "enable_physical_features",
    "set_physical_feature_normalization",
    "BUILD_OPTION_KEYS",
    "build_operator_model",
    "ResonatorSetEncoder",
    "SortedResonatorEncoder",
    "SetAndSortedResonatorEncoder",
    "build_resonator_encoder",
    "ResonanceQueryEncoder",
    "ERPSpectrumDataset",
    "build_spectrum_loaders",
    "prepare_operator_data",
    "erp_spectrum_loss",
    "UncertaintyWeightedERPLoss",
    "DWAWeightedERPLoss",
    "permute_resonators",
    "train_operator",
    "spectrum_error_diagnostics",
    "evaluate_operator",
    "predict_erp_spectrum",
    "parameter_count",
    "save_operator_checkpoint",
    "load_operator_checkpoint",
    "run_operator_experiment",
    "make_operator_runner",
]
