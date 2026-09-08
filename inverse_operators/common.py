"""Shared building blocks for probabilistic inverse-design models.

Every inverse model here answers the same question: given a target ERP
spectrum, what resonator configuration ([m,k,f_t,x,y] per resonator) would
produce it? The forward problem (config -> spectrum) is what operators/
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
from utils.neural_operator_utils import (
    MLP,
    _configuration_features,
    _split_ids,
    prepare_operator_data,
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


class SpectrumEncoder(nn.Module):
    """Encodes a full normalized ERP spectrum into a fixed-size embedding.

    Every inverse model conditions on this same embedding -- a small Conv1d
    stack over the frequency axis (local peak/notch shape) followed by
    mean+max pooling (global summary), matching the "shared tools, same
    information" convention the forward operators already use.
    """

    def __init__(self, width: int = 48, depth: int = 3, embed_dim: int = 96) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Conv1d(1, width, kernel_size=5, padding=2), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Conv1d(width, width, kernel_size=5, padding=2), nn.SiLU()]
        self.conv = nn.Sequential(*layers)
        self.head = MLP([2 * width, embed_dim, embed_dim], activation=nn.SiLU)
        self.embed_dim = embed_dim

    def forward(self, spectrum: torch.Tensor) -> torch.Tensor:
        # spectrum: (B, n_freq) normalized ERP values.
        x = self.conv(spectrum[:, None, :])  # (B, width, n_freq)
        pooled = torch.cat((x.mean(dim=-1), x.amax(dim=-1)), dim=-1)
        return self.head(pooled)


class InverseDesignDataset(Dataset):
    """One item = (normalized ERP spectrum, normalized canonical design vector)."""

    def __init__(self, dataset, configuration_ids: np.ndarray) -> None:
        ids = np.asarray(configuration_ids, dtype=np.int64)
        norm = dataset.norm_params
        raw = canonicalize_by_ft(_configuration_features(dataset)[ids])
        configuration = normalize_configuration_array(raw, norm)
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
    dataset_file: str = "datasets/dataset_erp_ft.pth",
    seed: int = 727,
):
    """Build spectrum->design train/val/test loaders sharing the forward split."""
    dataset, _ = prepare_operator_data(
        num_configurations=num_configurations,
        batch_size=16,
        dataset_file=dataset_file,
        regenerate_dataset=False,
        seed=seed,
    )
    splits = _split_ids(dataset)
    loaders = {
        name: DataLoader(
            InverseDesignDataset(dataset, ids),
            batch_size=batch_size,
            shuffle=(name == "train"),
        )
        for name, ids in splits.items()
    }
    return dataset, loaders


def denormalize_design(flat_design: np.ndarray, num_res: int, norm_params: Mapping[str, object]) -> np.ndarray:
    """Normalized flat design vector(s) -> physical ``(..., num_res, 5)`` [m,k,f_t,x,y]."""
    configuration = flat_design.reshape(*flat_design.shape[:-1], num_res, 5)
    return denormalize_configuration_array(configuration, norm_params)


def save_checkpoint(model: nn.Module, norm_params: Mapping[str, object], filename: str) -> None:
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "norm_params": dict(norm_params)}, path)
    print(f"Checkpoint saved: {path}")
